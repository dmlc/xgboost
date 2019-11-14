#include <dmlc/filesystem.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>


#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include "xgboost/c_api.h"

#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/hist_util.h"

#include "../helpers.h"

namespace xgboost {
namespace common {

void TestDeviceSketch(bool use_external_memory) {
  // create the data
  int nrows = 10001;
  std::shared_ptr<xgboost::DMatrix> *dmat = nullptr;

  size_t num_cols = 1;
  dmlc::TemporaryDirectory tmpdir;
  std::string file = tmpdir.path + "/big.libsvm";
  if (use_external_memory) {
    auto sp_dmat = CreateSparsePageDMatrix(nrows * 3, 128UL, file); // 3 entries/row
    dmat = new std::shared_ptr<xgboost::DMatrix>(std::move(sp_dmat));
    num_cols = 5;
  } else {
     std::vector<float> test_data(nrows);
     auto count_iter = thrust::make_counting_iterator(0);
     // fill in reverse order
     std::copy(count_iter, count_iter + nrows, test_data.rbegin());

     // create the DMatrix
     DMatrixHandle dmat_handle;
     XGDMatrixCreateFromMat(test_data.data(), nrows, 1, -1,
                            &dmat_handle);
     dmat = static_cast<std::shared_ptr<xgboost::DMatrix> *>(dmat_handle);
  }

  int device{0};
  int max_bin{20};
  int gpu_batch_nrows{0};

  // find quantiles on the CPU
  HistogramCuts hmat_cpu;
  hmat_cpu.Build((*dmat).get(), max_bin);

  // find the cuts on the GPU
  HistogramCuts hmat_gpu;
  size_t row_stride = DeviceSketch(device, max_bin, gpu_batch_nrows, dmat->get(), &hmat_gpu);

  // compare the row stride with the one obtained from the dmatrix
  bst_row_t expected_row_stride = 0;
  for (const auto &batch : dmat->get()->GetBatches<xgboost::SparsePage>()) {
    const auto &offset_vec = batch.offset.ConstHostVector();
    for (int i = 1; i <= offset_vec.size() -1; ++i) {
      expected_row_stride = std::max(expected_row_stride, offset_vec[i] - offset_vec[i-1]);
    }
  }

  ASSERT_EQ(expected_row_stride, row_stride);

  // compare the cuts
  double eps = 1e-2;
  ASSERT_EQ(hmat_gpu.MinValues().size(), num_cols);
  ASSERT_EQ(hmat_gpu.Ptrs().size(), num_cols + 1);
  ASSERT_EQ(hmat_gpu.Values().size(), hmat_cpu.Values().size());
  ASSERT_LT(fabs(hmat_cpu.MinValues()[0] - hmat_gpu.MinValues()[0]), eps * nrows);
  for (int i = 0; i < hmat_gpu.Values().size(); ++i) {
    ASSERT_LT(fabs(hmat_cpu.Values()[i] - hmat_gpu.Values()[i]), eps * nrows);
  }

  delete dmat;
}

TEST(gpu_hist_util, DeviceSketch) {
  TestDeviceSketch(false);
}

TEST(gpu_hist_util, DeviceSketch_ExternalMemory) {
  TestDeviceSketch(true);
}

}  // namespace common
}  // namespace xgboost
