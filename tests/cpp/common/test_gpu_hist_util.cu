#include <algorithm>
#include <cmath>

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include "xgboost/c_api.h"

#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/hist_util.h"

#include "../helpers.h"

namespace xgboost {
namespace common {

void TestDeviceSketch(const GPUSet& devices, bool use_external_memory) {
  // create the data
  int nrows = 10001;
  std::shared_ptr<xgboost::DMatrix> *dmat = nullptr;

  size_t num_cols = 1;
  if (use_external_memory) {
     auto sp_dmat = CreateSparsePageDMatrix(nrows * 3, 128UL); // 3 entries/row
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

  tree::TrainParam p;
  p.max_bin = 20;
  p.gpu_id = 0;
  p.n_gpus = devices.Size();
  int gpu_batch_nrows = 0;

  // find quantiles on the CPU
  HistCutMatrix hmat_cpu;
  hmat_cpu.Init((*dmat).get(), p.max_bin);

  // find the cuts on the GPU
  HistCutMatrix hmat_gpu;
  (void)DeviceSketch(p, gpu_batch_nrows, dmat->get(), &hmat_gpu);

  // compare the cuts
  double eps = 1e-2;
  ASSERT_EQ(hmat_gpu.min_val.size(), num_cols);
  ASSERT_EQ(hmat_gpu.row_ptr.size(), num_cols + 1);
  ASSERT_EQ(hmat_gpu.cut.size(), hmat_cpu.cut.size());
  ASSERT_LT(fabs(hmat_cpu.min_val[0] - hmat_gpu.min_val[0]), eps * nrows);
  for (int i = 0; i < hmat_gpu.cut.size(); ++i) {
    ASSERT_LT(fabs(hmat_cpu.cut[i] - hmat_gpu.cut[i]), eps * nrows);
  }

  delete dmat;
}

TEST(gpu_hist_util, DeviceSketch) {
  TestDeviceSketch(GPUSet::Range(0, 1), false);
}

TEST(gpu_hist_util, DeviceSketch_ExternalMemory) {
  TestDeviceSketch(GPUSet::Range(0, 1), true);
}

#if defined(XGBOOST_USE_NCCL)
TEST(gpu_hist_util, MGPU_DeviceSketch) {
  auto devices = GPUSet::AllVisible();
  CHECK_GT(devices.Size(), 1);
  TestDeviceSketch(devices, false);
}

TEST(gpu_hist_util, MGPU_DeviceSketch_ExternalMemory) {
  auto devices = GPUSet::AllVisible();
  CHECK_GT(devices.Size(), 1);
  TestDeviceSketch(devices, true);
}
#endif

}  // namespace common
}  // namespace xgboost
