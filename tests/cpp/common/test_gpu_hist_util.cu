#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/hist_util.h"
#include "gtest/gtest.h"
#include "xgboost/c_api.h"
#include <algorithm>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

namespace xgboost {
namespace common {

void TestDeviceSketch(const GPUSet& devices) {
  // create the data
  int nrows = 10001;
  std::vector<float> test_data(nrows);
  auto count_iter = thrust::make_counting_iterator(0);
  // fill in reverse order
  std::copy(count_iter, count_iter + nrows, test_data.rbegin());

  // create the DMatrix
  DMatrixHandle dmat_handle;
  XGDMatrixCreateFromMat(test_data.data(), nrows, 1, -1,
                         &dmat_handle);
  auto dmat = static_cast<std::shared_ptr<xgboost::DMatrix> *>(dmat_handle);

  // parameters for finding quantiles
  tree::TrainParam p;
  p.max_bin = 20;
  p.gpu_id = 0;
  p.n_gpus = devices.Size();
  // ensure that the exact quantiles are found
  p.gpu_batch_nrows = nrows * 10;

  // find quantiles on the CPU
  HistCutMatrix hmat_cpu;
  hmat_cpu.Init((*dmat).get(), p.max_bin);

  // find the cuts on the GPU
  const SparsePage& batch = *(*dmat)->GetRowBatches().begin();
  HistCutMatrix hmat_gpu;
  DeviceSketch(batch, (*dmat)->Info(), p, &hmat_gpu);

  // compare the cuts
  double eps = 1e-2;
  ASSERT_EQ(hmat_gpu.min_val.size(), 1);
  ASSERT_EQ(hmat_gpu.row_ptr.size(), 2);
  ASSERT_EQ(hmat_gpu.cut.size(), hmat_cpu.cut.size());
  ASSERT_LT(fabs(hmat_cpu.min_val[0] - hmat_gpu.min_val[0]), eps * nrows);
  for (int i = 0; i < hmat_gpu.cut.size(); ++i) {
    ASSERT_LT(fabs(hmat_cpu.cut[i] - hmat_gpu.cut[i]), eps * nrows);
  }

  delete dmat;
}

TEST(gpu_hist_util, DeviceSketch) {
  TestDeviceSketch(GPUSet::Range(0, 1));
}

#if defined(XGBOOST_USE_NCCL)
TEST(gpu_hist_util, MGPU_DeviceSketch) {
  auto devices = GPUSet::AllVisible();
  CHECK_GT(devices.Size(), 1);
  TestDeviceSketch(devices);
}
#endif

}  // namespace common
}  // namespace xgboost
