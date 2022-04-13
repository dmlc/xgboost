/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <cstddef>

#include "../../../src/common/stats.cuh"
#include "xgboost/host_device_vector.h"
namespace xgboost {
namespace common {
TEST(Stats, GPUQuantile) {
  linalg::Tensor<float, 1> arr({20.f, 0.f, 15.f, 50.f, 40.f, 0.f, 35.f}, {7}, Context::kCpuId);
  HostDeviceVector<float> resutls;

  auto d_arr = arr.View(0);
  auto val_it = dh::MakeTransformIterator<float>(thrust::make_counting_iterator(0ul),
                                                 [=](size_t i) { return d_arr(i); });
}

TEST(Stats, GPUWeightedQuantile) {}
}  // namespace common
}  // namespace xgboost
