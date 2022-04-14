/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <cstddef>

#include "../../../src/common/stats.cuh"
#include "xgboost/base.h"
#include "xgboost/generic_parameters.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/linalg.h"

namespace xgboost {
namespace common {
namespace {
void TestStatsGPUQuantile() {
  linalg::Tensor<float, 1> arr(
      {1.f, 2.f, 3.f, 4.f, 5.f,
       2.f, 4.f, 5.f, 3.f, 1.f},
      {10}, 0);
  linalg::Tensor<size_t, 1> indptr({0, 5, 10}, {3}, 0);
  HostDeviceVector<float> resutls;

  auto d_arr = arr.View(0);
  auto d_key = indptr.View(0);

  auto key_it = dh::MakeTransformIterator<size_t>(
      thrust::make_counting_iterator(0ul), [=] __device__(size_t i) { return d_key(i); });
  auto val_it = dh::MakeTransformIterator<float>(thrust::make_counting_iterator(0ul),
                                                 [=] XGBOOST_DEVICE(size_t i) { return d_arr(i); });

  Context ctx;
  ctx.gpu_id = 0;
  SegmentedQuantile(&ctx, 0.5, key_it, key_it + indptr.Size(), val_it, val_it + arr.Size(),
                    &resutls);

  auto const& h_results = resutls.HostVector();
  ASSERT_EQ(h_results.size(), indptr.Size() - 1);
  ASSERT_EQ(h_results.front(), 3.0f);
  ASSERT_EQ(h_results.back(), 3.0f);
}
}  // anonymous namespace

TEST(Stats, GPUQuantile) { TestStatsGPUQuantile(); }

TEST(Stats, GPUWeightedQuantile) {}
}  // namespace common
}  // namespace xgboost
