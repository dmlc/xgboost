/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/generic_parameters.h>

#include "../../../src/common/stats.h"

namespace xgboost {
namespace common {
TEST(Stats, Quantile) {
  {
    linalg::Tensor<float, 1> arr({20.f, 0.f, 15.f, 50.f, 40.f, 0.f, 35.f}, {7}, Context::kCpuId);
    std::vector<size_t> index{0, 2, 3, 4, 6};
    auto h_arr = arr.HostView();
    auto beg = MakeIndexTransformIter([&](size_t i) { return h_arr(index[i]); });
    auto end = beg + index.size();
    auto q = Quantile(0.40f, beg, end);
    ASSERT_EQ(q, 26.0);

    q = Quantile(0.20f, beg, end);
    ASSERT_EQ(q, 16.0);

    q = Quantile(0.10f, beg, end);
    ASSERT_EQ(q, 15.0);
  }

  {
    std::vector<float> vec{1., 2., 3., 4., 5.};
    auto beg = MakeIndexTransformIter([&](size_t i) { return vec[i]; });
    auto end = beg + vec.size();
    auto q = Quantile(0.5f, beg, end);
    ASSERT_EQ(q, 3.);
  }
}

TEST(Stats, WeightedQuantile) {
  linalg::Tensor<float, 1> arr({1.f, 2.f, 3.f, 4.f, 5.f}, {5}, Context::kCpuId);
  linalg::Tensor<float, 1> weight({1.f, 1.f, 1.f, 1.f, 1.f}, {5}, Context::kCpuId);

  auto h_arr = arr.HostView();
  auto h_weight = weight.HostView();

  auto beg = MakeIndexTransformIter([&](size_t i) { return h_arr(i); });
  auto end = beg + arr.Size();
  auto w = MakeIndexTransformIter([&](size_t i) { return h_weight(i); });

  auto q = WeightedQuantile(0.50f, beg, end, w);
  ASSERT_EQ(q, 3);

  q = WeightedQuantile(0.0, beg, end, w);
  ASSERT_EQ(q, 1);

  q = WeightedQuantile(1.0, beg, end, w);
  ASSERT_EQ(q, 5);
}

TEST(Stats, Median) {
  linalg::Tensor<float, 2> values{{.0f, .0f, 1.f, 2.f}, {4}, Context::kCpuId};
  Context ctx;
  HostDeviceVector<float> weights;
  auto m = Median(&ctx, values, weights);
  ASSERT_EQ(m, .5f);

#if defined(XGBOOST_USE_CUDA)
  ctx.gpu_id = 0;
  ASSERT_FALSE(ctx.IsCPU());
  m = Median(&ctx, values, weights);
  ASSERT_EQ(m, .5f);
#endif  // defined(XGBOOST_USE_CUDA)
}
}  // namespace common
}  // namespace xgboost
