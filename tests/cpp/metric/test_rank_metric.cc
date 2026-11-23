/**
 * Copyright 2016-2023, XGBoost Contributors
 */
#include "test_rank_metric.h"

#include <gtest/gtest.h>      // for Test, EXPECT_NEAR, ASSERT_STREQ
#include <xgboost/context.h>  // for Context
#include <xgboost/metric.h>   // for Metric

#include <cmath>
#include <memory>  // for unique_ptr
#include <vector>

#include "../helpers.h"    // for GetMetricEval, CreateEmptyGe...
#include "xgboost/base.h"  // for bst_float, kRtEps

namespace xgboost::metric {
TEST(Metric, RecallHostFallbackMutation) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", DeviceSym::SyclDefault()}});
  auto dmat = EmptyDMatrix();
  auto& info = dmat->Info();
  info.num_row_ = 4;
  info.labels = linalg::Matrix<float>{{0, 0, 1, 0}, {4, 1}, DeviceOrd::CPU()};
  info.group_ptr_ = {0, 4};
  HostDeviceVector<float> preds{4, 3, 2, 1};
  std::unique_ptr<Metric> metric{Metric::Create("recall@1-", &ctx)};
  EXPECT_NEAR(metric->Evaluate(preds, dmat), 0.0, kRtEps);
  info.group_ptr_ = {0, 2, 4};
  EXPECT_NEAR(metric->Evaluate(preds, dmat), 0.5, kRtEps);
  info.labels = linalg::Matrix<float>{{2, 0, 1, 0}, {4, 1}, DeviceOrd::CPU()};
  EXPECT_ANY_THROW(metric->Evaluate(preds, dmat));
  info.labels = linalg::Matrix<float>{{1, 0, 0, 1}, {4, 1}, DeviceOrd::CPU()};
  EXPECT_NEAR(metric->Evaluate(preds, dmat), 0.5, kRtEps);
}

TEST(Metric, AMS) {
  auto ctx = MakeCUDACtx(GPUIDX);
  EXPECT_ANY_THROW(Metric::Create("ams", &ctx));
  std::unique_ptr<Metric> metric{Metric::Create("ams@0.5f", &ctx)};
  ASSERT_STREQ(metric->Name(), "ams@0.5");
  EXPECT_NEAR(GetMetricEval(metric.get(), {0, 1}, {0, 1}), 0.311f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric.get(), {0.1f, 0.9f, 0.1f, 0.9f}, {0, 0, 1, 1}), 0.29710f,
              0.001f);

  metric.reset(Metric::Create("ams@0", &ctx));
  ASSERT_STREQ(metric->Name(), "ams@0");
  EXPECT_NEAR(GetMetricEval(metric.get(), {0, 1}, {0, 1}), 0.311f, 0.001f);
}
TEST(Metric, CoxKernel) {
  std::vector<int> ordinals{-1};
#if defined(XGBOOST_USE_CUDA)
  ordinals.push_back(0);
#endif
  for (auto ordinal : ordinals) {
    auto ctx = MakeCUDACtx(ordinal);
    std::unique_ptr<Metric> metric{Metric::Create("cox-nloglik", &ctx)};
    ASSERT_STREQ(metric->Name(), "cox-nloglik");
    // Tied event/censor times share the risk set; censored rows are not events.
    auto expected = (std::log(5.0) + std::log(7.0 / 4.0)) / 3.0;
    EXPECT_NEAR(GetMetricEval(metric.get(), {2, 1, 4, 3}, {1, -1, 2, 3}), expected, 1e-6);
    EXPECT_NEAR(GetMetricEval(metric.get(), {2, 1, 4, 3}, {1, -1, 2, 3}, {1, 2, 3, 4}), expected,
                1e-6);
  }
}

TEST(Metric, AMSWeightedKernel) {
  std::vector<int> ordinals{-1};
#if defined(XGBOOST_USE_CUDA)
  ordinals.push_back(0);
#endif
  for (auto ordinal : ordinals) {
    auto ctx = MakeCUDACtx(ordinal);
    std::unique_ptr<Metric> metric{Metric::Create("ams@0.5", &ctx)};
    auto expected = std::sqrt(2.0 * (13.0 * std::log(1.3) - 3.0));
    EXPECT_NEAR(GetMetricEval(metric.get(), {0, 1}, {0, 1}, {2, 3}), expected, 1e-6);
  }
}
}  // namespace xgboost::metric
