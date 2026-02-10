// Copyright by Contributors
#include <xgboost/metric.h>

#include "../helpers.h"
namespace xgboost {
TEST(Metric, UnknownMetric) {
  auto ctx = MakeCUDACtx(GPUIDX);
  xgboost::Metric* metric = nullptr;
  EXPECT_ANY_THROW(metric = xgboost::Metric::Create("unknown_name", &ctx));
  EXPECT_NO_THROW(metric = xgboost::Metric::Create("rmse", &ctx));
  delete metric;
  metric = nullptr;
  EXPECT_ANY_THROW(metric = xgboost::Metric::Create("unknown_name@1", &ctx));
  EXPECT_NO_THROW(metric = xgboost::Metric::Create("error@0.5f", &ctx));
  delete metric;
}

TEST(Metric, ExpectileLoadConfig) {
  auto ctx = MakeCUDACtx(GPUIDX);
  std::unique_ptr<xgboost::Metric> metric{xgboost::Metric::Create("expectile", &ctx)};
  metric->Configure({{"expectile_alpha", "0.8"}});
  Json config{Object{}};
  metric->SaveConfig(&config);

  std::unique_ptr<xgboost::Metric> loaded{xgboost::Metric::Create("expectile", &ctx)};
  loaded->LoadConfig(config);

  xgboost::HostDeviceVector<float> preds;
  preds.HostVector() = {0.1f, 0.9f};
  auto result = GetMetricEval(loaded.get(), preds, {0.0f, 1.0f}, {}, {}, DataSplitMode::kRow);
  EXPECT_NEAR(result, 0.005f, 1e-6f);
}
}  // namespace xgboost
