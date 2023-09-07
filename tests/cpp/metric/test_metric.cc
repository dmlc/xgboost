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
}  // namespace xgboost
