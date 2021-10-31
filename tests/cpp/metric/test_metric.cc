// Copyright by Contributors
#include <xgboost/metric.h>
#include <xgboost/task.h>

#include "../helpers.h"

TEST(Metric, UnknownMetric) {
  auto tparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = nullptr;
  using xgboost::ObjInfo;
  EXPECT_ANY_THROW(metric =
                       xgboost::Metric::Create("unknown_name", &tparam, {ObjInfo::kOther, true}));
  EXPECT_NO_THROW(metric = xgboost::Metric::Create("rmse", &tparam, {ObjInfo::kOther, true}));
  if (metric) {
    delete metric;
  }
  metric = nullptr;
  EXPECT_ANY_THROW(metric =
                       xgboost::Metric::Create("unknown_name@1", &tparam, {ObjInfo::kOther, true}));
  EXPECT_NO_THROW(metric = xgboost::Metric::Create("error@0.5f", &tparam, {ObjInfo::kOther, true}));
  if (metric) {
    delete metric;
  }
}
