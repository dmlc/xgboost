/**
 * Copyright 2016-2026, XGBoost contributors
 */
#include <dmlc/registry.h>
#include <xgboost/context.h>
#include <xgboost/linalg.h>
#include <xgboost/metric.h>

#include <memory>
#include <string>
#include <vector>

#include "../helpers.h"
namespace xgboost {
namespace {
bool AcceptsRowWeights(std::string const& name) {
  // Learning-to-rank metrics consume one weight per query. The Cox metric does not consume weights.
  return name != "pre" && name != "map" && name != "ndcg" && name != "cox-nloglik";
}

bool AcceptsQueryWeights(std::string const& name) { return name == "auc" || name == "aucpr"; }

std::unique_ptr<Metric> CreateMetricForTest(Context const* ctx, std::string const& name) {
  auto metric_name = name;
  if (name == "ams") {
    metric_name = "ams@0";
  } else if (name == "tweedie-nloglik") {
    metric_name = "tweedie-nloglik@1.5";
  }

  std::unique_ptr<Metric> metric{Metric::Create(metric_name, ctx)};
  Args args;
  if (name == "quantile") {
    args.emplace_back("quantile_alpha", "0.5");
  } else if (name == "expectile") {
    args.emplace_back("expectile_alpha", "0.5");
  }
  metric->Configure(args);
  return metric;
}

std::shared_ptr<DMatrix> MakeRowWeightData() {
  auto p_fmat = EmptyDMatrix();
  auto& info = p_fmat->Info();
  info.num_row_ = 2;
  info.labels.Reshape(info.num_row_, 1);
  info.labels.Data()->HostVector() = {0.0f, 1.0f};
  info.labels_lower_bound_.HostVector() = {1.0f, 1.0f};
  info.labels_upper_bound_.HostVector() = {1.0f, 1.0f};
  return p_fmat;
}

HostDeviceVector<float> MakePredictions(std::string const& name) {
  auto n_predictions = name == "merror" || name == "mlogloss" ? 4 : 2;
  return HostDeviceVector<float>(n_predictions, 0.5f);
}

void CheckInvalidRowWeights(Context const* ctx, std::string const& name) {
  SCOPED_TRACE(name);
  auto metric = CreateMetricForTest(ctx, name);
  auto p_fmat = MakeRowWeightData();
  auto& info = p_fmat->Info();
  auto predts = MakePredictions(name);

  // The number of row weights must match the number of rows.
  info.weights_.HostVector() = {1.0f};
  EXPECT_THROW(metric->Evaluate(predts, p_fmat), dmlc::Error);

  if (AcceptsQueryWeights(name)) {
    return;
  }

  // Row-wise metrics must not interpret query-group weights as row weights.
  info.weights_.HostVector() = {1.0f, 1.0f};
  info.group_ptr_ = {0, 2};
  EXPECT_THROW(metric->Evaluate(predts, p_fmat), dmlc::Error);
}

void CheckInvalidQuantileShape(Context const* ctx, std::string const& name,
                               std::string const& alpha_param) {
  SCOPED_TRACE(name);
  std::unique_ptr<Metric> metric{Metric::Create(name, ctx)};
  metric->Configure({{alpha_param, "[0.2, 0.8]"}});

  auto p_fmat = EmptyDMatrix();
  auto& info = p_fmat->Info();
  info.num_row_ = 2;
  info.labels.Reshape(1, 2);
  info.labels.Data()->HostVector() = {0.0f, 1.0f};
  HostDeviceVector<float> predts(4, 0.5f);

  EXPECT_THROW(metric->Evaluate(predts, p_fmat), dmlc::Error);

  info.labels.Reshape(2, 1);
  predts.Resize(3);
  EXPECT_THROW(metric->Evaluate(predts, p_fmat), dmlc::Error);
}
}  // namespace

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

TEST(MetricInvalidInput, RowWeights) {
  auto ctx = MakeCUDACtx(GPUIDX);
  for (auto const* entry : dmlc::Registry<MetricReg>::List()) {
    if (AcceptsRowWeights(entry->name)) {
      CheckInvalidRowWeights(&ctx, entry->name);
    }
  }
}

TEST(MetricInvalidInput, QuantileShape) {
  auto ctx = MakeCUDACtx(GPUIDX);
  CheckInvalidQuantileShape(&ctx, "quantile", "quantile_alpha");
  CheckInvalidQuantileShape(&ctx, "expectile", "expectile_alpha");
}

TEST(MetricInvalidInput, MultiTargetLabels) {
  auto ctx = MakeCUDACtx(GPUIDX);
  linalg::Tensor<float, 2> labels{{0.0f, 1.0f, 0.0f, 1.0f}, {2, 2}, DeviceOrd::CPU()};

  for (auto const& name : {"merror", "mlogloss"}) {
    SCOPED_TRACE(name);
    std::unique_ptr<Metric> metric{Metric::Create(name, &ctx)};
    HostDeviceVector<float> predts(8, 0.5f);
    ASSERT_THAT([&] { GetMultiMetricEval(metric.get(), predts, labels); },
                GMockThrow("multi-target"));
  }

  std::unique_ptr<Metric> rank_metric{Metric::Create("ndcg", &ctx)};
  HostDeviceVector<float> rank_predts(4, 0.5f);
  ASSERT_THAT([&] { GetMultiMetricEval(rank_metric.get(), rank_predts, labels); },
              GMockThrow("multi-target"));
}
}  // namespace xgboost
