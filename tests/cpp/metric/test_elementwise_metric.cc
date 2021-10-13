/*!
 * Copyright 2018-2019 XGBoost contributors
 */
#include <xgboost/metric.h>
#include <xgboost/json.h>

#include <map>
#include <memory>

#include "../helpers.h"

namespace xgboost {
namespace {
inline void CheckDeterministicMetricElementWise(StringView name, int32_t device) {
  auto lparam = CreateEmptyGenericParam(device);
  std::unique_ptr<Metric> metric{Metric::Create(name.c_str(), &lparam)};

  HostDeviceVector<float> predts;
  MetaInfo info;
  auto &h_labels = info.labels_.HostVector();
  auto &h_predts = predts.HostVector();

  SimpleLCG lcg;
  SimpleRealUniformDistribution<float> dist{0.0f, 1.0f};

  size_t n_samples = 2048;
  h_labels.resize(n_samples);
  h_predts.resize(n_samples);

  for (size_t i = 0; i < n_samples; ++i) {
    h_predts[i] = dist(&lcg);
    h_labels[i] = dist(&lcg);
  }

  auto result = metric->Eval(predts, info, false);
  for (size_t i = 0; i < 8; ++i) {
    ASSERT_EQ(metric->Eval(predts, info, false), result);
  }
}
}  // anonymous namespace
}  // namespace xgboost

TEST(Metric, DeclareUnifiedTest(RMSE)) {
  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("rmse", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "rmse");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.6403f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            { -1,   1,   9,  -9}),
              2.8284f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            {  1,   2,   9,   8}),
              0.6708f, 0.001f);
  delete metric;

  xgboost::CheckDeterministicMetricElementWise(xgboost::StringView{"rmse"}, GPUIDX);
}

TEST(Metric, DeclareUnifiedTest(RMSLE)) {
  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("rmsle", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "rmsle");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.2f, 0.4f, 0.8f, 1.6f},
                            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}),
              0.4063f, 1e-4);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.2f, 0.4f, 0.8f, 1.6f},
                            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                            {   0,   -1,    1,    -9,   9}),
              0.6212f, 1e-4);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.2f, 0.4f, 0.8f, 1.6f},
                            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                            {   0,    1,    2,    9,    8}),
              0.2415f, 1e-4);
  delete metric;

  xgboost::CheckDeterministicMetricElementWise(xgboost::StringView{"rmsle"}, GPUIDX);
}

TEST(Metric, DeclareUnifiedTest(MAE)) {
  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("mae", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "mae");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.5f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            { -1,   1,   9,  -9}),
              8.0f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            {  1,   2,   9,   8}),
              0.54f, 0.001f);
  delete metric;

  xgboost::CheckDeterministicMetricElementWise(xgboost::StringView{"mae"}, GPUIDX);
}

TEST(Metric, DeclareUnifiedTest(MAPE)) {
  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("mape", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "mape");
  EXPECT_NEAR(GetMetricEval(metric, {150, 300}, {100, 200}), 0.5f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {50, 400, 500, 4000},
                            {100, 200, 500, 1000}),
              1.125f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {50, 400, 500, 4000},
                            {100, 200, 500, 1000},
                            { -1,   1,   9,  -9}),
              -26.5f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {50, 400, 500, 4000},
                            {100, 200, 500, 1000},
                            {  1,   2,   9,   8}),
              1.3250f, 0.001f);
  delete metric;

  xgboost::CheckDeterministicMetricElementWise(xgboost::StringView{"mape"}, GPUIDX);
}

TEST(Metric, DeclareUnifiedTest(MPHE)) {
  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("mphe", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "mphe");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.1751f, 1e-4);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            { -1,   1,   9,  -9}),
              3.4037f, 1e-4);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            {  1,   2,   9,   8}),
              0.1922f, 1e-4);
  delete metric;

  xgboost::CheckDeterministicMetricElementWise(xgboost::StringView{"mphe"}, GPUIDX);
}

TEST(Metric, DeclareUnifiedTest(LogLoss)) {
  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("logloss", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "logloss");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.5f, 1e-17f, 1.0f+1e-17f, 0.9f},
                            {   0,      0,           1,    1}),
              0.1996f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              1.2039f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            { -1,   1,   9,  -9}),
              21.9722f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            {  1,   2,   9,   8}),
              1.3138f, 0.001f);
  delete metric;

  xgboost::CheckDeterministicMetricElementWise(xgboost::StringView{"logloss"}, GPUIDX);
}

TEST(Metric, DeclareUnifiedTest(Error)) {
  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("error", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "error");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.5f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                           {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            { -1,   1,   9,  -9}),
              10.0f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            {  1,   2,   9,   8}),
              0.55f, 0.001f);

  EXPECT_ANY_THROW(xgboost::Metric::Create("error@abc", &lparam));
  delete metric;

  metric = xgboost::Metric::Create("error@0.5f", &lparam);
  metric->Configure({});
  EXPECT_STREQ(metric->Name(), "error");

  delete metric;

  metric = xgboost::Metric::Create("error@0.1", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "error@0.1");
  EXPECT_STREQ(metric->Name(), "error@0.1");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {-0.1f, -0.9f, 0.1f, 0.9f},
                            {   0,    0,   1,   1}),
              0.25f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {-0.1f, -0.9f, 0.1f, 0.9f},
                            {   0,    0,   1,   1},
                            { -1,   1,   9,  -9}),
              9.0f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {-0.1f, -0.9f, 0.1f, 0.9f},
                            {   0,    0,   1,   1},
                            {  1,   2,   9,   8}),
              0.45f, 0.001f);
  delete metric;

  xgboost::CheckDeterministicMetricElementWise(xgboost::StringView{"error@0.5"}, GPUIDX);
}

TEST(Metric, DeclareUnifiedTest(PoissionNegLogLik)) {
  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("poisson-nloglik", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "poisson-nloglik");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0.5f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.5f, 1e-17f, 1.0f+1e-17f, 0.9f},
                            {   0,      0,           1,    1}),
              0.6263f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              1.1019f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            { -1,   1,   9,  -9}),
              13.3750f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            {  1,   2,   9,   8}),
              1.5783f, 0.001f);
  delete metric;

  xgboost::CheckDeterministicMetricElementWise(xgboost::StringView{"mphe"}, GPUIDX);
}
