/**
 * Copyright 2018-2023 by XGBoost contributors
 */
#pragma once
#include <xgboost/json.h>
#include <xgboost/metric.h>

#include <map>
#include <memory>

#include "../../../src/common/linalg_op.h"
#include "../helpers.h"

namespace xgboost {
namespace metric {

inline void CheckDeterministicMetricElementWise(StringView name, int32_t device) {
  auto ctx = CreateEmptyGenericParam(device);
  std::unique_ptr<Metric> metric{Metric::Create(name.c_str(), &ctx)};

  HostDeviceVector<float> predts;
  size_t n_samples = 2048;

  auto p_fmat = EmptyDMatrix();
  MetaInfo& info = p_fmat->Info();
  info.labels.Reshape(n_samples, 1);
  info.num_row_ = n_samples;
  auto &h_labels = info.labels.Data()->HostVector();
  auto &h_predts = predts.HostVector();

  SimpleLCG lcg;
  SimpleRealUniformDistribution<float> dist{0.0f, 1.0f};

  h_labels.resize(n_samples);
  h_predts.resize(n_samples);

  for (size_t i = 0; i < n_samples; ++i) {
    h_predts[i] = dist(&lcg);
    h_labels[i] = dist(&lcg);
  }

  auto result = metric->Evaluate(predts, p_fmat);
  for (size_t i = 0; i < 8; ++i) {
    ASSERT_EQ(metric->Evaluate(predts, p_fmat), result);
  }
}

inline void VerifyRMSE(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("rmse", &ctx);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "rmse");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
              0.6403f, 0.001f);
  auto expected = 2.8284f;
  if (collective::IsDistributed() && data_split_mode == DataSplitMode::kRow) {
    expected = sqrt(8.0f * collective::GetWorldSize());
  }
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            { -1,   1,   9,  -9}, {}, data_split_mode),
              expected, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            {  1,   2,   9,   8}, {}, data_split_mode),
              0.6708f, 0.001f);
  delete metric;

  CheckDeterministicMetricElementWise(StringView{"rmse"}, GPUIDX);
}

inline void VerifyRMSLE(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("rmsle", &ctx);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "rmsle");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.2f, 0.4f, 0.8f, 1.6f},
                            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {}, {}, data_split_mode),
              0.4063f, 1e-4);
  auto expected = 0.6212f;
  if (collective::IsDistributed() && data_split_mode == DataSplitMode::kRow) {
    expected = sqrt(0.3859f * collective::GetWorldSize());
  }
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.2f, 0.4f, 0.8f, 1.6f},
                            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                            {   0,   -1,    1,    -9,   9}, {}, data_split_mode),
              expected, 1e-4);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.2f, 0.4f, 0.8f, 1.6f},
                            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                            {   0,    1,    2,    9,    8}, {}, data_split_mode),
              0.2415f, 1e-4);
  delete metric;

  CheckDeterministicMetricElementWise(StringView{"rmsle"}, GPUIDX);
}

inline void VerifyMAE(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("mae", &ctx);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "mae");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
              0.5f, 0.001f);
  auto expected = 8.0f;
  if (collective::IsDistributed() && data_split_mode == DataSplitMode::kRow) {
    expected *= collective::GetWorldSize();
  }
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            { -1,   1,   9,  -9}, {}, data_split_mode),
              expected, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            {  1,   2,   9,   8}, {}, data_split_mode),
              0.54f, 0.001f);
  delete metric;

  CheckDeterministicMetricElementWise(StringView{"mae"}, GPUIDX);
}

inline void VerifyMAPE(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("mape", &ctx);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "mape");
  EXPECT_NEAR(GetMetricEval(metric, {150, 300}, {100, 200}, {}, {}, data_split_mode), 0.5f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {50, 400, 500, 4000},
                            {100, 200, 500, 1000}, {}, {}, data_split_mode),
              1.125f, 0.001f);
  auto expected = -26.5f;
  if (collective::IsDistributed() && data_split_mode == DataSplitMode::kRow) {
    expected *= collective::GetWorldSize();
  }
  EXPECT_NEAR(GetMetricEval(metric,
                            {50, 400, 500, 4000},
                            {100, 200, 500, 1000},
                            { -1,   1,   9,  -9}, {}, data_split_mode),
              expected, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {50, 400, 500, 4000},
                            {100, 200, 500, 1000},
                            {  1,   2,   9,   8}, {}, data_split_mode),
              1.3250f, 0.001f);
  delete metric;

  CheckDeterministicMetricElementWise(StringView{"mape"}, GPUIDX);
}

inline void VerifyMPHE(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<xgboost::Metric> metric{xgboost::Metric::Create("mphe", &ctx)};
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "mphe");
  EXPECT_NEAR(GetMetricEval(metric.get(), {0, 1}, {0, 1}, {}, {}, data_split_mode), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric.get(),
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
              0.1751f, 1e-4);
  auto expected = 3.40375f;
  if (collective::IsDistributed() && data_split_mode == DataSplitMode::kRow) {
    expected *= collective::GetWorldSize();
  }
  EXPECT_NEAR(GetMetricEval(metric.get(),
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            { -1,   1,   9,  -9}, {}, data_split_mode),
              expected, 1e-4);
  EXPECT_NEAR(GetMetricEval(metric.get(),
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            {  1,   2,   9,   8}, {}, data_split_mode),
              0.1922f, 1e-4);

  CheckDeterministicMetricElementWise(StringView{"mphe"}, GPUIDX);

  metric->Configure({{"huber_slope", "0.1"}});
  EXPECT_NEAR(GetMetricEval(metric.get(),
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            {  1,   2,   9,   8}, {}, data_split_mode),
              0.0461686f, 1e-4);
}

inline void VerifyLogLoss(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("logloss", &ctx);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "logloss");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.5f, 1e-17f, 1.0f+1e-17f, 0.9f},
                            {   0,      0,           1,    1}, {}, {}, data_split_mode),
              0.1996f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
              1.2039f, 0.001f);
  auto expected = 21.9722f;
  if (collective::IsDistributed() && data_split_mode == DataSplitMode::kRow) {
    expected *= collective::GetWorldSize();
  }
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            { -1,   1,   9,  -9}, {}, data_split_mode),
              expected, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            {  1,   2,   9,   8}, {}, data_split_mode),
              1.3138f, 0.001f);
  delete metric;

  CheckDeterministicMetricElementWise(StringView{"logloss"}, GPUIDX);
}

inline void VerifyError(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("error", &ctx);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "error");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
              0.5f, 0.001f);
  auto expected = 10.0f;
  if (collective::IsDistributed() && data_split_mode == DataSplitMode::kRow) {
    expected *= collective::GetWorldSize();
  }
  EXPECT_NEAR(GetMetricEval(metric,
                           {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            { -1,   1,   9,  -9}, {}, data_split_mode),
              expected, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            {  1,   2,   9,   8}, {}, data_split_mode),
              0.55f, 0.001f);

  EXPECT_ANY_THROW(xgboost::Metric::Create("error@abc", &ctx));
  delete metric;

  metric = xgboost::Metric::Create("error@0.5f", &ctx);
  metric->Configure({});
  EXPECT_STREQ(metric->Name(), "error");

  delete metric;

  metric = xgboost::Metric::Create("error@0.1", &ctx);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "error@0.1");
  EXPECT_STREQ(metric->Name(), "error@0.1");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {-0.1f, -0.9f, 0.1f, 0.9f},
                            {   0,    0,   1,   1}, {}, {}, data_split_mode),
              0.25f, 0.001f);
  expected = 9.0f;
  if (collective::IsDistributed() && data_split_mode == DataSplitMode::kRow) {
    expected *= collective::GetWorldSize();
  }
  EXPECT_NEAR(GetMetricEval(metric,
                            {-0.1f, -0.9f, 0.1f, 0.9f},
                            {   0,    0,   1,   1},
                            { -1,   1,   9,  -9}, {}, data_split_mode),
              expected, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {-0.1f, -0.9f, 0.1f, 0.9f},
                            {   0,    0,   1,   1},
                            {  1,   2,   9,   8}, {}, data_split_mode),
              0.45f, 0.001f);
  delete metric;

  CheckDeterministicMetricElementWise(StringView{"error@0.5"}, GPUIDX);
}

inline void VerifyPoissonNegLogLik(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("poisson-nloglik", &ctx);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "poisson-nloglik");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 0.5f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.5f, 1e-17f, 1.0f+1e-17f, 0.9f},
                            {   0,      0,           1,    1}, {}, {}, data_split_mode),
              0.6263f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
              1.1019f, 0.001f);
  auto expected = 13.3750f;
  if (collective::IsDistributed() && data_split_mode == DataSplitMode::kRow) {
    expected *= collective::GetWorldSize();
  }
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            { -1,   1,   9,  -9}, {}, data_split_mode),
              expected, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            {  1,   2,   9,   8}, {}, data_split_mode),
              1.5783f, 0.001f);
  delete metric;

  CheckDeterministicMetricElementWise(StringView{"poisson-nloglik"}, GPUIDX);
}

inline void VerifyMultiRMSE(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  size_t n_samples = 32, n_targets = 8;
  linalg::Tensor<float, 2> y{{n_samples, n_targets}, GPUIDX};
  auto &h_y = y.Data()->HostVector();
  std::iota(h_y.begin(), h_y.end(), 0);

  HostDeviceVector<float> predt(n_samples * n_targets, 0);

  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<Metric> metric{Metric::Create("rmse", &ctx)};
  metric->Configure({});

  auto loss = GetMultiMetricEval(metric.get(), predt, y, {}, {}, data_split_mode);
  std::vector<float> weights(n_samples, 1);
  auto loss_w = GetMultiMetricEval(metric.get(), predt, y, weights, {}, data_split_mode);

  std::transform(h_y.cbegin(), h_y.cend(), h_y.begin(), [](auto &v) { return v * v; });
  auto ret = std::sqrt(std::accumulate(h_y.cbegin(), h_y.cend(), 1.0, std::plus<>{}) / h_y.size());
  ASSERT_FLOAT_EQ(ret, loss);
  ASSERT_FLOAT_EQ(ret, loss_w);
}

inline void VerifyQuantile(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<Metric> metric{Metric::Create("quantile", &ctx)};

  HostDeviceVector<float> predts{0.1f, 0.9f, 0.1f, 0.9f};
  std::vector<float> labels{0.5f, 0.5f, 0.9f, 0.1f};
  std::vector<float> weights{0.2f, 0.4f, 0.6f, 0.8f};

  metric->Configure(Args{{"quantile_alpha", "[0.0]"}});
  EXPECT_NEAR(GetMetricEval(metric.get(), predts, labels, weights, {}, data_split_mode), 0.400f,
              0.001f);
  metric->Configure(Args{{"quantile_alpha", "[0.2]"}});
  EXPECT_NEAR(GetMetricEval(metric.get(), predts, labels, weights, {}, data_split_mode), 0.376f,
              0.001f);
  metric->Configure(Args{{"quantile_alpha", "[0.4]"}});
  EXPECT_NEAR(GetMetricEval(metric.get(), predts, labels, weights, {}, data_split_mode), 0.352f,
              0.001f);
  metric->Configure(Args{{"quantile_alpha", "[0.8]"}});
  EXPECT_NEAR(GetMetricEval(metric.get(), predts, labels, weights, {}, data_split_mode), 0.304f,
              0.001f);
  metric->Configure(Args{{"quantile_alpha", "[1.0]"}});
  EXPECT_NEAR(GetMetricEval(metric.get(), predts, labels, weights, {}, data_split_mode), 0.28f,
              0.001f);

  metric->Configure(Args{{"quantile_alpha", "[0.0]"}});
  EXPECT_NEAR(GetMetricEval(metric.get(), predts, labels, {}, {}, data_split_mode), 0.3f, 0.001f);
  metric->Configure(Args{{"quantile_alpha", "[0.2]"}});
  EXPECT_NEAR(GetMetricEval(metric.get(), predts, labels, {}, {}, data_split_mode), 0.3f, 0.001f);
  metric->Configure(Args{{"quantile_alpha", "[0.4]"}});
  EXPECT_NEAR(GetMetricEval(metric.get(), predts, labels, {}, {}, data_split_mode), 0.3f, 0.001f);
  metric->Configure(Args{{"quantile_alpha", "[0.8]"}});
  EXPECT_NEAR(GetMetricEval(metric.get(), predts, labels, {}, {}, data_split_mode), 0.3f, 0.001f);
  metric->Configure(Args{{"quantile_alpha", "[1.0]"}});
  EXPECT_NEAR(GetMetricEval(metric.get(), predts, labels, {}, {}, data_split_mode), 0.3f, 0.001f);
}
}  // namespace metric
}  // namespace xgboost
