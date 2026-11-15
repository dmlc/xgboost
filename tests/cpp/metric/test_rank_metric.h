/**
 * Copyright 2016-2023 by XGBoost Contributors
 */
#pragma once
#include <gtest/gtest.h>      // for Test, EXPECT_NEAR, ASSERT_STREQ
#include <xgboost/context.h>  // for Context
#include <xgboost/data.h>     // for MetaInfo, DMatrix
#include <xgboost/linalg.h>   // for Matrix
#include <xgboost/metric.h>   // for Metric

#include <algorithm>  // for max
#include <cmath>      // for log2
#include <cstddef>    // for size_t
#include <memory>     // for unique_ptr
#include <string>     // for string
#include <vector>     // for vector

#include "../helpers.h"                  // for GetMetricEval, CreateEmptyGe...
#include "xgboost/base.h"                // for bst_float, kRtEps
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/json.h"                // for Json, String, Object

namespace xgboost::metric {

inline void VerifyRankingKernelCache(DeviceOrd device, std::string const& name, double initial,
                                     double reordered) {
  auto ctx = MakeCUDACtx(device.ordinal);
  auto dmat = EmptyDMatrix();
  auto& info = dmat->Info();
  info.num_row_ = 6;
  info.labels = linalg::Matrix<float>{{1, 0, 1, 0, 1, 0}, {6, 1}, device};
  info.group_ptr_ = {0, 3, 6};
  info.weights_.HostVector() = {1, 3};
  HostDeviceVector<float> preds{3, 2, 1, 3, 2, 1};
  std::unique_ptr<Metric> metric{Metric::Create((name + "@2").c_str(), &ctx)};
  EXPECT_NEAR(metric->Evaluate(preds, dmat), initial, 1e-6);
  EXPECT_NEAR(metric->Evaluate(preds, dmat), initial, 1e-6);

  // Reuse the same DMatrix cache with a different prediction order.
  preds.HostVector() = {3, 1, 2, 1, 3, 2};
  EXPECT_NEAR(metric->Evaluate(preds, dmat), reordered, 1e-6);

  // Reloading top-k must reset the cached parameters used by each backend.
  Json config{Object{}};
  metric->SaveConfig(&config);
  config["lambdarank_param"]["lambdarank_num_pair_per_sample"] = String{"1"};
  metric->LoadConfig(config);
  preds.HostVector() = {3, 2, 1, 3, 2, 1};
  EXPECT_NEAR(metric->Evaluate(preds, dmat), 0.25, 1e-6);
}

inline void VerifyRankingCutoff(DeviceOrd device, std::string const& name, double expected) {
  auto ctx = MakeCUDACtx(device.ordinal);
  std::vector<float> labels(64);
  HostDeviceVector<float> preds(64);
  auto& h_preds = preds.HostVector();
  // The first 32 ranked documents are irrelevant; the last 32 are relevant.
  for (std::size_t i = 0; i < labels.size(); ++i) {
    labels[i] = i < 32 ? 0.0f : 1.0f;
    h_preds[i] = static_cast<float>(labels.size() - i);
  }
  for (auto cutoff : {0, 32, 64}) {
    auto metric_name = name;
    if (cutoff != 0) {
      metric_name += "@" + std::to_string(cutoff);
    }
    std::unique_ptr<Metric> metric{Metric::Create(metric_name, &ctx)};
    EXPECT_NEAR(GetMetricEval(metric.get(), preds, labels), cutoff == 32 ? 0.0 : expected, kRtEps)
        << metric_name;
  }
}

inline void VerifyPrecision(DeviceOrd device) {
  VerifyRankingCutoff(device, "pre", 0.5);
  VerifyRankingKernelCache(device, "pre", 0.5, 0.625);
  auto ctx = MakeCUDACtx(device.ordinal);
  std::unique_ptr<xgboost::Metric> metric{Metric::Create("pre", &ctx)};
  ASSERT_STREQ(metric->Name(), "pre");
  EXPECT_NEAR(GetMetricEval(metric.get(), {0, 1}, {0, 1}, {}, {}), 0.5, 1e-7);
  EXPECT_NEAR(GetMetricEval(metric.get(), {0.1f, 0.9f, 0.1f, 0.9f}, {0, 0, 1, 1}, {}, {}), 0.5,
              1e-7);

  metric.reset(xgboost::Metric::Create("pre@2", &ctx));
  ASSERT_STREQ(metric->Name(), "pre@2");
  EXPECT_NEAR(GetMetricEval(metric.get(), {0, 1}, {0, 1}, {}, {}), 0.5f, 1e-7);
  EXPECT_NEAR(GetMetricEval(metric.get(), {0.1f, 0.9f, 0.1f, 0.9f}, {0, 0, 1, 1}, {}, {}), 0.5f,
              0.001f);

  EXPECT_ANY_THROW(GetMetricEval(metric.get(), {0, 1}, {}, {}, {}));

  metric.reset(xgboost::Metric::Create("pre@4", &ctx));
  EXPECT_NEAR(GetMetricEval(metric.get(), {0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f},
                            {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f}, {}, {}),
              0.5f, 1e-7);
}

inline void VerifyNDCG(DeviceOrd device) {
  double dcg{0.0}, idcg{0.0};
  for (std::size_t i = 0; i < 32; ++i) {
    dcg += 1.0 / std::log2(i + 34.0);
    idcg += 1.0 / std::log2(i + 2.0);
  }
  VerifyRankingCutoff(device, "ndcg", dcg / idcg);
  auto discount = 1.0 / std::log2(3.0);
  VerifyRankingKernelCache(device, "ndcg", (1.0 / (1.0 + discount) + 3.0 * discount) / 4.0, 1.0);
  auto ctx = MakeCUDACtx(device.ordinal);
  Metric* metric = xgboost::Metric::Create("ndcg", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg");
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 1}, {}, {}, {}));
  ASSERT_NEAR(GetMetricEval(metric, xgboost::HostDeviceVector<xgboost::bst_float>{}, {}, {}, {}), 1,
              1e-10);
  ASSERT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0.1f, 0.9f, 0.1f, 0.9f}, {0, 0, 1, 1}, {}, {}), 0.6509f,
              0.001f);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@2", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg@2");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0.1f, 0.9f, 0.1f, 0.9f}, {0, 0, 1, 1}, {}, {}), 0.3868f,
              0.001f);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@-", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg-");
  EXPECT_NEAR(GetMetricEval(metric, xgboost::HostDeviceVector<xgboost::bst_float>{}, {}, {}, {}), 0,
              1e-10);
  ASSERT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}), 1.f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0.1f, 0.9f, 0.1f, 0.9f}, {0, 0, 1, 1}, {}, {}), 0.6509f,
              0.001f);
  delete metric;
  metric = xgboost::Metric::Create("ndcg-", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg-");
  EXPECT_NEAR(GetMetricEval(metric, xgboost::HostDeviceVector<xgboost::bst_float>{}, {}, {}, {}), 0,
              1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}), 1.f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0.1f, 0.9f, 0.1f, 0.9f}, {0, 0, 1, 1}, {}, {}), 0.6509f,
              0.001f);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@2-", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg@2-");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}), 1.f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0.1f, 0.9f, 0.1f, 0.9f}, {0, 0, 1, 1}, {}, {}), 1.f - 0.3868f,
              1.f - 0.001f);

  delete metric;
}

inline void VerifyMAP(DeviceOrd device) {
  double ap{0.0};
  for (std::size_t i = 1; i <= 32; ++i) {
    ap += static_cast<double>(i) / (32 + i);
  }
  VerifyRankingCutoff(device, "map", ap / 32.0);
  VerifyRankingKernelCache(device, "map", 0.5, 1.0);
  auto ctx = MakeCUDACtx(device.ordinal);
  Metric* metric = xgboost::Metric::Create("map", &ctx);
  ASSERT_STREQ(metric->Name(), "map");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}), 1, kRtEps);

  EXPECT_NEAR(GetMetricEval(metric, {0.1f, 0.9f, 0.1f, 0.9f}, {0, 0, 1, 1}, {}, {}), 0.5f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric, xgboost::HostDeviceVector<xgboost::bst_float>{},
                            std::vector<xgboost::bst_float>{}, {}, {}),
              1, 1e-10);

  // Rank metric with group info
  EXPECT_NEAR(
      GetMetricEval(metric, {0.1f, 0.9f, 0.2f, 0.8f, 0.4f, 1.7f}, {1, 1, 1, 0, 1, 0},  // Labels
                    {},                                                                // Weights
                    {0, 2, 5, 6}),                                                     // Group info
      0.8611f, 0.001f);

  delete metric;
  metric = xgboost::Metric::Create("map@-", &ctx);
  ASSERT_STREQ(metric->Name(), "map-");
  EXPECT_NEAR(GetMetricEval(metric, xgboost::HostDeviceVector<xgboost::bst_float>{}, {}, {}, {}), 0,
              1e-10);

  delete metric;
  metric = xgboost::Metric::Create("map-", &ctx);
  ASSERT_STREQ(metric->Name(), "map-");
  EXPECT_NEAR(GetMetricEval(metric, xgboost::HostDeviceVector<xgboost::bst_float>{}, {}, {}, {}), 0,
              1e-10);

  delete metric;
  metric = xgboost::Metric::Create("map@2", &ctx);
  ASSERT_STREQ(metric->Name(), "map@2");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0.1f, 0.9f, 0.1f, 0.9f}, {0, 0, 1, 1}, {}, {}), 0.25f, 0.001f);
  delete metric;
}

inline void VerifyNDCGExpGain(DeviceOrd device) {
  Context ctx = MakeCUDACtx(device.ordinal);

  auto p_fmat = xgboost::RandomDataGenerator{0, 0, 0}.GenerateDMatrix();
  MetaInfo& info = p_fmat->Info();
  info.labels = linalg::Matrix<float>{{10.0f, 0.0f, 0.0f, 1.0f, 5.0f}, {5}, ctx.Device()};
  info.num_row_ = info.labels.Shape(0);
  info.group_ptr_.resize(2);
  info.group_ptr_[0] = 0;
  info.group_ptr_[1] = info.num_row_;
  HostDeviceVector<float> predt{{0.1f, 0.2f, 0.3f, 4.0f, 70.0f}};

  std::unique_ptr<Metric> metric{Metric::Create("ndcg", &ctx)};
  Json config{Object{}};
  config["name"] = String{"ndcg"};
  config["lambdarank_param"] = Object{};
  config["lambdarank_param"]["ndcg_exp_gain"] = String{"true"};
  config["lambdarank_param"]["lambdarank_num_pair_per_sample"] = String{"32"};
  metric->LoadConfig(config);

  auto ndcg = metric->Evaluate(predt, p_fmat);
  ASSERT_NEAR(ndcg, 0.409738f, kRtEps);

  config["lambdarank_param"]["ndcg_exp_gain"] = String{"false"};
  metric->LoadConfig(config);

  ndcg = metric->Evaluate(predt, p_fmat);
  ASSERT_NEAR(ndcg, 0.695694f, kRtEps);

  predt.HostVector() = info.labels.Data()->HostVector();
  ndcg = metric->Evaluate(predt, p_fmat);
  ASSERT_NEAR(ndcg, 1.0, kRtEps);
}
}  // namespace xgboost::metric
