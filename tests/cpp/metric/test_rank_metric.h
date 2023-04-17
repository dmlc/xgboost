/**
 * Copyright 2016-2023 by XGBoost Contributors
 */
#pragma once
#include <gtest/gtest.h>                 // for Test, EXPECT_NEAR, ASSERT_STREQ
#include <xgboost/context.h>             // for Context
#include <xgboost/data.h>                // for MetaInfo, DMatrix
#include <xgboost/linalg.h>              // for Matrix
#include <xgboost/metric.h>              // for Metric

#include <algorithm>                     // for max
#include <memory>                        // for unique_ptr
#include <vector>                        // for vector

#include "../helpers.h"                  // for GetMetricEval, CreateEmptyGe...
#include "xgboost/base.h"                // for bst_float, kRtEps
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/json.h"                // for Json, String, Object

namespace xgboost {
namespace metric {

inline void VerifyPrecision(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  // When the limit for precision is not given, it takes the limit at
  // std::numeric_limits<unsigned>::max(); hence all values are very small
  // NOTE(AbdealiJK): Maybe this should be fixed to be num_row by default.
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("pre", &ctx);
  ASSERT_STREQ(metric->Name(), "pre");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 0, 1e-7);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
              0, 1e-7);

  delete metric;
  metric = xgboost::Metric::Create("pre@2", &ctx);
  ASSERT_STREQ(metric->Name(), "pre@2");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 0.5f, 1e-7);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
              0.5f, 0.001f);

  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 1}, {}, {}, {}, data_split_mode));

  delete metric;
}

inline void VerifyNDCG(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  auto ctx = CreateEmptyGenericParam(GPUIDX);
  Metric * metric = xgboost::Metric::Create("ndcg", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg");
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 1}, {}, {}, {}, data_split_mode));
  ASSERT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            {}, {}, {}, data_split_mode), 1, 1e-10);
  ASSERT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
              0.6509f, 0.001f);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@2", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg@2");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
              0.3868f, 0.001f);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@-", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg-");
  EXPECT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            {}, {}, {}, data_split_mode), 0, 1e-10);
  ASSERT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 1.f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
              0.6509f, 0.001f);
  delete metric;
  metric = xgboost::Metric::Create("ndcg-", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg-");
  EXPECT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            {}, {}, {}, data_split_mode), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 1.f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
               0.6509f, 0.001f);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@2-", &ctx);
  ASSERT_STREQ(metric->Name(), "ndcg@2-");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 1.f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
              1.f - 0.3868f, 1.f - 0.001f);

  delete metric;
}

inline void VerifyMAP(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);
  Metric * metric = xgboost::Metric::Create("map", &ctx);
  ASSERT_STREQ(metric->Name(), "map");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 1, kRtEps);

  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
              0.5f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            std::vector<xgboost::bst_float>{}, {}, {}, data_split_mode), 1, 1e-10);

  // Rank metric with group info
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.2f, 0.8f, 0.4f, 1.7f},
                            {1, 1, 1, 0, 1, 0},  // Labels
                            {},  // Weights
                            {0, 2, 5, 6},  // Group info
                            data_split_mode),
              0.8611f, 0.001f);

  delete metric;
  metric = xgboost::Metric::Create("map@-", &ctx);
  ASSERT_STREQ(metric->Name(), "map-");
  EXPECT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            {}, {}, {}, data_split_mode), 0, 1e-10);

  delete metric;
  metric = xgboost::Metric::Create("map-", &ctx);
  ASSERT_STREQ(metric->Name(), "map-");
  EXPECT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            {}, {}, {}, data_split_mode), 0, 1e-10);

  delete metric;
  metric = xgboost::Metric::Create("map@2", &ctx);
  ASSERT_STREQ(metric->Name(), "map@2");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}, {}, {}, data_split_mode), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}, {}, {}, data_split_mode),
              0.25f, 0.001f);
  delete metric;
}

inline void VerifyNDCGExpGain(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  Context ctx = xgboost::CreateEmptyGenericParam(GPUIDX);

  auto p_fmat = xgboost::RandomDataGenerator{0, 0, 0}.GenerateDMatrix();
  MetaInfo& info = p_fmat->Info();
  info.labels = linalg::Matrix<float>{{10.0f, 0.0f, 0.0f, 1.0f, 5.0f}, {5}, ctx.gpu_id};
  info.num_row_ = info.labels.Shape(0);
  info.group_ptr_.resize(2);
  info.group_ptr_[0] = 0;
  info.group_ptr_[1] = info.num_row_;
  info.data_split_mode = data_split_mode;
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
}  // namespace metric
}  // namespace xgboost
