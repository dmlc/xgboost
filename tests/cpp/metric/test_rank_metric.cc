// Copyright by Contributors
#include <xgboost/metric.h>

#include "../helpers.h"

TEST(Metric, AMS) {
  auto tparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  EXPECT_ANY_THROW(xgboost::Metric::Create("ams", &tparam));
  xgboost::Metric * metric = xgboost::Metric::Create("ams@0.5f", &tparam);
  ASSERT_STREQ(metric->Name(), "ams@0.5");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0.311f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.29710f, 0.001f);

  delete metric;
  metric = xgboost::Metric::Create("ams@0", &tparam);
  ASSERT_STREQ(metric->Name(), "ams@0");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0.311f, 0.001f);

  delete metric;
}

TEST(Metric, AUC) {
  auto tparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("auc", &tparam);
  ASSERT_STREQ(metric->Name(), "auc");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.5f, 0.001f);
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 1}, {}));
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 0}, {0, 0}));

  delete metric;
}

TEST(Metric, AUCPR) {
  auto tparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric *metric = xgboost::Metric::Create("aucpr", &tparam);
  ASSERT_STREQ(metric->Name(), "aucpr");
  EXPECT_NEAR(GetMetricEval(metric, {0, 0, 1, 1}, {0, 0, 1, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0.1f, 0.9f, 0.1f, 0.9f}, {0, 0, 1, 1}),
              0.5f, 0.001f);
  EXPECT_NEAR(
      GetMetricEval(metric,
                    {0.4f, 0.2f, 0.9f, 0.1f, 0.2f, 0.4f, 0.1f, 0.1f, 0.2f, 0.1f},
                    {0, 0, 0, 0, 0, 1, 0, 0, 1, 1}),
      0.2908445f, 0.001f);
  EXPECT_NEAR(GetMetricEval(
                  metric, {0.87f, 0.31f, 0.40f, 0.42f, 0.25f, 0.66f, 0.95f,
                           0.09f, 0.10f, 0.97f, 0.76f, 0.69f, 0.15f, 0.20f,
                           0.30f, 0.14f, 0.07f, 0.58f, 0.61f, 0.08f},
                  {0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1}),
              0.2769199f, 0.001f);
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 1}, {}));
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 0}, {0, 0}));

  delete metric;
}

TEST(Metric, Precision) {
  // When the limit for precision is not given, it takes the limit at
  // std::numeric_limits<unsigned>::max(); hence all values are very small
  // NOTE(AbdealiJK): Maybe this should be fixed to be num_row by default.
  auto tparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("pre", &tparam);
  ASSERT_STREQ(metric->Name(), "pre");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-7);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0, 1e-7);

  delete metric;
  metric = xgboost::Metric::Create("pre@2", &tparam);
  ASSERT_STREQ(metric->Name(), "pre@2");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0.5f, 1e-7);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.5f, 0.001f);

  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 1}, {}));

  delete metric;
}

TEST(Metric, NDCG) {
  auto tparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("ndcg", &tparam);
  ASSERT_STREQ(metric->Name(), "ndcg");
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 1}, {}));
  EXPECT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            {}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.6509f, 0.001f);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@2", &tparam);
  ASSERT_STREQ(metric->Name(), "ndcg@2");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.3868f, 0.001f);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@-", &tparam);
  ASSERT_STREQ(metric->Name(), "ndcg-");
  EXPECT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            {}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.6509f, 0.001f);
  delete metric;
  metric = xgboost::Metric::Create("ndcg-", &tparam);
  ASSERT_STREQ(metric->Name(), "ndcg-");
  EXPECT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            {}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.6509f, 0.001f);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@2-", &tparam);
  ASSERT_STREQ(metric->Name(), "ndcg@2-");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.3868f, 0.001f);

  delete metric;
}

TEST(Metric, MAP) {
  auto tparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = xgboost::Metric::Create("map", &tparam);
  ASSERT_STREQ(metric->Name(), "map");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.5f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            std::vector<xgboost::bst_float>{}), 1, 1e-10);

  delete metric;
  metric = xgboost::Metric::Create("map@-", &tparam);
  ASSERT_STREQ(metric->Name(), "map-");
  EXPECT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            {}), 0, 1e-10);

  delete metric;
  metric = xgboost::Metric::Create("map-", &tparam);
  ASSERT_STREQ(metric->Name(), "map-");
  EXPECT_NEAR(GetMetricEval(metric,
                            xgboost::HostDeviceVector<xgboost::bst_float>{},
                            {}), 0, 1e-10);

  delete metric;
  metric = xgboost::Metric::Create("map@2", &tparam);
  ASSERT_STREQ(metric->Name(), "map@2");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.25f, 0.001f);
  delete metric;
}
