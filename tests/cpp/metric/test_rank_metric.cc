// Copyright by Contributors
#include <xgboost/metric.h>

#include "../helpers.h"

#if !defined(__CUDACC__)
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
#endif

TEST(Metric, DeclareUnifiedTest(AUC)) {
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
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 1}, {1, 1}));

  // AUC with instance weights
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.9f, 0.1f, 0.4f, 0.3f},
                            {0,    0,    1,    1},
                            {1.0f, 3.0f, 2.0f, 4.0f}),
              0.75f, 0.001f);

  // AUC for a ranking task without weights
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.9f, 0.1f, 0.4f, 0.3f, 0.7f},
                            {0,    1,    0,    1,    1},
                            {},
                            {0, 2, 5}),
              0.25f, 0.001f);

  // AUC for a ranking task with weights/group
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.9f, 0.1f, 0.4f, 0.3f, 0.7f},
                            {1,    0,    1,    0,    0},
                            {1, 2},
                            {0, 2, 5}),
              0.75f, 0.001f);

  // AUC metric for grouped datasets - exception scenarios
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 1, 2}, {0, 0, 0}, {}, {0, 2, 3}));
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 1, 2}, {1, 1, 1}, {}, {0, 2, 3}));

  delete metric;
}

TEST(Metric, DeclareUnifiedTest(AUCPR)) {
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
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 0}, {1, 1}));

  // AUCPR with instance weights
  EXPECT_NEAR(GetMetricEval(
                  metric, {0.29f, 0.52f, 0.11f, 0.21f, 0.219f, 0.93f, 0.493f,
                           0.17f, 0.47f, 0.13f, 0.43f, 0.59f, 0.87f, 0.007f},
                  {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0},
                  {1, 2, 7, 4, 5, 2.2f, 3.2f, 5, 6, 1, 2, 1.1f, 3.2f, 4.5f}),  // weights
              0.694435f, 0.001f);

  // AUCPR with groups and no weights
  EXPECT_NEAR(GetMetricEval(
                  metric, {0.87f, 0.31f, 0.40f, 0.42f, 0.25f, 0.66f, 0.95f,
                           0.09f, 0.10f, 0.97f, 0.76f, 0.69f, 0.15f, 0.20f,
                           0.30f, 0.14f, 0.07f, 0.58f, 0.61f, 0.08f},
                  {0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1},
                  {},  // weights
                  {0, 2, 5, 9, 14, 20}),  // group info
              0.556021f, 0.001f);

  // AUCPR with groups and weights
  EXPECT_NEAR(GetMetricEval(
                  metric, {0.29f, 0.52f, 0.11f, 0.21f, 0.219f, 0.93f, 0.493f,
                           0.17f, 0.47f, 0.13f, 0.43f, 0.59f, 0.87f, 0.007f},  // predictions
                  {0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0},
                  {1, 2, 7, 4, 5, 2.2f, 3.2f, 5, 6, 1, 2, 1.1f, 3.2f, 4.5f},  // weights
                  {0, 2, 5, 9, 14}),  // group info
              0.8150615f, 0.001f);

  // Exception scenarios for grouped datasets
  EXPECT_ANY_THROW(GetMetricEval(metric,
                                 {0, 0.1f, 0.3f, 0.5f, 0.7f},
                                 {1, 1, 0, 0, 0},
                                 {},
                                 {0, 2, 5}));

  delete metric;
}


TEST(Metric, DeclareUnifiedTest(Precision)) {
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

TEST(Metric, DeclareUnifiedTest(NDCG)) {
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

TEST(Metric, DeclareUnifiedTest(MAP)) {
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

  // Rank metric with group info
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.2f, 0.8f, 0.4f, 1.7f},
                            {2, 7, 1, 0, 5, 0},  // Labels
                            {},  // Weights
                            {0, 2, 5, 6}),  // Group info
              0.8611f, 0.001f);

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
