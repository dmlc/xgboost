// Copyright by Contributors
#include <xgboost/metric.h>

#include "../helpers.h"

TEST(Metric, AMS) {
  EXPECT_ANY_THROW(xgboost::Metric::Create("ams"));
  xgboost::Metric * metric = xgboost::Metric::Create("ams@0.5");
  ASSERT_STREQ(metric->Name(), "ams@0.5");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0.311, 0.001);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.9, 0.1, 0.9},
                            {  0,   0,   1,   1}),
              0.29710, 0.001);

  delete metric;
  metric = xgboost::Metric::Create("ams@0");
  ASSERT_STREQ(metric->Name(), "ams@0");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0.311, 0.001);
}

TEST(Metric, AUC) {
  xgboost::Metric * metric = xgboost::Metric::Create("auc");
  ASSERT_STREQ(metric->Name(), "auc");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.9, 0.1, 0.9},
                            {  0,   0,   1,   1}),
              0.5, 0.001);
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 1}, {}));
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 0}, {0, 0}));
}

TEST(Metric, Precision) {
  // When the limit for precision is not given, it takes the limit at
  // std::numeric_limits<unsigned>::max(); hence all values are very small
  // NOTE(AbdealiJK): Maybe this should be fixed to be num_row by default.
  xgboost::Metric * metric = xgboost::Metric::Create("pre");
  ASSERT_STREQ(metric->Name(), "pre");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-7);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.9, 0.1, 0.9},
                            {  0,   0,   1,   1}),
              0, 1e-7);

  delete metric;
  metric = xgboost::Metric::Create("pre@2");
  ASSERT_STREQ(metric->Name(), "pre@2");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0.5, 1e-7);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.9, 0.1, 0.9},
                            {  0,   0,   1,   1}),
              0.5, 0.001);

  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 1}, {}));
}

TEST(Metric, NDCG) {
  xgboost::Metric * metric = xgboost::Metric::Create("ndcg");
  ASSERT_STREQ(metric->Name(), "ndcg");
  EXPECT_ANY_THROW(GetMetricEval(metric, {0, 1}, {}));
  EXPECT_NEAR(GetMetricEval(metric, {}, {}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.9, 0.1, 0.9},
                            {  0,   0,   1,   1}),
              0.6509, 0.001);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@2");
  ASSERT_STREQ(metric->Name(), "ndcg@2");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.9, 0.1, 0.9},
                            {  0,   0,   1,   1}),
              0.3868, 0.001);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@-");
  ASSERT_STREQ(metric->Name(), "ndcg@-");
  EXPECT_NEAR(GetMetricEval(metric, {}, {}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.9, 0.1, 0.9},
                            {  0,   0,   1,   1}),
              0.6509, 0.001);

  delete metric;
  metric = xgboost::Metric::Create("ndcg@2-");
  ASSERT_STREQ(metric->Name(), "ndcg@2-");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.9, 0.1, 0.9},
                            {  0,   0,   1,   1}),
              0.3868, 0.001);
}

TEST(Metric, MAP) {
  xgboost::Metric * metric = xgboost::Metric::Create("map");
  ASSERT_STREQ(metric->Name(), "map");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.9, 0.1, 0.9},
                            {  0,   0,   1,   1}),
              0.5, 0.001);
  EXPECT_NEAR(GetMetricEval(metric, {}, {}), 1, 1e-10);

  delete metric;
  metric = xgboost::Metric::Create("map@-");
  ASSERT_STREQ(metric->Name(), "map@-");
  EXPECT_NEAR(GetMetricEval(metric, {}, {}), 0, 1e-10);

  delete metric;
  metric = xgboost::Metric::Create("map@2");
  ASSERT_STREQ(metric->Name(), "map@2");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.9, 0.1, 0.9},
                            {  0,   0,   1,   1}),
              0.25, 0.001);
}
