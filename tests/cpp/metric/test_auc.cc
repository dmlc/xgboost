#include <xgboost/metric.h>
#include "../helpers.h"

namespace xgboost {
namespace metric {

TEST(Metric, DeclareUnifiedTest(BinaryAUC)) {
  auto tparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<Metric> uni_ptr {Metric::Create("auc", &tparam)};
  Metric * metric = uni_ptr.get();
  ASSERT_STREQ(metric->Name(), "auc");

  // Binary
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 1.0f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {1, 0}), 0.0f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0, 0}, {0, 1}), 0.5f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {1, 1}, {0, 1}), 0.5f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0, 0}, {1, 0}), 0.5f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {1, 1}, {1, 0}), 0.5f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {1, 0, 0}, {0, 0, 1}), 0.25f, 1e-10);

  // Invalid dataset
  MetaInfo info;
  info.labels = linalg::Tensor<float, 2>{{0.0f, 0.0f}, {2}, -1};
  float auc = metric->Eval({1, 1}, info);
  ASSERT_TRUE(std::isnan(auc));
  *info.labels.Data() = HostDeviceVector<float>{};
  auc = metric->Eval(HostDeviceVector<float>{}, info);
  ASSERT_TRUE(std::isnan(auc));

  EXPECT_NEAR(GetMetricEval(metric, {0, 1, 0, 1}, {0, 1, 0, 1}), 1.0f, 1e-10);

  // AUC with instance weights
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.9f, 0.1f, 0.4f, 0.3f},
                            {0,    0,    1,    1},
                            {1.0f, 3.0f, 2.0f, 4.0f}),
              0.75f, 0.001f);

  // regression test case
  ASSERT_NEAR(GetMetricEval(
                  metric,
                  {0.79523796, 0.5201713,  0.79523796, 0.24273258, 0.53452194,
                   0.53452194, 0.24273258, 0.5201713,  0.79523796, 0.53452194,
                   0.24273258, 0.53452194, 0.79523796, 0.5201713,  0.24273258,
                   0.5201713,  0.5201713,  0.53452194, 0.5201713,  0.53452194},
                  {0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0}),
              0.5, 1e-10);
}

TEST(Metric, DeclareUnifiedTest(MultiClassAUC)) {
  auto tparam = CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<Metric> uni_ptr{
      Metric::Create("auc", &tparam)};
  auto metric = uni_ptr.get();

  // MultiClass
  // 3x3
  EXPECT_NEAR(GetMetricEval(metric,
                            {
                                1.0f, 0.0f, 0.0f, // p_0
                                0.0f, 1.0f, 0.0f, // p_1
                                0.0f, 0.0f, 1.0f  // p_2
                            },
                            {0, 1, 2}),
              1.0f, 1e-10);

  EXPECT_NEAR(GetMetricEval(metric,
                            {
                                1.0f, 0.0f, 0.0f, // p_0
                                0.0f, 1.0f, 0.0f, // p_1
                                0.0f, 0.0f, 1.0f  // p_2
                            },
                            {0, 1, 2},
                            {1.0f, 1.0f, 1.0f}),
              1.0f, 1e-10);

  EXPECT_NEAR(GetMetricEval(metric,
                            {
                                1.0f, 0.0f, 0.0f, // p_0
                                0.0f, 1.0f, 0.0f, // p_1
                                0.0f, 0.0f, 1.0f  // p_2
                            },
                            {2, 1, 0}),
              0.5f, 1e-10);

  EXPECT_NEAR(GetMetricEval(metric,
                            {
                                1.0f, 0.0f, 0.0f, // p_0
                                0.0f, 1.0f, 0.0f, // p_1
                                0.0f, 0.0f, 1.0f  // p_2
                            },
                            {2, 0, 1}),
              0.25f, 1e-10);

  // invalid dataset
  float auc = GetMetricEval(metric,
                            {
                                1.0f, 0.0f, 0.0f, // p_0
                                0.0f, 1.0f, 0.0f, // p_1
                                0.0f, 0.0f, 1.0f  // p_2
                            },
                            {0, 1, 1});  // no class 2.
  EXPECT_TRUE(std::isnan(auc)) << auc;

  HostDeviceVector<float> predts{
    0.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 1.0f,
  };
  std::vector<float> labels {1.0f, 0.0f, 2.0f, 1.0f};
  auc = GetMetricEval(metric, predts, labels, {1.0f, 2.0f, 3.0f, 4.0f});
  ASSERT_GT(auc, 0.714);
}

TEST(Metric, DeclareUnifiedTest(RankingAUC)) {
  auto tparam = CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<Metric> metric{Metric::Create("auc", &tparam)};

  // single group
  EXPECT_NEAR(GetMetricEval(metric.get(), {0.7f, 0.2f, 0.3f, 0.6f},
                            {1.0f, 0.8f, 0.4f, 0.2f}, /*weights=*/{},
                            {0, 4}),
              0.5f, 1e-10);

  // multi group
  EXPECT_NEAR(GetMetricEval(metric.get(), {0, 1, 2, 0, 1, 2},
                            {0, 1, 2, 0, 1, 2}, /*weights=*/{}, {0, 3, 6}),
              1.0f, 1e-10);

  EXPECT_NEAR(GetMetricEval(metric.get(), {0, 1, 2, 0, 1, 2},
                            {0, 1, 2, 0, 1, 2}, /*weights=*/{1.0f, 2.0f},
                            {0, 3, 6}),
              1.0f, 1e-10);

  // AUC metric for grouped datasets - exception scenarios
  ASSERT_TRUE(std::isnan(
      GetMetricEval(metric.get(), {0, 1, 2}, {0, 0, 0}, {}, {0, 2, 3})));

  // regression case
  HostDeviceVector<float> predt{0.33935383, 0.5149714,  0.32138085, 1.4547751,
                                1.2010975,  0.42651367, 0.23104341, 0.83610827,
                                0.8494239,  0.07136688, 0.5623144,  0.8086237,
                                1.5066161,  -4.094787,  0.76887935, -2.4082742};
  std::vector<bst_group_t> groups{0, 7, 16};
  std::vector<float> labels{1., 0., 0., 1., 2., 1., 0., 0.,
                            0., 0., 0., 0., 1., 0., 1., 0.};

  EXPECT_NEAR(GetMetricEval(metric.get(), std::move(predt), labels,
                            /*weights=*/{}, groups),
              0.769841f, 1e-6);
}

TEST(Metric, DeclareUnifiedTest(PRAUC)) {
  auto tparam = xgboost::CreateEmptyGenericParam(GPUIDX);

  xgboost::Metric *metric = xgboost::Metric::Create("aucpr", &tparam);
  ASSERT_STREQ(metric->Name(), "aucpr");
  EXPECT_NEAR(GetMetricEval(metric, {0, 0, 1, 1}, {0, 0, 1, 1}), 1, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0.1f, 0.9f, 0.1f, 0.9f}, {0, 0, 1, 1}),
              0.5f, 0.001f);
  EXPECT_NEAR(GetMetricEval(
                  metric,
                  {0.4f, 0.2f, 0.9f, 0.1f, 0.2f, 0.4f, 0.1f, 0.1f, 0.2f, 0.1f},
                  {0, 0, 0, 0, 0, 1, 0, 0, 1, 1}),
              0.2908445f, 0.001f);
  EXPECT_NEAR(GetMetricEval(
                  metric, {0.87f, 0.31f, 0.40f, 0.42f, 0.25f, 0.66f, 0.95f,
                           0.09f, 0.10f, 0.97f, 0.76f, 0.69f, 0.15f, 0.20f,
                           0.30f, 0.14f, 0.07f, 0.58f, 0.61f, 0.08f},
                  {0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1}),
              0.2769199f, 0.001f);
  auto auc = GetMetricEval(metric, {0, 1}, {});
  ASSERT_TRUE(std::isnan(auc));

  // AUCPR with instance weights
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.29f, 0.52f, 0.11f, 0.21f, 0.219f, 0.93f, 0.493f,
                             0.17f, 0.47f, 0.13f, 0.43f, 0.59f, 0.87f, 0.007f},
                            {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0},
                            {1, 2, 7, 4, 5, 2.2f, 3.2f, 5, 6, 1, 2, 1.1f, 3.2f,
                             4.5f}), // weights
              0.694435f, 0.001f);

  // Both groups contain only pos or neg samples.
  auc = GetMetricEval(metric,
                      {0, 0.1f, 0.3f, 0.5f, 0.7f},
                      {1, 1, 0, 0, 0},
                      {},
                      {0, 2, 5});
  ASSERT_TRUE(std::isnan(auc));
  delete metric;
}

TEST(Metric, DeclareUnifiedTest(MultiClassPRAUC)) {
  auto tparam = xgboost::CreateEmptyGenericParam(GPUIDX);

  std::unique_ptr<Metric> metric{Metric::Create("aucpr", &tparam)};

  float auc = 0;
  std::vector<float> labels {1.0f, 0.0f, 2.0f};
  HostDeviceVector<float> predts{
    0.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f,
  };
  auc = GetMetricEval(metric.get(), predts, labels, {});
  EXPECT_EQ(auc, 1.0f);

  auc = GetMetricEval(metric.get(), predts, labels, {1.0f, 1.0f, 1.0f});
  EXPECT_EQ(auc, 1.0f);

  predts.HostVector() =  {
    0.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 1.0f,
  };
  labels = {1.0f, 0.0f, 2.0f, 1.0f};
  auc = GetMetricEval(metric.get(), predts, labels, {1.0f, 2.0f, 3.0f, 4.0f});
  ASSERT_GT(auc, 0.699);
}

TEST(Metric, DeclareUnifiedTest(RankingPRAUC)) {
  auto tparam = xgboost::CreateEmptyGenericParam(GPUIDX);

  std::unique_ptr<Metric> metric{Metric::Create("aucpr", &tparam)};

  std::vector<float> labels {1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
  std::vector<uint32_t> groups {0, 2, 6};

  float auc = 0;
  auc = GetMetricEval(metric.get(), {1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f}, labels, {}, groups);
  EXPECT_EQ(auc, 1.0f);

  auc = GetMetricEval(metric.get(), {1.0f, 0.5f, 0.8f, 0.3f, 0.2f, 1.0f}, labels, {}, groups);
  EXPECT_EQ(auc, 1.0f);

  auc = GetMetricEval(metric.get(), {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                      {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {}, groups);
  ASSERT_TRUE(std::isnan(auc));

  // Incorrect label
  ASSERT_THROW(GetMetricEval(metric.get(), {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                             {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 3.0f}, {}, groups),
               dmlc::Error);

  // AUCPR with groups and no weights
  EXPECT_NEAR(GetMetricEval(
      metric.get(), {0.87f, 0.31f, 0.40f, 0.42f, 0.25f, 0.66f, 0.95f,
                     0.09f, 0.10f, 0.97f, 0.76f, 0.69f, 0.15f, 0.20f,
                     0.30f, 0.14f, 0.07f, 0.58f, 0.61f, 0.08f},
                  {0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1},
                  {},  // weights
                  {0, 2, 5, 9, 14, 20}),  // group info
              0.556021f, 0.001f);
}
}  // namespace metric
}  // namespace xgboost
