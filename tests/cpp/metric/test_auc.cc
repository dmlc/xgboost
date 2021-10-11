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
  info.labels_ = {0, 0};
  float auc = metric->Eval({1, 1}, info, false);
  ASSERT_TRUE(std::isnan(auc));
  info.labels_ = HostDeviceVector<float>{};
  auc = metric->Eval(HostDeviceVector<float>{}, info, false);
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

TEST(Metric, DeclareUnifiedTest(MultiAUC)) {
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
}  // namespace metric
}  // namespace xgboost
