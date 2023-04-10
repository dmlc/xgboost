// Copyright by Contributors
#include <xgboost/metric.h>
#include <string>

#include "../helpers.h"

namespace xgboost {
namespace metric {

inline void CheckDeterministicMetricMultiClass(StringView name, int32_t device) {
  auto ctx = CreateEmptyGenericParam(device);
  std::unique_ptr<Metric> metric{Metric::Create(name.c_str(), &ctx)};

  HostDeviceVector<float> predts;
  auto p_fmat = EmptyDMatrix();
  MetaInfo& info = p_fmat->Info();
  auto &h_predts = predts.HostVector();

  SimpleLCG lcg;

  size_t n_samples = 2048, n_classes = 4;

  info.labels.Reshape(n_samples);
  auto &h_labels = info.labels.Data()->HostVector();
  h_predts.resize(n_samples * n_classes);

  {
    SimpleRealUniformDistribution<float> dist{0.0f, static_cast<float>(n_classes)};
    for (size_t i = 0; i < n_samples; ++i) {
      h_labels[i] = dist(&lcg);
    }
  }

  {
    SimpleRealUniformDistribution<float> dist{0.0f, 1.0f};
    for (size_t i = 0; i < n_samples * n_classes; ++i) {
      h_predts[i] = dist(&lcg);
    }
  }

  auto result = metric->Evaluate(predts, p_fmat);
  for (size_t i = 0; i < 8; ++i) {
    ASSERT_EQ(metric->Evaluate(predts, p_fmat), result);
  }
}

inline void TestMultiClassError(int device, DataSplitMode data_split_mode) {
  auto ctx = xgboost::CreateEmptyGenericParam(device);
  ctx.gpu_id = device;
  xgboost::Metric * metric = xgboost::Metric::Create("merror", &ctx);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "merror");
  EXPECT_ANY_THROW(GetMetricEval(metric, {0}, {0, 0}, {}, {}, data_split_mode));
  EXPECT_NEAR(GetMetricEval(
      metric, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 1, 2}, {}, {}, data_split_mode), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f},
                            {0, 1, 2}, {}, {}, data_split_mode),
              0.666f, 0.001f);
  delete metric;
}

inline void VerifyMultiClassError(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  TestMultiClassError(GPUIDX, data_split_mode);
  CheckDeterministicMetricMultiClass(StringView{"merror"}, GPUIDX);
}

inline void TestMultiClassLogLoss(int device, DataSplitMode data_split_mode) {
  auto ctx = xgboost::CreateEmptyGenericParam(device);
  ctx.gpu_id = device;
  xgboost::Metric * metric = xgboost::Metric::Create("mlogloss", &ctx);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "mlogloss");
  EXPECT_ANY_THROW(GetMetricEval(metric, {0}, {0, 0}, {}, {}, data_split_mode));
  EXPECT_NEAR(GetMetricEval(
    metric, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 1, 2}, {}, {}, data_split_mode), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f},
                            {0, 1, 2}, {}, {}, data_split_mode),
              2.302f, 0.001f);

  delete metric;
}

inline void VerifyMultiClassLogLoss(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  TestMultiClassLogLoss(GPUIDX, data_split_mode);
  CheckDeterministicMetricMultiClass(StringView{"mlogloss"}, GPUIDX);
}

}  // namespace metric
}  // namespace xgboost
