/*!
 * Copyright 2021 by XGBoost Contributors
 */
#include <xgboost/host_device_vector.h>
#include <xgboost/json.h>
#include <xgboost/metric.h>

#include "../helpers.h"

namespace xgboost {
template <bool is_aft>
inline void CheckDeterministicMetricElementWise(StringView name, int32_t device) {
  auto lparam = CreateEmptyGenericParam(device);
  std::unique_ptr<Metric> metric{Metric::Create(name.c_str(), &lparam)};
  metric->Configure(Args{});

  HostDeviceVector<float> predts;
  MetaInfo info;
  auto &h_predts = predts.HostVector();

  SimpleLCG lcg;
  SimpleRealUniformDistribution<float> dist{0.0f, 1.0f};

  size_t n_samples = 2048;
  h_predts.resize(n_samples);

  for (size_t i = 0; i < n_samples; ++i) {
    h_predts[i] = dist(&lcg);
  }

  if (is_aft) {
    auto &h_upper = info.labels_upper_bound_.HostVector();
    auto &h_lower = info.labels_lower_bound_.HostVector();
    h_lower.resize(n_samples);
    h_upper.resize(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
      h_lower[i] = 1;
      h_upper[i] = 10;
    }
  } else {
    auto &h_labels = info.labels_.HostVector();
    h_labels.resize(n_samples);
    for (auto &v : h_labels) {
      v = dist(&lcg);
    }
  }

  auto result = metric->Eval(predts, info, false);
  for (size_t i = 0; i < 8; ++i) {
    ASSERT_EQ(metric->Eval(predts, info, false), result);
  }
}
}  // namespace xgboost
