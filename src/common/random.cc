/**
 * Copyright 2020-2023, XGBoost Contributors
 */
#include "random.h"

namespace xgboost::common {
std::shared_ptr<HostDeviceVector<bst_feature_t>> ColumnSampler::ColSample(
    std::shared_ptr<HostDeviceVector<bst_feature_t>> p_features, float colsample) {
  if (colsample == 1.0f) {
    return p_features;
  }
  const auto &features = p_features->HostVector();
  CHECK_GT(features.size(), 0);

  int n = std::max(1, static_cast<int>(colsample * features.size()));
  auto p_new_features = std::make_shared<HostDeviceVector<bst_feature_t>>();
  auto &new_features = *p_new_features;

  if (!feature_weights_.Empty()) {
    auto const &h_features = p_features->HostVector();
    auto const &h_feature_weight = feature_weights_.ConstHostVector();
    std::vector<float> weights(h_features.size());
    for (size_t i = 0; i < h_features.size(); ++i) {
      weights[i] = h_feature_weight[h_features[i]];
    }
    CHECK(ctx_);
    new_features.HostVector() =
        WeightedSamplingWithoutReplacement(ctx_, p_features->HostVector(), weights, n);
  } else {
    new_features.Resize(features.size());
    std::copy(features.begin(), features.end(), new_features.HostVector().begin());
    std::shuffle(new_features.HostVector().begin(), new_features.HostVector().end(), rng_);
    new_features.Resize(n);
  }
  std::sort(new_features.HostVector().begin(), new_features.HostVector().end());
  return p_new_features;
}
}  // namespace xgboost::common
