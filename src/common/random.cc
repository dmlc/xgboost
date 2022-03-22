/*!
 * Copyright 2020 by XGBoost Contributors
 * \file random.cc
 */
#include "random.h"

namespace xgboost {
namespace common {
std::shared_ptr<HostDeviceVector<bst_feature_t>> ColumnSampler::ColSample(
    std::shared_ptr<HostDeviceVector<bst_feature_t>> p_features,
    float colsample) {
  if (colsample == 1.0f) {
    return p_features;
  }
  const auto &features = p_features->HostVector();
  CHECK_GT(features.size(), 0);

  int n = std::max(1, static_cast<int>(colsample * features.size()));
  auto p_new_features = std::make_shared<HostDeviceVector<bst_feature_t>>();
  auto &new_features = *p_new_features;

  if (feature_weights_.size() != 0) {
    new_features.HostVector() = WeightedSamplingWithoutReplacement(
        p_features->HostVector(), feature_weights_, n);
  } else {
    new_features.Resize(features.size());
    std::copy(features.begin(), features.end(),
              new_features.HostVector().begin());
    std::shuffle(new_features.HostVector().begin(),
                 new_features.HostVector().end(), rng_);
    new_features.Resize(n);
  }
  std::sort(new_features.HostVector().begin(), new_features.HostVector().end());
  return p_new_features;
}

}  // namespace common
}  // namespace xgboost
