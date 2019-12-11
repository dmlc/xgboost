/*!
 * Copyright 2019 by Contributors
 */
#include <utility>
#include <limits>
#include "xgboost/json.h"
#include "gblinear_model.h"

namespace xgboost {
namespace gbm {

void GBLinearModel::SaveModel(Json* p_out) const {
  using WeightType = std::remove_reference<decltype(std::declval<decltype(weight)>().back())>::type;
  using JsonFloat = Number::Float;
  static_assert(std::is_same<WeightType, JsonFloat>::value,
                "Weight type should be of the same type with JSON float");
  auto& out = *p_out;

  size_t const n_weights = weight.size();
  std::vector<Json> j_weights(n_weights);
  for (size_t i = 0; i < n_weights; ++i) {
    j_weights[i] = weight[i];
  }
  out["weights"] = std::move(j_weights);
}

void GBLinearModel::LoadModel(Json const& in) {
  auto const& j_weights = get<Array const>(in["weights"]);
  auto n_weights = j_weights.size();
  weight.resize(n_weights);
  for (size_t i = 0; i < n_weights; ++i) {
    weight[i] = get<Number const>(j_weights[i]);
  }
}

DMLC_REGISTER_PARAMETER(DeprecatedGBLinearModelParam);
}  // namespace gbm
}  // namespace xgboost
