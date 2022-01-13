/*!
 * Copyright 2019-2021 by Contributors
 */
#include <utility>
#include <limits>
#include "xgboost/json.h"
#include "gblinear_model.h"

namespace xgboost {
namespace gbm {

void GBLinearModel::SaveModel(Json* p_out) const {
  auto& out = *p_out;

  size_t const n_weights = weight.size();
  std::vector<Json> j_weights(n_weights);
  for (size_t i = 0; i < n_weights; ++i) {
    j_weights[i] = weight[i];
  }
  out["weights"] = std::move(j_weights);
  out["boosted_rounds"] = Json{this->num_boosted_rounds};
}

void GBLinearModel::LoadModel(Json const& in) {
  auto const& j_weights = get<Array const>(in["weights"]);
  auto n_weights = j_weights.size();
  weight.resize(n_weights);
  for (size_t i = 0; i < n_weights; ++i) {
    weight[i] = get<Number const>(j_weights[i]);
  }
  auto const& obj = get<Object const>(in);
  auto boosted_rounds = obj.find("boosted_rounds");
  if (boosted_rounds != obj.cend()) {
    this->num_boosted_rounds = get<Integer const>(boosted_rounds->second);
  } else {
    this->num_boosted_rounds = 0;
  }
}

DMLC_REGISTER_PARAMETER(DeprecatedGBLinearModelParam);
}  // namespace gbm
}  // namespace xgboost
