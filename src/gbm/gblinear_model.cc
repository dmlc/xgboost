/*!
 * Copyright 2019-2022 by Contributors
 */
#include <algorithm>
#include <utility>
#include <limits>
#include "xgboost/json.h"
#include "gblinear_model.h"

namespace xgboost {
namespace gbm {

void GBLinearModel::SaveModel(Json* p_out) const {
  auto& out = *p_out;

  size_t const n_weights = weight.size();
  F32Array j_weights{n_weights};
  std::copy(weight.begin(), weight.end(), j_weights.GetArray().begin());
  out["weights"] = std::move(j_weights);
  out["boosted_rounds"] = Json{this->num_boosted_rounds};
}

void GBLinearModel::LoadModel(Json const& in) {
  auto const& obj = get<Object const>(in);
  auto weight_it = obj.find("weights");
  if (IsA<F32Array>(weight_it->second)) {
    auto const& j_weights = get<F32Array const>(weight_it->second);
    weight.resize(j_weights.size());
    std::copy(j_weights.begin(), j_weights.end(), weight.begin());
  } else {
    auto const& j_weights = get<Array const>(weight_it->second);
    auto n_weights = j_weights.size();
    weight.resize(n_weights);
    for (size_t i = 0; i < n_weights; ++i) {
      weight[i] = get<Number const>(j_weights[i]);
    }
  }

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
