/*!
 * Copyright 2021 by XGBoost Contributors
 *
 * \brief Utilities for testing categorical data support.
 */
#include <numeric>
#include <vector>

#include "xgboost/span.h"
#include "helpers.h"
#include "../../src/common/categorical.h"

namespace xgboost {
inline std::vector<float> OneHotEncodeFeature(std::vector<float> x,
                                              size_t num_cat) {
  std::vector<float> ret(x.size() * num_cat, 0);
  size_t n_rows = x.size();
  for (size_t r = 0; r < n_rows; ++r) {
    bst_cat_t cat = common::AsCat(x[r]);
    ret.at(num_cat * r + cat) = 1;
  }
  return ret;
}

template <typename GradientSumT>
void ValidateCategoricalHistogram(size_t n_categories,
                                  common::Span<GradientSumT> onehot,
                                  common::Span<GradientSumT> cat) {
  auto cat_sum = std::accumulate(cat.cbegin(), cat.cend(), GradientPairPrecise{});
  for (size_t c = 0; c < n_categories; ++c) {
    auto zero = onehot[c * 2];
    auto one = onehot[c * 2 + 1];

    auto chosen = cat[c];
    auto not_chosen = cat_sum - chosen;

    ASSERT_LE(RelError(zero.GetGrad(), not_chosen.GetGrad()), kRtEps);
    ASSERT_LE(RelError(zero.GetHess(), not_chosen.GetHess()), kRtEps);

    ASSERT_LE(RelError(one.GetGrad(), chosen.GetGrad()), kRtEps);
    ASSERT_LE(RelError(one.GetHess(), chosen.GetHess()), kRtEps);
  }
}
} // namespace xgboost
