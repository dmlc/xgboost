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

} // namespace xgboost
