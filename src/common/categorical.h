/*!
 * Copyright 2020 by XGBoost Contributors
 * \file categorical.h
 */
#ifndef XGBOOST_COMMON_CATEGORICAL_H_
#define XGBOOST_COMMON_CATEGORICAL_H_

#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/span.h"
#include "xgboost/parameter.h"
#include "bitfield.h"

namespace xgboost {
namespace common {
// Cast the categorical type.
template <typename T>
XGBOOST_DEVICE bst_cat_t AsCat(T const& v) {
  return static_cast<bst_cat_t>(v);
}

/* \brief Whether is fidx a categorical feature.
 *
 * \param ft   Feature type for all features.
 * \param fidx Feature index.
 * \return Whether feature pointed by fidx is categorical feature.
 */
inline XGBOOST_DEVICE bool IsCat(Span<FeatureType const> ft, bst_feature_t fidx) {
  return !ft.empty() && ft[fidx] == FeatureType::kCategorical;
}

/* \brief Whether should it traverse to left branch of a tree.
 *
 *  For one hot split, go to left if it's NOT the matching category.
 */
inline XGBOOST_DEVICE bool Decision(common::Span<uint32_t const> cats, bst_cat_t cat) {
  auto pos = CLBitField32::ToBitPos(cat);
  if (pos.int_pos >= cats.size()) {
    return true;
  }
  CLBitField32 const s_cats(cats);
  return !s_cats.Check(cat);
}

using CatBitField = LBitField32;
using KCatBitField = CLBitField32;
}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_CATEGORICAL_H_
