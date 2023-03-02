/*!
 * Copyright 2020-2022 by XGBoost Contributors
 * \file categorical.h
 */
#ifndef XGBOOST_COMMON_CATEGORICAL_H_
#define XGBOOST_COMMON_CATEGORICAL_H_

#include <limits>

#include "bitfield.h"
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/parameter.h"
#include "xgboost/span.h"

namespace xgboost {
namespace common {

using CatBitField = LBitField32;
using KCatBitField = CLBitField32;

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

constexpr inline bst_cat_t OutOfRangeCat() {
  // See the round trip assert in `InvalidCat`.
  return static_cast<bst_cat_t>(16777217) - static_cast<bst_cat_t>(1);
}

inline XGBOOST_DEVICE bool InvalidCat(float cat) {
  constexpr auto kMaxCat = OutOfRangeCat();
  static_assert(static_cast<bst_cat_t>(static_cast<float>(kMaxCat)) == kMaxCat);
  static_assert(static_cast<bst_cat_t>(static_cast<float>(kMaxCat + 1)) != kMaxCat + 1);
  static_assert(static_cast<float>(kMaxCat + 1) == kMaxCat);
  return cat < 0 || cat >= kMaxCat;
}

/**
 * \brief Whether should it traverse to left branch of a tree.
 *
 *   Go to left if it's NOT the matching category, which matches one-hot encoding.
 */
inline XGBOOST_DEVICE bool Decision(common::Span<uint32_t const> cats, float cat) {
  KCatBitField const s_cats(cats);
  if (XGBOOST_EXPECT(InvalidCat(cat), false)) {
    return true;
  }

  auto pos = KCatBitField::ToBitPos(cat);
  // If the input category is larger than the size of the bit field, it implies that the
  // category is not chosen. Otherwise the bit field would have the category instead of
  // being smaller than the category value.
  if (pos.int_pos >= cats.size()) {
    return true;
  }
  return !s_cats.Check(AsCat(cat));
}

inline void InvalidCategory() {
  // OutOfRangeCat() can be accurately represented, but everything after it will be
  // rounded toward it, so we use >= for comparison check.  As a result, we require input
  // values to be less than this last representable value.
  auto str = std::to_string(OutOfRangeCat());
  LOG(FATAL) << "Invalid categorical value detected.  Categorical value should be non-negative, "
                "less than total number of categories in training data and less than " +
                    str;
}

inline void CheckMaxCat(float max_cat, size_t n_categories) {
  CHECK_GE(max_cat + 1, n_categories)
      << "Maximum cateogry should not be lesser than the total number of categories.";
}

/*!
 * \brief Whether should we use onehot encoding for categorical data.
 */
XGBOOST_DEVICE inline bool UseOneHot(uint32_t n_cats, uint32_t max_cat_to_onehot) {
  bool use_one_hot = n_cats < max_cat_to_onehot;
  return use_one_hot;
}

struct IsCatOp {
  XGBOOST_DEVICE bool operator()(FeatureType ft) { return ft == FeatureType::kCategorical; }
};
}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_CATEGORICAL_H_
