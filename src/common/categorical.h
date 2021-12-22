/*!
 * Copyright 2020-2021 by XGBoost Contributors
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
#include "xgboost/task.h"

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

template <typename SizeT>
bool InvalidCat(float cat, SizeT n) {
  return cat < 0 || cat > static_cast<float>(std::numeric_limits<bst_cat_t>::max()) || cat >= n;
}

/* \brief Whether should it traverse to left branch of a tree.
 *
 *  For one hot split, go to left if it's NOT the matching category.
 */
template <bool validate = true>
inline XGBOOST_DEVICE bool Decision(common::Span<uint32_t const> cats, float cat, bool dft_left) {
  CLBitField32 const s_cats(cats);
  // FIXME: Size() is not accurate since it represents the size of bit set instead of
  // actual number of categories.
  if (XGBOOST_EXPECT(validate && InvalidCat(cat, s_cats.Size()), false)) {
    return dft_left;
  }
  return !s_cats.Check(AsCat(cat));
}

inline void InvalidCategory() {
  LOG(FATAL) << "Invalid categorical value detected.  Categorical value "
                "should be non-negative, less than maximum size of int32 and less than total "
                "number of categories in training data.";
}

/*!
 * \brief Whether should we use onehot encoding for categorical data.
 */
inline bool UseOneHot(uint32_t n_cats, uint32_t max_cat_to_onehot, ObjInfo task) {
  bool use_one_hot = n_cats < max_cat_to_onehot ||
                     (task.task != ObjInfo::kRegression && task.task != ObjInfo::kBinary);
  return use_one_hot;
}

struct IsCatOp {
  XGBOOST_DEVICE bool operator()(FeatureType ft) { return ft == FeatureType::kCategorical; }
};

using CatBitField = LBitField32;
using KCatBitField = CLBitField32;
}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_CATEGORICAL_H_
