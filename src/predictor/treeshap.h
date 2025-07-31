/**
 * Copyright 2017-2025, XGBoost Contributors
 */
#pragma once

#include <vector>  // for vector

#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost {
/**
 * @brief calculate the approximate feature contributions for the given root
 *
 *   This follows the idea of http://blog.datadive.net/interpreting-random-forests/
 *
 * @param feat dense feature vector, if the feature is missing the field is set to NaN
 * @param out_contribs output vector to hold the contributions
 */
void CalculateContributionsApprox(RegTree const& tree, const RegTree::FVec& feat,
                                  std::vector<float>* mean_values, float* out_contribs);

/**
 * \brief calculate the feature contributions (https://arxiv.org/abs/1706.06060) for the tree
 * \param feat dense feature vector, if the feature is missing the field is set to NaN
 * \param out_contribs output vector to hold the contributions
 * \param condition fix one feature to either off (-1) on (1) or not fixed (0 default)
 * \param condition_feature the index of the feature to fix
 */
void CalculateContributions(RegTree const& tree, const RegTree::FVec& feat,
                            std::vector<float>* mean_values, float* out_contribs, int condition,
                            unsigned condition_feature);
}  // namespace xgboost
