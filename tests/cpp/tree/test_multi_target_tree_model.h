/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once
#include <xgboost/tree_model.h>  // for RegTree

#include <memory>  // for unique_ptr

namespace xgboost {
std::unique_ptr<RegTree> MakeMtTreeForTest(bst_target_t n_targets);
}  // namespace xgboost
