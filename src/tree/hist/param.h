/**
 * Copyright 2021-2023, XGBoost Contributors
 */
#pragma once
#include "xgboost/parameter.h"
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost::tree {
struct HistMakerTrainParam : public XGBoostParameter<HistMakerTrainParam> {
  constexpr static std::int64_t DefaultNodes() { return static_cast<std::int64_t>(1) << 16; }

  bool debug_synchronize{false};
  std::int64_t max_cached_n_nodes{DefaultNodes()};

  void CheckTreesSynchronized(RegTree const* local_tree) const;
  // declare parameters
  DMLC_DECLARE_PARAMETER(HistMakerTrainParam) {
    DMLC_DECLARE_FIELD(debug_synchronize)
        .set_default(false)
        .describe("Check if all distributed tree are identical after tree construction.");
    DMLC_DECLARE_FIELD(max_cached_n_nodes)
        .set_default(DefaultNodes())
        .describe(
            "Number of cached nodes before clearing the cache. This is not an absolute limit as "
            "XGBoost can still grow the cache during histogram construction, but clear it right "
            "after once the limit is exceeded.");
  }
};
}  // namespace xgboost::tree
