/**
 * Copyright 2021-2023, XGBoost Contributors
 */
#pragma once

#include <cstddef>  // for size_t

#include "xgboost/parameter.h"   // for XGBoostParameter
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost::tree {
struct HistMakerTrainParam : public XGBoostParameter<HistMakerTrainParam> {
  constexpr static std::size_t DefaultNodes() { return static_cast<std::size_t>(1) << 16; }

  bool debug_synchronize{false};
  std::size_t max_cached_hist_node{DefaultNodes()};

  void CheckTreesSynchronized(RegTree const* local_tree) const;

  // declare parameters
  DMLC_DECLARE_PARAMETER(HistMakerTrainParam) {
    DMLC_DECLARE_FIELD(debug_synchronize)
        .set_default(false)
        .describe("Check if all distributed tree are identical after tree construction.");
    DMLC_DECLARE_FIELD(max_cached_hist_node)
        .set_default(DefaultNodes())
        .set_lower_bound(1)
        .describe("Maximum number of nodes in CPU histogram cache. Only for internal usage.");
  }
};
}  // namespace xgboost::tree
