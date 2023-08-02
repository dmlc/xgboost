/**
 * Copyright 2021-2023, XGBoost Contributors
 */
#pragma once
#include "xgboost/parameter.h"
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost::tree {
struct HistMakerTrainParam : public XGBoostParameter<HistMakerTrainParam> {
  bool debug_synchronize;
  void CheckTreesSynchronized(RegTree const* local_tree) const;

  // declare parameters
  DMLC_DECLARE_PARAMETER(HistMakerTrainParam) {
    DMLC_DECLARE_FIELD(debug_synchronize)
        .set_default(false)
        .describe("Check if all distributed tree are identical after tree construction.");
  }
};
}  // namespace xgboost::tree
