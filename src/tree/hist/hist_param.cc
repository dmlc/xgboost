/**
 * Copyright 2021-2025, XGBoost Contributors
 */
#include "hist_param.h"

#include "../../collective/communicator-inl.h"  // for GetRank
#include "../model_utils.h"                     // for BroadcastTreeModel
#include "xgboost/json.h"                       // for Object, Json
#include "xgboost/tree_model.h"                 // for RegTree

namespace xgboost::tree {
DMLC_REGISTER_PARAMETER(HistMakerTrainParam);

void HistMakerTrainParam::CheckTreesSynchronized(Context const* ctx,
                                                 RegTree const* local_tree) const {
  if (!this->debug_synchronize) {
    return;
  }

  Json model{Object{}};
  int rank = collective::GetRank();
  if (rank == 0) {
    local_tree->SaveModel(&model);
  }

  RegTree ref_tree{};  // rank 0 tree
  auto j_ref_tree = BroadcastTreeModel(ctx, model);
  ref_tree.LoadModel(j_ref_tree);
  CHECK(local_tree->Equal(ref_tree));
}
}  // namespace xgboost::tree
