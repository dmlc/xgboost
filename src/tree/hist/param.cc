/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#include "param.h"

#include <string>  // for string

#include "../../collective/broadcast.h"         // for Broadcast
#include "../../collective/communicator-inl.h"  // for GetRank
#include "xgboost/json.h"                       // for Object, Json
#include "xgboost/tree_model.h"                 // for RegTree

namespace xgboost::tree {
DMLC_REGISTER_PARAMETER(HistMakerTrainParam);

void HistMakerTrainParam::CheckTreesSynchronized(Context const* ctx,
                                                 RegTree const* local_tree) const {
  if (!this->debug_synchronize) {
    return;
  }

  std::string s_model;
  Json model{Object{}};
  int rank = collective::GetRank();
  if (rank == 0) {
    local_tree->SaveModel(&model);
  }
  Json::Dump(model, &s_model, std::ios::binary);
  auto rc = collective::Broadcast(ctx, linalg::MakeVec(s_model.data(), s_model.size()), 0);
  CHECK(rc.OK()) << rc.Report();

  RegTree ref_tree{};  // rank 0 tree
  auto j_ref_tree = Json::Load(StringView{s_model}, std::ios::binary);
  ref_tree.LoadModel(j_ref_tree);
  CHECK(*local_tree == ref_tree);
}
}  // namespace xgboost::tree
