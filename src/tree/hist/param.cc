/**
 * Copyright 2021-2023, XGBoost Contributors
 */
#include "param.h"

#include <string>  // for string

#include "../../collective/communicator-inl.h"  // for GetRank, Broadcast
#include "xgboost/json.h"                       // for Object, Json
#include "xgboost/tree_model.h"                 // for RegTree

namespace xgboost::tree {
DMLC_REGISTER_PARAMETER(HistMakerTrainParam);

void HistMakerTrainParam::CheckTreesSynchronized(Context const*, RegTree const* local_tree) const {
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
  collective::Broadcast(&s_model, 0);

  RegTree ref_tree{};  // rank 0 tree
  auto j_ref_tree = Json::Load(StringView{s_model}, std::ios::binary);
  ref_tree.LoadModel(j_ref_tree);
  CHECK(*local_tree == ref_tree);
}
}  // namespace xgboost::tree
