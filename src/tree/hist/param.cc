/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#include "param.h"

#include <ios>     // for binary
#include <string>  // for string

#include "../../collective/broadcast.h"         // for Broadcast
#include "../../collective/communicator-inl.h"  // for GetRank
#include "xgboost/json.h"                       // for Object, Json
#include "xgboost/linalg.h"                     // for MakeVec
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

  auto nchars{static_cast<std::int64_t>(s_model.size())};
  auto rc = collective::Success() << [&] {
    return collective::Broadcast(ctx, linalg::MakeVec(&nchars, 1), 0);
  } << [&] {
    s_model.resize(nchars);
    return collective::Broadcast(ctx, linalg::MakeVec(s_model.data(), s_model.size()), 0);
  };
  collective::SafeColl(rc);

  RegTree ref_tree{};  // rank 0 tree
  auto j_ref_tree = Json::Load(StringView{s_model}, std::ios::binary);
  ref_tree.LoadModel(j_ref_tree);
  CHECK(*local_tree == ref_tree);
}
}  // namespace xgboost::tree
