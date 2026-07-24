/**
 * Copyright 2026, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_MODEL_UTILS_H_
#define XGBOOST_TREE_MODEL_UTILS_H_

#include <cstdint>
#include <ios>
#include <string>

#include "../collective/broadcast.h"         // for Broadcast
#include "../collective/communicator-inl.h"  // for GetRank
#include "xgboost/context.h"                 // for Context
#include "xgboost/json.h"                    // for Json
#include "xgboost/linalg.h"                  // for MakeVec

namespace xgboost::tree {
inline Json BroadcastTreeModel(Context const* ctx, Json const& model) {
  std::string serialized;
  if (collective::GetRank() == 0) {
    Json::Dump(model, &serialized, std::ios::binary);
  }

  auto size = static_cast<std::int64_t>(serialized.size());
  auto rc = collective::Success() << [&] {
    return collective::Broadcast(ctx, linalg::MakeVec(&size, 1), 0);
  } << [&] {
    serialized.resize(size);
    return collective::Broadcast(ctx, linalg::MakeVec(serialized.data(), serialized.size()), 0);
  };
  collective::SafeColl(rc);
  return Json::Load(StringView{serialized}, std::ios::binary);
}
}  // namespace xgboost::tree

#endif  // XGBOOST_TREE_MODEL_UTILS_H_
