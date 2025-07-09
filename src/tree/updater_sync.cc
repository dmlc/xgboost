/**
 * Copyright 2014-2025, XBGoost Contributors
 * \file updater_sync.cc
 * \brief synchronize the tree in all distributed nodes
 */
#include <string>
#include <vector>

#include "../collective/broadcast.h"         // for Broadcast
#include "../collective/communicator-inl.h"  // for GetRank, GetWorldSize
#include "xgboost/context.h"                 // for Context
#include "xgboost/json.h"                    // for Json, Object
#include "xgboost/linalg.h"                  // for Matrix
#include "xgboost/tree_updater.h"            // for TreeUpdater

namespace xgboost::tree {

DMLC_REGISTRY_FILE_TAG(updater_sync);

/*!
 * \brief syncher that synchronize the tree in all distributed nodes
 * can implement various strategies, so far it is always set to node 0's tree
 */
class TreeSyncher : public TreeUpdater {
 public:
  explicit TreeSyncher(Context const* tparam) : TreeUpdater{tparam} {}
  void Configure(Args const&) override {}

  void LoadConfig(Json const&) override {}
  void SaveConfig(Json*) const override {}

  [[nodiscard]] char const* Name() const override { return "sync"; }

  void Update(TrainParam const*, linalg::Matrix<GradientPair>*, DMatrix*,
              common::Span<HostDeviceVector<bst_node_t>> /*out_position*/,
              const std::vector<RegTree*>& trees) override {
    if (collective::GetWorldSize() == 1) {
      return;
    }

    Json model{Object{}};
    auto rank = collective::GetRank();
    if (rank == 0) {
      for (auto tree : trees) {
        tree->SaveModel(&model);
      }
    }
    std::vector<char> jmodel;
    Json::Dump(model, &jmodel, std::ios::binary);
    auto rc = collective::Broadcast(ctx_, linalg::MakeVec(jmodel.data(), jmodel.size()), 0);
    SafeColl(rc);

    for (auto tree : trees) {
      tree->LoadModel(model);
    }
  }
};

XGBOOST_REGISTER_TREE_UPDATER(TreeSyncher, "sync")
    .describe("Syncher that synchronize the tree in all distributed nodes.")
    .set_body([](Context const* ctx, auto) { return new TreeSyncher(ctx); });
}  // namespace xgboost::tree
