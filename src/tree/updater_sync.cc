/**
 * Copyright 2014-2023 by XBGoost Contributors
 * \file updater_sync.cc
 * \brief synchronize the tree in all distributed nodes
 */
#include <xgboost/tree_updater.h>

#include <limits>
#include <string>
#include <vector>

#include "../collective/communicator-inl.h"
#include "../common/io.h"
#include "xgboost/json.h"

namespace xgboost::tree {

DMLC_REGISTRY_FILE_TAG(updater_sync);

/*!
 * \brief syncher that synchronize the tree in all distributed nodes
 * can implement various strategies, so far it is always set to node 0's tree
 */
class TreeSyncher : public TreeUpdater {
 public:
  explicit TreeSyncher(Context const* tparam) : TreeUpdater(tparam) {}
  void Configure(const Args&) override {}

  void LoadConfig(Json const&) override {}
  void SaveConfig(Json*) const override {}

  [[nodiscard]] char const* Name() const override { return "prune"; }

  void Update(TrainParam const*, HostDeviceVector<GradientPair>*, DMatrix*,
              common::Span<HostDeviceVector<bst_node_t>> /*out_position*/,
              const std::vector<RegTree*>& trees) override {
    if (collective::GetWorldSize() == 1) return;
    std::string s_model;
    common::MemoryBufferStream fs(&s_model);
    int rank = collective::GetRank();
    if (rank == 0) {
      for (auto tree : trees) {
        tree->Save(&fs);
      }
    }
    fs.Seek(0);
    collective::Broadcast(&s_model, 0);
    for (auto tree : trees) {
      tree->Load(&fs);
    }
  }
};

XGBOOST_REGISTER_TREE_UPDATER(TreeSyncher, "sync")
    .describe("Syncher that synchronize the tree in all distributed nodes.")
    .set_body([](Context const* ctx, auto) { return new TreeSyncher(ctx); });
}  // namespace xgboost::tree
