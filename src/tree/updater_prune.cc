/**
 * Copyright 2014-2026, XGBoost Contributors
 * \file updater_prune.cc
 * \brief prune a tree given the statistics
 * \author Tianqi Chen
 */
#include <xgboost/tree_updater.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "../collective/broadcast.h"         // for Broadcast
#include "../collective/communicator-inl.h"  // for GetRank, GetWorldSize
#include "../common/timer.h"
#include "./param.h"
#include "xgboost/base.h"
#include "xgboost/gradient.h"  // for GradientContainer
#include "xgboost/json.h"
#include "xgboost/linalg.h"  // for MakeVec

namespace xgboost::tree {
DMLC_REGISTRY_FILE_TAG(updater_prune);

/*! \brief pruner that prunes a tree after growing finishes */
class TreePruner : public TreeUpdater {
 public:
  explicit TreePruner(Context const* ctx) : TreeUpdater(ctx) { pruner_monitor_.Init("TreePruner"); }
  [[nodiscard]] char const* Name() const override { return "prune"; }
  // set training parameter
  void Configure(const Args&) override {}

  void LoadConfig(Json const&) override {}
  void SaveConfig(Json*) const override {}
  [[nodiscard]] bool CanModifyTree() const override { return true; }

  // update the tree, do pruning
  void Update(TrainParam const* param, GradientContainer*, DMatrix*,
              common::Span<HostDeviceVector<bst_node_t>>,
              std::vector<RegTree*> const& trees) override {
    pruner_monitor_.Start("PrunerUpdate");
    for (auto tree : trees) {
      this->DoPrune(param, tree);
    }
    this->Synchronize(trees);
    pruner_monitor_.Stop("PrunerUpdate");
  }

 private:
  // try to prune off current leaf
  bst_node_t TryPruneLeaf(TrainParam const* param, RegTree* p_tree, int nid, int depth,
                          int npruned) {
    auto& tree = *p_tree;
    CHECK(tree[nid].IsLeaf());
    if (tree[nid].IsRoot()) {
      return npruned;
    }
    bst_node_t pid = tree[nid].Parent();
    CHECK(!tree[pid].IsLeaf());
    RTreeNodeStat const& s = tree.Stat(pid);
    // Only prune when both child are leaf.
    auto left = tree[pid].LeftChild();
    auto right = tree[pid].RightChild();
    bool balanced = tree[left].IsLeaf() && right != RegTree::kInvalidNodeId && tree[right].IsLeaf();
    if (balanced && param->NeedPrune(s.loss_chg, depth)) {
      // need to be pruned
      tree.ChangeToLeaf(pid, param->learning_rate * s.base_weight);
      // tail recursion
      return this->TryPruneLeaf(param, p_tree, pid, depth - 1, npruned + 2);
    } else {
      return npruned;
    }
  }
  /*! \brief do pruning of a tree */
  void DoPrune(TrainParam const* param, RegTree* p_tree) {
    auto& tree = *p_tree;
    CHECK(!tree.IsMultiTarget()) << "Pruning" << MTNotImplemented();
    bst_node_t npruned = 0;
    for (int nid = 0; nid < tree.NumNodes(); ++nid) {
      if (tree[nid].IsLeaf() && !tree[nid].IsDeleted()) {
        npruned = this->TryPruneLeaf(param, p_tree, nid, tree.GetDepth(nid), npruned);
      }
    }
    LOG(INFO) << "tree pruning end, " << tree.NumExtraNodes() << " extra nodes, " << npruned
              << " pruned nodes, max_depth=" << tree.MaxDepth();
  }

  void Synchronize(std::vector<RegTree*> const& trees) {
    if (collective::GetWorldSize() == 1) {
      return;
    }

    auto rank = collective::GetRank();
    std::vector<char> serialized;
    if (rank == 0) {
      Json model{Array{}};
      auto& tree_models = get<Array>(model);
      for (auto tree : trees) {
        Json tree_model{Object{}};
        tree->SaveModel(&tree_model);
        tree_models.emplace_back(std::move(tree_model));
      }
      Json::Dump(model, &serialized, std::ios::binary);
    }

    std::size_t size = serialized.size();
    auto rc = collective::Broadcast(ctx_, linalg::MakeVec(&size, 1), 0);
    SafeColl(rc);
    serialized.resize(size);
    rc = collective::Broadcast(ctx_, linalg::MakeVec(serialized.data(), serialized.size()), 0);
    SafeColl(rc);

    if (rank != 0) {
      auto model = Json::Load(StringView{serialized.data(), serialized.size()}, std::ios::binary);
      auto const& tree_models = get<Array const>(model);
      CHECK_EQ(tree_models.size(), trees.size());
      for (std::size_t i = 0; i < trees.size(); ++i) {
        trees[i]->LoadModel(tree_models[i]);
      }
    }
  }

 private:
  common::Monitor pruner_monitor_;
};

XGBOOST_REGISTER_TREE_UPDATER(TreePruner, "prune")
    .describe("Pruner that prune the tree according to statistics.")
    .set_body([](Context const* ctx, ObjInfo const*) { return new TreePruner{ctx}; });
}  // namespace xgboost::tree
