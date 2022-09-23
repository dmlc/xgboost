/*!
 * Copyright 2014-2022 by XGBoost Contributors
 * \file updater_refresh.cc
 * \brief refresh the statistics and leaf value on the tree on the dataset
 * \author Tianqi Chen
 */
#include <xgboost/tree_updater.h>

#include <limits>
#include <vector>

#include "../collective/communicator-inl.h"
#include "../common/io.h"
#include "../common/threading_utils.h"
#include "../predictor/predict_fn.h"
#include "./param.h"
#include "xgboost/json.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_refresh);

/*! \brief pruner that prunes a tree after growing finishs */
class TreeRefresher : public TreeUpdater {
 public:
  explicit TreeRefresher(GenericParameter const *ctx) : TreeUpdater(ctx) {}
  void Configure(const Args &args) override { param_.UpdateAllowUnknown(args); }
  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = ToJson(param_);
  }
  char const* Name() const override {
    return "refresh";
  }
  bool CanModifyTree() const override {
    return true;
  }
  // update the tree, do pruning
  void Update(HostDeviceVector<GradientPair> *gpair, DMatrix *p_fmat,
              common::Span<HostDeviceVector<bst_node_t>> /*out_position*/,
              const std::vector<RegTree *> &trees) override {
    if (trees.size() == 0) return;
    const std::vector<GradientPair> &gpair_h = gpair->ConstHostVector();
    // thread temporal space
    std::vector<std::vector<GradStats> > stemp;
    std::vector<RegTree::FVec> fvec_temp;
    // setup temp space for each thread
    const int nthread = ctx_->Threads();
    fvec_temp.resize(nthread, RegTree::FVec());
    stemp.resize(nthread, std::vector<GradStats>());
    dmlc::OMPException exc;
#pragma omp parallel num_threads(nthread)
    {
      exc.Run([&]() {
        int tid = omp_get_thread_num();
        int num_nodes = 0;
        for (auto tree : trees) {
          num_nodes += tree->param.num_nodes;
        }
        stemp[tid].resize(num_nodes, GradStats());
        std::fill(stemp[tid].begin(), stemp[tid].end(), GradStats());
        fvec_temp[tid].Init(trees[0]->param.num_feature);
      });
    }
    exc.Rethrow();
    // if it is C++11, use lazy evaluation for Allreduce,
    // to gain speedup in recovery
    auto lazy_get_stats = [&]() {
      const MetaInfo &info = p_fmat->Info();
      // start accumulating statistics
      for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
        auto page = batch.GetView();
        CHECK_LT(batch.Size(), std::numeric_limits<unsigned>::max());
        const auto nbatch = static_cast<bst_omp_uint>(batch.Size());
        common::ParallelFor(nbatch, ctx_->Threads(), [&](bst_omp_uint i) {
          SparsePage::Inst inst = page[i];
          const int tid = omp_get_thread_num();
          const auto ridx = static_cast<bst_uint>(batch.base_rowid + i);
          RegTree::FVec &feats = fvec_temp[tid];
          feats.Fill(inst);
          int offset = 0;
          for (auto tree : trees) {
            AddStats(*tree, feats, gpair_h, info, ridx,
                     dmlc::BeginPtr(stemp[tid]) + offset);
            offset += tree->param.num_nodes;
          }
          feats.Drop(inst);
        });
      }
      // aggregate the statistics
      auto num_nodes = static_cast<int>(stemp[0].size());
      common::ParallelFor(num_nodes, ctx_->Threads(), [&](int nid) {
        for (int tid = 1; tid < nthread; ++tid) {
          stemp[0][nid].Add(stemp[tid][nid]);
        }
      });
    };
    lazy_get_stats();
    collective::Allreduce<collective::Operation::kSum>(&dmlc::BeginPtr(stemp[0])->sum_grad,
                                                       stemp[0].size() * 2);
    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    int offset = 0;
    for (auto tree : trees) {
      this->Refresh(dmlc::BeginPtr(stemp[0]) + offset, 0, tree);
      offset += tree->param.num_nodes;
    }
    // set learning rate back
    param_.learning_rate = lr;
  }

 private:
  inline static void AddStats(const RegTree &tree,
                              const RegTree::FVec &feat,
                              const std::vector<GradientPair> &gpair,
                              const MetaInfo&,
                              const bst_uint ridx,
                              GradStats *gstats) {
    // start from groups that belongs to current data
    auto pid = 0;
    gstats[pid].Add(gpair[ridx]);
    auto const& cats = tree.GetCategoriesMatrix();
    // traverse tree
    while (!tree[pid].IsLeaf()) {
      unsigned split_index = tree[pid].SplitIndex();
      pid = predictor::GetNextNode<true, true>(
          tree[pid], pid, feat.GetFvalue(split_index), feat.IsMissing(split_index),
          cats);
      gstats[pid].Add(gpair[ridx]);
    }
  }
  inline void Refresh(const GradStats *gstats,
                      int nid, RegTree *p_tree) {
    RegTree &tree = *p_tree;
    tree.Stat(nid).base_weight =
        static_cast<bst_float>(CalcWeight(param_, gstats[nid]));
    tree.Stat(nid).sum_hess = static_cast<bst_float>(gstats[nid].sum_hess);
    if (tree[nid].IsLeaf()) {
      if (param_.refresh_leaf) {
        tree[nid].SetLeaf(tree.Stat(nid).base_weight * param_.learning_rate);
      }
    } else {
      tree.Stat(nid).loss_chg = static_cast<bst_float>(
          xgboost::tree::CalcGain(param_, gstats[tree[nid].LeftChild()]) +
          xgboost::tree::CalcGain(param_, gstats[tree[nid].RightChild()]) -
          xgboost::tree::CalcGain(param_, gstats[nid]));
      this->Refresh(gstats, tree[nid].LeftChild(), p_tree);
      this->Refresh(gstats, tree[nid].RightChild(), p_tree);
    }
  }
  // training parameter
  TrainParam param_;
};

XGBOOST_REGISTER_TREE_UPDATER(TreeRefresher, "refresh")
    .describe("Refresher that refreshes the weight and statistics according to data.")
    .set_body([](GenericParameter const *ctx, ObjInfo) { return new TreeRefresher(ctx); });
}  // namespace tree
}  // namespace xgboost
