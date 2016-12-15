/*!
 * Copyright 2014 by Contributors
 * \file updater_refresh.cc
 * \brief refresh the statistics and leaf value on the tree on the dataset
 * \author Tianqi Chen
 */

#include <xgboost/tree_updater.h>
#include <vector>
#include <limits>
#include "./param.h"
#include "../common/sync.h"
#include "../common/io.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_refresh);

/*! \brief pruner that prunes a tree after growing finishs */
template<typename TStats>
class TreeRefresher: public TreeUpdater {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    param.InitAllowUnknown(args);
  }
  // update the tree, do pruning
  void Update(const std::vector<bst_gpair> &gpair,
              DMatrix *p_fmat,
              const std::vector<RegTree*> &trees) override {
    if (trees.size() == 0) return;
    // number of threads
    // thread temporal space
    std::vector<std::vector<TStats> > stemp;
    std::vector<RegTree::FVec> fvec_temp;
    // setup temp space for each thread
    const int nthread = omp_get_max_threads();
    fvec_temp.resize(nthread, RegTree::FVec());
    stemp.resize(nthread, std::vector<TStats>());
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int num_nodes = 0;
      for (size_t i = 0; i < trees.size(); ++i) {
        num_nodes += trees[i]->param.num_nodes;
      }
      stemp[tid].resize(num_nodes, TStats(param));
      std::fill(stemp[tid].begin(), stemp[tid].end(), TStats(param));
      fvec_temp[tid].Init(trees[0]->param.num_feature);
    }
    // if it is C++11, use lazy evaluation for Allreduce,
    // to gain speedup in recovery
#if __cplusplus >= 201103L
    auto lazy_get_stats = [&]()
#endif
    {
      const MetaInfo &info = p_fmat->info();
      // start accumulating statistics
      dmlc::DataIter<RowBatch> *iter = p_fmat->RowIterator();
      iter->BeforeFirst();
      while (iter->Next()) {
        const RowBatch &batch = iter->Value();
        CHECK_LT(batch.size, std::numeric_limits<unsigned>::max());
        const bst_omp_uint nbatch = static_cast<bst_omp_uint>(batch.size);
        #pragma omp parallel for schedule(static)
        for (bst_omp_uint i = 0; i < nbatch; ++i) {
          RowBatch::Inst inst = batch[i];
          const int tid = omp_get_thread_num();
          const bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
          RegTree::FVec &feats = fvec_temp[tid];
          feats.Fill(inst);
          int offset = 0;
          for (size_t j = 0; j < trees.size(); ++j) {
            AddStats(*trees[j], feats, gpair, info, ridx,
                     dmlc::BeginPtr(stemp[tid]) + offset);
            offset += trees[j]->param.num_nodes;
          }
          feats.Drop(inst);
        }
      }
      // aggregate the statistics
      int num_nodes = static_cast<int>(stemp[0].size());
      #pragma omp parallel for schedule(static)
      for (int nid = 0; nid < num_nodes; ++nid) {
        for (int tid = 1; tid < nthread; ++tid) {
          stemp[0][nid].Add(stemp[tid][nid]);
        }
      }
    };
#if __cplusplus >= 201103L
    reducer.Allreduce(dmlc::BeginPtr(stemp[0]), stemp[0].size(), lazy_get_stats);
#else
    reducer.Allreduce(dmlc::BeginPtr(stemp[0]), stemp[0].size());
#endif
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    int offset = 0;
    for (size_t i = 0; i < trees.size(); ++i) {
      for (int rid = 0; rid < trees[i]->param.num_roots; ++rid) {
        this->Refresh(dmlc::BeginPtr(stemp[0]) + offset, rid, trees[i]);
      }
      offset += trees[i]->param.num_nodes;
    }
    // set learning rate back
    param.learning_rate = lr;
  }

 private:
  inline static void AddStats(const RegTree &tree,
                              const RegTree::FVec &feat,
                              const std::vector<bst_gpair> &gpair,
                              const MetaInfo &info,
                              const bst_uint ridx,
                              TStats *gstats) {
    // start from groups that belongs to current data
    int pid = static_cast<int>(info.GetRoot(ridx));
    gstats[pid].Add(gpair, info, ridx);
    // tranverse tree
    while (!tree[pid].is_leaf()) {
      unsigned split_index = tree[pid].split_index();
      pid = tree.GetNext(pid, feat.fvalue(split_index), feat.is_missing(split_index));
      gstats[pid].Add(gpair, info, ridx);
    }
  }
  inline void Refresh(const TStats *gstats,
                      int nid, RegTree *p_tree) {
    RegTree &tree = *p_tree;
    tree.stat(nid).base_weight = static_cast<bst_float>(gstats[nid].CalcWeight(param));
    tree.stat(nid).sum_hess = static_cast<bst_float>(gstats[nid].sum_hess);
    gstats[nid].SetLeafVec(param, tree.leafvec(nid));
    if (tree[nid].is_leaf()) {
      if (param.refresh_leaf) {
        tree[nid].set_leaf(tree.stat(nid).base_weight * param.learning_rate);
      }
    } else {
      tree.stat(nid).loss_chg = static_cast<bst_float>(
          gstats[tree[nid].cleft()].CalcGain(param) +
          gstats[tree[nid].cright()].CalcGain(param) -
          gstats[nid].CalcGain(param));
      this->Refresh(gstats, tree[nid].cleft(), p_tree);
      this->Refresh(gstats, tree[nid].cright(), p_tree);
    }
  }
  // training parameter
  TrainParam param;
  // reducer
  rabit::Reducer<TStats, TStats::Reduce> reducer;
};

XGBOOST_REGISTER_TREE_UPDATER(TreeRefresher, "refresh")
.describe("Refresher that refreshes the weight and statistics according to data.")
.set_body([]() {
    return new TreeRefresher<GradStats>();
  });
}  // namespace tree
}  // namespace xgboost
