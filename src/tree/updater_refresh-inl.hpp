#ifndef XGBOOST_TREE_UPDATER_REFRESH_INL_HPP_
#define XGBOOST_TREE_UPDATER_REFRESH_INL_HPP_
/*!
 * \file updater_refresh-inl.hpp
 * \brief refresh the statistics and leaf value on the tree on the dataset
 * \author Tianqi Chen
 */
#include <vector>
#include <limits>
#include "./param.h"
#include "./updater.h"
#include "../utils/omp.h"

namespace xgboost {
namespace tree {
/*! \brief pruner that prunes a tree after growing finishs */
template<typename TStats>
class TreeRefresher: public IUpdater {
 public:
  virtual ~TreeRefresher(void) {}
  // set training parameter
  virtual void SetParam(const char *name, const char *val) {
    param.SetParam(name, val);
  }
  // update the tree, do pruning
  virtual void Update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {    
    if (trees.size() == 0) return;
    // number of threads
    // thread temporal space
    std::vector< std::vector<TStats> > stemp;
    std::vector<RegTree::FVec> fvec_temp;
    // setup temp space for each thread
    int nthread;
    #pragma omp parallel
    {
      nthread = omp_get_num_threads();
    }
    fvec_temp.resize(nthread, RegTree::FVec());
    stemp.resize(trees.size() * nthread, std::vector<TStats>());
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      for (size_t i = 0; i < trees.size(); ++i) {
        std::vector<TStats> &vec = stemp[tid * trees.size() + i];
        vec.resize(trees[i]->param.num_nodes, TStats(param));
        std::fill(vec.begin(), vec.end(), TStats(param));
      }
      fvec_temp[tid].Init(trees[0]->param.num_feature);
    }
    // start accumulating statistics
    utils::IIterator<RowBatch> *iter = p_fmat->RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch &batch = iter->Value();
      utils::Check(batch.size < std::numeric_limits<unsigned>::max(),
                   "too large batch size ");
      const bst_omp_uint nbatch = static_cast<bst_omp_uint>(batch.size);
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nbatch; ++i) {
        RowBatch::Inst inst = batch[i];
        const int tid = omp_get_thread_num();
        const bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
        RegTree::FVec &feats = fvec_temp[tid];
        feats.Fill(inst);
        for (size_t j = 0; j < trees.size(); ++j) {
          AddStats(*trees[j], feats, gpair, info, ridx,
                   &stemp[tid * trees.size() + j]);
        }
        feats.Drop(inst);
      }
    }
    // start update the trees using the statistics
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    for (size_t i = 0; i < trees.size(); ++i) {
      // aggregate
      #pragma omp parallel for schedule(static)
      for (int nid = 0; nid < trees[i]->param.num_nodes; ++nid) {
        for (int tid = 1; tid < nthread; ++tid) {
          stemp[i][nid].Add(stemp[tid * trees.size() + i][nid]);
        }
      }
      for (int rid = 0; rid < trees[i]->param.num_roots; ++rid) {
        this->Refresh(stemp[i], rid, trees[i]);
      }
    }
    // set learning rate back
    param.learning_rate = lr;
  }

 private:
  inline static void AddStats(const RegTree &tree,
                              const RegTree::FVec &feat,
                              const std::vector<bst_gpair> &gpair,
                              const BoosterInfo &info,
                              const bst_uint ridx,
                              std::vector<TStats> *p_gstats) {
    std::vector<TStats> &gstats = *p_gstats;
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
  inline void Refresh(const std::vector<TStats> &gstats,
                      int nid, RegTree *p_tree) {
    RegTree &tree = *p_tree;
    tree.stat(nid).base_weight = static_cast<float>(gstats[nid].CalcWeight(param));
    tree.stat(nid).sum_hess = static_cast<float>(gstats[nid].sum_hess);
    gstats[nid].SetLeafVec(param, tree.leafvec(nid));
    if (tree[nid].is_leaf()) {
      tree[nid].set_leaf(tree.stat(nid).base_weight * param.learning_rate);
    } else {
      tree.stat(nid).loss_chg = static_cast<float>(
          gstats[tree[nid].cleft()].CalcGain(param) +
          gstats[tree[nid].cright()].CalcGain(param) -
          gstats[nid].CalcGain(param));
      this->Refresh(gstats, tree[nid].cleft(), p_tree);
      this->Refresh(gstats, tree[nid].cright(), p_tree);
    }
  }
  // training parameter
  TrainParam param;
};

}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_REFRESH_INL_HPP_
