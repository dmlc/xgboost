/*!
 * Copyright 2014 by Contributors
 * \file updater_distcol-inl.hpp
 * \brief beta distributed version that takes a sub-column
 *        and construct a tree
 * \author Tianqi Chen
 */
#ifndef XGBOOST_TREE_UPDATER_DISTCOL_INL_HPP_
#define XGBOOST_TREE_UPDATER_DISTCOL_INL_HPP_

#include <vector>
#include <algorithm>
#include "../sync/sync.h"
#include "../utils/bitmap.h"
#include "../utils/io.h"
#include "./updater_colmaker-inl.hpp"
#include "./updater_prune-inl.hpp"

namespace xgboost {
namespace tree {
template<typename TStats>
class DistColMaker : public ColMaker<TStats> {
 public:
  DistColMaker(void) : builder(param) {}
  virtual ~DistColMaker(void) {}
  // set training parameter
  virtual void SetParam(const char *name, const char *val) {
    param.SetParam(name, val);
    pruner.SetParam(name, val);
  }
  virtual void Update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {
    TStats::CheckInfo(info);
    utils::Check(trees.size() == 1, "DistColMaker: only support one tree at a time");
    // build the tree
    builder.Update(gpair, p_fmat, info, trees[0]);
    //// prune the tree, note that pruner will sync the tree
    pruner.Update(gpair, p_fmat, info, trees);
    // update position after the tree is pruned
    builder.UpdatePosition(p_fmat, *trees[0]);
  }
  virtual const int* GetLeafPosition(void) const {
    return builder.GetLeafPosition();
  }

 private:
  struct Builder : public ColMaker<TStats>::Builder {
   public:
    explicit Builder(const TrainParam &param)
        : ColMaker<TStats>::Builder(param) {
    }
    inline void UpdatePosition(IFMatrix *p_fmat, const RegTree &tree) {
      const std::vector<bst_uint> &rowset = p_fmat->buffered_rowset();
      const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        int nid = this->DecodePosition(ridx);
        while (tree[nid].is_deleted()) {
          nid = tree[nid].parent();
          utils::Assert(nid >=0, "distributed learning error");
        }
        this->position[ridx] = nid;
      }
    }
    virtual const int* GetLeafPosition(void) const {
      return BeginPtr(this->position);
    }

   protected:
    virtual void SetNonDefaultPosition(const std::vector<int> &qexpand,
                                       IFMatrix *p_fmat, const RegTree &tree) {
      // step 2, classify the non-default data into right places
      std::vector<unsigned> fsplits;
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        if (!tree[nid].is_leaf()) {
          fsplits.push_back(tree[nid].split_index());
        }
      }
      // get the candidate split index
      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());
      while (fsplits.size() != 0 && fsplits.back() >= p_fmat->NumCol()) {
        fsplits.pop_back();
      }
      // bitmap is only word concurrent, set to bool first
      {
        bst_omp_uint ndata = static_cast<bst_omp_uint>(this->position.size());
        boolmap.resize(ndata);
        #pragma omp parallel for schedule(static)
        for (bst_omp_uint j = 0; j < ndata; ++j) {
            boolmap[j] = 0;
        }
      }
      utils::IIterator<ColBatch> *iter = p_fmat->ColIterator(fsplits);
      while (iter->Next()) {
        const ColBatch &batch = iter->Value();
        for (size_t i = 0; i < batch.size; ++i) {
          ColBatch::Inst col = batch[i];
          const bst_uint fid = batch.col_index[i];
          const bst_omp_uint ndata = static_cast<bst_omp_uint>(col.length);
          #pragma omp parallel for schedule(static)
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const float fvalue = col[j].fvalue;
            const int nid = this->DecodePosition(ridx);
            if (!tree[nid].is_leaf() && tree[nid].split_index() == fid) {
              if (fvalue < tree[nid].split_cond()) {
                if (!tree[nid].default_left()) boolmap[ridx] = 1;
              } else {
                if (tree[nid].default_left()) boolmap[ridx] = 1;
              }
            }
          }
        }
      }

      bitmap.InitFromBool(boolmap);
      // communicate bitmap
      rabit::Allreduce<rabit::op::BitOR>(BeginPtr(bitmap.data), bitmap.data.size());
      const std::vector<bst_uint> &rowset = p_fmat->buffered_rowset();
      // get the new position
      const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        const int nid = this->DecodePosition(ridx);
        if (bitmap.Get(ridx)) {
          utils::Assert(!tree[nid].is_leaf(), "inconsistent reduce information");
          if (tree[nid].default_left()) {
            this->SetEncodePosition(ridx, tree[nid].cright());
          } else {
            this->SetEncodePosition(ridx, tree[nid].cleft());
          }
        }
      }
    }
    // synchronize the best solution of each node
    virtual void SyncBestSolution(const std::vector<int> &qexpand) {
      std::vector<SplitEntry> vec;
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        for (int tid = 0; tid < this->nthread; ++tid) {
          this->snode[nid].best.Update(this->stemp[tid][nid].best);
        }
        vec.push_back(this->snode[nid].best);
      }
      // TODO(tqchen) lazy version
      // communicate best solution
      reducer.Allreduce(BeginPtr(vec), vec.size());
      // assign solution back
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        this->snode[nid].best = vec[i];
      }
    }

   private:
    utils::BitMap bitmap;
    std::vector<int> boolmap;
    rabit::Reducer<SplitEntry, SplitEntry::Reduce> reducer;
  };
  // we directly introduce pruner here
  TreePruner pruner;
  // training parameter
  TrainParam param;
  // pointer to the builder
  Builder builder;
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_DISTCOL_INL_HPP_
