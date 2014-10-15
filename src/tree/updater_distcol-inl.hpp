#ifndef XGBOOST_TREE_UPDATER_DISTCOL_INL_HPP_
#define XGBOOST_TREE_UPDATER_DISTCOL_INL_HPP_
/*!
 * \file updater_distcol-inl.hpp
 * \brief beta distributed version that takes a sub-column 
 *        and construct a tree
 * \author Tianqi Chen
 */
#include "../utils/bitmap.h"
#include "./updater_colmaker-inl.hpp"

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
  }
  virtual void Update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {    
    TStats::CheckInfo(info);
    utils::Check(trees.size() == 1, "DistColMaker: only support one tree at a time");
    builder.Update(gpair, p_fmat, info, trees[0]);
  }
 private:
  struct Builder : public ColMaker<TStats>::Builder {
   public:
    Builder(const TrainParam &param) 
        : ColMaker<TStats>::Builder(param) {
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
      // setup BitMap
      bitmap.Resize(this->position.size());
      bitmap.Clear();
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
            int nid = this->position[ridx];
            if (nid < 0) continue;
            // go back to parent, correct those who are not default
            nid = tree[nid].parent();
            if (tree[nid].split_index() == fid) {
              if (fvalue < tree[nid].split_cond()) {
                if (!tree[nid].default_left()) bitmap.SetTrue(ridx);
              } else {
                if (tree[nid].default_left()) bitmap.SetTrue(ridx);
              }
            }
          }
        }
      }
      // communicate bitmap
      //sync::AllReduce();
      const std::vector<bst_uint> &rowset = p_fmat->buffered_rowset();   
      // get the new position
      const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        int nid = this->position[ridx];
        if (nid >= 0 && bitmap.Get(ridx)) {
          nid = tree[nid].parent();
          if (tree[nid].default_left()) {
            this->position[ridx] = tree[nid].cright();
          } else {
            this->position[ridx] = tree[nid].cleft();
          }
        }
      }
    }
    // synchronize the best solution of each node
    virtual void SyncBestSolution(const std::vector<int> &qexpand) {
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        for (int tid = 0; tid < this->nthread; ++tid) {
          this->snode[nid].best.Update(this->stemp[tid][nid].best);
        }
      }
      // communicate best solution
      // sync::AllReduce
    }
    
   private:
    utils::BitMap bitmap;
  };
  // training parameter
  TrainParam param;
  // pointer to the builder
  Builder builder; 
};
}  // namespace tree
}  // namespace xgboost
#endif
