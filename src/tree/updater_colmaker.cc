/*!
 * Copyright 2014 by Contributors
 * \file updater_colmaker.cc
 * \brief use columnwise update to construct a tree
 * \author Tianqi Chen
 */
#include <xgboost/tree_updater.h>
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include "./param.h"
#include "../common/random.h"
#include "../common/bitmap.h"
#include "../common/sync.h"
#include "split_evaluator.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_colmaker);

/*! \brief column-wise update to construct a tree */
class ColMaker: public TreeUpdater {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
    spliteval_.reset(SplitEvaluator::Create(param_.split_evaluator));
    spliteval_->Init(args);
  }

  void Update(HostDeviceVector<GradientPair> *gpair,
              DMatrix* dmat,
              const std::vector<RegTree*> &trees) override {
    GradStats::CheckInfo(dmat->Info());
    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    // build tree
    for (auto tree : trees) {
      Builder builder(
        param_,
        std::unique_ptr<SplitEvaluator>(spliteval_->GetHostClone()));
      builder.Update(gpair->HostVector(), dmat, tree);
    }
    param_.learning_rate = lr;
  }

 protected:
  // training parameter
  TrainParam param_;
  // SplitEvaluator that will be cloned for each Builder
  std::unique_ptr<SplitEvaluator> spliteval_;
  // data structure
  /*! \brief per thread x per node entry to store tmp data */
  struct ThreadEntry {
    /*! \brief statistics of data */
    GradStats stats;
    /*! \brief extra statistics of data */
    GradStats stats_extra;
    /*! \brief last feature value scanned */
    bst_float last_fvalue;
    /*! \brief first feature value scanned */
    bst_float first_fvalue;
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    explicit ThreadEntry(const TrainParam &param)
        : stats(param), stats_extra(param) {
    }
  };
  struct NodeEntry {
    /*! \brief statics for node entry */
    GradStats stats;
    /*! \brief loss of this node, without split */
    bst_float root_gain;
    /*! \brief weight calculated related to current data */
    bst_float weight;
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    explicit NodeEntry(const TrainParam& param)
        : stats(param), root_gain(0.0f), weight(0.0f){
    }
  };
  // actual builder that runs the algorithm
  class Builder {
   public:
    // constructor
    explicit Builder(const TrainParam& param,
                     std::unique_ptr<SplitEvaluator> spliteval)
        : param_(param), nthread_(omp_get_max_threads()),
          spliteval_(std::move(spliteval)) {}
    // update one tree, growing
    virtual void Update(const std::vector<GradientPair>& gpair,
                        DMatrix* p_fmat,
                        RegTree* p_tree) {
      std::vector<int> newnodes;
      this->InitData(gpair, *p_fmat, *p_tree);
      this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
      for (int depth = 0; depth < param_.max_depth; ++depth) {
        this->FindSplit(depth, qexpand_, gpair, p_fmat, p_tree);
        this->ResetPosition(qexpand_, p_fmat, *p_tree);
        this->UpdateQueueExpand(*p_tree, qexpand_, &newnodes);
        this->InitNewNode(newnodes, gpair, *p_fmat, *p_tree);
        for (auto nid : qexpand_) {
          if ((*p_tree)[nid].IsLeaf()) {
            continue;
          }
          int cleft = (*p_tree)[nid].LeftChild();
          int cright = (*p_tree)[nid].RightChild();
          spliteval_->AddSplit(nid,
                               cleft,
                               cright,
                               snode_[nid].best.SplitIndex(),
                               snode_[cleft].weight,
                               snode_[cright].weight);
        }
        qexpand_ = newnodes;
        // if nothing left to be expand, break
        if (qexpand_.size() == 0) break;
      }
      // set all the rest expanding nodes to leaf
      for (const int nid : qexpand_) {
        (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
      }
      // remember auxiliary statistics in the tree node
      for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
        p_tree->Stat(nid).loss_chg = snode_[nid].best.loss_chg;
        p_tree->Stat(nid).base_weight = snode_[nid].weight;
        p_tree->Stat(nid).sum_hess = static_cast<float>(snode_[nid].stats.sum_hess);
        snode_[nid].stats.SetLeafVec(param_, p_tree->Leafvec(nid));
      }
    }

   protected:
    // initialize temp data structure
    inline void InitData(const std::vector<GradientPair>& gpair,
                         const DMatrix& fmat,
                         const RegTree& tree) {
      CHECK_EQ(tree.param.num_nodes, tree.param.num_roots)
          << "ColMaker: can only grow new tree";
      const std::vector<unsigned>& root_index = fmat.Info().root_index_;
      const RowSet& rowset = fmat.BufferedRowset();
      {
        // setup position
        position_.resize(gpair.size());
        if (root_index.size() == 0) {
          for (size_t i = 0; i < rowset.Size(); ++i) {
            position_[rowset[i]] = 0;
          }
        } else {
          for (size_t i = 0; i < rowset.Size(); ++i) {
            const bst_uint ridx = rowset[i];
            position_[ridx] = root_index[ridx];
            CHECK_LT(root_index[ridx], (unsigned)tree.param.num_roots);
          }
        }
        // mark delete for the deleted datas
        for (size_t i = 0; i < rowset.Size(); ++i) {
          const bst_uint ridx = rowset[i];
          if (gpair[ridx].GetHess() < 0.0f) position_[ridx] = ~position_[ridx];
        }
        // mark subsample
        if (param_.subsample < 1.0f) {
          std::bernoulli_distribution coin_flip(param_.subsample);
          auto& rnd = common::GlobalRandom();
          for (size_t i = 0; i < rowset.Size(); ++i) {
            const bst_uint ridx = rowset[i];
            if (gpair[ridx].GetHess() < 0.0f) continue;
            if (!coin_flip(rnd)) position_[ridx] = ~position_[ridx];
          }
        }
      }
      {
        // initialize feature index
        auto ncol = static_cast<unsigned>(fmat.Info().num_col_);
        for (unsigned i = 0; i < ncol; ++i) {
          if (fmat.GetColSize(i) != 0) {
            feat_index_.push_back(i);
          }
        }
        unsigned n = std::max(static_cast<unsigned>(1),
                              static_cast<unsigned>(param_.colsample_bytree * feat_index_.size()));
        std::shuffle(feat_index_.begin(), feat_index_.end(), common::GlobalRandom());
        CHECK_GT(param_.colsample_bytree, 0U)
            << "colsample_bytree cannot be zero.";
        feat_index_.resize(n);
      }
      {
        // setup temp space for each thread
        // reserve a small space
        stemp_.clear();
        stemp_.resize(this->nthread_, std::vector<ThreadEntry>());
        for (auto& i : stemp_) {
          i.clear(); i.reserve(256);
        }
        snode_.reserve(256);
      }
      {
        // expand query
        qexpand_.reserve(256); qexpand_.clear();
        for (int i = 0; i < tree.param.num_roots; ++i) {
          qexpand_.push_back(i);
        }
      }
    }
    /*!
     * \brief initialize the base_weight, root_gain,
     *  and NodeEntry for all the new nodes in qexpand
     */
    inline void InitNewNode(const std::vector<int>& qexpand,
                            const std::vector<GradientPair>& gpair,
                            const DMatrix& fmat,
                            const RegTree& tree) {
      {
        // setup statistics space for each tree node
        for (auto& i : stemp_) {
          i.resize(tree.param.num_nodes, ThreadEntry(param_));
        }
        snode_.resize(tree.param.num_nodes, NodeEntry(param_));
      }
      const RowSet &rowset = fmat.BufferedRowset();
      const MetaInfo& info = fmat.Info();
      // setup position
      const auto ndata = static_cast<bst_omp_uint>(rowset.Size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        const int tid = omp_get_thread_num();
        if (position_[ridx] < 0) continue;
        stemp_[tid][position_[ridx]].stats.Add(gpair, info, ridx);
      }
      // sum the per thread statistics together
      for (int nid : qexpand) {
        GradStats stats(param_);
        for (auto& s : stemp_) {
          stats.Add(s[nid].stats);
        }
        // update node statistics
        snode_[nid].stats = stats;
      }
      // calculating the weights
      for (int nid : qexpand) {
        bst_uint parentid = tree[nid].Parent();
        snode_[nid].weight = static_cast<float>(
            spliteval_->ComputeWeight(parentid, snode_[nid].stats));
        snode_[nid].root_gain = static_cast<float>(
            spliteval_->ComputeScore(parentid, snode_[nid].stats, snode_[nid].weight));
      }
    }
    /*! \brief update queue expand add in new leaves */
    inline void UpdateQueueExpand(const RegTree& tree,
                                  const std::vector<int> &qexpand,
                                  std::vector<int>* p_newnodes) {
      p_newnodes->clear();
      for (int nid : qexpand) {
        if (!tree[ nid ].IsLeaf()) {
          p_newnodes->push_back(tree[nid].LeftChild());
          p_newnodes->push_back(tree[nid].RightChild());
        }
      }
    }
    // parallel find the best split of current fid
    // this function does not support nested functions
    inline void ParallelFindSplit(const SparsePage::Inst &col,
                                  bst_uint fid,
                                  const DMatrix &fmat,
                                  const std::vector<GradientPair> &gpair) {
      // TODO(tqchen): double check stats order.
      const MetaInfo& info = fmat.Info();
      const bool ind = col.length != 0 && col.data[0].fvalue == col.data[col.length - 1].fvalue;
      bool need_forward = param_.NeedForwardSearch(fmat.GetColDensity(fid), ind);
      bool need_backward = param_.NeedBackwardSearch(fmat.GetColDensity(fid), ind);
      const std::vector<int> &qexpand = qexpand_;
      #pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        std::vector<ThreadEntry> &temp = stemp_[tid];
        // cleanup temp statistics
        for (int j : qexpand) {
          temp[j].stats.Clear();
        }
        bst_uint step = (col.length + this->nthread_ - 1) / this->nthread_;
        bst_uint end = std::min(col.length, step * (tid + 1));
        for (bst_uint i = tid * step; i < end; ++i) {
          const bst_uint ridx = col[i].index;
          const int nid = position_[ridx];
          if (nid < 0) continue;
          const bst_float fvalue = col[i].fvalue;
          if (temp[nid].stats.Empty()) {
            temp[nid].first_fvalue = fvalue;
          }
          temp[nid].stats.Add(gpair, info, ridx);
          temp[nid].last_fvalue = fvalue;
        }
      }
      // start collecting the partial sum statistics
      auto nnode = static_cast<bst_omp_uint>(qexpand.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint j = 0; j < nnode; ++j) {
        const int nid = qexpand[j];
        GradStats sum(param_), tmp(param_), c(param_);
        for (int tid = 0; tid < this->nthread_; ++tid) {
          tmp = stemp_[tid][nid].stats;
          stemp_[tid][nid].stats = sum;
          sum.Add(tmp);
          if (tid != 0) {
            std::swap(stemp_[tid - 1][nid].last_fvalue, stemp_[tid][nid].first_fvalue);
          }
        }
        for (int tid = 0; tid < this->nthread_; ++tid) {
          stemp_[tid][nid].stats_extra = sum;
          ThreadEntry &e = stemp_[tid][nid];
          bst_float fsplit;
          if (tid != 0) {
            if (stemp_[tid - 1][nid].last_fvalue != e.first_fvalue) {
              fsplit = (stemp_[tid - 1][nid].last_fvalue + e.first_fvalue) * 0.5f;
            } else {
              continue;
            }
          } else {
            fsplit = e.first_fvalue - kRtEps;
          }
          if (need_forward && tid != 0) {
            c.SetSubstract(snode_[nid].stats, e.stats);
            if (c.sum_hess >= param_.min_child_weight &&
                e.stats.sum_hess >= param_.min_child_weight) {
              auto loss_chg = static_cast<bst_float>(
                  spliteval_->ComputeSplitScore(nid, fid, e.stats, c) -
                  snode_[nid].root_gain);
              e.best.Update(loss_chg, fid, fsplit, false);
            }
          }
          if (need_backward) {
            tmp.SetSubstract(sum, e.stats);
            c.SetSubstract(snode_[nid].stats, tmp);
            if (c.sum_hess >= param_.min_child_weight &&
                tmp.sum_hess >= param_.min_child_weight) {
              auto loss_chg = static_cast<bst_float>(
                  spliteval_->ComputeSplitScore(nid, fid, tmp, c) -
                  snode_[nid].root_gain);
              e.best.Update(loss_chg, fid, fsplit, true);
            }
          }
        }
        if (need_backward) {
          tmp = sum;
          ThreadEntry &e = stemp_[this->nthread_-1][nid];
          c.SetSubstract(snode_[nid].stats, tmp);
          if (c.sum_hess >= param_.min_child_weight &&
              tmp.sum_hess >= param_.min_child_weight) {
            auto loss_chg = static_cast<bst_float>(
                spliteval_->ComputeSplitScore(nid, fid, tmp, c) -
                snode_[nid].root_gain);
            e.best.Update(loss_chg, fid, e.last_fvalue + kRtEps, true);
          }
        }
      }
      // rescan, generate candidate split
      #pragma omp parallel
      {
        GradStats c(param_), cright(param_);
        const int tid = omp_get_thread_num();
        std::vector<ThreadEntry> &temp = stemp_[tid];
        bst_uint step = (col.length + this->nthread_ - 1) / this->nthread_;
        bst_uint end = std::min(col.length, step * (tid + 1));
        for (bst_uint i = tid * step; i < end; ++i) {
          const bst_uint ridx = col[i].index;
          const int nid = position_[ridx];
          if (nid < 0) continue;
          const bst_float fvalue = col[i].fvalue;
          // get the statistics of nid
          ThreadEntry &e = temp[nid];
          if (e.stats.Empty()) {
            e.stats.Add(gpair, info, ridx);
            e.first_fvalue = fvalue;
          } else {
            // forward default right
            if (fvalue != e.first_fvalue) {
              if (need_forward) {
                c.SetSubstract(snode_[nid].stats, e.stats);
                if (c.sum_hess >= param_.min_child_weight &&
                    e.stats.sum_hess >= param_.min_child_weight) {
                  auto loss_chg = static_cast<bst_float>(
                      spliteval_->ComputeSplitScore(nid, fid, e.stats, c) -
                      snode_[nid].root_gain);
                  e.best.Update(loss_chg, fid, (fvalue + e.first_fvalue) * 0.5f,
                                false);
                }
              }
              if (need_backward) {
                cright.SetSubstract(e.stats_extra, e.stats);
                c.SetSubstract(snode_[nid].stats, cright);
                if (c.sum_hess >= param_.min_child_weight &&
                    cright.sum_hess >= param_.min_child_weight) {
                  auto loss_chg = static_cast<bst_float>(
                      spliteval_->ComputeSplitScore(nid, fid, c, cright) -
                      snode_[nid].root_gain);
                  e.best.Update(loss_chg, fid, (fvalue + e.first_fvalue) * 0.5f, true);
                }
              }
            }
            e.stats.Add(gpair, info, ridx);
            e.first_fvalue = fvalue;
          }
        }
      }
    }
    // update enumeration solution
    inline void UpdateEnumeration(int nid, GradientPair gstats,
                                  bst_float fvalue, int d_step, bst_uint fid,
                                  GradStats &c, std::vector<ThreadEntry> &temp) { // NOLINT(*)
      // get the statistics of nid
      ThreadEntry &e = temp[nid];
      // test if first hit, this is fine, because we set 0 during init
      if (e.stats.Empty()) {
        e.stats.Add(gstats);
        e.last_fvalue = fvalue;
      } else {
        // try to find a split
        if (fvalue != e.last_fvalue &&
            e.stats.sum_hess >= param_.min_child_weight) {
          c.SetSubstract(snode_[nid].stats, e.stats);
          if (c.sum_hess >= param_.min_child_weight) {
            bst_float loss_chg;
            if (d_step == -1) {
              loss_chg = static_cast<bst_float>(
                  spliteval_->ComputeSplitScore(nid, fid, c, e.stats) -
                  snode_[nid].root_gain);
            } else {
              loss_chg = static_cast<bst_float>(
                  spliteval_->ComputeSplitScore(nid, fid, e.stats, c) -
                  snode_[nid].root_gain);
            }
            e.best.Update(loss_chg, fid, (fvalue + e.last_fvalue) * 0.5f,
                          d_step == -1);
          }
        }
        // update the statistics
        e.stats.Add(gstats);
        e.last_fvalue = fvalue;
      }
    }
    // same as EnumerateSplit, with cacheline prefetch optimization
    inline void EnumerateSplitCacheOpt(const Entry *begin,
                                       const Entry *end,
                                       int d_step,
                                       bst_uint fid,
                                       const std::vector<GradientPair> &gpair,
                                       std::vector<ThreadEntry> &temp) { // NOLINT(*)
      const std::vector<int> &qexpand = qexpand_;
      // clear all the temp statistics
      for (auto nid : qexpand) {
        temp[nid].stats.Clear();
      }
      // left statistics
      GradStats c(param_);
      // local cache buffer for position and gradient pair
      constexpr int kBuffer = 32;
      int buf_position[kBuffer] = {};
      GradientPair buf_gpair[kBuffer] = {};
      // aligned ending position
      const Entry *align_end;
      if (d_step > 0) {
        align_end = begin + (end - begin) / kBuffer * kBuffer;
      } else {
        align_end = begin - (begin - end) / kBuffer * kBuffer;
      }
      int i;
      const Entry *it;
      const int align_step = d_step * kBuffer;
      // internal cached loop
      for (it = begin; it != align_end; it += align_step) {
        const Entry *p;
        for (i = 0, p = it; i < kBuffer; ++i, p += d_step) {
          buf_position[i] = position_[p->index];
          buf_gpair[i] = gpair[p->index];
        }
        for (i = 0, p = it; i < kBuffer; ++i, p += d_step) {
          const int nid = buf_position[i];
          if (nid < 0) continue;
          this->UpdateEnumeration(nid, buf_gpair[i],
                                  p->fvalue, d_step,
                                  fid, c, temp);
        }
      }
      // finish up the ending piece
      for (it = align_end, i = 0; it != end; ++i, it += d_step) {
        buf_position[i] = position_[it->index];
        buf_gpair[i] = gpair[it->index];
      }
      for (it = align_end, i = 0; it != end; ++i, it += d_step) {
        const int nid = buf_position[i];
        if (nid < 0) continue;
        this->UpdateEnumeration(nid, buf_gpair[i],
                                it->fvalue, d_step,
                                fid, c, temp);
      }
      // finish updating all statistics, check if it is possible to include all sum statistics
      for (int nid : qexpand) {
        ThreadEntry &e = temp[nid];
        c.SetSubstract(snode_[nid].stats, e.stats);
        if (e.stats.sum_hess >= param_.min_child_weight &&
            c.sum_hess >= param_.min_child_weight) {
          bst_float loss_chg;
          if (d_step == -1) {
            loss_chg = static_cast<bst_float>(
                spliteval_->ComputeSplitScore(nid, fid, c, e.stats) -
                snode_[nid].root_gain);
          } else {
            loss_chg = static_cast<bst_float>(
                spliteval_->ComputeSplitScore(nid, fid, e.stats, c) -
                snode_[nid].root_gain);
          }
          const bst_float gap = std::abs(e.last_fvalue) + kRtEps;
          const bst_float delta = d_step == +1 ? gap: -gap;
          e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1);
        }
      }
    }

    // enumerate the split values of specific feature
    inline void EnumerateSplit(const Entry *begin,
                               const Entry *end,
                               int d_step,
                               bst_uint fid,
                               const std::vector<GradientPair> &gpair,
                               const MetaInfo &info,
                               std::vector<ThreadEntry> &temp) { // NOLINT(*)
      // use cacheline aware optimization
      if (GradStats::kSimpleStats != 0 && param_.cache_opt != 0) {
        EnumerateSplitCacheOpt(begin, end, d_step, fid, gpair, temp);
        return;
      }
      const std::vector<int> &qexpand = qexpand_;
      // clear all the temp statistics
      for (auto nid : qexpand) {
        temp[nid].stats.Clear();
      }
      // left statistics
      GradStats c(param_);
      for (const Entry *it = begin; it != end; it += d_step) {
        const bst_uint ridx = it->index;
        const int nid = position_[ridx];
        if (nid < 0) continue;
        // start working
        const bst_float fvalue = it->fvalue;
        // get the statistics of nid
        ThreadEntry &e = temp[nid];
        // test if first hit, this is fine, because we set 0 during init
        if (e.stats.Empty()) {
          e.stats.Add(gpair, info, ridx);
          e.last_fvalue = fvalue;
        } else {
          // try to find a split
          if (fvalue != e.last_fvalue &&
              e.stats.sum_hess >= param_.min_child_weight) {
            c.SetSubstract(snode_[nid].stats, e.stats);
            if (c.sum_hess >= param_.min_child_weight) {
              bst_float loss_chg;
              if (d_step == -1) {
                loss_chg = static_cast<bst_float>(
                    spliteval_->ComputeSplitScore(nid, fid, c, e.stats) -
                    snode_[nid].root_gain);
              } else {
                loss_chg = static_cast<bst_float>(
                    spliteval_->ComputeSplitScore(nid, fid, e.stats, c) -
                    snode_[nid].root_gain);
              }
              e.best.Update(loss_chg, fid, (fvalue + e.last_fvalue) * 0.5f, d_step == -1);
            }
          }
          // update the statistics
          e.stats.Add(gpair, info, ridx);
          e.last_fvalue = fvalue;
        }
      }
      // finish updating all statistics, check if it is possible to include all sum statistics
      for (int nid : qexpand) {
        ThreadEntry &e = temp[nid];
        c.SetSubstract(snode_[nid].stats, e.stats);
        if (e.stats.sum_hess >= param_.min_child_weight &&
            c.sum_hess >= param_.min_child_weight) {
          bst_float loss_chg;
          if (d_step == -1) {
            loss_chg = static_cast<bst_float>(
                spliteval_->ComputeSplitScore(nid, fid, c, e.stats) -
                snode_[nid].root_gain);
          } else {
            loss_chg = static_cast<bst_float>(
                spliteval_->ComputeSplitScore(nid, fid, e.stats, c) -
                snode_[nid].root_gain);
          }
          const bst_float gap = std::abs(e.last_fvalue) + kRtEps;
          const bst_float delta = d_step == +1 ? gap: -gap;
          e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1);
        }
      }
    }

    // update the solution candidate
    virtual void UpdateSolution(const SparsePage &batch,
                                const std::vector<bst_uint> &feat_set,
                                const std::vector<GradientPair> &gpair,
                                const DMatrix &fmat) {
      const MetaInfo& info = fmat.Info();
      // start enumeration
      const auto num_features = static_cast<bst_omp_uint>(feat_set.size());
      #if defined(_OPENMP)
      const int batch_size = std::max(static_cast<int>(num_features / this->nthread_ / 32), 1);
      #endif
      int poption = param_.parallel_option;
      if (poption == 2) {
        poption = static_cast<int>(num_features) * 2 < this->nthread_ ? 1 : 0;
      }
      if (poption == 0) {
        #pragma omp parallel for schedule(dynamic, batch_size)
        for (bst_omp_uint i = 0; i < num_features; ++i) {
          int fid = feat_set[i];
          const int tid = omp_get_thread_num();
          auto c = batch[fid];
          const bool ind = c.length != 0 && c.data[0].fvalue == c.data[c.length - 1].fvalue;
          if (param_.NeedForwardSearch(fmat.GetColDensity(fid), ind)) {
            this->EnumerateSplit(c.data, c.data + c.length, +1,
                                 fid, gpair, info, stemp_[tid]);
          }
          if (param_.NeedBackwardSearch(fmat.GetColDensity(fid), ind)) {
            this->EnumerateSplit(c.data + c.length - 1, c.data - 1, -1,
                                 fid, gpair, info, stemp_[tid]);
          }
        }
      } else {
        for (bst_omp_uint fid = 0; fid < num_features; ++fid) {
          this->ParallelFindSplit(batch[fid], fid,
                                  fmat, gpair);
        }
      }
    }
    // find splits at current level, do split per level
    inline void FindSplit(int depth,
                          const std::vector<int> &qexpand,
                          const std::vector<GradientPair> &gpair,
                          DMatrix *p_fmat,
                          RegTree *p_tree) {
      std::vector<bst_uint> feat_set = feat_index_;
      if (param_.colsample_bylevel != 1.0f) {
        std::shuffle(feat_set.begin(), feat_set.end(), common::GlobalRandom());
        unsigned n = std::max(static_cast<unsigned>(1),
                              static_cast<unsigned>(param_.colsample_bylevel * feat_index_.size()));
        CHECK_GT(param_.colsample_bylevel, 0U)
            << "colsample_bylevel cannot be zero.";
        feat_set.resize(n);
      }
      auto iter = p_fmat->ColIterator();
      while (iter->Next()) {
        this->UpdateSolution(iter->Value(), feat_set, gpair, *p_fmat);
      }
      // after this each thread's stemp will get the best candidates, aggregate results
      this->SyncBestSolution(qexpand);
      // get the best result, we can synchronize the solution
      for (int nid : qexpand) {
        NodeEntry &e = snode_[nid];
        // now we know the solution in snode[nid], set split
        if (e.best.loss_chg > kRtEps) {
          p_tree->AddChilds(nid);
          (*p_tree)[nid].SetSplit(e.best.SplitIndex(), e.best.split_value, e.best.DefaultLeft());
          // mark right child as 0, to indicate fresh leaf
          (*p_tree)[(*p_tree)[nid].LeftChild()].SetLeaf(0.0f, 0);
          (*p_tree)[(*p_tree)[nid].RightChild()].SetLeaf(0.0f, 0);
        } else {
          (*p_tree)[nid].SetLeaf(e.weight * param_.learning_rate);
        }
      }
    }
    // reset position of each data points after split is created in the tree
    inline void ResetPosition(const std::vector<int> &qexpand,
                              DMatrix* p_fmat,
                              const RegTree& tree) {
      // set the positions in the nondefault
      this->SetNonDefaultPosition(qexpand, p_fmat, tree);
      // set rest of instances to default position
      const RowSet &rowset = p_fmat->BufferedRowset();
      // set default direct nodes to default
      // for leaf nodes that are not fresh, mark then to ~nid,
      // so that they are ignored in future statistics collection
      const auto ndata = static_cast<bst_omp_uint>(rowset.Size());

      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        CHECK_LT(ridx, position_.size())
            << "ridx exceed bound " << "ridx="<<  ridx << " pos=" << position_.size();
        const int nid = this->DecodePosition(ridx);
        if (tree[nid].IsLeaf()) {
          // mark finish when it is not a fresh leaf
          if (tree[nid].RightChild() == -1) {
            position_[ridx] = ~nid;
          }
        } else {
          // push to default branch
          if (tree[nid].DefaultLeft()) {
            this->SetEncodePosition(ridx, tree[nid].LeftChild());
          } else {
            this->SetEncodePosition(ridx, tree[nid].RightChild());
          }
        }
      }
    }
    // customization part
    // synchronize the best solution of each node
    virtual void SyncBestSolution(const std::vector<int> &qexpand) {
      for (int nid : qexpand) {
        NodeEntry &e = snode_[nid];
        for (int tid = 0; tid < this->nthread_; ++tid) {
          e.best.Update(stemp_[tid][nid].best);
        }
      }
    }
    virtual void SetNonDefaultPosition(const std::vector<int> &qexpand,
                                       DMatrix *p_fmat,
                                       const RegTree &tree) {
      // step 1, classify the non-default data into right places
      std::vector<unsigned> fsplits;
      for (int nid : qexpand) {
        if (!tree[nid].IsLeaf()) {
          fsplits.push_back(tree[nid].SplitIndex());
        }
      }
      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());
      auto iter = p_fmat->ColIterator();
      while (iter->Next()) {
        auto &batch = iter->Value();
        for (auto fid : fsplits) {
          auto col = batch[fid];
          const auto ndata = static_cast<bst_omp_uint>(col.length);
          #pragma omp parallel for schedule(static)
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const int nid = this->DecodePosition(ridx);
            const bst_float fvalue = col[j].fvalue;
            // go back to parent, correct those who are not default
            if (!tree[nid].IsLeaf() && tree[nid].SplitIndex() == fid) {
              if (fvalue < tree[nid].SplitCond()) {
                this->SetEncodePosition(ridx, tree[nid].LeftChild());
              } else {
                this->SetEncodePosition(ridx, tree[nid].RightChild());
              }
            }
          }
        }
      }
    }
    // utils to get/set position, with encoded format
    // return decoded position
    inline int DecodePosition(bst_uint ridx) const {
      const int pid = position_[ridx];
      return pid < 0 ? ~pid : pid;
    }
    // encode the encoded position value for ridx
    inline void SetEncodePosition(bst_uint ridx, int nid) {
      if (position_[ridx] < 0) {
        position_[ridx] = ~nid;
      } else {
        position_[ridx] = nid;
      }
    }
    //  --data fields--
    const TrainParam& param_;
    // number of omp thread used during training
    const int nthread_;
    // Per feature: shuffle index of each feature index
    std::vector<bst_uint> feat_index_;
    // Instance Data: current node position in the tree of each instance
    std::vector<int> position_;
    // PerThread x PerTreeNode: statistics for per thread construction
    std::vector< std::vector<ThreadEntry> > stemp_;
    /*! \brief TreeNode Data: statistics for each constructed node */
    std::vector<NodeEntry> snode_;
    /*! \brief queue of nodes to be expanded */
    std::vector<int> qexpand_;
    // Evaluates splits and computes optimal weights for a given split
    std::unique_ptr<SplitEvaluator> spliteval_;
  };
};

// distributed column maker
class DistColMaker : public ColMaker {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
    pruner_.reset(TreeUpdater::Create("prune"));
    pruner_->Init(args);
    spliteval_.reset(SplitEvaluator::Create(param_.split_evaluator));
    spliteval_->Init(args);
  }
  void Update(HostDeviceVector<GradientPair> *gpair,
              DMatrix* dmat,
              const std::vector<RegTree*> &trees) override {
    GradStats::CheckInfo(dmat->Info());
    CHECK_EQ(trees.size(), 1U) << "DistColMaker: only support one tree at a time";
    Builder builder(
      param_,
      std::unique_ptr<SplitEvaluator>(spliteval_->GetHostClone()));
    // build the tree
    builder.Update(gpair->HostVector(), dmat, trees[0]);
    //// prune the tree, note that pruner will sync the tree
    pruner_->Update(gpair, dmat, trees);
    // update position after the tree is pruned
    builder.UpdatePosition(dmat, *trees[0]);
  }

 private:
  class Builder : public ColMaker::Builder {
   public:
    explicit Builder(const TrainParam &param,
                     std::unique_ptr<SplitEvaluator> spliteval)
        : ColMaker::Builder(param, std::move(spliteval)) {}
    inline void UpdatePosition(DMatrix* p_fmat, const RegTree &tree) {
      const RowSet &rowset = p_fmat->BufferedRowset();
      const auto ndata = static_cast<bst_omp_uint>(rowset.Size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        int nid = this->DecodePosition(ridx);
        while (tree[nid].IsDeleted()) {
          nid = tree[nid].Parent();
          CHECK_GE(nid, 0);
        }
        this->position_[ridx] = nid;
      }
    }
    inline const int* GetLeafPosition() const {
      return dmlc::BeginPtr(this->position_);
    }

   protected:
    void SetNonDefaultPosition(const std::vector<int> &qexpand, DMatrix *p_fmat,
                               const RegTree &tree) override {
      // step 2, classify the non-default data into right places
      std::vector<unsigned> fsplits;
      for (int nid : qexpand) {
        if (!tree[nid].IsLeaf()) {
          fsplits.push_back(tree[nid].SplitIndex());
        }
      }
      // get the candidate split index
      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());
      while (fsplits.size() != 0 && fsplits.back() >= p_fmat->Info().num_col_) {
        fsplits.pop_back();
      }
      // bitmap is only word concurrent, set to bool first
      {
        auto ndata = static_cast<bst_omp_uint>(this->position_.size());
        boolmap_.resize(ndata);
        #pragma omp parallel for schedule(static)
        for (bst_omp_uint j = 0; j < ndata; ++j) {
            boolmap_[j] = 0;
        }
      }
      auto iter = p_fmat->ColIterator();
      while (iter->Next()) {
        auto &batch = iter->Value();
        for (auto fid : fsplits) {
          auto col = batch[fid];
          const auto ndata = static_cast<bst_omp_uint>(col.length);
          #pragma omp parallel for schedule(static)
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const bst_float fvalue = col[j].fvalue;
            const int nid = this->DecodePosition(ridx);
            if (!tree[nid].IsLeaf() && tree[nid].SplitIndex() == fid) {
              if (fvalue < tree[nid].SplitCond()) {
                if (!tree[nid].DefaultLeft()) boolmap_[ridx] = 1;
              } else {
                if (tree[nid].DefaultLeft()) boolmap_[ridx] = 1;
              }
            }
          }
        }
      }

      bitmap_.InitFromBool(boolmap_);
      // communicate bitmap
      rabit::Allreduce<rabit::op::BitOR>(dmlc::BeginPtr(bitmap_.data), bitmap_.data.size());
      const RowSet &rowset = p_fmat->BufferedRowset();
      // get the new position
      const auto ndata = static_cast<bst_omp_uint>(rowset.Size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        const int nid = this->DecodePosition(ridx);
        if (bitmap_.Get(ridx)) {
          CHECK(!tree[nid].IsLeaf()) << "inconsistent reduce information";
          if (tree[nid].DefaultLeft()) {
            this->SetEncodePosition(ridx, tree[nid].RightChild());
          } else {
            this->SetEncodePosition(ridx, tree[nid].LeftChild());
          }
        }
      }
    }
    // synchronize the best solution of each node
    void SyncBestSolution(const std::vector<int> &qexpand) override {
      std::vector<SplitEntry> vec;
      for (int nid : qexpand) {
        for (int tid = 0; tid < this->nthread_; ++tid) {
          this->snode_[nid].best.Update(this->stemp_[tid][nid].best);
        }
        vec.push_back(this->snode_[nid].best);
      }
      // TODO(tqchen) lazy version
      // communicate best solution
      reducer_.Allreduce(dmlc::BeginPtr(vec), vec.size());
      // assign solution back
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        this->snode_[nid].best = vec[i];
      }
    }

   private:
    common::BitMap bitmap_;
    std::vector<int> boolmap_;
    rabit::Reducer<SplitEntry, SplitEntry::Reduce> reducer_;
  };
  // we directly introduce pruner here
  std::unique_ptr<TreeUpdater> pruner_;
  // training parameter
  TrainParam param_;
  // Cloned for each builder instantiation
  std::unique_ptr<SplitEvaluator> spliteval_;
};

XGBOOST_REGISTER_TREE_UPDATER(ColMaker, "grow_colmaker")
.describe("Grow tree with parallelization over columns.")
.set_body([]() {
    return new ColMaker();
  });

XGBOOST_REGISTER_TREE_UPDATER(DistColMaker, "distcol")
.describe("Distributed column split version of tree maker.")
.set_body([]() {
    return new DistColMaker();
  });
}  // namespace tree
}  // namespace xgboost
