/*!
 * Copyright 2014 by Contributors
 * \file updater_colmaker.cc
 * \brief use columnwise update to construct a tree
 * \author Tianqi Chen
 */
#include <xgboost/tree_updater.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "./param.h"
#include "../common/random.h"
#include "../common/bitmap.h"
#include "../common/sync.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_colmaker);

/*! \brief column-wise update to construct a tree */
template<typename TStats, typename TConstraint>
class ColMaker: public TreeUpdater {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    param.InitAllowUnknown(args);
  }

  void Update(const std::vector<bst_gpair> &gpair,
              DMatrix* dmat,
              const std::vector<RegTree*> &trees) override {
    TStats::CheckInfo(dmat->info());
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    TConstraint::Init(&param, dmat->info().num_col);
    // build tree
    for (size_t i = 0; i < trees.size(); ++i) {
      Builder builder(param);
      builder.Update(gpair, dmat, trees[i]);
    }
    param.learning_rate = lr;
  }

 protected:
  // training parameter
  TrainParam param;
  // data structure
  /*! \brief per thread x per node entry to store tmp data */
  struct ThreadEntry {
    /*! \brief statistics of data */
    TStats stats;
    /*! \brief extra statistics of data */
    TStats stats_extra;
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
    TStats stats;
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
  struct Builder {
   public:
    // constructor
    explicit Builder(const TrainParam& param) : param(param), nthread(omp_get_max_threads()) {}
    // update one tree, growing
    virtual void Update(const std::vector<bst_gpair>& gpair,
                        DMatrix* p_fmat,
                        RegTree* p_tree) {
      this->InitData(gpair, *p_fmat, *p_tree);
      this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
      for (int depth = 0; depth < param.max_depth; ++depth) {
        this->FindSplit(depth, qexpand_, gpair, p_fmat, p_tree);
        this->ResetPosition(qexpand_, p_fmat, *p_tree);
        this->UpdateQueueExpand(*p_tree, &qexpand_);
        this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
        // if nothing left to be expand, break
        if (qexpand_.size() == 0) break;
      }
      // set all the rest expanding nodes to leaf
      for (size_t i = 0; i < qexpand_.size(); ++i) {
        const int nid = qexpand_[i];
        (*p_tree)[nid].set_leaf(snode[nid].weight * param.learning_rate);
      }
      // remember auxiliary statistics in the tree node
      for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
        p_tree->stat(nid).loss_chg = snode[nid].best.loss_chg;
        p_tree->stat(nid).base_weight = snode[nid].weight;
        p_tree->stat(nid).sum_hess = static_cast<float>(snode[nid].stats.sum_hess);
        snode[nid].stats.SetLeafVec(param, p_tree->leafvec(nid));
      }
    }

   protected:
    // initialize temp data structure
    inline void InitData(const std::vector<bst_gpair>& gpair,
                         const DMatrix& fmat,
                         const RegTree& tree) {
      CHECK_EQ(tree.param.num_nodes, tree.param.num_roots)
          << "ColMaker: can only grow new tree";
      const std::vector<unsigned>& root_index = fmat.info().root_index;
      const RowSet& rowset = fmat.buffered_rowset();
      {
        // setup position
        position.resize(gpair.size());
        if (root_index.size() == 0) {
          for (size_t i = 0; i < rowset.size(); ++i) {
            position[rowset[i]] = 0;
          }
        } else {
          for (size_t i = 0; i < rowset.size(); ++i) {
            const bst_uint ridx = rowset[i];
            position[ridx] = root_index[ridx];
            CHECK_LT(root_index[ridx], (unsigned)tree.param.num_roots);
          }
        }
        // mark delete for the deleted datas
        for (size_t i = 0; i < rowset.size(); ++i) {
          const bst_uint ridx = rowset[i];
          if (gpair[ridx].GetHess() < 0.0f) position[ridx] = ~position[ridx];
        }
        // mark subsample
        if (param.subsample < 1.0f) {
          std::bernoulli_distribution coin_flip(param.subsample);
          auto& rnd = common::GlobalRandom();
          for (size_t i = 0; i < rowset.size(); ++i) {
            const bst_uint ridx = rowset[i];
            if (gpair[ridx].GetHess() < 0.0f) continue;
            if (!coin_flip(rnd)) position[ridx] = ~position[ridx];
          }
        }
      }
      {
        // initialize feature index
        unsigned ncol = static_cast<unsigned>(fmat.info().num_col);
        for (unsigned i = 0; i < ncol; ++i) {
          if (fmat.GetColSize(i) != 0) {
            feat_index.push_back(i);
          }
        }
        unsigned n = std::max(static_cast<unsigned>(1),
                              static_cast<unsigned>(param.colsample_bytree * feat_index.size()));
        std::shuffle(feat_index.begin(), feat_index.end(), common::GlobalRandom());
        CHECK_GT(param.colsample_bytree, 0U)
            << "colsample_bytree cannot be zero.";
        feat_index.resize(n);
      }
      {
        // setup temp space for each thread
        // reserve a small space
        stemp.clear();
        stemp.resize(this->nthread, std::vector<ThreadEntry>());
        for (size_t i = 0; i < stemp.size(); ++i) {
          stemp[i].clear(); stemp[i].reserve(256);
        }
        snode.reserve(256);
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
                            const std::vector<bst_gpair>& gpair,
                            const DMatrix& fmat,
                            const RegTree& tree) {
      {
        // setup statistics space for each tree node
        for (size_t i = 0; i < stemp.size(); ++i) {
          stemp[i].resize(tree.param.num_nodes, ThreadEntry(param));
        }
        snode.resize(tree.param.num_nodes, NodeEntry(param));
        constraints_.resize(tree.param.num_nodes);
      }
      const RowSet &rowset = fmat.buffered_rowset();
      const MetaInfo& info = fmat.info();
      // setup position
      const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        const int tid = omp_get_thread_num();
        if (position[ridx] < 0) continue;
        stemp[tid][position[ridx]].stats.Add(gpair, info, ridx);
      }
      // sum the per thread statistics together
      for (size_t j = 0; j < qexpand.size(); ++j) {
        const int nid = qexpand[j];
        TStats stats(param);
        for (size_t tid = 0; tid < stemp.size(); ++tid) {
          stats.Add(stemp[tid][nid].stats);
        }
        // update node statistics
        snode[nid].stats = stats;
      }
      // setup constraints before calculating the weight
      for (size_t j = 0; j < qexpand.size(); ++j) {
        const int nid = qexpand[j];
        if (tree[nid].is_root()) continue;
        const int pid = tree[nid].parent();
        constraints_[pid].SetChild(param, tree[pid].split_index(),
                                   snode[tree[pid].cleft()].stats,
                                   snode[tree[pid].cright()].stats,
                                   &constraints_[tree[pid].cleft()],
                                   &constraints_[tree[pid].cright()]);
      }
      // calculating the weights
      for (size_t j = 0; j < qexpand.size(); ++j) {
        const int nid = qexpand[j];
        snode[nid].root_gain = static_cast<float>(
            constraints_[nid].CalcGain(param, snode[nid].stats));
        snode[nid].weight = static_cast<float>(
            constraints_[nid].CalcWeight(param, snode[nid].stats));
      }
    }
    /*! \brief update queue expand add in new leaves */
    inline void UpdateQueueExpand(const RegTree& tree, std::vector<int>* p_qexpand) {
      std::vector<int> &qexpand = *p_qexpand;
      std::vector<int> newnodes;
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        if (!tree[ nid ].is_leaf()) {
          newnodes.push_back(tree[nid].cleft());
          newnodes.push_back(tree[nid].cright());
        }
      }
      // use new nodes for qexpand
      qexpand = newnodes;
    }
    // parallel find the best split of current fid
    // this function does not support nested functions
    inline void ParallelFindSplit(const ColBatch::Inst &col,
                                  bst_uint fid,
                                  const DMatrix &fmat,
                                  const std::vector<bst_gpair> &gpair) {
      // TODO(tqchen): double check stats order.
      const MetaInfo& info = fmat.info();
      const bool ind = col.length != 0 && col.data[0].fvalue == col.data[col.length - 1].fvalue;
      bool need_forward = param.need_forward_search(fmat.GetColDensity(fid), ind);
      bool need_backward = param.need_backward_search(fmat.GetColDensity(fid), ind);
      const std::vector<int> &qexpand = qexpand_;
      #pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        std::vector<ThreadEntry> &temp = stemp[tid];
        // cleanup temp statistics
        for (size_t j = 0; j < qexpand.size(); ++j) {
          temp[qexpand[j]].stats.Clear();
        }
        bst_uint step = (col.length + this->nthread - 1) / this->nthread;
        bst_uint end = std::min(col.length, step * (tid + 1));
        for (bst_uint i = tid * step; i < end; ++i) {
          const bst_uint ridx = col[i].index;
          const int nid = position[ridx];
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
      bst_omp_uint nnode = static_cast<bst_omp_uint>(qexpand.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint j = 0; j < nnode; ++j) {
        const int nid = qexpand[j];
        TStats sum(param), tmp(param), c(param);
        for (int tid = 0; tid < this->nthread; ++tid) {
          tmp = stemp[tid][nid].stats;
          stemp[tid][nid].stats = sum;
          sum.Add(tmp);
          if (tid != 0) {
            std::swap(stemp[tid - 1][nid].last_fvalue, stemp[tid][nid].first_fvalue);
          }
        }
        for (int tid = 0; tid < this->nthread; ++tid) {
          stemp[tid][nid].stats_extra = sum;
          ThreadEntry &e = stemp[tid][nid];
          bst_float fsplit;
          if (tid != 0) {
            if (stemp[tid - 1][nid].last_fvalue != e.first_fvalue) {
              fsplit = (stemp[tid - 1][nid].last_fvalue + e.first_fvalue) * 0.5f;
            } else {
              continue;
            }
          } else {
            fsplit = e.first_fvalue - rt_eps;
          }
          if (need_forward && tid != 0) {
            c.SetSubstract(snode[nid].stats, e.stats);
            if (c.sum_hess >= param.min_child_weight &&
                e.stats.sum_hess >= param.min_child_weight) {
              bst_float loss_chg = static_cast<bst_float>(
                  constraints_[nid].CalcSplitGain(param, fid, e.stats, c) - snode[nid].root_gain);
              e.best.Update(loss_chg, fid, fsplit, false);
            }
          }
          if (need_backward) {
            tmp.SetSubstract(sum, e.stats);
            c.SetSubstract(snode[nid].stats, tmp);
            if (c.sum_hess >= param.min_child_weight &&
                tmp.sum_hess >= param.min_child_weight) {
              bst_float loss_chg = static_cast<bst_float>(
                  constraints_[nid].CalcSplitGain(param, fid, tmp, c) - snode[nid].root_gain);
              e.best.Update(loss_chg, fid, fsplit, true);
            }
          }
        }
        if (need_backward) {
          tmp = sum;
          ThreadEntry &e = stemp[this->nthread-1][nid];
          c.SetSubstract(snode[nid].stats, tmp);
          if (c.sum_hess >= param.min_child_weight &&
              tmp.sum_hess >= param.min_child_weight) {
            bst_float loss_chg = static_cast<bst_float>(
                constraints_[nid].CalcSplitGain(param, fid, tmp, c) - snode[nid].root_gain);
            e.best.Update(loss_chg, fid, e.last_fvalue + rt_eps, true);
          }
        }
      }
      // rescan, generate candidate split
      #pragma omp parallel
      {
        TStats c(param), cright(param);
        const int tid = omp_get_thread_num();
        std::vector<ThreadEntry> &temp = stemp[tid];
        bst_uint step = (col.length + this->nthread - 1) / this->nthread;
        bst_uint end = std::min(col.length, step * (tid + 1));
        for (bst_uint i = tid * step; i < end; ++i) {
          const bst_uint ridx = col[i].index;
          const int nid = position[ridx];
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
                c.SetSubstract(snode[nid].stats, e.stats);
                if (c.sum_hess >= param.min_child_weight &&
                    e.stats.sum_hess >= param.min_child_weight) {
                  bst_float loss_chg = static_cast<bst_float>(
                      constraints_[nid].CalcSplitGain(param, fid, e.stats, c) -
                      snode[nid].root_gain);
                  e.best.Update(loss_chg, fid, (fvalue + e.first_fvalue) * 0.5f, false);
                }
              }
              if (need_backward) {
                cright.SetSubstract(e.stats_extra, e.stats);
                c.SetSubstract(snode[nid].stats, cright);
                if (c.sum_hess >= param.min_child_weight &&
                    cright.sum_hess >= param.min_child_weight) {
                  bst_float loss_chg = static_cast<bst_float>(
                      constraints_[nid].CalcSplitGain(param, fid, c, cright) -
                      snode[nid].root_gain);
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
    inline void UpdateEnumeration(int nid, bst_gpair gstats,
                                  bst_float fvalue, int d_step, bst_uint fid,
                                  TStats &c, std::vector<ThreadEntry> &temp) { // NOLINT(*)
      // get the statistics of nid
      ThreadEntry &e = temp[nid];
      // test if first hit, this is fine, because we set 0 during init
      if (e.stats.Empty()) {
        e.stats.Add(gstats);
        e.last_fvalue = fvalue;
      } else {
        // try to find a split
        if (fvalue != e.last_fvalue &&
            e.stats.sum_hess >= param.min_child_weight) {
          c.SetSubstract(snode[nid].stats, e.stats);
          if (c.sum_hess >= param.min_child_weight) {
            bst_float loss_chg;
            if (d_step == -1) {
              loss_chg = static_cast<bst_float>(
                  constraints_[nid].CalcSplitGain(param, fid, c, e.stats) - snode[nid].root_gain);
            } else {
              loss_chg = static_cast<bst_float>(
                  constraints_[nid].CalcSplitGain(param, fid, e.stats, c) - snode[nid].root_gain);
            }
            e.best.Update(loss_chg, fid, (fvalue + e.last_fvalue) * 0.5f, d_step == -1);
          }
        }
        // update the statistics
        e.stats.Add(gstats);
        e.last_fvalue = fvalue;
      }
    }
    // same as EnumerateSplit, with cacheline prefetch optimization
    inline void EnumerateSplitCacheOpt(const ColBatch::Entry *begin,
                                       const ColBatch::Entry *end,
                                       int d_step,
                                       bst_uint fid,
                                       const std::vector<bst_gpair> &gpair,
                                       std::vector<ThreadEntry> &temp) { // NOLINT(*)
      const std::vector<int> &qexpand = qexpand_;
      // clear all the temp statistics
      for (size_t j = 0; j < qexpand.size(); ++j) {
        temp[qexpand[j]].stats.Clear();
      }
      // left statistics
      TStats c(param);
      // local cache buffer for position and gradient pair
      const int kBuffer = 32;
      int buf_position[kBuffer];
      bst_gpair buf_gpair[kBuffer];
      // aligned ending position
      const ColBatch::Entry *align_end;
      if (d_step > 0) {
        align_end = begin + (end - begin) / kBuffer * kBuffer;
      } else {
        align_end = begin - (begin - end) / kBuffer * kBuffer;
      }
      int i;
      const ColBatch::Entry *it;
      const int align_step = d_step * kBuffer;
      // internal cached loop
      for (it = begin; it != align_end; it += align_step) {
        const ColBatch::Entry *p;
        for (i = 0, p = it; i < kBuffer; ++i, p += d_step) {
          buf_position[i] = position[p->index];
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
        buf_position[i] = position[it->index];
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
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        ThreadEntry &e = temp[nid];
        c.SetSubstract(snode[nid].stats, e.stats);
        if (e.stats.sum_hess >= param.min_child_weight &&
            c.sum_hess >= param.min_child_weight) {
          bst_float loss_chg;
          if (d_step == -1) {
            loss_chg = static_cast<bst_float>(
                constraints_[nid].CalcSplitGain(param, fid, c, e.stats) - snode[nid].root_gain);
          } else {
            loss_chg = static_cast<bst_float>(
                constraints_[nid].CalcSplitGain(param, fid, e.stats, c) - snode[nid].root_gain);
          }
          const bst_float gap = std::abs(e.last_fvalue) + rt_eps;
          const bst_float delta = d_step == +1 ? gap: -gap;
          e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1);
        }
      }
    }

    // enumerate the split values of specific feature
    inline void EnumerateSplit(const ColBatch::Entry *begin,
                               const ColBatch::Entry *end,
                               int d_step,
                               bst_uint fid,
                               const std::vector<bst_gpair> &gpair,
                               const MetaInfo &info,
                               std::vector<ThreadEntry> &temp) { // NOLINT(*)
      // use cacheline aware optimization
      if (TStats::kSimpleStats != 0 && param.cache_opt != 0) {
        EnumerateSplitCacheOpt(begin, end, d_step, fid, gpair, temp);
        return;
      }
      const std::vector<int> &qexpand = qexpand_;
      // clear all the temp statistics
      for (size_t j = 0; j < qexpand.size(); ++j) {
        temp[qexpand[j]].stats.Clear();
      }
      // left statistics
      TStats c(param);
      for (const ColBatch::Entry *it = begin; it != end; it += d_step) {
        const bst_uint ridx = it->index;
        const int nid = position[ridx];
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
              e.stats.sum_hess >= param.min_child_weight) {
            c.SetSubstract(snode[nid].stats, e.stats);
            if (c.sum_hess >= param.min_child_weight) {
              bst_float loss_chg;
              if (d_step == -1) {
                loss_chg = static_cast<bst_float>(
                    constraints_[nid].CalcSplitGain(param, fid, c, e.stats) -
                    snode[nid].root_gain);
              } else {
                loss_chg = static_cast<bst_float>(
                    constraints_[nid].CalcSplitGain(param, fid, e.stats, c) -
                    snode[nid].root_gain);
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
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        ThreadEntry &e = temp[nid];
        c.SetSubstract(snode[nid].stats, e.stats);
        if (e.stats.sum_hess >= param.min_child_weight && c.sum_hess >= param.min_child_weight) {
          bst_float loss_chg;
          if (d_step == -1) {
            loss_chg = static_cast<bst_float>(
                constraints_[nid].CalcSplitGain(param, fid, c, e.stats) - snode[nid].root_gain);
          } else {
            loss_chg = static_cast<bst_float>(
                constraints_[nid].CalcSplitGain(param, fid, e.stats, c) - snode[nid].root_gain);
          }
          const bst_float gap = std::abs(e.last_fvalue) + rt_eps;
          const bst_float delta = d_step == +1 ? gap: -gap;
          e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1);
        }
      }
    }

    // update the solution candidate
    virtual void UpdateSolution(const ColBatch& batch,
                                const std::vector<bst_gpair>& gpair,
                                const DMatrix& fmat) {
      const MetaInfo& info = fmat.info();
      // start enumeration
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
      #if defined(_OPENMP)
      const int batch_size = std::max(static_cast<int>(nsize / this->nthread / 32), 1);
      #endif
      int poption = param.parallel_option;
      if (poption == 2) {
        poption = static_cast<int>(nsize) * 2 < this->nthread ? 1 : 0;
      }
      if (poption == 0) {
        #pragma omp parallel for schedule(dynamic, batch_size)
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          const bst_uint fid = batch.col_index[i];
          const int tid = omp_get_thread_num();
          const ColBatch::Inst c = batch[i];
          const bool ind = c.length != 0 && c.data[0].fvalue == c.data[c.length - 1].fvalue;
          if (param.need_forward_search(fmat.GetColDensity(fid), ind)) {
            this->EnumerateSplit(c.data, c.data + c.length, +1,
                                 fid, gpair, info, stemp[tid]);
          }
          if (param.need_backward_search(fmat.GetColDensity(fid), ind)) {
            this->EnumerateSplit(c.data + c.length - 1, c.data - 1, -1,
                                 fid, gpair, info, stemp[tid]);
          }
        }
      } else {
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          this->ParallelFindSplit(batch[i], batch.col_index[i],
                                  fmat, gpair);
        }
      }
    }
    // find splits at current level, do split per level
    inline void FindSplit(int depth,
                          const std::vector<int> &qexpand,
                          const std::vector<bst_gpair> &gpair,
                          DMatrix *p_fmat,
                          RegTree *p_tree) {
      std::vector<bst_uint> feat_set = feat_index;
      if (param.colsample_bylevel != 1.0f) {
        std::shuffle(feat_set.begin(), feat_set.end(), common::GlobalRandom());
        unsigned n = std::max(static_cast<unsigned>(1),
                              static_cast<unsigned>(param.colsample_bylevel * feat_index.size()));
        CHECK_GT(param.colsample_bylevel, 0U)
            << "colsample_bylevel cannot be zero.";
        feat_set.resize(n);
      }
      dmlc::DataIter<ColBatch>* iter = p_fmat->ColIterator(feat_set);
      while (iter->Next()) {
        this->UpdateSolution(iter->Value(), gpair, *p_fmat);
      }
      // after this each thread's stemp will get the best candidates, aggregate results
      this->SyncBestSolution(qexpand);
      // get the best result, we can synchronize the solution
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        NodeEntry &e = snode[nid];
        // now we know the solution in snode[nid], set split
        if (e.best.loss_chg > rt_eps) {
          p_tree->AddChilds(nid);
          (*p_tree)[nid].set_split(e.best.split_index(), e.best.split_value, e.best.default_left());
          // mark right child as 0, to indicate fresh leaf
          (*p_tree)[(*p_tree)[nid].cleft()].set_leaf(0.0f, 0);
          (*p_tree)[(*p_tree)[nid].cright()].set_leaf(0.0f, 0);
        } else {
          (*p_tree)[nid].set_leaf(e.weight * param.learning_rate);
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
      const RowSet &rowset = p_fmat->buffered_rowset();
      // set default direct nodes to default
      // for leaf nodes that are not fresh, mark then to ~nid,
      // so that they are ignored in future statistics collection
      const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());

      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        CHECK_LT(ridx, position.size())
            << "ridx exceed bound " << "ridx="<<  ridx << " pos=" << position.size();
        const int nid = this->DecodePosition(ridx);
        if (tree[nid].is_leaf()) {
          // mark finish when it is not a fresh leaf
          if (tree[nid].cright() == -1) {
            position[ridx] = ~nid;
          }
        } else {
          // push to default branch
          if (tree[nid].default_left()) {
            this->SetEncodePosition(ridx, tree[nid].cleft());
          } else {
            this->SetEncodePosition(ridx, tree[nid].cright());
          }
        }
      }
    }
    // customization part
    // synchronize the best solution of each node
    virtual void SyncBestSolution(const std::vector<int> &qexpand) {
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        NodeEntry &e = snode[nid];
        for (int tid = 0; tid < this->nthread; ++tid) {
          e.best.Update(stemp[tid][nid].best);
        }
      }
    }
    virtual void SetNonDefaultPosition(const std::vector<int> &qexpand,
                                       DMatrix *p_fmat,
                                       const RegTree &tree) {
      // step 1, classify the non-default data into right places
      std::vector<unsigned> fsplits;
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        if (!tree[nid].is_leaf()) {
          fsplits.push_back(tree[nid].split_index());
        }
      }
      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());
      dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator(fsplits);
      while (iter->Next()) {
        const ColBatch &batch = iter->Value();
        for (size_t i = 0; i < batch.size; ++i) {
          ColBatch::Inst col = batch[i];
          const bst_uint fid = batch.col_index[i];
          const bst_omp_uint ndata = static_cast<bst_omp_uint>(col.length);
          #pragma omp parallel for schedule(static)
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const int nid = this->DecodePosition(ridx);
            const bst_float fvalue = col[j].fvalue;
            // go back to parent, correct those who are not default
            if (!tree[nid].is_leaf() && tree[nid].split_index() == fid) {
              if (fvalue < tree[nid].split_cond()) {
                this->SetEncodePosition(ridx, tree[nid].cleft());
              } else {
                this->SetEncodePosition(ridx, tree[nid].cright());
              }
            }
          }
        }
      }
    }
    // utils to get/set position, with encoded format
    // return decoded position
    inline int DecodePosition(bst_uint ridx) const {
      const int pid = position[ridx];
      return pid < 0 ? ~pid : pid;
    }
    // encode the encoded position value for ridx
    inline void SetEncodePosition(bst_uint ridx, int nid) {
      if (position[ridx] < 0) {
        position[ridx] = ~nid;
      } else {
        position[ridx] = nid;
      }
    }
    //  --data fields--
    const TrainParam& param;
    // number of omp thread used during training
    const int nthread;
    // Per feature: shuffle index of each feature index
    std::vector<bst_uint> feat_index;
    // Instance Data: current node position in the tree of each instance
    std::vector<int> position;
    // PerThread x PerTreeNode: statistics for per thread construction
    std::vector< std::vector<ThreadEntry> > stemp;
    /*! \brief TreeNode Data: statistics for each constructed node */
    std::vector<NodeEntry> snode;
    /*! \brief queue of nodes to be expanded */
    std::vector<int> qexpand_;
    // constraint value
    std::vector<TConstraint> constraints_;
  };
};

// distributed column maker
template<typename TStats, typename TConstraint>
class DistColMaker : public ColMaker<TStats, TConstraint> {
 public:
  DistColMaker() : builder(param) {
    pruner.reset(TreeUpdater::Create("prune"));
  }
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    param.InitAllowUnknown(args);
    pruner->Init(args);
  }
  void Update(const std::vector<bst_gpair> &gpair,
              DMatrix* dmat,
              const std::vector<RegTree*> &trees) override {
    TStats::CheckInfo(dmat->info());
    CHECK_EQ(trees.size(), 1U) << "DistColMaker: only support one tree at a time";
    // build the tree
    builder.Update(gpair, dmat, trees[0]);
    //// prune the tree, note that pruner will sync the tree
    pruner->Update(gpair, dmat, trees);
    // update position after the tree is pruned
    builder.UpdatePosition(dmat, *trees[0]);
  }

 private:
  struct Builder : public ColMaker<TStats, TConstraint>::Builder {
   public:
    explicit Builder(const TrainParam &param)
        : ColMaker<TStats, TConstraint>::Builder(param) {
    }
    inline void UpdatePosition(DMatrix* p_fmat, const RegTree &tree) {
      const RowSet &rowset = p_fmat->buffered_rowset();
      const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        int nid = this->DecodePosition(ridx);
        while (tree[nid].is_deleted()) {
          nid = tree[nid].parent();
          CHECK_GE(nid, 0);
        }
        this->position[ridx] = nid;
      }
    }
    inline const int* GetLeafPosition() const {
      return dmlc::BeginPtr(this->position);
    }

   protected:
    void SetNonDefaultPosition(const std::vector<int> &qexpand,
                               DMatrix *p_fmat,
                               const RegTree &tree) override {
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
      while (fsplits.size() != 0 && fsplits.back() >= p_fmat->info().num_col) {
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
      dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator(fsplits);
      while (iter->Next()) {
        const ColBatch &batch = iter->Value();
        for (size_t i = 0; i < batch.size; ++i) {
          ColBatch::Inst col = batch[i];
          const bst_uint fid = batch.col_index[i];
          const bst_omp_uint ndata = static_cast<bst_omp_uint>(col.length);
          #pragma omp parallel for schedule(static)
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const bst_float fvalue = col[j].fvalue;
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
      rabit::Allreduce<rabit::op::BitOR>(dmlc::BeginPtr(bitmap.data), bitmap.data.size());
      const RowSet &rowset = p_fmat->buffered_rowset();
      // get the new position
      const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        const int nid = this->DecodePosition(ridx);
        if (bitmap.Get(ridx)) {
          CHECK(!tree[nid].is_leaf()) << "inconsistent reduce information";
          if (tree[nid].default_left()) {
            this->SetEncodePosition(ridx, tree[nid].cright());
          } else {
            this->SetEncodePosition(ridx, tree[nid].cleft());
          }
        }
      }
    }
    // synchronize the best solution of each node
    void SyncBestSolution(const std::vector<int> &qexpand) override {
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
      reducer.Allreduce(dmlc::BeginPtr(vec), vec.size());
      // assign solution back
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        this->snode[nid].best = vec[i];
      }
    }

   private:
    common::BitMap bitmap;
    std::vector<int> boolmap;
    rabit::Reducer<SplitEntry, SplitEntry::Reduce> reducer;
  };
  // we directly introduce pruner here
  std::unique_ptr<TreeUpdater> pruner;
  // training parameter
  TrainParam param;
  // pointer to the builder
  Builder builder;
};

// simple switch to defer implementation.
class TreeUpdaterSwitch : public TreeUpdater {
 public:
  TreeUpdaterSwitch() : monotone_(false) {}
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    for (auto &kv : args) {
      if (kv.first == "monotone_constraints" && kv.second.length() != 0) {
        monotone_ = true;
      }
    }
    if (inner_.get() == nullptr) {
      if (monotone_) {
        inner_.reset(new ColMaker<GradStats, ValueConstraint>());
      } else {
        inner_.reset(new ColMaker<GradStats, NoConstraint>());
      }
    }

    inner_->Init(args);
  }

  void Update(const std::vector<bst_gpair>& gpair,
              DMatrix* data,
              const std::vector<RegTree*>& trees) override {
    CHECK(inner_ != nullptr);
    inner_->Update(gpair, data, trees);
  }

 private:
  //  monotone constraints
  bool monotone_;
  // internal implementation
  std::unique_ptr<TreeUpdater> inner_;
};

XGBOOST_REGISTER_TREE_UPDATER(ColMaker, "grow_colmaker")
.describe("Grow tree with parallelization over columns.")
.set_body([]() {
    return new TreeUpdaterSwitch();
  });

XGBOOST_REGISTER_TREE_UPDATER(DistColMaker, "distcol")
.describe("Distributed column split version of tree maker.")
.set_body([]() {
    return new DistColMaker<GradStats, NoConstraint>();
  });
}  // namespace tree
}  // namespace xgboost
