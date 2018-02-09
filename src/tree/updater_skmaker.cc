/*!
 * Copyright 2014 by Contributors
 * \file updater_skmaker.cc
 * \brief use approximation sketch to construct a tree,
          a refresh is needed to make the statistics exactly correct
 * \author Tianqi Chen
 */

#include <xgboost/base.h>
#include <xgboost/tree_updater.h>
#include <vector>
#include <algorithm>
#include "../common/sync.h"
#include "../common/quantile.h"
#include "../common/group_data.h"
#include "./updater_basemaker-inl.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_skmaker);

class SketchMaker: public BaseMaker {
 public:
  void Update(const std::vector<bst_gpair> &gpair,
              DMatrix *p_fmat,
              const std::vector<RegTree*> &trees) override {
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    // build tree
    for (size_t i = 0; i < trees.size(); ++i) {
      this->Update(gpair, p_fmat, trees[i]);
    }
    param.learning_rate = lr;
  }

 protected:
  inline void Update(const std::vector<bst_gpair> &gpair,
                     DMatrix *p_fmat,
                     RegTree *p_tree) {
    this->InitData(gpair, *p_fmat, *p_tree);
    for (int depth = 0; depth < param.max_depth; ++depth) {
      this->GetNodeStats(gpair, *p_fmat, *p_tree,
                         &thread_stats, &node_stats);
      this->BuildSketch(gpair, p_fmat, *p_tree);
      this->SyncNodeStats();
      this->FindSplit(depth, gpair, p_fmat, p_tree);
      this->ResetPositionCol(qexpand, p_fmat, *p_tree);
      this->UpdateQueueExpand(*p_tree);
      // if nothing left to be expand, break
      if (qexpand.size() == 0) break;
    }
    if (qexpand.size() != 0) {
      this->GetNodeStats(gpair, *p_fmat, *p_tree,
                         &thread_stats, &node_stats);
      this->SyncNodeStats();
    }
    // set all statistics correctly
    for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
      this->SetStats(nid, node_stats[nid], p_tree);
      if (!(*p_tree)[nid].is_leaf()) {
        p_tree->stat(nid).loss_chg = static_cast<bst_float>(
            node_stats[(*p_tree)[nid].cleft()].CalcGain(param) +
            node_stats[(*p_tree)[nid].cright()].CalcGain(param) -
            node_stats[nid].CalcGain(param));
      }
    }
    // set left leaves
    for (size_t i = 0; i < qexpand.size(); ++i) {
      const int nid = qexpand[i];
      (*p_tree)[nid].set_leaf(p_tree->stat(nid).base_weight * param.learning_rate);
    }
  }
  // define the sketch we want to use
  typedef common::WXQuantileSketch<bst_float, bst_float> WXQSketch;

 private:
  // statistics needed in the gradient calculation
  struct SKStats {
    /*! \brief sum of all positive gradient */
    double pos_grad;
    /*! \brief sum of all negative gradient */
    double neg_grad;
    /*! \brief sum of hessian statistics */
    double sum_hess;
    SKStats(void) {}
    // constructor
    explicit SKStats(const TrainParam &param) {
      this->Clear();
    }
    /*! \brief clear the statistics */
    inline void Clear(void) {
      neg_grad = pos_grad = sum_hess = 0.0f;
    }
    // accumulate statistics
    inline void Add(const std::vector<bst_gpair> &gpair,
                    const MetaInfo &info,
                    bst_uint ridx) {
      const bst_gpair &b = gpair[ridx];
      if (b.GetGrad() >= 0.0f) {
        pos_grad += b.GetGrad();
      } else {
        neg_grad -= b.GetGrad();
      }
      sum_hess += b.GetHess();
    }
    /*! \brief calculate gain of the solution */
    inline double CalcGain(const TrainParam &param) const {
      return xgboost::tree::CalcGain(param, pos_grad - neg_grad, sum_hess);
    }
    /*! \brief set current value to a - b */
    inline void SetSubstract(const SKStats &a, const SKStats &b) {
      pos_grad = a.pos_grad - b.pos_grad;
      neg_grad = a.neg_grad - b.neg_grad;
      sum_hess = a.sum_hess - b.sum_hess;
    }
    // calculate leaf weight
    inline double CalcWeight(const TrainParam &param) const {
      return xgboost::tree::CalcWeight(param, pos_grad - neg_grad, sum_hess);
    }
    /*! \brief add statistics to the data */
    inline void Add(const SKStats &b) {
      pos_grad += b.pos_grad;
      neg_grad += b.neg_grad;
      sum_hess += b.sum_hess;
    }
    /*! \brief same as add, reduce is used in All Reduce */
    inline static void Reduce(SKStats &a, const SKStats &b) { // NOLINT(*)
      a.Add(b);
    }
    /*! \brief set leaf vector value based on statistics */
    inline void SetLeafVec(const TrainParam &param, bst_float *vec) const {
    }
  };
  inline void BuildSketch(const std::vector<bst_gpair> &gpair,
                          DMatrix *p_fmat,
                          const RegTree &tree) {
    const MetaInfo& info = p_fmat->info();
    sketchs.resize(this->qexpand.size() * tree.param.num_feature * 3);
    for (size_t i = 0; i < sketchs.size(); ++i) {
      sketchs[i].Init(info.num_row, this->param.sketch_eps);
    }
    thread_sketch.resize(omp_get_max_threads());
    // number of rows in
    const size_t nrows = p_fmat->buffered_rowset().size();
    // start accumulating statistics
    dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const ColBatch &batch = iter->Value();
      // start enumeration
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
      #pragma omp parallel for schedule(dynamic, 1)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        this->UpdateSketchCol(gpair, batch[i], tree,
                              node_stats,
                              batch.col_index[i],
                              batch[i].length == nrows,
                              &thread_sketch[omp_get_thread_num()]);
      }
    }
    // setup maximum size
    unsigned max_size = param.max_sketch_size();
    // synchronize sketch
    summary_array.resize(sketchs.size());
    for (size_t i = 0; i < sketchs.size(); ++i) {
      common::WXQuantileSketch<bst_float, bst_float>::SummaryContainer out;
      sketchs[i].GetSummary(&out);
      summary_array[i].Reserve(max_size);
      summary_array[i].SetPrune(out, max_size);
    }
    size_t nbytes = WXQSketch::SummaryContainer::CalcMemCost(max_size);
    sketch_reducer.Allreduce(dmlc::BeginPtr(summary_array), nbytes, summary_array.size());
  }
  // update sketch information in column fid
  inline void UpdateSketchCol(const std::vector<bst_gpair> &gpair,
                              const ColBatch::Inst &c,
                              const RegTree &tree,
                              const std::vector<SKStats> &nstats,
                              bst_uint fid,
                              bool col_full,
                              std::vector<SketchEntry> *p_temp) {
    if (c.length == 0) return;
    // initialize sbuilder for use
    std::vector<SketchEntry> &sbuilder = *p_temp;
    sbuilder.resize(tree.param.num_nodes * 3);
    for (size_t i = 0; i < this->qexpand.size(); ++i) {
      const unsigned nid = this->qexpand[i];
      const unsigned wid = this->node2workindex[nid];
      for (int k = 0; k < 3; ++k) {
        sbuilder[3 * nid + k].sum_total = 0.0f;
        sbuilder[3 * nid + k].sketch = &sketchs[(wid * tree.param.num_feature + fid) * 3 + k];
      }
    }
    if (!col_full) {
      for (bst_uint j = 0; j < c.length; ++j) {
        const bst_uint ridx = c[j].index;
        const int nid = this->position[ridx];
        if (nid >= 0) {
          const bst_gpair &e = gpair[ridx];
          if (e.GetGrad() >= 0.0f) {
            sbuilder[3 * nid + 0].sum_total += e.GetGrad();
          } else {
            sbuilder[3 * nid + 1].sum_total -= e.GetGrad();
          }
          sbuilder[3 * nid + 2].sum_total += e.GetHess();
        }
      }
    } else {
      for (size_t i = 0; i < this->qexpand.size(); ++i) {
        const unsigned nid = this->qexpand[i];
        sbuilder[3 * nid + 0].sum_total = static_cast<bst_float>(nstats[nid].pos_grad);
        sbuilder[3 * nid + 1].sum_total = static_cast<bst_float>(nstats[nid].neg_grad);
        sbuilder[3 * nid + 2].sum_total = static_cast<bst_float>(nstats[nid].sum_hess);
      }
    }
    // if only one value, no need to do second pass
    if (c[0].fvalue  == c[c.length-1].fvalue) {
      for (size_t i = 0; i < this->qexpand.size(); ++i) {
        const int nid = this->qexpand[i];
        for (int k = 0; k < 3; ++k) {
          sbuilder[3 * nid + k].sketch->Push(c[0].fvalue,
                                             static_cast<bst_float>(
                                                 sbuilder[3 * nid + k].sum_total));
        }
      }
      return;
    }
    // two pass scan
    unsigned max_size = param.max_sketch_size();
    for (size_t i = 0; i < this->qexpand.size(); ++i) {
      const int nid = this->qexpand[i];
      for (int k = 0; k < 3; ++k) {
        sbuilder[3 * nid + k].Init(max_size);
      }
    }
    // second pass, build the sketch
    for (bst_uint j = 0; j < c.length; ++j) {
      const bst_uint ridx = c[j].index;
      const int nid = this->position[ridx];
      if (nid >= 0) {
        const bst_gpair &e = gpair[ridx];
        if (e.GetGrad() >= 0.0f) {
          sbuilder[3 * nid + 0].Push(c[j].fvalue, e.GetGrad(), max_size);
        } else {
          sbuilder[3 * nid + 1].Push(c[j].fvalue, -e.GetGrad(), max_size);
        }
        sbuilder[3 * nid + 2].Push(c[j].fvalue, e.GetHess(), max_size);
      }
    }
    for (size_t i = 0; i < this->qexpand.size(); ++i) {
      const int nid = this->qexpand[i];
      for (int k = 0; k < 3; ++k) {
        sbuilder[3 * nid + k].Finalize(max_size);
      }
    }
  }
  inline void SyncNodeStats(void) {
    CHECK_NE(qexpand.size(), 0U);
    std::vector<SKStats> tmp(qexpand.size());
    for (size_t i = 0; i < qexpand.size(); ++i) {
      tmp[i] = node_stats[qexpand[i]];
    }
    stats_reducer.Allreduce(dmlc::BeginPtr(tmp), tmp.size());
    for (size_t i = 0; i < qexpand.size(); ++i) {
      node_stats[qexpand[i]] = tmp[i];
    }
  }
  inline void FindSplit(int depth,
                        const std::vector<bst_gpair> &gpair,
                        DMatrix *p_fmat,
                        RegTree *p_tree) {
    const bst_uint num_feature = p_tree->param.num_feature;
    // get the best split condition for each node
    std::vector<SplitEntry> sol(qexpand.size());
    bst_omp_uint nexpand = static_cast<bst_omp_uint>(qexpand.size());
    #pragma omp parallel for schedule(dynamic, 1)
    for (bst_omp_uint wid = 0; wid < nexpand; ++wid) {
      const int nid = qexpand[wid];
      CHECK_EQ(node2workindex[nid], static_cast<int>(wid));
      SplitEntry &best = sol[wid];
      for (bst_uint fid = 0; fid < num_feature; ++fid) {
        unsigned base = (wid * p_tree->param.num_feature + fid) * 3;
        EnumerateSplit(summary_array[base + 0],
                       summary_array[base + 1],
                       summary_array[base + 2],
                       node_stats[nid], fid, &best);
      }
    }
    // get the best result, we can synchronize the solution
    for (bst_omp_uint wid = 0; wid < nexpand; ++wid) {
      const int nid = qexpand[wid];
      const SplitEntry &best = sol[wid];
      // set up the values
      p_tree->stat(nid).loss_chg = best.loss_chg;
      this->SetStats(nid, node_stats[nid], p_tree);
      // now we know the solution in snode[nid], set split
      if (best.loss_chg > rt_eps) {
        p_tree->AddChilds(nid);
        (*p_tree)[nid].set_split(best.split_index(),
                                 best.split_value, best.default_left());
        // mark right child as 0, to indicate fresh leaf
        (*p_tree)[(*p_tree)[nid].cleft()].set_leaf(0.0f, 0);
        (*p_tree)[(*p_tree)[nid].cright()].set_leaf(0.0f, 0);
      } else {
        (*p_tree)[nid].set_leaf(p_tree->stat(nid).base_weight * param.learning_rate);
      }
    }
  }
  // set statistics on ptree
  inline void SetStats(int nid, const SKStats &node_sum, RegTree *p_tree) {
    p_tree->stat(nid).base_weight = static_cast<bst_float>(node_sum.CalcWeight(param));
    p_tree->stat(nid).sum_hess = static_cast<bst_float>(node_sum.sum_hess);
    node_sum.SetLeafVec(param, p_tree->leafvec(nid));
  }
  inline void EnumerateSplit(const WXQSketch::Summary &pos_grad,
                             const WXQSketch::Summary &neg_grad,
                             const WXQSketch::Summary &sum_hess,
                             const SKStats &node_sum,
                             bst_uint fid,
                             SplitEntry *best) {
    if (sum_hess.size == 0) return;
    double root_gain = node_sum.CalcGain(param);
    std::vector<bst_float> fsplits;
    for (size_t i = 0; i < pos_grad.size; ++i) {
      fsplits.push_back(pos_grad.data[i].value);
    }
    for (size_t i = 0; i < neg_grad.size; ++i) {
      fsplits.push_back(neg_grad.data[i].value);
    }
    for (size_t i = 0; i < sum_hess.size; ++i) {
      fsplits.push_back(sum_hess.data[i].value);
    }
    std::sort(fsplits.begin(), fsplits.end());
    fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());
    // sum feature
    SKStats feat_sum;
    feat_sum.pos_grad = pos_grad.data[pos_grad.size - 1].rmax;
    feat_sum.neg_grad = neg_grad.data[neg_grad.size - 1].rmax;
    feat_sum.sum_hess = sum_hess.data[sum_hess.size - 1].rmax;
    size_t ipos = 0, ineg = 0, ihess = 0;
    for (size_t i = 1; i < fsplits.size(); ++i) {
      WXQSketch::Entry pos = pos_grad.Query(fsplits[i], ipos);
      WXQSketch::Entry neg = neg_grad.Query(fsplits[i], ineg);
      WXQSketch::Entry hess = sum_hess.Query(fsplits[i], ihess);
      SKStats s, c;
      s.pos_grad = 0.5f * (pos.rmin + pos.rmax - pos.wmin);
      s.neg_grad = 0.5f * (neg.rmin + neg.rmax - neg.wmin);
      s.sum_hess = 0.5f * (hess.rmin + hess.rmax - hess.wmin);
      c.SetSubstract(node_sum, s);
      // forward
      if (s.sum_hess >= param.min_child_weight &&
          c.sum_hess >= param.min_child_weight) {
        double loss_chg = s.CalcGain(param) + c.CalcGain(param) - root_gain;
        best->Update(static_cast<bst_float>(loss_chg), fid, fsplits[i], false);
      }
      // backward
      c.SetSubstract(feat_sum, s);
      s.SetSubstract(node_sum, c);
      if (s.sum_hess >= param.min_child_weight &&
          c.sum_hess >= param.min_child_weight) {
        double loss_chg = s.CalcGain(param) + c.CalcGain(param) - root_gain;
        best->Update(static_cast<bst_float>(loss_chg), fid, fsplits[i], true);
      }
    }
    {
      // all including
      SKStats s = feat_sum, c;
      c.SetSubstract(node_sum, s);
      if (s.sum_hess >= param.min_child_weight &&
          c.sum_hess >= param.min_child_weight) {
        bst_float cpt = fsplits.back();
        double loss_chg = s.CalcGain(param) + c.CalcGain(param) - root_gain;
        best->Update(static_cast<bst_float>(loss_chg),
                     fid, cpt + std::abs(cpt) + 1.0f, false);
      }
    }
  }

  // thread temp data
  // used to hold temporal sketch
  std::vector<std::vector<SketchEntry> > thread_sketch;
  // used to hold statistics
  std::vector<std::vector<SKStats> > thread_stats;
  // node statistics
  std::vector<SKStats> node_stats;
  // summary array
  std::vector<WXQSketch::SummaryContainer> summary_array;
  // reducer for summary
  rabit::Reducer<SKStats, SKStats::Reduce> stats_reducer;
  // reducer for summary
  rabit::SerializeReducer<WXQSketch::SummaryContainer> sketch_reducer;
  // per node, per feature sketch
  std::vector<common::WXQuantileSketch<bst_float, bst_float> > sketchs;
};

XGBOOST_REGISTER_TREE_UPDATER(SketchMaker, "grow_skmaker")
.describe("Approximate sketching maker.")
.set_body([]() {
    return new SketchMaker();
  });
}  // namespace tree
}  // namespace xgboost
