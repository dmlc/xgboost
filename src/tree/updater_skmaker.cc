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
  void Update(HostDeviceVector<GradientPair> *gpair,
              DMatrix *p_fmat,
              const std::vector<RegTree*> &trees) override {
    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    // build tree
    for (auto tree : trees) {
      this->Update(gpair->ConstHostVector(), p_fmat, tree);
    }
    param_.learning_rate = lr;
  }

 protected:
  inline void Update(const std::vector<GradientPair> &gpair,
                     DMatrix *p_fmat,
                     RegTree *p_tree) {
    this->InitData(gpair, *p_fmat, *p_tree);
    for (int depth = 0; depth < param_.max_depth; ++depth) {
      this->GetNodeStats(gpair, *p_fmat, *p_tree,
                         &thread_stats_, &node_stats_);
      this->BuildSketch(gpair, p_fmat, *p_tree);
      this->SyncNodeStats();
      this->FindSplit(depth, gpair, p_fmat, p_tree);
      this->ResetPositionCol(qexpand_, p_fmat, *p_tree);
      this->UpdateQueueExpand(*p_tree);
      // if nothing left to be expand, break
      if (qexpand_.size() == 0) break;
    }
    if (qexpand_.size() != 0) {
      this->GetNodeStats(gpair, *p_fmat, *p_tree,
                         &thread_stats_, &node_stats_);
      this->SyncNodeStats();
    }
    // set all statistics correctly
    for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
      this->SetStats(nid, node_stats_[nid], p_tree);
      if (!(*p_tree)[nid].IsLeaf()) {
        p_tree->Stat(nid).loss_chg = static_cast<bst_float>(
            node_stats_[(*p_tree)[nid].LeftChild()].CalcGain(param_) +
            node_stats_[(*p_tree)[nid].RightChild()].CalcGain(param_) -
            node_stats_[nid].CalcGain(param_));
      }
    }
    // set left leaves
    for (int nid : qexpand_) {
      (*p_tree)[nid].SetLeaf(p_tree->Stat(nid).base_weight * param_.learning_rate);
    }
  }
  // define the sketch we want to use
  using WXQSketch = common::WXQuantileSketch<bst_float, bst_float>;

 private:
  // statistics needed in the gradient calculation
  struct SKStats {
    /*! \brief sum of all positive gradient */
    double pos_grad;
    /*! \brief sum of all negative gradient */
    double neg_grad;
    /*! \brief sum of hessian statistics */
    double sum_hess;
    SKStats() = default;
    // constructor
    explicit SKStats(const TrainParam &param) {
      this->Clear();
    }
    /*! \brief clear the statistics */
    inline void Clear() {
      neg_grad = pos_grad = sum_hess = 0.0f;
    }
    // accumulate statistics
    inline void Add(const std::vector<GradientPair> &gpair,
                    const MetaInfo &info,
                    bst_uint ridx) {
      const GradientPair &b = gpair[ridx];
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
  inline void BuildSketch(const std::vector<GradientPair> &gpair,
                          DMatrix *p_fmat,
                          const RegTree &tree) {
    const MetaInfo& info = p_fmat->Info();
    sketchs_.resize(this->qexpand_.size() * tree.param.num_feature * 3);
    for (auto & sketch : sketchs_) {
      sketch.Init(info.num_row_, this->param_.sketch_eps);
    }
    thread_sketch_.resize(omp_get_max_threads());
    // number of rows in
    const size_t nrows = p_fmat->Info().num_row_;
    // start accumulating statistics
    for (const auto &batch : p_fmat->GetSortedColumnBatches()) {
      // start enumeration
      const auto nsize = static_cast<bst_omp_uint>(batch.Size());
      #pragma omp parallel for schedule(dynamic, 1)
      for (bst_omp_uint fidx = 0; fidx < nsize; ++fidx) {
        this->UpdateSketchCol(gpair, batch[fidx], tree,
                              node_stats_,
                              fidx,
                              batch[fidx].size() == nrows,
                              &thread_sketch_[omp_get_thread_num()]);
      }
    }
    // setup maximum size
    unsigned max_size = param_.MaxSketchSize();
    // synchronize sketch
    summary_array_.resize(sketchs_.size());
    for (size_t i = 0; i < sketchs_.size(); ++i) {
      common::WXQuantileSketch<bst_float, bst_float>::SummaryContainer out;
      sketchs_[i].GetSummary(&out);
      summary_array_[i].Reserve(max_size);
      summary_array_[i].SetPrune(out, max_size);
    }
    size_t nbytes = WXQSketch::SummaryContainer::CalcMemCost(max_size);
    sketch_reducer_.Allreduce(dmlc::BeginPtr(summary_array_), nbytes, summary_array_.size());
  }
  // update sketch information in column fid
  inline void UpdateSketchCol(const std::vector<GradientPair> &gpair,
                              const SparsePage::Inst &col,
                              const RegTree &tree,
                              const std::vector<SKStats> &nstats,
                              bst_uint fid,
                              bool col_full,
                              std::vector<SketchEntry> *p_temp) {
    if (col.size() == 0) return;
    // initialize sbuilder for use
    std::vector<SketchEntry> &sbuilder = *p_temp;
    sbuilder.resize(tree.param.num_nodes * 3);
    for (unsigned int nid : this->qexpand_) {
      const unsigned wid = this->node2workindex_[nid];
      for (int k = 0; k < 3; ++k) {
        sbuilder[3 * nid + k].sum_total = 0.0f;
        sbuilder[3 * nid + k].sketch = &sketchs_[(wid * tree.param.num_feature + fid) * 3 + k];
      }
    }
    if (!col_full) {
      for (const auto& c : col) {
        const bst_uint ridx = c.index;
        const int nid = this->position_[ridx];
        if (nid > 0) {
          const GradientPair &e = gpair[ridx];
          if (e.GetGrad() >= 0.0f) {
            sbuilder[3 * nid + 0].sum_total += e.GetGrad();
          } else {
            sbuilder[3 * nid + 1].sum_total -= e.GetGrad();
          }
          sbuilder[3 * nid + 2].sum_total += e.GetHess();
        }
      }
    } else {
      for (unsigned int nid : this->qexpand_) {
        sbuilder[3 * nid + 0].sum_total = static_cast<bst_float>(nstats[nid].pos_grad);
        sbuilder[3 * nid + 1].sum_total = static_cast<bst_float>(nstats[nid].neg_grad);
        sbuilder[3 * nid + 2].sum_total = static_cast<bst_float>(nstats[nid].sum_hess);
      }
    }
    // if only one value, no need to do second pass
    if (col[0].fvalue  == col[col.size()-1].fvalue) {
      for (int nid : this->qexpand_) {
        for (int k = 0; k < 3; ++k) {
          sbuilder[3 * nid + k].sketch->Push(col[0].fvalue,
                                             static_cast<bst_float>(
                                                 sbuilder[3 * nid + k].sum_total));
        }
      }
      return;
    }
    // two pass scan
    unsigned max_size = param_.MaxSketchSize();
    for (int nid : this->qexpand_) {
      for (int k = 0; k < 3; ++k) {
        sbuilder[3 * nid + k].Init(max_size);
      }
    }
    // second pass, build the sketch
    for (const auto& c : col) {
      const bst_uint ridx = c.index;
      const int nid = this->position_[ridx];
      if (nid >= 0) {
        const GradientPair &e = gpair[ridx];
        if (e.GetGrad() >= 0.0f) {
          sbuilder[3 * nid + 0].Push(c.fvalue, e.GetGrad(), max_size);
        } else {
          sbuilder[3 * nid + 1].Push(c.fvalue, -e.GetGrad(), max_size);
        }
        sbuilder[3 * nid + 2].Push(c.fvalue, e.GetHess(), max_size);
      }
    }
    for (int nid : this->qexpand_) {
      for (int k = 0; k < 3; ++k) {
        sbuilder[3 * nid + k].Finalize(max_size);
      }
    }
  }
  inline void SyncNodeStats() {
    CHECK_NE(qexpand_.size(), 0U);
    std::vector<SKStats> tmp(qexpand_.size());
    for (size_t i = 0; i < qexpand_.size(); ++i) {
      tmp[i] = node_stats_[qexpand_[i]];
    }
    stats_reducer_.Allreduce(dmlc::BeginPtr(tmp), tmp.size());
    for (size_t i = 0; i < qexpand_.size(); ++i) {
      node_stats_[qexpand_[i]] = tmp[i];
    }
  }
  inline void FindSplit(int depth,
                        const std::vector<GradientPair> &gpair,
                        DMatrix *p_fmat,
                        RegTree *p_tree) {
    const bst_uint num_feature = p_tree->param.num_feature;
    // get the best split condition for each node
    std::vector<SplitEntry> sol(qexpand_.size());
    auto nexpand = static_cast<bst_omp_uint>(qexpand_.size());
    #pragma omp parallel for schedule(dynamic, 1)
    for (bst_omp_uint wid = 0; wid < nexpand; ++wid) {
      const int nid = qexpand_[wid];
      CHECK_EQ(node2workindex_[nid], static_cast<int>(wid));
      SplitEntry &best = sol[wid];
      for (bst_uint fid = 0; fid < num_feature; ++fid) {
        unsigned base = (wid * p_tree->param.num_feature + fid) * 3;
        EnumerateSplit(summary_array_[base + 0],
                       summary_array_[base + 1],
                       summary_array_[base + 2],
                       node_stats_[nid], fid, &best);
      }
    }
    // get the best result, we can synchronize the solution
    for (bst_omp_uint wid = 0; wid < nexpand; ++wid) {
      const int nid = qexpand_[wid];
      const SplitEntry &best = sol[wid];
      // set up the values
      p_tree->Stat(nid).loss_chg = best.loss_chg;
      this->SetStats(nid, node_stats_[nid], p_tree);
      // now we know the solution in snode[nid], set split
      if (best.loss_chg > kRtEps) {
        p_tree->AddChilds(nid);
        (*p_tree)[nid].SetSplit(best.SplitIndex(),
                                 best.split_value, best.DefaultLeft());
        // mark right child as 0, to indicate fresh leaf
        (*p_tree)[(*p_tree)[nid].LeftChild()].SetLeaf(0.0f, 0);
        (*p_tree)[(*p_tree)[nid].RightChild()].SetLeaf(0.0f, 0);
      } else {
        (*p_tree)[nid].SetLeaf(p_tree->Stat(nid).base_weight * param_.learning_rate);
      }
    }
  }
  // set statistics on ptree
  inline void SetStats(int nid, const SKStats &node_sum, RegTree *p_tree) {
    p_tree->Stat(nid).base_weight = static_cast<bst_float>(node_sum.CalcWeight(param_));
    p_tree->Stat(nid).sum_hess = static_cast<bst_float>(node_sum.sum_hess);
    node_sum.SetLeafVec(param_, p_tree->Leafvec(nid));
  }
  inline void EnumerateSplit(const WXQSketch::Summary &pos_grad,
                             const WXQSketch::Summary &neg_grad,
                             const WXQSketch::Summary &sum_hess,
                             const SKStats &node_sum,
                             bst_uint fid,
                             SplitEntry *best) {
    if (sum_hess.size == 0) return;
    double root_gain = node_sum.CalcGain(param_);
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
      if (s.sum_hess >= param_.min_child_weight &&
          c.sum_hess >= param_.min_child_weight) {
        double loss_chg = s.CalcGain(param_) + c.CalcGain(param_) - root_gain;
        best->Update(static_cast<bst_float>(loss_chg), fid, fsplits[i], false);
      }
      // backward
      c.SetSubstract(feat_sum, s);
      s.SetSubstract(node_sum, c);
      if (s.sum_hess >= param_.min_child_weight &&
          c.sum_hess >= param_.min_child_weight) {
        double loss_chg = s.CalcGain(param_) + c.CalcGain(param_) - root_gain;
        best->Update(static_cast<bst_float>(loss_chg), fid, fsplits[i], true);
      }
    }
    {
      // all including
      SKStats s = feat_sum, c;
      c.SetSubstract(node_sum, s);
      if (s.sum_hess >= param_.min_child_weight &&
          c.sum_hess >= param_.min_child_weight) {
        bst_float cpt = fsplits.back();
        double loss_chg = s.CalcGain(param_) + c.CalcGain(param_) - root_gain;
        best->Update(static_cast<bst_float>(loss_chg),
                     fid, cpt + std::abs(cpt) + 1.0f, false);
      }
    }
  }

  // thread temp data
  // used to hold temporal sketch
  std::vector<std::vector<SketchEntry> > thread_sketch_;
  // used to hold statistics
  std::vector<std::vector<SKStats> > thread_stats_;
  // node statistics
  std::vector<SKStats> node_stats_;
  // summary array
  std::vector<WXQSketch::SummaryContainer> summary_array_;
  // reducer for summary
  rabit::Reducer<SKStats, SKStats::Reduce> stats_reducer_;
  // reducer for summary
  rabit::SerializeReducer<WXQSketch::SummaryContainer> sketch_reducer_;
  // per node, per feature sketch
  std::vector<common::WXQuantileSketch<bst_float, bst_float> > sketchs_;
};

XGBOOST_REGISTER_TREE_UPDATER(SketchMaker, "grow_skmaker")
.describe("Approximate sketching maker.")
.set_body([]() {
    return new SketchMaker();
  });
}  // namespace tree
}  // namespace xgboost
