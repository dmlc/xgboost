/*!
 * Copyright 2014-2019 by Contributors
 * \file updater_histmaker.cc
 * \brief use histogram counting to construct a tree
 * \author Tianqi Chen
 */
#include <rabit/rabit.h>
#include <vector>
#include <algorithm>

#include "xgboost/tree_updater.h"
#include "xgboost/base.h"
#include "xgboost/logging.h"

#include "../common/quantile.h"
#include "../common/group_data.h"
#include "./updater_basemaker-inl.h"
#include "constraints.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_histmaker);

class HistMaker: public BaseMaker {
 public:
  void Update(HostDeviceVector<GradientPair> *gpair,
              DMatrix *p_fmat,
              const std::vector<RegTree*> &trees) override {
    interaction_constraints_.Configure(param_, p_fmat->Info().num_col_);
    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    // build tree
    for (auto tree : trees) {
      this->UpdateTree(gpair->ConstHostVector(), p_fmat, tree);
    }
    param_.learning_rate = lr;
  }
  char const* Name() const override {
    return "grow_histmaker";
  }

 protected:
    /*! \brief a single column of histogram cuts */
  struct HistUnit {
    /*! \brief cutting point of histogram, contains maximum point */
    const float *cut;
    /*! \brief content of statistics data */
    GradStats *data;
    /*! \brief size of histogram */
    uint32_t size;
    // default constructor
    HistUnit() = default;
    // constructor
    HistUnit(const float *cut, GradStats *data, uint32_t size)
        : cut{cut}, data{data}, size{size} {}
    /*! \brief add a histogram to data */
  };
  /*! \brief a set of histograms from different index */
  struct HistSet {
    /*! \brief the index pointer of each histunit */
    const uint32_t *rptr;
    /*! \brief cutting points in each histunit */
    const bst_float *cut;
    /*! \brief data in different hist unit */
    std::vector<GradStats> data;
    /*! \brief return a column of histogram cuts */
    inline HistUnit operator[](size_t fid) {
      return {cut + rptr[fid], &data[0] + rptr[fid], rptr[fid+1] - rptr[fid]};
    }
  };
  // thread workspace
  struct ThreadWSpace {
    /*! \brief actual unit pointer */
    std::vector<unsigned> rptr;
    /*! \brief cut field */
    std::vector<bst_float> cut;
    // per thread histset
    std::vector<HistSet> hset;
    // initialize the hist set
    inline void Configure(int nthread) {
      hset.resize(nthread);
      // cleanup statistics
      for (int tid = 0; tid < nthread; ++tid) {
        for (auto& d : hset[tid].data) { d = GradStats(); }
        hset[tid].rptr = dmlc::BeginPtr(rptr);
        hset[tid].cut = dmlc::BeginPtr(cut);
        hset[tid].data.resize(cut.size(), GradStats());
      }
    }
    /*! \brief clear the workspace */
    inline void Clear() {
      cut.clear(); rptr.resize(1); rptr[0] = 0;
    }
    /*! \brief total size */
    inline size_t Size() const {
      return rptr.size() - 1;
    }
  };
  // workspace of thread
  ThreadWSpace wspace_;
  // reducer for histogram
  rabit::Reducer<GradStats, GradStats::Reduce> histred_;
  // set of working features
  std::vector<bst_feature_t> selected_features_;
  // update function implementation
  virtual void UpdateTree(const std::vector<GradientPair> &gpair,
                          DMatrix *p_fmat,
                          RegTree *p_tree) {
    CHECK(param_.max_depth > 0) << "max_depth must be larger than 0";
    this->InitData(gpair, *p_fmat, *p_tree);
    this->InitWorkSet(p_fmat, *p_tree, &selected_features_);
    // mark root node as fresh.
    (*p_tree)[0].SetLeaf(0.0f, 0);

    for (int depth = 0; depth < param_.max_depth; ++depth) {
      // reset and propose candidate split
      this->ResetPosAndPropose(gpair, p_fmat, selected_features_, *p_tree);
      // create histogram
      this->CreateHist(gpair, p_fmat, selected_features_, *p_tree);
      // find split based on histogram statistics
      this->FindSplit(selected_features_, p_tree);
      // reset position after split
      this->ResetPositionAfterSplit(p_fmat, *p_tree);
      this->UpdateQueueExpand(*p_tree);
      // if nothing left to be expand, break
      if (qexpand_.size() == 0) break;
    }
    for (int const nid : qexpand_) {
      (*p_tree)[nid].SetLeaf(p_tree->Stat(nid).base_weight * param_.learning_rate);
    }
  }
  // this function does two jobs
  // (1) reset the position in array position, to be the latest leaf id
  // (2) propose a set of candidate cuts and set wspace.rptr wspace.cut correctly
  virtual void ResetPosAndPropose(const std::vector<GradientPair> &gpair,
                                  DMatrix *p_fmat,
                                  const std::vector <bst_feature_t> &fset,
                                  const RegTree &tree) = 0;
  // initialize the current working set of features in this round
  virtual void InitWorkSet(DMatrix *,
                           const RegTree &tree,
                           std::vector<bst_feature_t> *p_fset) {
    p_fset->resize(tree.param.num_feature);
    for (size_t i = 0; i < p_fset->size(); ++i) {
      (*p_fset)[i] = static_cast<unsigned>(i);
    }
  }
  // reset position after split, this is not a must, depending on implementation
  virtual void ResetPositionAfterSplit(DMatrix *p_fmat,
                                       const RegTree &tree) {
  }
  virtual void CreateHist(const std::vector<GradientPair> &gpair,
                          DMatrix *,
                          const std::vector <bst_feature_t> &fset,
                          const RegTree &)  = 0;

 private:
  void EnumerateSplit(const HistUnit &hist,
                      const GradStats &node_sum,
                      bst_uint fid,
                      SplitEntry *best,
                      GradStats *left_sum) const {
    if (hist.size == 0) return;

    double root_gain = CalcGain(param_, node_sum.GetGrad(), node_sum.GetHess());
    GradStats s, c;
    for (bst_uint i = 0; i < hist.size; ++i) {
      s.Add(hist.data[i]);
      if (s.sum_hess >= param_.min_child_weight) {
        c.SetSubstract(node_sum, s);
        if (c.sum_hess >= param_.min_child_weight) {
          double loss_chg = CalcGain(param_, s.GetGrad(), s.GetHess()) +
                            CalcGain(param_, c.GetGrad(), c.GetHess()) - root_gain;
          if (best->Update(static_cast<bst_float>(loss_chg), fid, hist.cut[i], false, s, c)) {
            *left_sum = s;
          }
        }
      }
    }
    s = GradStats();
    for (bst_uint i = hist.size - 1; i != 0; --i) {
      s.Add(hist.data[i]);
      if (s.sum_hess >= param_.min_child_weight) {
        c.SetSubstract(node_sum, s);
        if (c.sum_hess >= param_.min_child_weight) {
          double loss_chg = CalcGain(param_, s.GetGrad(), s.GetHess()) +
                            CalcGain(param_, c.GetGrad(), c.GetHess()) - root_gain;
          if (best->Update(static_cast<bst_float>(loss_chg), fid, hist.cut[i-1], true, c, s)) {
            *left_sum = c;
          }
        }
      }
    }
  }

  void FindSplit(const std::vector <bst_feature_t> &feature_set,
                 RegTree *p_tree) {
    const size_t num_feature = feature_set.size();
    // get the best split condition for each node
    std::vector<SplitEntry> sol(qexpand_.size());
    std::vector<GradStats> left_sum(qexpand_.size());
    auto nexpand = static_cast<bst_omp_uint>(qexpand_.size());
#pragma omp parallel for schedule(dynamic, 1)
    for (bst_omp_uint wid = 0; wid < nexpand; ++wid) {
      const int nid = qexpand_[wid];
      CHECK_EQ(node2workindex_[nid], static_cast<int>(wid));
      SplitEntry &best = sol[wid];
      GradStats &node_sum = wspace_.hset[0][num_feature + wid * (num_feature + 1)].data[0];
      for (size_t i = 0; i < feature_set.size(); ++i) {
        // Query is thread safe as it's a const function.
        if (!this->interaction_constraints_.Query(nid, feature_set[i])) {
          continue;
        }

        EnumerateSplit(this->wspace_.hset[0][i + wid * (num_feature+1)],
                       node_sum, feature_set[i], &best, &left_sum[wid]);
      }
    }
    // get the best result, we can synchronize the solution
    for (bst_omp_uint wid = 0; wid < nexpand; ++wid) {
      const bst_node_t nid = qexpand_[wid];
      SplitEntry const& best = sol[wid];
      const GradStats &node_sum = wspace_.hset[0][num_feature + wid * (num_feature + 1)].data[0];
      this->SetStats(p_tree, nid, node_sum);
      // set up the values
      p_tree->Stat(nid).loss_chg = best.loss_chg;
      // now we know the solution in snode[nid], set split
      if (best.loss_chg > kRtEps) {
        bst_float base_weight = CalcWeight(param_, node_sum);
        bst_float left_leaf_weight =
            CalcWeight(param_, best.left_sum.sum_grad, best.left_sum.sum_hess) *
            param_.learning_rate;
        bst_float right_leaf_weight =
            CalcWeight(param_, best.right_sum.sum_grad,
                       best.right_sum.sum_hess) *
            param_.learning_rate;
        p_tree->ExpandNode(nid, best.SplitIndex(), best.split_value,
                           best.DefaultLeft(), base_weight, left_leaf_weight,
                           right_leaf_weight, best.loss_chg,
                           node_sum.sum_hess,
                           best.left_sum.GetHess(), best.right_sum.GetHess());
        GradStats right_sum;
        right_sum.SetSubstract(node_sum, left_sum[wid]);
        auto left_child = (*p_tree)[nid].LeftChild();
        auto right_child = (*p_tree)[nid].RightChild();
        this->SetStats(p_tree, left_child, left_sum[wid]);
        this->SetStats(p_tree, right_child, right_sum);
        this->interaction_constraints_.Split(nid, best.SplitIndex(), left_child, right_child);
      } else {
        (*p_tree)[nid].SetLeaf(p_tree->Stat(nid).base_weight * param_.learning_rate);
      }
    }
  }

  inline void SetStats(RegTree *p_tree, int nid, const GradStats &node_sum) {
    p_tree->Stat(nid).base_weight =
        static_cast<bst_float>(CalcWeight(param_, node_sum));
    p_tree->Stat(nid).sum_hess = static_cast<bst_float>(node_sum.sum_hess);
  }
};

class CQHistMaker: public HistMaker {
 public:
  CQHistMaker()  = default;
  char const* Name() const override {
    return "grow_local_histmaker";
  }

 protected:
  struct HistEntry {
    HistMaker::HistUnit hist;
    unsigned istart;
    /*!
     * \brief add a histogram to data,
     * do linear scan, start from istart
     */
    inline void Add(bst_float fv,
                    const std::vector<GradientPair> &gpair,
                    const bst_uint ridx) {
      while (istart < hist.size && !(fv < hist.cut[istart])) ++istart;
      CHECK_NE(istart, hist.size);
      hist.data[istart].Add(gpair[ridx]);
    }
    /*!
     * \brief add a histogram to data,
     * do linear scan, start from istart
     */
    inline void Add(bst_float fv,
                    GradientPair gstats) {
      if (fv < hist.cut[istart]) {
        hist.data[istart].Add(gstats);
      } else {
        while (istart < hist.size && !(fv < hist.cut[istart])) ++istart;
        if (istart != hist.size) {
          hist.data[istart].Add(gstats);
        } else {
          LOG(INFO) << "fv=" << fv << ", hist.size=" << hist.size;
          for (size_t i = 0; i < hist.size; ++i) {
            LOG(INFO) << "hist[" << i << "]=" << hist.cut[i];
          }
          LOG(FATAL) << "fv=" << fv << ", hist.last=" << hist.cut[hist.size - 1];
        }
      }
    }
  };
  // sketch type used for this
  using WXQSketch = common::WXQuantileSketch<bst_float, bst_float>;
  // initialize the work set of tree
  void InitWorkSet(DMatrix *p_fmat,
                   const RegTree &tree,
                   std::vector<bst_feature_t> *p_fset) override {
    if (p_fmat != cache_dmatrix_) {
      feat_helper_.InitByCol(p_fmat, tree);
      cache_dmatrix_ = p_fmat;
    }
    feat_helper_.SyncInfo();
    feat_helper_.SampleCol(this->param_.colsample_bytree, p_fset);
  }
  // code to create histogram
  void CreateHist(const std::vector<GradientPair> &gpair,
                  DMatrix *p_fmat,
                  const std::vector<bst_feature_t> &fset,
                  const RegTree &tree) override {
    const MetaInfo &info = p_fmat->Info();
    // fill in reverse map
    feat2workindex_.resize(tree.param.num_feature);
    std::fill(feat2workindex_.begin(), feat2workindex_.end(), -1);
    for (size_t i = 0; i < fset.size(); ++i) {
      feat2workindex_[fset[i]] = static_cast<int>(i);
    }
    // start to work
    this->wspace_.Configure(1);
    // if it is C++11, use lazy evaluation for Allreduce,
    // to gain speedup in recovery
    auto lazy_get_hist = [&]() {
      thread_hist_.resize(omp_get_max_threads());
      // start accumulating statistics
      for (const auto &batch : p_fmat->GetBatches<SortedCSCPage>()) {
        // start enumeration
        const auto nsize = static_cast<bst_omp_uint>(fset.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          int fid = fset[i];
          int offset = feat2workindex_[fid];
          if (offset >= 0) {
            this->UpdateHistCol(gpair, batch[fid], info, tree,
                                fset, offset,
                                &thread_hist_[omp_get_thread_num()]);
          }
        }
      }
      // update node statistics.
      this->GetNodeStats(gpair, *p_fmat, tree,
                         &thread_stats_, &node_stats_);
      for (int const nid : this->qexpand_) {
        const int wid = this->node2workindex_[nid];
        this->wspace_.hset[0][fset.size() + wid * (fset.size() + 1)]
                .data[0] = node_stats_[nid];
      }
    };
    // sync the histogram
    this->histred_.Allreduce(dmlc::BeginPtr(this->wspace_.hset[0].data),
                             this->wspace_.hset[0].data.size(), lazy_get_hist);
  }

  void ResetPositionAfterSplit(DMatrix *,
                                 const RegTree &tree) override {
    this->GetSplitSet(this->qexpand_, tree, &fsplit_set_);
  }
  void ResetPosAndPropose(const std::vector<GradientPair> &gpair,
                          DMatrix *p_fmat,
                          const std::vector<bst_feature_t> &fset,
                          const RegTree &tree) override {
    const MetaInfo &info = p_fmat->Info();
    // fill in reverse map
    feat2workindex_.resize(tree.param.num_feature);
    std::fill(feat2workindex_.begin(), feat2workindex_.end(), -1);
    work_set_.clear();
    for (auto fidx : fset) {
      if (feat_helper_.Type(fidx) == 2) {
        feat2workindex_[fidx] = static_cast<int>(work_set_.size());
        work_set_.push_back(fidx);
      } else {
        feat2workindex_[fidx] = -2;
      }
    }
    const size_t work_set_size = work_set_.size();

    sketchs_.resize(this->qexpand_.size() * work_set_size);
    for (auto& sketch : sketchs_) {
      sketch.Init(info.num_row_, this->param_.sketch_eps);
    }
    // intitialize the summary array
    summary_array_.resize(sketchs_.size());
    // setup maximum size
    unsigned max_size = this->param_.MaxSketchSize();
    for (size_t i = 0; i < sketchs_.size(); ++i) {
      summary_array_[i].Reserve(max_size);
    }
    {
      // get smmary
      thread_sketch_.resize(omp_get_max_threads());

      // TWOPASS: use the real set + split set in the column iteration.
      this->SetDefaultPostion(p_fmat, tree);
      work_set_.insert(work_set_.end(), fsplit_set_.begin(), fsplit_set_.end());
      std::sort(work_set_.begin(), work_set_.end());
      work_set_.resize(std::unique(work_set_.begin(), work_set_.end()) - work_set_.begin());

      // start accumulating statistics
      for (const auto &batch : p_fmat->GetBatches<SortedCSCPage>()) {
        // TWOPASS: use the real set + split set in the column iteration.
        this->CorrectNonDefaultPositionByBatch(batch, fsplit_set_, tree);

        // start enumeration
        const auto nsize = static_cast<bst_omp_uint>(work_set_.size());
        #pragma omp parallel for schedule(dynamic, 1)
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          int fid = work_set_[i];
          int offset = feat2workindex_[fid];
          if (offset >= 0) {
            this->UpdateSketchCol(gpair, batch[fid], tree,
                                  work_set_size, offset,
                                  &thread_sketch_[omp_get_thread_num()]);
          }
        }
      }
      for (size_t i = 0; i < sketchs_.size(); ++i) {
        common::WXQuantileSketch<bst_float, bst_float>::SummaryContainer out;
        sketchs_[i].GetSummary(&out);
        summary_array_[i].SetPrune(out, max_size);
      }
      CHECK_EQ(summary_array_.size(), sketchs_.size());
    }
    if (summary_array_.size() != 0) {
      size_t nbytes = WXQSketch::SummaryContainer::CalcMemCost(max_size);
      sreducer_.Allreduce(dmlc::BeginPtr(summary_array_), nbytes, summary_array_.size());
    }
    // now we get the final result of sketch, setup the cut
    this->wspace_.cut.clear();
    this->wspace_.rptr.clear();
    this->wspace_.rptr.push_back(0);
    for (size_t wid = 0; wid < this->qexpand_.size(); ++wid) {
      for (unsigned int i : fset) {
        int offset = feat2workindex_[i];
        if (offset >= 0) {
          const WXQSketch::Summary &a = summary_array_[wid * work_set_size + offset];
          for (size_t i = 1; i < a.size; ++i) {
            bst_float cpt = a.data[i].value - kRtEps;
            if (i == 1 || cpt > this->wspace_.cut.back()) {
              this->wspace_.cut.push_back(cpt);
            }
          }
          // push a value that is greater than anything
          if (a.size != 0) {
            bst_float cpt = a.data[a.size - 1].value;
            // this must be bigger than last value in a scale
            bst_float last = cpt + fabs(cpt) + kRtEps;
            this->wspace_.cut.push_back(last);
          }
          this->wspace_.rptr.push_back(static_cast<unsigned>(this->wspace_.cut.size()));
        } else {
          CHECK_EQ(offset, -2);
          bst_float cpt = feat_helper_.MaxValue(i);
          this->wspace_.cut.push_back(cpt + fabs(cpt) + kRtEps);
          this->wspace_.rptr.push_back(static_cast<unsigned>(this->wspace_.cut.size()));
        }
      }
      // reserve last value for global statistics
      this->wspace_.cut.push_back(0.0f);
      this->wspace_.rptr.push_back(static_cast<unsigned>(this->wspace_.cut.size()));
    }
    CHECK_EQ(this->wspace_.rptr.size(),
             (fset.size() + 1) * this->qexpand_.size() + 1);
  }

  inline void UpdateHistCol(const std::vector<GradientPair> &gpair,
                            const SparsePage::Inst &col,
                            const MetaInfo &info,
                            const RegTree &tree,
                            const std::vector<bst_feature_t> &fset,
                            bst_uint fid_offset,
                            std::vector<HistEntry> *p_temp) {
    if (col.size() == 0) return;
    // initialize sbuilder for use
    std::vector<HistEntry> &hbuilder = *p_temp;
    hbuilder.resize(tree.param.num_nodes);
    for (int const nid : this->qexpand_) {
      const unsigned wid = this->node2workindex_[nid];
      hbuilder[nid].istart = 0;
      hbuilder[nid].hist = this->wspace_.hset[0][fid_offset + wid * (fset.size()+1)];
    }
    if (this->param_.cache_opt != 0) {
      constexpr bst_uint kBuffer = 32;
      bst_uint align_length = col.size() / kBuffer * kBuffer;
      int buf_position[kBuffer];
      GradientPair buf_gpair[kBuffer];
      for (bst_uint j = 0; j < align_length; j += kBuffer) {
        for (bst_uint i = 0; i < kBuffer; ++i) {
          bst_uint ridx = col[j + i].index;
          buf_position[i] = this->position_[ridx];
          buf_gpair[i] = gpair[ridx];
        }
        for (bst_uint i = 0; i < kBuffer; ++i) {
          const int nid = buf_position[i];
          if (nid >= 0) {
            hbuilder[nid].Add(col[j + i].fvalue, buf_gpair[i]);
          }
        }
      }
      for (bst_uint j = align_length; j < col.size(); ++j) {
        const bst_uint ridx = col[j].index;
        const int nid = this->position_[ridx];
        if (nid >= 0) {
          hbuilder[nid].Add(col[j].fvalue, gpair[ridx]);
        }
      }
    } else {
      for (const auto& c : col) {
        const bst_uint ridx = c.index;
        const int nid = this->position_[ridx];
        if (nid >= 0) {
          hbuilder[nid].Add(c.fvalue, gpair, ridx);
        }
      }
    }
  }
  inline void UpdateSketchCol(const std::vector<GradientPair> &gpair,
                              const SparsePage::Inst &col,
                              const RegTree &tree,
                              size_t work_set_size,
                              bst_uint offset,
                              std::vector<BaseMaker::SketchEntry> *p_temp) {
    if (col.size() == 0) return;
    // initialize sbuilder for use
    std::vector<BaseMaker::SketchEntry> &sbuilder = *p_temp;
    sbuilder.resize(tree.param.num_nodes);
    for (int const nid : this->qexpand_) {
      const unsigned wid = this->node2workindex_[nid];
      sbuilder[nid].sum_total = 0.0f;
      sbuilder[nid].sketch = &sketchs_[wid * work_set_size + offset];
    }
    // first pass, get sum of weight, TODO, optimization to skip first pass
    for (const auto& c : col) {
        const bst_uint ridx = c.index;
        const int nid = this->position_[ridx];
        if (nid >= 0) {
          sbuilder[nid].sum_total += gpair[ridx].GetHess();
      }
    }
    // if only one value, no need to do second pass
    if (col[0].fvalue  == col[col.size()-1].fvalue) {
      for (int const nid : this->qexpand_) {
        sbuilder[nid].sketch->Push(
            col[0].fvalue, static_cast<bst_float>(sbuilder[nid].sum_total));
      }
      return;
    }
    // two pass scan
    unsigned max_size = this->param_.MaxSketchSize();
    for (int const nid : this->qexpand_) {
      sbuilder[nid].Init(max_size);
    }
    // second pass, build the sketch
    if (this->param_.cache_opt != 0) {
      constexpr bst_uint kBuffer = 32;
      bst_uint align_length = col.size() / kBuffer * kBuffer;
      int buf_position[kBuffer];
      bst_float buf_hess[kBuffer];
      for (bst_uint j = 0; j < align_length; j += kBuffer) {
        for (bst_uint i = 0; i < kBuffer; ++i) {
          bst_uint ridx = col[j + i].index;
          buf_position[i] = this->position_[ridx];
          buf_hess[i] = gpair[ridx].GetHess();
        }
        for (bst_uint i = 0; i < kBuffer; ++i) {
          const int nid = buf_position[i];
          if (nid >= 0) {
            sbuilder[nid].Push(col[j + i].fvalue, buf_hess[i], max_size);
          }
        }
      }
      for (bst_uint j = align_length; j < col.size(); ++j) {
        const bst_uint ridx = col[j].index;
        const int nid = this->position_[ridx];
        if (nid >= 0) {
          sbuilder[nid].Push(col[j].fvalue, gpair[ridx].GetHess(), max_size);
        }
      }
    } else {
      for (const auto& c : col) {
        const bst_uint ridx = c.index;
        const int nid = this->position_[ridx];
        if (nid >= 0) {
          sbuilder[nid].Push(c.fvalue, gpair[ridx].GetHess(), max_size);
        }
      }
    }
    for (int const nid : this->qexpand_) { sbuilder[nid].Finalize(max_size); }
  }
  // cached dmatrix where we initialized the feature on.
  const DMatrix* cache_dmatrix_{nullptr};
  // feature helper
  BaseMaker::FMetaHelper feat_helper_;
  // temp space to map feature id to working index
  std::vector<int> feat2workindex_;
  // set of index from fset that are current work set
  std::vector<bst_feature_t> work_set_;
  // set of index from that are split candidates.
  std::vector<bst_uint> fsplit_set_;
  // thread temp data
  std::vector<std::vector<BaseMaker::SketchEntry> > thread_sketch_;
  // used to hold statistics
  std::vector<std::vector<GradStats> > thread_stats_;
  // used to hold start pointer
  std::vector<std::vector<HistEntry> > thread_hist_;
  // node statistics
  std::vector<GradStats> node_stats_;
  // summary array
  std::vector<WXQSketch::SummaryContainer> summary_array_;
  // reducer for summary
  rabit::SerializeReducer<WXQSketch::SummaryContainer> sreducer_;
  // per node, per feature sketch
  std::vector<common::WXQuantileSketch<bst_float, bst_float> > sketchs_;
};

// global proposal
class GlobalProposalHistMaker: public CQHistMaker {
 public:
  char const* Name() const override {
    return "grow_histmaker";
  }

 protected:
  void ResetPosAndPropose(const std::vector<GradientPair> &gpair,
                          DMatrix *p_fmat,
                          const std::vector<bst_feature_t> &fset,
                          const RegTree &tree) override {
    if (this->qexpand_.size() == 1) {
      cached_rptr_.clear();
      cached_cut_.clear();
    }
    if (cached_rptr_.size() == 0) {
      CHECK_EQ(this->qexpand_.size(), 1U);
      CQHistMaker::ResetPosAndPropose(gpair, p_fmat, fset, tree);
      cached_rptr_ = this->wspace_.rptr;
      cached_cut_ = this->wspace_.cut;
    } else {
      this->wspace_.cut.clear();
      this->wspace_.rptr.clear();
      this->wspace_.rptr.push_back(0);
      for (size_t i = 0; i < this->qexpand_.size(); ++i) {
        for (size_t j = 0; j < cached_rptr_.size() - 1; ++j) {
          this->wspace_.rptr.push_back(
              this->wspace_.rptr.back() + cached_rptr_[j + 1] - cached_rptr_[j]);
        }
        this->wspace_.cut.insert(this->wspace_.cut.end(), cached_cut_.begin(), cached_cut_.end());
      }
      CHECK_EQ(this->wspace_.rptr.size(),
               (fset.size() + 1) * this->qexpand_.size() + 1);
      CHECK_EQ(this->wspace_.rptr.back(), this->wspace_.cut.size());
    }
  }

  // code to create histogram
  void CreateHist(const std::vector<GradientPair> &gpair,
                  DMatrix *p_fmat,
                  const std::vector<bst_feature_t> &fset,
                  const RegTree &tree) override {
    const MetaInfo &info = p_fmat->Info();
    // fill in reverse map
    this->feat2workindex_.resize(tree.param.num_feature);
    this->work_set_ = fset;
    std::fill(this->feat2workindex_.begin(), this->feat2workindex_.end(), -1);
    for (size_t i = 0; i < fset.size(); ++i) {
      this->feat2workindex_[fset[i]] = static_cast<int>(i);
    }
    // start to work
    this->wspace_.Configure(1);
    // to gain speedup in recovery
    {
      this->thread_hist_.resize(omp_get_max_threads());

      // TWOPASS: use the real set + split set in the column iteration.
      this->SetDefaultPostion(p_fmat, tree);
      this->work_set_.insert(this->work_set_.end(), this->fsplit_set_.begin(),
                             this->fsplit_set_.end());
      XGBOOST_PARALLEL_SORT(this->work_set_.begin(), this->work_set_.end(),
                            std::less<>{});
      this->work_set_.resize(
          std::unique(this->work_set_.begin(), this->work_set_.end()) - this->work_set_.begin());

      // start accumulating statistics
      for (const auto &batch : p_fmat->GetBatches<SortedCSCPage>()) {
        // TWOPASS: use the real set + split set in the column iteration.
        this->CorrectNonDefaultPositionByBatch(batch, this->fsplit_set_, tree);

        // start enumeration
        const auto nsize = static_cast<bst_omp_uint>(this->work_set_.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          int fid = this->work_set_[i];
          int offset = this->feat2workindex_[fid];
          if (offset >= 0) {
            this->UpdateHistCol(gpair, batch[fid], info, tree,
                                fset, offset,
                                &this->thread_hist_[omp_get_thread_num()]);
          }
        }
      }

      // update node statistics.
      this->GetNodeStats(gpair, *p_fmat, tree,
                         &(this->thread_stats_), &(this->node_stats_));
      for (const int nid : this->qexpand_) {
        const int wid = this->node2workindex_[nid];
        this->wspace_.hset[0][fset.size() + wid * (fset.size()+1)]
            .data[0] = this->node_stats_[nid];
      }
    }
    this->histred_.Allreduce(dmlc::BeginPtr(this->wspace_.hset[0].data),
                            this->wspace_.hset[0].data.size());
  }

  // cached unit pointer
  std::vector<unsigned> cached_rptr_;
  // cached cut value.
  std::vector<bst_float> cached_cut_;
};

XGBOOST_REGISTER_TREE_UPDATER(LocalHistMaker, "grow_local_histmaker")
.describe("Tree constructor that uses approximate histogram construction.")
.set_body([]() {
    return new CQHistMaker();
  });

// The updater for approx tree method.
XGBOOST_REGISTER_TREE_UPDATER(HistMaker, "grow_histmaker")
.describe("Tree constructor that uses approximate global of histogram construction.")
.set_body([]() {
    return new GlobalProposalHistMaker();
  });
}  // namespace tree
}  // namespace xgboost
