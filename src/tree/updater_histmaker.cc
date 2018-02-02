/*!
 * Copyright 2014 by Contributors
 * \file updater_histmaker.cc
 * \brief use histogram counting to construct a tree
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

DMLC_REGISTRY_FILE_TAG(updater_histmaker);

template<typename TStats>
class HistMaker: public BaseMaker {
 public:
  void Update(const std::vector<bst_gpair> &gpair,
              DMatrix *p_fmat,
              const std::vector<RegTree*> &trees) override {
    TStats::CheckInfo(p_fmat->info());
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
  /*! \brief a single histogram */
  struct HistUnit {
    /*! \brief cutting point of histogram, contains maximum point */
    const bst_float *cut;
    /*! \brief content of statistics data */
    TStats *data;
    /*! \brief size of histogram */
    unsigned size;
    // default constructor
    HistUnit() {}
    // constructor
    HistUnit(const bst_float *cut, TStats *data, unsigned size)
        : cut(cut), data(data), size(size) {}
    /*! \brief add a histogram to data */
    inline void Add(bst_float fv,
                    const std::vector<bst_gpair> &gpair,
                    const MetaInfo &info,
                    const bst_uint ridx) {
      unsigned i = std::upper_bound(cut, cut + size, fv) - cut;
      CHECK_NE(size, 0U) << "try insert into size=0";
      CHECK_LT(i, size);
      data[i].Add(gpair, info, ridx);
    }
  };
  /*! \brief a set of histograms from different index */
  struct HistSet {
    /*! \brief the index pointer of each histunit */
    const unsigned *rptr;
    /*! \brief cutting points in each histunit */
    const bst_float *cut;
    /*! \brief data in different hist unit */
    std::vector<TStats> data;
    /*! \brief */
    inline HistUnit operator[](size_t fid) {
      return HistUnit(cut + rptr[fid],
                      &data[0] + rptr[fid],
                      rptr[fid+1] - rptr[fid]);
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
    inline void Init(const TrainParam &param, int nthread) {
      hset.resize(nthread);
      // cleanup statistics
      for (int tid = 0; tid < nthread; ++tid) {
        for (size_t i = 0; i < hset[tid].data.size(); ++i) {
          hset[tid].data[i].Clear();
        }
        hset[tid].rptr = dmlc::BeginPtr(rptr);
        hset[tid].cut = dmlc::BeginPtr(cut);
        hset[tid].data.resize(cut.size(), TStats(param));
      }
    }
    // aggregate all statistics to hset[0]
    inline void Aggregate() {
      bst_omp_uint nsize = static_cast<bst_omp_uint>(cut.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        for (size_t tid = 1; tid < hset.size(); ++tid) {
          hset[0].data[i].Add(hset[tid].data[i]);
        }
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
  ThreadWSpace wspace;
  // reducer for histogram
  rabit::Reducer<TStats, TStats::Reduce> histred;
  // set of working features
  std::vector<bst_uint> fwork_set;
  // update function implementation
  virtual void Update(const std::vector<bst_gpair> &gpair,
                      DMatrix *p_fmat,
                      RegTree *p_tree) {
    this->InitData(gpair, *p_fmat, *p_tree);
    this->InitWorkSet(p_fmat, *p_tree, &fwork_set);
    // mark root node as fresh.
    for (int i = 0; i < p_tree->param.num_roots; ++i) {
      (*p_tree)[i].set_leaf(0.0f, 0);
    }

    for (int depth = 0; depth < param.max_depth; ++depth) {
      // reset and propose candidate split
      this->ResetPosAndPropose(gpair, p_fmat, fwork_set, *p_tree);
      // create histogram
      this->CreateHist(gpair, p_fmat, fwork_set, *p_tree);
      // find split based on histogram statistics
      this->FindSplit(depth, gpair, p_fmat, fwork_set, p_tree);
      // reset position after split
      this->ResetPositionAfterSplit(p_fmat, *p_tree);
      this->UpdateQueueExpand(*p_tree);
      // if nothing left to be expand, break
      if (qexpand.size() == 0) break;
    }
    for (size_t i = 0; i < qexpand.size(); ++i) {
      const int nid = qexpand[i];
      (*p_tree)[nid].set_leaf(p_tree->stat(nid).base_weight * param.learning_rate);
    }
  }
  // this function does two jobs
  // (1) reset the position in array position, to be the latest leaf id
  // (2) propose a set of candidate cuts and set wspace.rptr wspace.cut correctly
  virtual void ResetPosAndPropose(const std::vector<bst_gpair> &gpair,
                                  DMatrix *p_fmat,
                                  const std::vector <bst_uint> &fset,
                                  const RegTree &tree) = 0;
  // initialize the current working set of features in this round
  virtual void InitWorkSet(DMatrix *p_fmat,
                           const RegTree &tree,
                           std::vector<bst_uint> *p_fset) {
    p_fset->resize(tree.param.num_feature);
    for (size_t i = 0; i < p_fset->size(); ++i) {
      (*p_fset)[i] = static_cast<unsigned>(i);
    }
  }
  // reset position after split, this is not a must, depending on implementation
  virtual void ResetPositionAfterSplit(DMatrix *p_fmat,
                                       const RegTree &tree) {
  }
  virtual void CreateHist(const std::vector<bst_gpair> &gpair,
                          DMatrix *p_fmat,
                          const std::vector <bst_uint> &fset,
                          const RegTree &tree)  = 0;

 private:
  inline void EnumerateSplit(const HistUnit &hist,
                             const TStats &node_sum,
                             bst_uint fid,
                             SplitEntry *best,
                             TStats *left_sum) {
    if (hist.size == 0) return;

    double root_gain = node_sum.CalcGain(param);
    TStats s(param), c(param);
    for (bst_uint i = 0; i < hist.size; ++i) {
      s.Add(hist.data[i]);
      if (s.sum_hess >= param.min_child_weight) {
        c.SetSubstract(node_sum, s);
        if (c.sum_hess >= param.min_child_weight) {
          double loss_chg = s.CalcGain(param) + c.CalcGain(param) - root_gain;
          if (best->Update(static_cast<bst_float>(loss_chg), fid, hist.cut[i], false)) {
            *left_sum = s;
          }
        }
      }
    }
    s.Clear();
    for (bst_uint i = hist.size - 1; i != 0; --i) {
      s.Add(hist.data[i]);
      if (s.sum_hess >= param.min_child_weight) {
        c.SetSubstract(node_sum, s);
        if (c.sum_hess >= param.min_child_weight) {
          double loss_chg = s.CalcGain(param) + c.CalcGain(param) - root_gain;
          if (best->Update(static_cast<bst_float>(loss_chg), fid, hist.cut[i-1], true)) {
            *left_sum = c;
          }
        }
      }
    }
  }
  inline void FindSplit(int depth,
                        const std::vector<bst_gpair> &gpair,
                        DMatrix *p_fmat,
                        const std::vector <bst_uint> &fset,
                        RegTree *p_tree) {
    const size_t num_feature = fset.size();
    // get the best split condition for each node
    std::vector<SplitEntry> sol(qexpand.size());
    std::vector<TStats> left_sum(qexpand.size());
    bst_omp_uint nexpand = static_cast<bst_omp_uint>(qexpand.size());
    #pragma omp parallel for schedule(dynamic, 1)
    for (bst_omp_uint wid = 0; wid < nexpand; ++wid) {
      const int nid = qexpand[wid];
      CHECK_EQ(node2workindex[nid], static_cast<int>(wid));
      SplitEntry &best = sol[wid];
      TStats &node_sum = wspace.hset[0][num_feature + wid * (num_feature + 1)].data[0];
      for (size_t i = 0; i < fset.size(); ++i) {
        EnumerateSplit(this->wspace.hset[0][i + wid * (num_feature+1)],
                       node_sum, fset[i], &best, &left_sum[wid]);
      }
    }
    // get the best result, we can synchronize the solution
    for (bst_omp_uint wid = 0; wid < nexpand; ++wid) {
      const int nid = qexpand[wid];
      const SplitEntry &best = sol[wid];
      const TStats &node_sum = wspace.hset[0][num_feature + wid * (num_feature + 1)].data[0];
      this->SetStats(p_tree, nid, node_sum);
      // set up the values
      p_tree->stat(nid).loss_chg = best.loss_chg;
      // now we know the solution in snode[nid], set split
      if (best.loss_chg > rt_eps) {
        p_tree->AddChilds(nid);
        (*p_tree)[nid].set_split(best.split_index(),
                                 best.split_value, best.default_left());
        // mark right child as 0, to indicate fresh leaf
        (*p_tree)[(*p_tree)[nid].cleft()].set_leaf(0.0f, 0);
        (*p_tree)[(*p_tree)[nid].cright()].set_leaf(0.0f, 0);
        // right side sum
        TStats right_sum;
        right_sum.SetSubstract(node_sum, left_sum[wid]);
        this->SetStats(p_tree, (*p_tree)[nid].cleft(), left_sum[wid]);
        this->SetStats(p_tree, (*p_tree)[nid].cright(), right_sum);
      } else {
        (*p_tree)[nid].set_leaf(p_tree->stat(nid).base_weight * param.learning_rate);
      }
    }
  }

  inline void SetStats(RegTree *p_tree, int nid, const TStats &node_sum) {
    p_tree->stat(nid).base_weight = static_cast<bst_float>(node_sum.CalcWeight(param));
    p_tree->stat(nid).sum_hess = static_cast<bst_float>(node_sum.sum_hess);
    node_sum.SetLeafVec(param, p_tree->leafvec(nid));
  }
};

template<typename TStats>
class CQHistMaker: public HistMaker<TStats> {
 public:
  CQHistMaker() : cache_dmatrix_(nullptr) {
  }

 protected:
  struct HistEntry {
    typename HistMaker<TStats>::HistUnit hist;
    unsigned istart;
    /*!
     * \brief add a histogram to data,
     * do linear scan, start from istart
     */
    inline void Add(bst_float fv,
                    const std::vector<bst_gpair> &gpair,
                    const MetaInfo &info,
                    const bst_uint ridx) {
      while (istart < hist.size && !(fv < hist.cut[istart])) ++istart;
      CHECK_NE(istart, hist.size);
      hist.data[istart].Add(gpair, info, ridx);
    }
    /*!
     * \brief add a histogram to data,
     * do linear scan, start from istart
     */
    inline void Add(bst_float fv,
                    bst_gpair gstats) {
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
  typedef common::WXQuantileSketch<bst_float, bst_float> WXQSketch;
  // initialize the work set of tree
  void InitWorkSet(DMatrix *p_fmat,
                   const RegTree &tree,
                   std::vector<bst_uint> *p_fset) override {
    if (p_fmat != cache_dmatrix_) {
      feat_helper.InitByCol(p_fmat, tree);
      cache_dmatrix_ = p_fmat;
    }
    feat_helper.SyncInfo();
    feat_helper.SampleCol(this->param.colsample_bytree, p_fset);
  }
  // code to create histogram
  void CreateHist(const std::vector<bst_gpair> &gpair,
                  DMatrix *p_fmat,
                  const std::vector<bst_uint> &fset,
                  const RegTree &tree) override {
    const MetaInfo &info = p_fmat->info();
    // fill in reverse map
    feat2workindex.resize(tree.param.num_feature);
    std::fill(feat2workindex.begin(), feat2workindex.end(), -1);
    for (size_t i = 0; i < fset.size(); ++i) {
      feat2workindex[fset[i]] = static_cast<int>(i);
    }
    // start to work
    this->wspace.Init(this->param, 1);
    // if it is C++11, use lazy evaluation for Allreduce,
    // to gain speedup in recovery
#if __cplusplus >= 201103L
    auto lazy_get_hist = [&]()
#endif
    {
      thread_hist.resize(omp_get_max_threads());
      // start accumulating statistics
      dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator(fset);
      iter->BeforeFirst();
      while (iter->Next()) {
        const ColBatch &batch = iter->Value();
        // start enumeration
        const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
        #pragma omp parallel for schedule(dynamic, 1)
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          int offset = feat2workindex[batch.col_index[i]];
          if (offset >= 0) {
            this->UpdateHistCol(gpair, batch[i], info, tree,
                                fset, offset,
                                &thread_hist[omp_get_thread_num()]);
          }
        }
      }
      // update node statistics.
      this->GetNodeStats(gpair, *p_fmat, tree,
                         &thread_stats, &node_stats);
      for (size_t i = 0; i < this->qexpand.size(); ++i) {
        const int nid = this->qexpand[i];
        const int wid = this->node2workindex[nid];
        this->wspace.hset[0][fset.size() + wid * (fset.size()+1)]
            .data[0] = node_stats[nid];
      }
    };
    // sync the histogram
    // if it is C++11, use lazy evaluation for Allreduce
#if __cplusplus >= 201103L
    this->histred.Allreduce(dmlc::BeginPtr(this->wspace.hset[0].data),
                            this->wspace.hset[0].data.size(), lazy_get_hist);
#else
    this->histred.Allreduce(dmlc::BeginPtr(this->wspace.hset[0].data),
                            this->wspace.hset[0].data.size());
#endif
  }
  void ResetPositionAfterSplit(DMatrix *p_fmat,
                               const RegTree &tree) override {
    this->GetSplitSet(this->qexpand, tree, &fsplit_set);
  }
  void ResetPosAndPropose(const std::vector<bst_gpair> &gpair,
                          DMatrix *p_fmat,
                          const std::vector<bst_uint> &fset,
                          const RegTree &tree) override {
    const MetaInfo &info = p_fmat->info();
    // fill in reverse map
    feat2workindex.resize(tree.param.num_feature);
    std::fill(feat2workindex.begin(), feat2workindex.end(), -1);
    work_set.clear();
    for (size_t i = 0; i < fset.size(); ++i) {
      if (feat_helper.Type(fset[i]) == 2) {
        feat2workindex[fset[i]] = static_cast<int>(work_set.size());
        work_set.push_back(fset[i]);
      } else {
        feat2workindex[fset[i]] = -2;
      }
    }
    const size_t work_set_size = work_set.size();

    sketchs.resize(this->qexpand.size() * work_set_size);
    for (size_t i = 0; i < sketchs.size(); ++i) {
      sketchs[i].Init(info.num_row, this->param.sketch_eps);
    }
    // intitialize the summary array
    summary_array.resize(sketchs.size());
    // setup maximum size
    unsigned max_size = this->param.max_sketch_size();
    for (size_t i = 0; i < sketchs.size(); ++i) {
      summary_array[i].Reserve(max_size);
    }
    {
      // get smmary
      thread_sketch.resize(omp_get_max_threads());

      // TWOPASS: use the real set + split set in the column iteration.
      this->SetDefaultPostion(p_fmat, tree);
      work_set.insert(work_set.end(), fsplit_set.begin(), fsplit_set.end());
      std::sort(work_set.begin(), work_set.end());
      work_set.resize(std::unique(work_set.begin(), work_set.end()) - work_set.begin());

      // start accumulating statistics
      dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator(work_set);
      iter->BeforeFirst();
      while (iter->Next()) {
        const ColBatch &batch = iter->Value();
        // TWOPASS: use the real set + split set in the column iteration.
        this->CorrectNonDefaultPositionByBatch(batch, fsplit_set, tree);

        // start enumeration
        const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
        #pragma omp parallel for schedule(dynamic, 1)
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          int offset = feat2workindex[batch.col_index[i]];
          if (offset >= 0) {
            this->UpdateSketchCol(gpair, batch[i], tree,
                                  work_set_size, offset,
                                  &thread_sketch[omp_get_thread_num()]);
          }
        }
      }
      for (size_t i = 0; i < sketchs.size(); ++i) {
        common::WXQuantileSketch<bst_float, bst_float>::SummaryContainer out;
        sketchs[i].GetSummary(&out);
        summary_array[i].SetPrune(out, max_size);
      }
      CHECK_EQ(summary_array.size(), sketchs.size());
    }
    if (summary_array.size() != 0) {
      size_t nbytes = WXQSketch::SummaryContainer::CalcMemCost(max_size);
      sreducer.Allreduce(dmlc::BeginPtr(summary_array), nbytes, summary_array.size());
    }
    // now we get the final result of sketch, setup the cut
    this->wspace.cut.clear();
    this->wspace.rptr.clear();
    this->wspace.rptr.push_back(0);
    for (size_t wid = 0; wid < this->qexpand.size(); ++wid) {
      for (size_t i = 0; i < fset.size(); ++i) {
        int offset = feat2workindex[fset[i]];
        if (offset >= 0) {
          const WXQSketch::Summary &a = summary_array[wid * work_set_size + offset];
          for (size_t i = 1; i < a.size; ++i) {
            bst_float cpt = a.data[i].value - rt_eps;
            if (i == 1 || cpt > this->wspace.cut.back()) {
              this->wspace.cut.push_back(cpt);
            }
          }
          // push a value that is greater than anything
          if (a.size != 0) {
            bst_float cpt = a.data[a.size - 1].value;
            // this must be bigger than last value in a scale
            bst_float last = cpt + fabs(cpt) + rt_eps;
            this->wspace.cut.push_back(last);
          }
          this->wspace.rptr.push_back(static_cast<unsigned>(this->wspace.cut.size()));
        } else {
          CHECK_EQ(offset, -2);
          bst_float cpt = feat_helper.MaxValue(fset[i]);
          this->wspace.cut.push_back(cpt + fabs(cpt) + rt_eps);
          this->wspace.rptr.push_back(static_cast<unsigned>(this->wspace.cut.size()));
        }
      }
      // reserve last value for global statistics
      this->wspace.cut.push_back(0.0f);
      this->wspace.rptr.push_back(static_cast<unsigned>(this->wspace.cut.size()));
    }
    CHECK_EQ(this->wspace.rptr.size(),
             (fset.size() + 1) * this->qexpand.size() + 1);
  }

  inline void UpdateHistCol(const std::vector<bst_gpair> &gpair,
                            const ColBatch::Inst &c,
                            const MetaInfo &info,
                            const RegTree &tree,
                            const std::vector<bst_uint> &fset,
                            bst_uint fid_offset,
                            std::vector<HistEntry> *p_temp) {
    if (c.length == 0) return;
    // initialize sbuilder for use
    std::vector<HistEntry> &hbuilder = *p_temp;
    hbuilder.resize(tree.param.num_nodes);
    for (size_t i = 0; i < this->qexpand.size(); ++i) {
      const unsigned nid = this->qexpand[i];
      const unsigned wid = this->node2workindex[nid];
      hbuilder[nid].istart = 0;
      hbuilder[nid].hist = this->wspace.hset[0][fid_offset + wid * (fset.size()+1)];
    }
    if (TStats::kSimpleStats != 0 && this->param.cache_opt != 0) {
      const bst_uint kBuffer = 32;
      bst_uint align_length = c.length / kBuffer * kBuffer;
      int buf_position[kBuffer];
      bst_gpair buf_gpair[kBuffer];
      for (bst_uint j = 0; j < align_length; j += kBuffer) {
        for (bst_uint i = 0; i < kBuffer; ++i) {
          bst_uint ridx = c[j + i].index;
          buf_position[i] = this->position[ridx];
          buf_gpair[i] = gpair[ridx];
        }
        for (bst_uint i = 0; i < kBuffer; ++i) {
          const int nid = buf_position[i];
          if (nid >= 0) {
            hbuilder[nid].Add(c[j + i].fvalue, buf_gpair[i]);
          }
        }
      }
      for (bst_uint j = align_length; j < c.length; ++j) {
        const bst_uint ridx = c[j].index;
        const int nid = this->position[ridx];
        if (nid >= 0) {
          hbuilder[nid].Add(c[j].fvalue, gpair[ridx]);
        }
      }
    } else {
      for (bst_uint j = 0; j < c.length; ++j) {
        const bst_uint ridx = c[j].index;
        const int nid = this->position[ridx];
        if (nid >= 0) {
          hbuilder[nid].Add(c[j].fvalue, gpair, info, ridx);
        }
      }
    }
  }
  inline void UpdateSketchCol(const std::vector<bst_gpair> &gpair,
                              const ColBatch::Inst &c,
                              const RegTree &tree,
                              size_t work_set_size,
                              bst_uint offset,
                              std::vector<BaseMaker::SketchEntry> *p_temp) {
    if (c.length == 0) return;
    // initialize sbuilder for use
    std::vector<BaseMaker::SketchEntry> &sbuilder = *p_temp;
    sbuilder.resize(tree.param.num_nodes);
    for (size_t i = 0; i < this->qexpand.size(); ++i) {
      const unsigned nid = this->qexpand[i];
      const unsigned wid = this->node2workindex[nid];
      sbuilder[nid].sum_total = 0.0f;
      sbuilder[nid].sketch = &sketchs[wid * work_set_size + offset];
    }

    // first pass, get sum of weight, TODO, optimization to skip first pass
    for (bst_uint j = 0; j < c.length; ++j) {
        const bst_uint ridx = c[j].index;
        const int nid = this->position[ridx];
        if (nid >= 0) {
        sbuilder[nid].sum_total += gpair[ridx].GetHess();
      }
    }
    // if only one value, no need to do second pass
    if (c[0].fvalue  == c[c.length-1].fvalue) {
      for (size_t i = 0; i < this->qexpand.size(); ++i) {
        const int nid = this->qexpand[i];
        sbuilder[nid].sketch->Push(c[0].fvalue, static_cast<bst_float>(sbuilder[nid].sum_total));
      }
      return;
    }
    // two pass scan
    unsigned max_size = this->param.max_sketch_size();
    for (size_t i = 0; i < this->qexpand.size(); ++i) {
      const int nid = this->qexpand[i];
      sbuilder[nid].Init(max_size);
    }
    // second pass, build the sketch
    if (TStats::kSimpleStats != 0 && this->param.cache_opt != 0) {
      const bst_uint kBuffer = 32;
      bst_uint align_length = c.length / kBuffer * kBuffer;
      int buf_position[kBuffer];
      bst_float buf_hess[kBuffer];
      for (bst_uint j = 0; j < align_length; j += kBuffer) {
        for (bst_uint i = 0; i < kBuffer; ++i) {
          bst_uint ridx = c[j + i].index;
          buf_position[i] = this->position[ridx];
          buf_hess[i] = gpair[ridx].GetHess();
        }
        for (bst_uint i = 0; i < kBuffer; ++i) {
          const int nid = buf_position[i];
          if (nid >= 0) {
            sbuilder[nid].Push(c[j + i].fvalue, buf_hess[i], max_size);
          }
        }
      }
      for (bst_uint j = align_length; j < c.length; ++j) {
        const bst_uint ridx = c[j].index;
        const int nid = this->position[ridx];
        if (nid >= 0) {
          sbuilder[nid].Push(c[j].fvalue, gpair[ridx].GetHess(), max_size);
        }
      }
    } else {
      for (bst_uint j = 0; j < c.length; ++j) {
        const bst_uint ridx = c[j].index;
        const int nid = this->position[ridx];
        if (nid >= 0) {
          sbuilder[nid].Push(c[j].fvalue, gpair[ridx].GetHess(), max_size);
        }
      }
    }
    for (size_t i = 0; i < this->qexpand.size(); ++i) {
      const int nid = this->qexpand[i];
      sbuilder[nid].Finalize(max_size);
    }
  }
  // cached dmatrix where we initialized the feature on.
  const DMatrix* cache_dmatrix_;
  // feature helper
  BaseMaker::FMetaHelper feat_helper;
  // temp space to map feature id to working index
  std::vector<int> feat2workindex;
  // set of index from fset that are current work set
  std::vector<bst_uint> work_set;
  // set of index from that are split candidates.
  std::vector<bst_uint> fsplit_set;
  // thread temp data
  std::vector<std::vector<BaseMaker::SketchEntry> > thread_sketch;
  // used to hold statistics
  std::vector<std::vector<TStats> > thread_stats;
  // used to hold start pointer
  std::vector<std::vector<HistEntry> > thread_hist;
  // node statistics
  std::vector<TStats> node_stats;
  // summary array
  std::vector<WXQSketch::SummaryContainer> summary_array;
  // reducer for summary
  rabit::SerializeReducer<WXQSketch::SummaryContainer> sreducer;
  // per node, per feature sketch
  std::vector<common::WXQuantileSketch<bst_float, bst_float> > sketchs;
};

// global proposal
template<typename TStats>
class GlobalProposalHistMaker: public CQHistMaker<TStats> {
 protected:
  void ResetPosAndPropose(const std::vector<bst_gpair> &gpair,
                          DMatrix *p_fmat,
                          const std::vector<bst_uint> &fset,
                          const RegTree &tree) override {
    if (this->qexpand.size() == 1) {
      cached_rptr_.clear();
      cached_cut_.clear();
    }
    if (cached_rptr_.size() == 0) {
      CHECK_EQ(this->qexpand.size(), 1U);
      CQHistMaker<TStats>::ResetPosAndPropose(gpair, p_fmat, fset, tree);
      cached_rptr_ = this->wspace.rptr;
      cached_cut_ = this->wspace.cut;
    } else {
      this->wspace.cut.clear();
      this->wspace.rptr.clear();
      this->wspace.rptr.push_back(0);
      for (size_t i = 0; i < this->qexpand.size(); ++i) {
        for (size_t j = 0; j < cached_rptr_.size() - 1; ++j) {
          this->wspace.rptr.push_back(
              this->wspace.rptr.back() + cached_rptr_[j + 1] - cached_rptr_[j]);
        }
        this->wspace.cut.insert(this->wspace.cut.end(), cached_cut_.begin(), cached_cut_.end());
      }
      CHECK_EQ(this->wspace.rptr.size(),
               (fset.size() + 1) * this->qexpand.size() + 1);
      CHECK_EQ(this->wspace.rptr.back(), this->wspace.cut.size());
    }
  }

  // code to create histogram
  void CreateHist(const std::vector<bst_gpair> &gpair,
                  DMatrix *p_fmat,
                  const std::vector<bst_uint> &fset,
                  const RegTree &tree) override {
    const MetaInfo &info = p_fmat->info();
    // fill in reverse map
    this->feat2workindex.resize(tree.param.num_feature);
    this->work_set = fset;
    std::fill(this->feat2workindex.begin(), this->feat2workindex.end(), -1);
    for (size_t i = 0; i < fset.size(); ++i) {
      this->feat2workindex[fset[i]] = static_cast<int>(i);
    }
    // start to work
    this->wspace.Init(this->param, 1);
    // to gain speedup in recovery
    {
      this->thread_hist.resize(omp_get_max_threads());

      // TWOPASS: use the real set + split set in the column iteration.
      this->SetDefaultPostion(p_fmat, tree);
      this->work_set.insert(this->work_set.end(), this->fsplit_set.begin(), this->fsplit_set.end());
      std::sort(this->work_set.begin(), this->work_set.end());
      this->work_set.resize(
          std::unique(this->work_set.begin(), this->work_set.end()) - this->work_set.begin());

      // start accumulating statistics
      dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator(this->work_set);
      iter->BeforeFirst();
      while (iter->Next()) {
        const ColBatch &batch = iter->Value();
        // TWOPASS: use the real set + split set in the column iteration.
        this->CorrectNonDefaultPositionByBatch(batch, this->fsplit_set, tree);

        // start enumeration
        const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
        #pragma omp parallel for schedule(dynamic, 1)
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          int offset = this->feat2workindex[batch.col_index[i]];
          if (offset >= 0) {
            this->UpdateHistCol(gpair, batch[i], info, tree,
                                fset, offset,
                                &this->thread_hist[omp_get_thread_num()]);
          }
        }
      }

      // update node statistics.
      this->GetNodeStats(gpair, *p_fmat, tree,
                         &(this->thread_stats), &(this->node_stats));
      for (size_t i = 0; i < this->qexpand.size(); ++i) {
        const int nid = this->qexpand[i];
        const int wid = this->node2workindex[nid];
        this->wspace.hset[0][fset.size() + wid * (fset.size()+1)]
            .data[0] = this->node_stats[nid];
      }
    }
    this->histred.Allreduce(dmlc::BeginPtr(this->wspace.hset[0].data),
                            this->wspace.hset[0].data.size());
  }

  // cached unit pointer
  std::vector<unsigned> cached_rptr_;
  // cached cut value.
  std::vector<bst_float> cached_cut_;
};


template<typename TStats>
class QuantileHistMaker: public HistMaker<TStats> {
 protected:
  typedef common::WXQuantileSketch<bst_float, bst_float> WXQSketch;
  void ResetPosAndPropose(const std::vector<bst_gpair> &gpair,
                          DMatrix *p_fmat,
                          const std::vector <bst_uint> &fset,
                          const RegTree &tree) override {
    const MetaInfo &info = p_fmat->info();
    // initialize the data structure
    const int nthread = omp_get_max_threads();
    sketchs.resize(this->qexpand.size() * tree.param.num_feature);
    for (size_t i = 0; i < sketchs.size(); ++i) {
      sketchs[i].Init(info.num_row, this->param.sketch_eps);
    }
    // start accumulating statistics
    dmlc::DataIter<RowBatch> *iter = p_fmat->RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch &batch = iter->Value();
      // parallel convert to column major format
      common::ParallelGroupBuilder<SparseBatch::Entry>
          builder(&col_ptr, &col_data, &thread_col_ptr);
      builder.InitBudget(tree.param.num_feature, nthread);

      const bst_omp_uint nbatch = static_cast<bst_omp_uint>(batch.size);
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nbatch; ++i) {
        RowBatch::Inst inst = batch[i];
        const bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
        int nid = this->position[ridx];
        if (nid >= 0) {
          if (!tree[nid].is_leaf()) {
            this->position[ridx] = nid = HistMaker<TStats>::NextLevel(inst, tree, nid);
          }
          if (this->node2workindex[nid] < 0) {
            this->position[ridx] = ~nid;
          } else {
            for (bst_uint j = 0; j < inst.length; ++j) {
              builder.AddBudget(inst[j].index, omp_get_thread_num());
            }
          }
        }
      }
      builder.InitStorage();
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nbatch; ++i) {
        RowBatch::Inst inst = batch[i];
        const bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
        const int nid = this->position[ridx];
        if (nid >= 0) {
          for (bst_uint j = 0; j < inst.length; ++j) {
            builder.Push(inst[j].index,
                         SparseBatch::Entry(nid, inst[j].fvalue),
                         omp_get_thread_num());
          }
        }
      }
      // start putting things into sketch
      const bst_omp_uint nfeat = col_ptr.size() - 1;
      #pragma omp parallel for schedule(dynamic, 1)
      for (bst_omp_uint k = 0; k < nfeat; ++k) {
        for (size_t i = col_ptr[k]; i < col_ptr[k+1]; ++i) {
          const SparseBatch::Entry &e = col_data[i];
          const int wid = this->node2workindex[e.index];
          sketchs[wid * tree.param.num_feature + k].Push(e.fvalue, gpair[e.index].GetHess());
        }
      }
    }
    // setup maximum size
    unsigned max_size = this->param.max_sketch_size();
    // synchronize sketch
    summary_array.resize(sketchs.size());
    for (size_t i = 0; i < sketchs.size(); ++i) {
      common::WQuantileSketch<bst_float, bst_float>::SummaryContainer out;
      sketchs[i].GetSummary(&out);
      summary_array[i].Reserve(max_size);
      summary_array[i].SetPrune(out, max_size);
    }

    size_t nbytes = WXQSketch::SummaryContainer::CalcMemCost(max_size);
    sreducer.Allreduce(dmlc::BeginPtr(summary_array), nbytes, summary_array.size());
    // now we get the final result of sketch, setup the cut
    this->wspace.cut.clear();
    this->wspace.rptr.clear();
    this->wspace.rptr.push_back(0);
    for (size_t wid = 0; wid < this->qexpand.size(); ++wid) {
      for (int fid = 0; fid < tree.param.num_feature; ++fid) {
        const WXQSketch::Summary &a = summary_array[wid * tree.param.num_feature + fid];
        for (size_t i = 1; i < a.size; ++i) {
          bst_float cpt = a.data[i].value - rt_eps;
          if (i == 1 || cpt > this->wspace.cut.back()) {
            this->wspace.cut.push_back(cpt);
          }
        }
        // push a value that is greater than anything
        if (a.size != 0) {
          bst_float cpt = a.data[a.size - 1].value;
          // this must be bigger than last value in a scale
          bst_float last = cpt + fabs(cpt) + rt_eps;
          this->wspace.cut.push_back(last);
        }
        this->wspace.rptr.push_back(this->wspace.cut.size());
      }
      // reserve last value for global statistics
      this->wspace.cut.push_back(0.0f);
      this->wspace.rptr.push_back(this->wspace.cut.size());
    }
    CHECK_EQ(this->wspace.rptr.size(),
             (tree.param.num_feature + 1) * this->qexpand.size() + 1);
  }

 private:
  // summary array
  std::vector<WXQSketch::SummaryContainer> summary_array;
  // reducer for summary
  rabit::SerializeReducer<WXQSketch::SummaryContainer> sreducer;
  // local temp column data structure
  std::vector<size_t> col_ptr;
  // local storage of column data
  std::vector<SparseBatch::Entry> col_data;
  std::vector<std::vector<size_t> > thread_col_ptr;
  // per node, per feature sketch
  std::vector<common::WQuantileSketch<bst_float, bst_float> > sketchs;
};

XGBOOST_REGISTER_TREE_UPDATER(LocalHistMaker, "grow_local_histmaker")
.describe("Tree constructor that uses approximate histogram construction.")
.set_body([]() {
    return new CQHistMaker<GradStats>();
  });

XGBOOST_REGISTER_TREE_UPDATER(GlobalHistMaker, "grow_global_histmaker")
.describe("Tree constructor that uses approximate global proposal of histogram construction.")
.set_body([]() {
    return new GlobalProposalHistMaker<GradStats>();
  });

XGBOOST_REGISTER_TREE_UPDATER(HistMaker, "grow_histmaker")
.describe("Tree constructor that uses approximate global of histogram construction.")
.set_body([]() {
    return new GlobalProposalHistMaker<GradStats>();
  });
}  // namespace tree
}  // namespace xgboost
