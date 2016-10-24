/*!
 * Copyright 2017 by Contributors
 * \file updater_fast_hist.cc
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Checn
 */
#include <dmlc/timer.h>
#include <xgboost/tree_updater.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <queue>
#include <iomanip>
#include "./param.h"
#include "../common/random.h"
#include "../common/bitmap.h"
#include "../common/sync.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"

namespace xgboost {
namespace tree {

using xgboost::common::HistCutMatrix;
using xgboost::common::GHistIndexMatrix;
using xgboost::common::GHistIndexRow;
using xgboost::common::HistEntry;
using xgboost::common::HistCollection;
using xgboost::common::RowSetCollection;
using xgboost::common::HistRow;
using xgboost::common::HistMaker;

DMLC_REGISTRY_FILE_TAG(updater_fast_hist);

/*! \brief construct a tree using quantized feature values */
template<typename TStats, typename TConstraint>
class FastHistMaker: public TreeUpdater {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    param.InitAllowUnknown(args);
    is_gmat_initialized_ = false;
  }

  void Update(const std::vector<bst_gpair>& gpair,
              DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    TStats::CheckInfo(dmat->info());
    if (is_gmat_initialized_ == false) {
      double tstart = dmlc::GetTime();
      hmat_.Init(dmat, param.max_bin);
      gmat_.cut = &hmat_;
      gmat_.Init(dmat);
      is_gmat_initialized_ = true;
      if (param.verbose > 0) {
        LOG(INFO) << "Generating gmat: " << dmlc::GetTime()-tstart << " sec";
      }
    }
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    TConstraint::Init(&param, dmat->info().num_col);
    // build tree
    if (!builder_) {
      builder_.reset(new Builder(param));
    }
    for (size_t i = 0; i < trees.size(); ++i) {
      builder_->Update(gmat_, gpair, dmat, trees[i]);
    }
    param.learning_rate = lr;
  }

 protected:
  // training parameter
  TrainParam param;
  // data sketch
  HistCutMatrix hmat_;
  GHistIndexMatrix gmat_;
  bool is_gmat_initialized_;

  // data structure
  /*! \brief per thread x per node entry to store tmp data */
  struct ThreadEntry {
    /*! \brief statistics of data */
    TStats stats;
    /*! \brief extra statistics of data */
    TStats stats_extra;
    /*! \brief last feature value scanned */
    float  last_fvalue;
    /*! \brief first feature value scanned */
    float  first_fvalue;
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    explicit ThreadEntry(const TrainParam& param)
        : stats(param), stats_extra(param) {
    }
  };
  struct NodeEntry {
    /*! \brief statics for node entry */
    TStats stats;
    /*! \brief loss of this node, without split */
    bst_float root_gain;
    /*! \brief weight calculated related to current data */
    float weight;
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
    explicit Builder(const TrainParam& param) : param(param) {}
    // update one tree, growing
    virtual void Update(const GHistIndexMatrix& gmat,
                        const std::vector<bst_gpair>& gpair,
                        DMatrix* p_fmat,
                        RegTree* p_tree) {
      double gstart = dmlc::GetTime();

      std::vector<int> feat_set(p_fmat->info().num_col);
      std::iota(feat_set.begin(), feat_set.end(), 0);
      int num_leaves = 0;

      double tstart;
      double init_data = 0;
      double init_new_node = 0;
      double make_hist = 0;
      double evaluate_split = 0;
      double apply_split = 0;

      tstart = dmlc::GetTime();
      this->InitData(gmat, gpair, *p_fmat, *p_tree);
      init_data = dmlc::GetTime() - tstart;
      for (int nid = 0; nid < p_tree->param.num_roots; ++nid) {
        tstart = dmlc::GetTime();
        hist_.AddHistRow(nid);
        maker_.MakeHist(gpair, row_set_collection_[nid], gmat, hist_[nid]);
        make_hist += dmlc::GetTime() - tstart;

        tstart = dmlc::GetTime();
        this->InitNewNode(nid, gmat, gpair, *p_fmat, *p_tree);
        init_new_node += dmlc::GetTime() - tstart;

        tstart = dmlc::GetTime();
        this->EvaluateSplit(nid, gmat, hist_, *p_fmat, *p_tree, feat_set);
        evaluate_split += dmlc::GetTime() - tstart;
        qexpand_->push(ExpandEntry(nid, p_tree->GetDepth(nid), snode[nid].best.loss_chg));
        ++num_leaves;
      }

      while (!qexpand_->empty()) {
        const ExpandEntry candidate = qexpand_->top();
        const int nid = candidate.nid;
        qexpand_->pop();
        if (candidate.loss_chg <= rt_eps
            || (param.max_depth > 0 && candidate.depth == param.max_depth)
            || (param.max_leaves > 0 && num_leaves == param.max_leaves) ) {
          (*p_tree)[nid].set_leaf(snode[nid].weight * param.learning_rate);
        } else {
          tstart = dmlc::GetTime();
          this->ApplySplit(nid, gmat, hist_, *p_fmat, p_tree);
          apply_split += dmlc::GetTime() - tstart;

          tstart = dmlc::GetTime();
          const int cleft = (*p_tree)[nid].cleft();
          const int cright = (*p_tree)[nid].cright();
          hist_.AddHistRow(cleft);
          hist_.AddHistRow(cright);
          if (row_set_collection_[cleft].size() < row_set_collection_[cright].size()) {
            maker_.MakeHist(gpair, row_set_collection_[cleft], gmat, hist_[cleft]);
            maker_.SubtractionTrick(hist_[cright], hist_[cleft], hist_[nid]);
          } else {
            maker_.MakeHist(gpair, row_set_collection_[cright], gmat, hist_[cright]);
            maker_.SubtractionTrick(hist_[cleft], hist_[cright], hist_[nid]);
          }
          make_hist += dmlc::GetTime() - tstart;

          tstart = dmlc::GetTime();
          this->InitNewNode(cleft, gmat, gpair, *p_fmat, *p_tree);
          this->InitNewNode(cright, gmat, gpair, *p_fmat, *p_tree);
          init_new_node += dmlc::GetTime() - tstart;

          tstart = dmlc::GetTime();
          this->EvaluateSplit(cleft, gmat, hist_, *p_fmat, *p_tree, feat_set);
          this->EvaluateSplit(cright, gmat, hist_, *p_fmat, *p_tree, feat_set);
          evaluate_split += dmlc::GetTime() - tstart;

          qexpand_->push(ExpandEntry(cleft, p_tree->GetDepth(cleft),
                                     snode[cleft].best.loss_chg));
          qexpand_->push(ExpandEntry(cright, p_tree->GetDepth(cright),
                                     snode[cright].best.loss_chg));

          ++num_leaves;  // give two and take one, as parent is no longer a leaf
        }
      }

      // set all the rest expanding nodes to leaf
      while (!qexpand_->empty()) {
        const int nid = qexpand_->top().nid;
        qexpand_->pop();
        (*p_tree)[nid].set_leaf(snode[nid].weight * param.learning_rate);
      }
      // remember auxiliary statistics in the tree node
      for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
        p_tree->stat(nid).loss_chg = snode[nid].best.loss_chg;
        p_tree->stat(nid).base_weight = snode[nid].weight;
        p_tree->stat(nid).sum_hess = static_cast<float>(snode[nid].stats.sum_hess);
        snode[nid].stats.SetLeafVec(param, p_tree->leafvec(nid));
      }

      if (param.verbose > 0) {
        double total_time = dmlc::GetTime() - gstart;
        LOG(INFO) << "\nInitData:          "
                  << std::fixed << std::setw(4) << std::setprecision(2) << init_data
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << init_data/total_time*100 << "%)\n"
                  << "InitNewNode:       "
                  << std::fixed << std::setw(4) << std::setprecision(2) << init_new_node
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << init_new_node/total_time*100 << "%)\n"
                  << "MakeHist:          "
                  << std::fixed << std::setw(4) << std::setprecision(2) << make_hist
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << make_hist/total_time*100 << "%)\n"
                  << "EvaluateSplit:     "
                  << std::fixed << std::setw(4) << std::setprecision(2) << evaluate_split
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << evaluate_split/total_time*100 << "%)\n"
                  << "ApplySplit:        "
                  << std::fixed << std::setw(4) << std::setprecision(2) << apply_split
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << apply_split/total_time*100 << "%)\n"
                  << "========================================\n"
                  << "Total:             "
                  << std::fixed << std::setw(4) << std::setprecision(2) << total_time;
      }
    }

   protected:
    // initialize temp data structure
    inline void InitData(const GHistIndexMatrix& gmat,
                         const std::vector<bst_gpair>& gpair,
                         const DMatrix& fmat,
                         const RegTree& tree) {
      CHECK_EQ(tree.param.num_nodes, tree.param.num_roots)
          << "ColMakerHist: can only grow new tree";
      CHECK((param.max_depth > 0 || param.max_leaves > 0))
          << "max_depth or max_leaves cannot be both 0 (unlimited); "
          << "at least one should be a positive quantity.";
      if (param.grow_policy == TrainParam::kDepthWise) {
        CHECK(param.max_depth > 0) << "max_depth cannot be 0 (unlimited) "
          << "when grow_policy is depthwise.";
      }
      const auto& info = fmat.info();

      {
        // initialize the row set
        row_set_collection_.Clear();
        // initialize histogram collection
        size_t nbins = gmat.cut->row_ptr.back();
        hist_.Init(nbins);

        #pragma omp parallel
        {
          this->nthread = omp_get_num_threads();
        }
        maker_.Init(this->nthread, nbins);

        CHECK_EQ(info.root_index.size(), 0);
        std::vector<bst_uint>& row_indices = row_set_collection_.row_indices_;
        // mark subsample and build list of member rows
        if (param.subsample < 1.0f) {
          std::bernoulli_distribution coin_flip(param.subsample);
          auto& rnd = common::GlobalRandom();
          for (bst_uint i = 0; i < info.num_row; ++i) {
            if (gpair[i].hess >= 0.0f && coin_flip(rnd)) {
              row_indices.push_back(i);
            }
          }
        } else {
          for (bst_uint i = 0; i < info.num_row; ++i) {
            if (gpair[i].hess >= 0.0f) {
              row_indices.push_back(i);
            }
          }
        }
        row_set_collection_.Init();
      }

      {
        // initialize feature index
        unsigned ncol = static_cast<unsigned>(info.num_col);
        feat_index.clear();
        for (unsigned i = 0; i < ncol; ++i) {
          feat_index.push_back(i);
        }
        unsigned n = static_cast<unsigned>(param.colsample_bytree * feat_index.size());
        std::shuffle(feat_index.begin(), feat_index.end(), common::GlobalRandom());
        CHECK_GT(n, 0)
            << "colsample_bytree=" << param.colsample_bytree
            << " is too small that no feature can be included";
        feat_index.resize(n);
      }
      {
        /* determine layout of data */
        const auto nrow = info.num_row;
        const auto ncol = info.num_col;
        const auto nnz = info.num_nonzero;
        // number of discrete bins for feature 0
        const unsigned nbins_f0 = gmat.cut->row_ptr[1] - gmat.cut->row_ptr[0];
        if (nrow*ncol == nnz) {
          // dense data with zero-based indexing
          data_layout_ = kDenseDataZeroBased;
        } else if (nbins_f0 == 0 && nrow*(ncol-1) == nnz) {
          // dense data with one-based indexing
          data_layout_ = kDenseDataOneBased;
        } else {
          // sparse data
          data_layout_ = kSparseData;
        }
      }
      if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
        const std::vector<unsigned>& row_ptr = gmat.cut->row_ptr;
        const size_t nfeature = row_ptr.size()-1;
        size_t min_nbins_per_feature = 0;
        for (size_t i = 0; i < nfeature; ++i) {
          const unsigned nbins = row_ptr[i+1] - row_ptr[i];
          if (nbins > 0) {
            if (min_nbins_per_feature == 0 || min_nbins_per_feature > nbins) {
              min_nbins_per_feature = nbins;
              fid_least_bins_ = i;
            }
          }
        }
        CHECK_GT(min_nbins_per_feature, 0);
      }
      {
        snode.reserve(256);
        snode.clear();
      }
      {
        if (param.grow_policy == TrainParam::kLossGuide) {
          qexpand_.reset(new ExpandQueue(loss_guide));
        } else {
          qexpand_.reset(new ExpandQueue(depth_wise));
        }
      }
    }

    inline void EvaluateSplit(int nid,
                              const GHistIndexMatrix& gmat,
                              const HistCollection& hist,
                              const DMatrix& fmat,
                              const RegTree& tree,
                              const std::vector<int>& feat_set) {
      // start enumeration
      const MetaInfo& info = fmat.info();
      for (int fid : feat_set) {
        this->EnumerateSplit(-1, gmat, hist[nid], snode[nid], constraints_[nid], info,
          &snode[nid].best, fid);
        this->EnumerateSplit(+1, gmat, hist[nid], snode[nid], constraints_[nid], info,
          &snode[nid].best, fid);
      }
    }

    inline void ApplySplit(int nid,
                           const GHistIndexMatrix& gmat,
                           const HistCollection& hist,
                           const DMatrix& fmat,
                           RegTree *p_tree) {
      // TODO(hcho3): support feature sampling by levels

      /* 1. Create child nodes */
      NodeEntry& e = snode[nid];

      p_tree->AddChilds(nid);
      (*p_tree)[nid].set_split(e.best.split_index(), e.best.split_value, e.best.default_left());
      // mark right child as 0, to indicate fresh leaf
      int cleft = (*p_tree)[nid].cleft();
      int cright = (*p_tree)[nid].cright();
      (*p_tree)[cleft].set_leaf(0.0f, 0);
      (*p_tree)[cright].set_leaf(0.0f, 0);

      /* 2. Categorize member rows */
      const bst_omp_uint nthread = static_cast<bst_omp_uint>(this->nthread);
      row_split_tloc_.resize(nthread);
      for (bst_omp_uint i = 0; i < nthread; ++i) {
        row_split_tloc_[i].left.clear();
        row_split_tloc_[i].right.clear();
      }
      const bool default_left = (*p_tree)[nid].default_left();
      const bst_uint fid = (*p_tree)[nid].split_index();
      const bst_float split_pt = (*p_tree)[nid].split_cond();
      const bst_uint lower_bound = gmat.cut->row_ptr[fid];
      const bst_uint upper_bound = gmat.cut->row_ptr[fid + 1];
      // set the split condition correctly
      bst_uint split_cond = 0;
      // set the condition
      for (unsigned i = gmat.cut->row_ptr[fid]; i < gmat.cut->row_ptr[fid + 1]; ++i) {
        if (split_pt == gmat.cut->cut[i]) split_cond = i;
      }

      const auto& rowset = row_set_collection_[nid];
      if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
        /* specialized code for dense data */
        const size_t column_offset = (data_layout_ == kDenseDataOneBased) ? (fid-1): fid;
        ApplySplitDenseData(rowset, gmat, &row_split_tloc_, column_offset, split_cond);
      } else {
        ApplySplitSparseData(rowset, gmat, &row_split_tloc_, lower_bound, upper_bound,
          split_cond, default_left);
      }
      row_set_collection_.AddSplit(
          nid, row_split_tloc_, (*p_tree)[nid].cleft(), (*p_tree)[nid].cright());
    }

    inline void ApplySplitDenseData(const RowSetCollection::Elem& rowset,
                                    const GHistIndexMatrix& gmat,
                                    std::vector<RowSetCollection::Split> *p_row_split_tloc,
                                    size_t column_offset,
                                    bst_uint split_cond) {
      std::vector<RowSetCollection::Split>& row_split_tloc = *p_row_split_tloc;
      const int K = 8;  // loop unrolling factor
      const bst_omp_uint nrows = rowset.end - rowset.begin;
      const bst_omp_uint rest = nrows % K;
      #pragma omp parallel for num_threads(nthread) schedule(static)
      for (bst_omp_uint i = 0; i < nrows - rest; i += K) {
        bst_uint rid[K];
        unsigned rbin[K];
        bst_uint tid = omp_get_thread_num();
        auto& left = row_split_tloc[tid].left;
        auto& right = row_split_tloc[tid].right;
        for (int k = 0; k < K; ++k) {
          rid[k] = rowset.begin[i+k];
        }
        for (int k = 0; k < K; ++k) {
          rbin[k] = gmat[rid[k]].index[column_offset];
        }
        for (int k = 0; k < K; ++k) {
          if (rbin[k] <= split_cond) {
            left.push_back(rid[k]);
          } else {
            right.push_back(rid[k]);
          }
        }
      }
      for (bst_omp_uint i = nrows - rest; i < nrows; ++i) {
        const bst_uint rid = rowset.begin[i];
        const unsigned rbin = gmat[rid].index[column_offset];
        if (rbin <= split_cond) {
          row_split_tloc[0].left.push_back(rid);
        } else {
          row_split_tloc[0].right.push_back(rid);
        }
      }
    }

    inline void ApplySplitSparseData(const RowSetCollection::Elem& rowset,
                                     const GHistIndexMatrix& gmat,
                                     std::vector<RowSetCollection::Split> *p_row_split_tloc,
                                     bst_uint lower_bound,
                                     bst_uint upper_bound,
                                     bst_uint split_cond,
                                     bool default_left) {
      std::vector<RowSetCollection::Split>& row_split_tloc = *p_row_split_tloc;
      const int K = 8;  // loop unrolling factor
      const bst_omp_uint nrows = rowset.end - rowset.begin;
      const bst_omp_uint rest = nrows % K;
      #pragma omp parallel for num_threads(nthread) schedule(static)
      for (bst_omp_uint i = 0; i < nrows - rest; i += K) {
        bst_uint rid[K];
        GHistIndexRow row[K];
        const unsigned *p[K];
        bst_uint tid = omp_get_thread_num();
        auto& left = row_split_tloc[tid].left;
        auto& right = row_split_tloc[tid].right;
        for (int k = 0; k < K; ++k) {
          rid[k] = rowset.begin[i+k];
        }
        for (int k = 0; k < K; ++k) {
          row[k] = gmat[rid[k]];
        }
        for (int k = 0; k < K; ++k) {
          p[k] = std::lower_bound(row[k].index, row[k].index + row[k].size, lower_bound);
        }
        for (int k = 0; k < K; ++k) {
          if (p[k] != row[k].index + row[k].size && *p[k] < upper_bound) {
            if (*p[k] <= split_cond) {
              left.push_back(rid[k]);
            } else {
              right.push_back(rid[k]);
            }
          } else {
            if (default_left) {
              left.push_back(rid[k]);
            } else {
              right.push_back(rid[k]);
            }
          }
        }
      }
      for (bst_omp_uint i = nrows - rest; i < nrows; ++i) {
        const bst_uint rid = rowset.begin[i];
        const auto row = gmat[rid];
        const auto p = std::lower_bound(row.index, row.index + row.size, lower_bound);
        auto& left = row_split_tloc[0].left;
        auto& right = row_split_tloc[0].right;
        if (p != row.index + row.size && *p < upper_bound) {
          if (*p <= split_cond) {
            left.push_back(rid);
          } else {
            right.push_back(rid);
          }
        } else {
          if (default_left) {
            left.push_back(rid);
          } else {
            right.push_back(rid);
          }
        }
      }
    }

    inline void InitNewNode(int nid,
                            const GHistIndexMatrix& gmat,
                            const std::vector<bst_gpair>& gpair,
                            const DMatrix& fmat,
                            const RegTree& tree) {
      {
        snode.resize(tree.param.num_nodes, NodeEntry(param));
        constraints_.resize(tree.param.num_nodes);
      }

      // setup constraints before calculating the weight
      {
        auto& stats = snode[nid].stats;
        if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
          // specialized code for dense data
          HistRow hist = hist_[nid];
          const std::vector<unsigned>& row_ptr = gmat.cut->row_ptr;

          const size_t ibegin = row_ptr[fid_least_bins_];
          const size_t iend = row_ptr[fid_least_bins_+1];
          for (size_t i = ibegin; i < iend; ++i) {
            const HistEntry et = hist.begin[i];
            stats.Add(et.sum_grad, et.sum_hess);
          }
        } else {
          const auto& e = row_set_collection_[nid];
          for (const bst_uint *it = e.begin; it < e.end; ++it) {
            stats.Add(gpair[*it]);
          }
        }
        if (!tree[nid].is_root()) {
          const int pid = tree[nid].parent();
          constraints_[pid].SetChild(param, tree[pid].split_index(),
                                     snode[tree[pid].cleft()].stats,
                                     snode[tree[pid].cright()].stats,
                                     &constraints_[tree[pid].cleft()],
                                     &constraints_[tree[pid].cright()]);
        }
      }

      // calculating the weights
      {
        snode[nid].root_gain = static_cast<float>(
            constraints_[nid].CalcGain(param, snode[nid].stats));
        snode[nid].weight = static_cast<float>(
            constraints_[nid].CalcWeight(param, snode[nid].stats));
      }
    }

    // enumerate the split values of specific feature
    inline void EnumerateSplit(int d_step,
                               const GHistIndexMatrix& gmat,
                               const HistRow& hist,
                               const NodeEntry& snode,
                               const TConstraint& constraint,
                               const MetaInfo& info,
                               SplitEntry *p_best,
                               int fid) {
      CHECK(d_step == +1 || d_step == -1);

      // aliases
      const std::vector<unsigned>& cut_ptr = gmat.cut->row_ptr;
      const std::vector<bst_float>& cut_val = gmat.cut->cut;

      // statistics on both sides of split
      TStats c(param);
      TStats e(param);
      // best split so far
      SplitEntry best;

      // bin boundaries
      // imin: index (offset) of the minimum value for feature fid
      //       need this for backward enumeration
      const int imin = cut_ptr[fid];
      // ibegin, iend: smallest/largest cut points for feature fid
      int ibegin, iend;
      if (d_step > 0) {
        ibegin = cut_ptr[fid];
        iend = cut_ptr[fid+1];
      } else {
        ibegin = cut_ptr[fid+1]-1;
        iend = cut_ptr[fid]-1;
      }

      for (int i = ibegin; i != iend; i += d_step) {
        // start working
        // try to find a split
        e.Add(hist.begin[i].sum_grad, hist.begin[i].sum_hess);
        if (e.sum_hess >= param.min_child_weight) {
          c.SetSubstract(snode.stats, e);
          if (c.sum_hess >= param.min_child_weight) {
            bst_float loss_chg;
            bst_float split_pt;
            if (d_step > 0) {
              // forward enumeration: split at right bound of each bin
              loss_chg = static_cast<bst_float>(
                  constraint.CalcSplitGain(param, fid, e, c) -
                  snode.root_gain);
              split_pt = cut_val[i];
            } else {
              // backward enumeration: split at left bound of each bin
              loss_chg = static_cast<bst_float>(
                  constraint.CalcSplitGain(param, fid, c, e) -
                  snode.root_gain);
              if (i == imin) {
                // for leftmost bin, left bound is the smallest feature value
                split_pt = gmat.cut->min_val[fid];
              } else {
                split_pt = cut_val[i-1];
              }
            }
            best.Update(loss_chg, fid, split_pt, d_step == -1);
          }
        }
      }
      p_best->Update(best);
    }

    /* tree growing policies */
    struct ExpandEntry {
      int nid;
      int depth;
      bst_float loss_chg;
      ExpandEntry(int nid, int depth, bst_float loss_chg)
        : nid(nid), depth(depth), loss_chg(loss_chg) {}
    };
    inline static bool depth_wise(ExpandEntry lhs, ExpandEntry rhs) {
      return lhs.depth > rhs.depth;
    }
    inline static bool loss_guide(ExpandEntry lhs, ExpandEntry rhs) {
      return lhs.loss_chg < rhs.loss_chg;
    }

    //  --data fields--
    const TrainParam& param;
    // number of omp thread used during training
    int nthread;
    // Per feature: shuffle index of each feature index
    std::vector<bst_uint> feat_index;
    // the internal row sets
    RowSetCollection row_set_collection_;
    // the temp space for split
    std::vector<RowSetCollection::Split> row_split_tloc_;
    /*! \brief TreeNode Data: statistics for each constructed node */
    std::vector<NodeEntry> snode;
    /*! \brief culmulative histogram of gradients. */
    HistCollection hist_;
    size_t fid_least_bins_;

    HistMaker maker_;

    // constraint value
    std::vector<TConstraint> constraints_;

    using ExpandQueue
      = std::priority_queue<ExpandEntry,
                            std::vector<ExpandEntry>,
                            std::function<bool(ExpandEntry, ExpandEntry)>>;
    std::unique_ptr<ExpandQueue> qexpand_;

    enum DataLayout { kDenseDataZeroBased, kDenseDataOneBased, kSparseData };
    DataLayout data_layout_;
  };

  std::unique_ptr<Builder> builder_;
};

XGBOOST_REGISTER_TREE_UPDATER(FastHistMaker, "grow_fast_histmaker")
.describe("Grow tree using quantized histogram.")
.set_body([]() {
    return new FastHistMaker<GradStats, NoConstraint>();
  });

}  // namespace tree
}  // namespace xgboost
