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
#include <numeric>
#include "./param.h"
#include "./fast_hist_param.h"
#include "../common/random.h"
#include "../common/bitmap.h"
#include "../common/sync.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/column_matrix.h"

namespace xgboost {
namespace tree {

using xgboost::common::HistCutMatrix;
using xgboost::common::GHistIndexMatrix;
using xgboost::common::GHistIndexBlockMatrix;
using xgboost::common::GHistIndexRow;
using xgboost::common::GHistEntry;
using xgboost::common::HistCollection;
using xgboost::common::RowSetCollection;
using xgboost::common::GHistRow;
using xgboost::common::GHistBuilder;
using xgboost::common::ColumnMatrix;
using xgboost::common::Column;

DMLC_REGISTRY_FILE_TAG(updater_fast_hist);

DMLC_REGISTER_PARAMETER(FastHistParam);

/*! \brief construct a tree using quantized feature values */
template<typename TStats, typename TConstraint>
class FastHistMaker: public TreeUpdater {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    // initialize pruner
    if (!pruner_) {
      pruner_.reset(TreeUpdater::Create("prune"));
    }
    pruner_->Init(args);
    param.InitAllowUnknown(args);
    fhparam.InitAllowUnknown(args);
    is_gmat_initialized_ = false;
  }

  void Update(const std::vector<bst_gpair>& gpair,
              DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    TStats::CheckInfo(dmat->info());
    if (is_gmat_initialized_ == false) {
      double tstart = dmlc::GetTime();
      hmat_.Init(dmat, static_cast<uint32_t>(param.max_bin));
      gmat_.cut = &hmat_;
      gmat_.Init(dmat);
      column_matrix_.Init(gmat_, fhparam);
      if (fhparam.enable_feature_grouping > 0) {
        gmatb_.Init(gmat_, column_matrix_, fhparam);
      }
      is_gmat_initialized_ = true;
      if (param.debug_verbose > 0) {
        LOG(INFO) << "Generating gmat: " << dmlc::GetTime() - tstart << " sec";
      }
    }
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    TConstraint::Init(&param, dmat->info().num_col);
    // build tree
    if (!builder_) {
      builder_.reset(new Builder(param, fhparam, std::move(pruner_)));
    }
    for (size_t i = 0; i < trees.size(); ++i) {
      builder_->Update(gmat_, gmatb_, column_matrix_, gpair, dmat, trees[i]);
    }
    param.learning_rate = lr;
  }

  bool UpdatePredictionCache(const DMatrix* data,
                             std::vector<bst_float>* out_preds) override {
    if (!builder_ || param.subsample < 1.0f) {
      return false;
    } else {
      return builder_->UpdatePredictionCache(data, out_preds);
    }
  }

 protected:
  // training parameter
  TrainParam param;
  FastHistParam fhparam;
  // data sketch
  HistCutMatrix hmat_;
  // quantized data matrix
  GHistIndexMatrix gmat_;
  // (optional) data matrix with feature grouping
  GHistIndexBlockMatrix gmatb_;
  // column accessor
  ColumnMatrix column_matrix_;
  bool is_gmat_initialized_;

  // data structure
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
        : stats(param), root_gain(0.0f), weight(0.0f) {
    }
  };
  // actual builder that runs the algorithm

  struct Builder {
   public:
    // constructor
    explicit Builder(const TrainParam& param,
                     const FastHistParam& fhparam,
                     std::unique_ptr<TreeUpdater> pruner)
      : param(param), fhparam(fhparam), pruner_(std::move(pruner)),
        p_last_tree_(nullptr), p_last_fmat_(nullptr) {}
    // update one tree, growing
    virtual void Update(const GHistIndexMatrix& gmat,
                        const GHistIndexBlockMatrix& gmatb,
                        const ColumnMatrix& column_matrix,
                        const std::vector<bst_gpair>& gpair,
                        DMatrix* p_fmat,
                        RegTree* p_tree) {
      double gstart = dmlc::GetTime();

      int num_leaves = 0;
      unsigned timestamp = 0;

      double tstart;
      double time_init_data = 0;
      double time_init_new_node = 0;
      double time_build_hist = 0;
      double time_evaluate_split = 0;
      double time_apply_split = 0;

      tstart = dmlc::GetTime();
      this->InitData(gmat, gpair, *p_fmat, *p_tree);
      std::vector<bst_uint> feat_set = feat_index;
      time_init_data = dmlc::GetTime() - tstart;

      // FIXME(hcho3): this code is broken when param.num_roots > 1. Please fix it
      CHECK_EQ(p_tree->param.num_roots, 1)
        << "tree_method=hist does not support multiple roots at this moment";
      for (int nid = 0; nid < p_tree->param.num_roots; ++nid) {
        tstart = dmlc::GetTime();
        hist_.AddHistRow(nid);
        BuildHist(gpair, row_set_collection_[nid], gmat, gmatb, feat_set, hist_[nid]);
        time_build_hist += dmlc::GetTime() - tstart;

        tstart = dmlc::GetTime();
        this->InitNewNode(nid, gmat, gpair, *p_fmat, *p_tree);
        time_init_new_node += dmlc::GetTime() - tstart;

        tstart = dmlc::GetTime();
        this->EvaluateSplit(nid, gmat, hist_, *p_fmat, *p_tree, feat_set);
        time_evaluate_split += dmlc::GetTime() - tstart;
        qexpand_->push(ExpandEntry(nid, p_tree->GetDepth(nid),
                                   snode[nid].best.loss_chg,
                                   timestamp++));
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
          this->ApplySplit(nid, gmat, column_matrix, hist_, *p_fmat, p_tree);
          time_apply_split += dmlc::GetTime() - tstart;

          tstart = dmlc::GetTime();
          const int cleft = (*p_tree)[nid].cleft();
          const int cright = (*p_tree)[nid].cright();
          hist_.AddHistRow(cleft);
          hist_.AddHistRow(cright);
          if (row_set_collection_[cleft].size() < row_set_collection_[cright].size()) {
            BuildHist(gpair, row_set_collection_[cleft], gmat, gmatb, feat_set, hist_[cleft]);
            SubtractionTrick(hist_[cright], hist_[cleft], hist_[nid]);
          } else {
            BuildHist(gpair, row_set_collection_[cright], gmat, gmatb, feat_set, hist_[cright]);
            SubtractionTrick(hist_[cleft], hist_[cright], hist_[nid]);
          }
          time_build_hist += dmlc::GetTime() - tstart;

          tstart = dmlc::GetTime();
          this->InitNewNode(cleft, gmat, gpair, *p_fmat, *p_tree);
          this->InitNewNode(cright, gmat, gpair, *p_fmat, *p_tree);
          time_init_new_node += dmlc::GetTime() - tstart;

          tstart = dmlc::GetTime();
          this->EvaluateSplit(cleft, gmat, hist_, *p_fmat, *p_tree, feat_set);
          this->EvaluateSplit(cright, gmat, hist_, *p_fmat, *p_tree, feat_set);
          time_evaluate_split += dmlc::GetTime() - tstart;

          qexpand_->push(ExpandEntry(cleft, p_tree->GetDepth(cleft),
                                     snode[cleft].best.loss_chg,
                                     timestamp++));
          qexpand_->push(ExpandEntry(cright, p_tree->GetDepth(cright),
                                     snode[cright].best.loss_chg,
                                     timestamp++));

          ++num_leaves;  // give two and take one, as parent is no longer a leaf
        }
      }

      // set all the rest expanding nodes to leaf
      // This post condition is not needed in current code, but may be necessary
      // when there are stopping rule that leaves qexpand non-empty
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

      pruner_->Update(gpair, p_fmat, std::vector<RegTree*>{p_tree});

      if (param.debug_verbose > 0) {
        double total_time = dmlc::GetTime() - gstart;
        LOG(INFO) << "\nInitData:          "
                  << std::fixed << std::setw(6) << std::setprecision(4) << time_init_data
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << time_init_data / total_time * 100 << "%)\n"
                  << "InitNewNode:       "
                  << std::fixed << std::setw(6) << std::setprecision(4) << time_init_new_node
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << time_init_new_node / total_time * 100 << "%)\n"
                  << "BuildHist:         "
                  << std::fixed << std::setw(6) << std::setprecision(4) << time_build_hist
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << time_build_hist / total_time * 100 << "%)\n"
                  << "EvaluateSplit:     "
                  << std::fixed << std::setw(6) << std::setprecision(4) << time_evaluate_split
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << time_evaluate_split / total_time * 100 << "%)\n"
                  << "ApplySplit:        "
                  << std::fixed << std::setw(6) << std::setprecision(4) << time_apply_split
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << time_apply_split / total_time * 100 << "%)\n"
                  << "========================================\n"
                  << "Total:             "
                  << std::fixed << std::setw(6) << std::setprecision(4) << total_time;
      }
    }

    inline void BuildHist(const std::vector<bst_gpair>& gpair,
                          const RowSetCollection::Elem row_indices,
                          const GHistIndexMatrix& gmat,
                          const GHistIndexBlockMatrix& gmatb,
                          const std::vector<bst_uint>& feat_set,
                          GHistRow hist) {
      if (fhparam.enable_feature_grouping > 0) {
        hist_builder_.BuildBlockHist(gpair, row_indices, gmatb, feat_set, hist);
      } else {
        hist_builder_.BuildHist(gpair, row_indices, gmat, feat_set, hist);
      }
    }

    inline void SubtractionTrick(GHistRow self, GHistRow sibling, GHistRow parent) {
      hist_builder_.SubtractionTrick(self, sibling, parent);
    }

    inline bool UpdatePredictionCache(const DMatrix* data,
                                      std::vector<bst_float>* p_out_preds) {
      std::vector<bst_float>& out_preds = *p_out_preds;

      // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
      // conjunction with Update().
      if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_) {
        return false;
      }

      if (leaf_value_cache_.empty()) {
        leaf_value_cache_.resize(p_last_tree_->param.num_nodes,
          std::numeric_limits<float>::infinity());
      }

      CHECK_GT(out_preds.size(), 0U);

      for (const RowSetCollection::Elem rowset : row_set_collection_) {
        if (rowset.begin != nullptr && rowset.end != nullptr) {
          int nid = rowset.node_id;
          bst_float leaf_value;
          // if a node is marked as deleted by the pruner, traverse upward to locate
          // a non-deleted leaf.
          if ((*p_last_tree_)[nid].is_deleted()) {
            while ((*p_last_tree_)[nid].is_deleted()) {
              nid = (*p_last_tree_)[nid].parent();
            }
            CHECK((*p_last_tree_)[nid].is_leaf());
          }
          leaf_value = (*p_last_tree_)[nid].leaf_value();

          for (const size_t* it = rowset.begin; it < rowset.end; ++it) {
            out_preds[*it] += leaf_value;
          }
        }
      }

      return true;
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
        // clear local prediction cache
        leaf_value_cache_.clear();
        // initialize histogram collection
        uint32_t nbins = gmat.cut->row_ptr.back();
        hist_.Init(nbins);

        // initialize histogram builder
        #pragma omp parallel
        {
          this->nthread = omp_get_num_threads();
        }
        hist_builder_.Init(this->nthread, nbins);

        CHECK_EQ(info.root_index.size(), 0U);
        std::vector<size_t>& row_indices = row_set_collection_.row_indices_;
        // mark subsample and build list of member rows
        if (param.subsample < 1.0f) {
          std::bernoulli_distribution coin_flip(param.subsample);
          auto& rnd = common::GlobalRandom();
          for (size_t i = 0; i < info.num_row; ++i) {
            if (gpair[i].GetHess() >= 0.0f && coin_flip(rnd)) {
              row_indices.push_back(i);
            }
          }
        } else {
          for (size_t i = 0; i < info.num_row; ++i) {
            if (gpair[i].GetHess() >= 0.0f) {
              row_indices.push_back(i);
            }
          }
        }
        row_set_collection_.Init();
      }

      {
        /* determine layout of data */
        const size_t nrow = info.num_row;
        const size_t ncol = info.num_col;
        const size_t nnz = info.num_nonzero;
        // number of discrete bins for feature 0
        const uint32_t nbins_f0 = gmat.cut->row_ptr[1] - gmat.cut->row_ptr[0];
        if (nrow * ncol == nnz) {
          // dense data with zero-based indexing
          data_layout_ = kDenseDataZeroBased;
        } else if (nbins_f0 == 0 && nrow * (ncol - 1) == nnz) {
          // dense data with one-based indexing
          data_layout_ = kDenseDataOneBased;
        } else {
          // sparse data
          data_layout_ = kSparseData;
        }
      }
      {
        // store a pointer to the tree
        p_last_tree_ = &tree;
        // store a pointer to training data
        p_last_fmat_ = &fmat;
        // initialize feature index
        bst_uint ncol = static_cast<bst_uint>(info.num_col);
        feat_index.clear();
        if (data_layout_ == kDenseDataOneBased) {
          for (bst_uint i = 1; i < ncol; ++i) {
            feat_index.push_back(i);
          }
        } else {
          for (bst_uint i = 0; i < ncol; ++i) {
            feat_index.push_back(i);
          }
        }
        bst_uint n = std::max(static_cast<bst_uint>(1),
                              static_cast<bst_uint>(param.colsample_bytree * feat_index.size()));
        std::shuffle(feat_index.begin(), feat_index.end(), common::GlobalRandom());
        CHECK_GT(param.colsample_bytree, 0U)
            << "colsample_bytree cannot be zero.";
        feat_index.resize(n);
      }
      if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
        /* specialized code for dense data:
           choose the column that has a least positive number of discrete bins.
           For dense data (with no missing value),
              the sum of gradient histogram is equal to snode[nid] */
        const std::vector<uint32_t>& row_ptr = gmat.cut->row_ptr;
        const bst_uint nfeature = static_cast<bst_uint>(row_ptr.size() - 1);
        uint32_t min_nbins_per_feature = 0;
        for (bst_uint i = 0; i < nfeature; ++i) {
          const uint32_t nbins = row_ptr[i + 1] - row_ptr[i];
          if (nbins > 0) {
            if (min_nbins_per_feature == 0 || min_nbins_per_feature > nbins) {
              min_nbins_per_feature = nbins;
              fid_least_bins_ = i;
            }
          }
        }
        CHECK_GT(min_nbins_per_feature, 0U);
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
                              const std::vector<bst_uint>& feat_set) {
      // start enumeration
      const MetaInfo& info = fmat.info();
      const bst_uint nfeature = static_cast<bst_uint>(feat_set.size());
      const bst_omp_uint nthread = static_cast<bst_omp_uint>(this->nthread);
      best_split_tloc_.resize(nthread);
      #pragma omp parallel for schedule(static) num_threads(nthread)
      for (bst_omp_uint tid = 0; tid < nthread; ++tid) {
        best_split_tloc_[tid] = snode[nid].best;
      }
      #pragma omp parallel for schedule(dynamic) num_threads(nthread)
      for (bst_omp_uint i = 0; i < nfeature; ++i) {
        const bst_uint fid = feat_set[i];
        const unsigned tid = omp_get_thread_num();
        this->EnumerateSplit(-1, gmat, hist[nid], snode[nid], constraints_[nid], info,
          &best_split_tloc_[tid], fid);
        this->EnumerateSplit(+1, gmat, hist[nid], snode[nid], constraints_[nid], info,
          &best_split_tloc_[tid], fid);
      }
      for (unsigned tid = 0; tid < nthread; ++tid) {
        snode[nid].best.Update(best_split_tloc_[tid]);
      }
    }

    inline void ApplySplit(int nid,
                           const GHistIndexMatrix& gmat,
                           const ColumnMatrix& column_matrix,
                           const HistCollection& hist,
                           const DMatrix& fmat,
                           RegTree* p_tree) {
      XGBOOST_TYPE_SWITCH(column_matrix.dtype, {
        ApplySplit_<DType>(nid, gmat, column_matrix, hist, fmat, p_tree);
      });
    }

    template <typename T>
    inline void ApplySplit_(int nid,
                            const GHistIndexMatrix& gmat,
                            const ColumnMatrix& column_matrix,
                            const HistCollection& hist,
                            const DMatrix& fmat,
                            RegTree* p_tree) {
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
      const uint32_t lower_bound = gmat.cut->row_ptr[fid];
      const uint32_t upper_bound = gmat.cut->row_ptr[fid + 1];
      int32_t split_cond = -1;
      // convert floating-point split_pt into corresponding bin_id
      // split_cond = -1 indicates that split_pt is less than all known cut points
      CHECK_LT(upper_bound,
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
      for (uint32_t i = lower_bound; i < upper_bound; ++i) {
        if (split_pt == gmat.cut->cut[i]) {
          split_cond = static_cast<int32_t>(i);
        }
      }

      const auto& rowset = row_set_collection_[nid];

      Column<T> column = column_matrix.GetColumn<T>(fid);
      if (column.type == xgboost::common::kDenseColumn) {
        ApplySplitDenseData(rowset, gmat, &row_split_tloc_, column, split_cond,
          default_left);
      } else {
        ApplySplitSparseData(rowset, gmat, &row_split_tloc_, column, lower_bound,
          upper_bound, split_cond, default_left);
      }

      row_set_collection_.AddSplit(
        nid, row_split_tloc_, (*p_tree)[nid].cleft(), (*p_tree)[nid].cright());
    }

    template<typename T>
    inline void ApplySplitDenseData(const RowSetCollection::Elem rowset,
                                    const GHistIndexMatrix& gmat,
                                    std::vector<RowSetCollection::Split>* p_row_split_tloc,
                                    const Column<T>& column,
                                    bst_int split_cond,
                                    bool default_left) {
      std::vector<RowSetCollection::Split>& row_split_tloc = *p_row_split_tloc;
      const int K = 8;  // loop unrolling factor
      const size_t nrows = rowset.end - rowset.begin;
      const size_t rest = nrows % K;

      #pragma omp parallel for num_threads(nthread) schedule(static)
      for (bst_omp_uint i = 0; i < nrows - rest; i += K) {
        const bst_uint tid = omp_get_thread_num();
        auto& left = row_split_tloc[tid].left;
        auto& right = row_split_tloc[tid].right;
        size_t rid[K];
        T rbin[K];
        for (int k = 0; k < K; ++k) {
          rid[k] = rowset.begin[i + k];
        }
        for (int k = 0; k < K; ++k) {
          rbin[k] = column.index[rid[k]];
        }
        for (int k = 0; k < K; ++k) {
          if (rbin[k] == std::numeric_limits<T>::max()) {  // missing value
            if (default_left) {
              left.push_back(rid[k]);
            } else {
              right.push_back(rid[k]);
            }
          } else {
            CHECK_LT(rbin[k] + column.index_base,
              static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
            if (static_cast<int32_t>(rbin[k] + column.index_base) <= split_cond) {
              left.push_back(rid[k]);
            } else {
              right.push_back(rid[k]);
            }
          }
        }
      }
      for (size_t i = nrows - rest; i < nrows; ++i) {
        auto& left = row_split_tloc[nthread-1].left;
        auto& right = row_split_tloc[nthread-1].right;
        const size_t rid = rowset.begin[i];
        const T rbin = column.index[rid];
        if (rbin == std::numeric_limits<T>::max()) {  // missing value
          if (default_left) {
            left.push_back(rid);
          } else {
            right.push_back(rid);
          }
        } else {
          CHECK_LT(rbin + column.index_base,
            static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
          if (static_cast<int32_t>(rbin + column.index_base) <= split_cond) {
            left.push_back(rid);
          } else {
            right.push_back(rid);
          }
        }
      }
    }

    inline void ApplySplitSparseDataOld(const RowSetCollection::Elem rowset,
                                        const GHistIndexMatrix& gmat,
                                        std::vector<RowSetCollection::Split>* p_row_split_tloc,
                                        bst_uint lower_bound,
                                        bst_uint upper_bound,
                                        bst_int split_cond,
                                        bool default_left) {
      std::vector<RowSetCollection::Split>& row_split_tloc = *p_row_split_tloc;
      const int K = 8;  // loop unrolling factor
      const size_t nrows = rowset.end - rowset.begin;
      const size_t rest = nrows % K;
      #pragma omp parallel for num_threads(nthread) schedule(static)
      for (bst_omp_uint i = 0; i < nrows - rest; i += K) {
        size_t rid[K];
        GHistIndexRow row[K];
        const uint32_t* p[K];
        bst_uint tid = omp_get_thread_num();
        auto& left = row_split_tloc[tid].left;
        auto& right = row_split_tloc[tid].right;
        for (int k = 0; k < K; ++k) {
          rid[k] = rowset.begin[i + k];
        }
        for (int k = 0; k < K; ++k) {
          row[k] = gmat[rid[k]];
        }
        for (int k = 0; k < K; ++k) {
          p[k] = std::lower_bound(row[k].index, row[k].index + row[k].size, lower_bound);
        }
        for (int k = 0; k < K; ++k) {
          if (p[k] != row[k].index + row[k].size && *p[k] < upper_bound) {
            CHECK_LT(*p[k],
              static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
            if (static_cast<int32_t>(*p[k]) <= split_cond) {
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
      for (size_t i = nrows - rest; i < nrows; ++i) {
        const size_t rid = rowset.begin[i];
        const auto row = gmat[rid];
        const auto p = std::lower_bound(row.index, row.index + row.size, lower_bound);
        auto& left = row_split_tloc[0].left;
        auto& right = row_split_tloc[0].right;
        if (p != row.index + row.size && *p < upper_bound) {
          CHECK_LT(*p, static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
          if (static_cast<int32_t>(*p) <= split_cond) {
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

    template<typename T>
    inline void ApplySplitSparseData(const RowSetCollection::Elem rowset,
                                    const GHistIndexMatrix& gmat,
                                    std::vector<RowSetCollection::Split>* p_row_split_tloc,
                                    const Column<T>& column,
                                    bst_uint lower_bound,
                                    bst_uint upper_bound,
                                    bst_int split_cond,
                                    bool default_left) {
      std::vector<RowSetCollection::Split>& row_split_tloc = *p_row_split_tloc;
      const size_t nrows = rowset.end - rowset.begin;

      #pragma omp parallel num_threads(nthread)
      {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        const size_t ibegin = tid * nrows / nthread;
        const size_t iend = (tid + 1) * nrows / nthread;
        if (ibegin < iend) {  // ensure that [ibegin, iend) is nonempty range
          // search first nonzero row with index >= rowset[ibegin]
          const size_t* p = std::lower_bound(column.row_ind,
                                             column.row_ind + column.len,
                                             rowset.begin[ibegin]);

          auto& left = row_split_tloc[tid].left;
          auto& right = row_split_tloc[tid].right;
          if (p != column.row_ind + column.len && *p <= rowset.begin[iend - 1]) {
            size_t cursor = p - column.row_ind;

            for (size_t i = ibegin; i < iend; ++i) {
              const size_t rid = rowset.begin[i];
              while (cursor < column.len
                     && column.row_ind[cursor] < rid
                     && column.row_ind[cursor] <= rowset.begin[iend - 1]) {
                ++cursor;
              }
              if (cursor < column.len && column.row_ind[cursor] == rid) {
                const T rbin = column.index[cursor];
                CHECK_LT(rbin + column.index_base,
                  static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
                if (static_cast<int32_t>(rbin + column.index_base) <= split_cond) {
                  left.push_back(rid);
                } else {
                  right.push_back(rid);
                }
                ++cursor;
              } else {
                // missing value
                if (default_left) {
                  left.push_back(rid);
                } else {
                  right.push_back(rid);
                }
              }
            }
          } else {  // all rows in [ibegin, iend) have missing values
            if (default_left) {
              for (size_t i = ibegin; i < iend; ++i) {
                const size_t rid = rowset.begin[i];
                left.push_back(rid);
              }
            } else {
              for (size_t i = ibegin; i < iend; ++i) {
                const size_t rid = rowset.begin[i];
                right.push_back(rid);
              }
            }
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
          /* specialized code for dense data
             For dense data (with no missing value),
                the sum of gradient histogram is equal to snode[nid] */
          GHistRow hist = hist_[nid];
          const std::vector<uint32_t>& row_ptr = gmat.cut->row_ptr;

          const uint32_t ibegin = row_ptr[fid_least_bins_];
          const uint32_t iend = row_ptr[fid_least_bins_ + 1];
          for (uint32_t i = ibegin; i < iend; ++i) {
            const GHistEntry et = hist.begin[i];
            stats.Add(et.sum_grad, et.sum_hess);
          }
        } else {
          const RowSetCollection::Elem e = row_set_collection_[nid];
          for (const size_t* it = e.begin; it < e.end; ++it) {
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
                               const GHistRow& hist,
                               const NodeEntry& snode,
                               const TConstraint& constraint,
                               const MetaInfo& info,
                               SplitEntry* p_best,
                               bst_uint fid) {
      CHECK(d_step == +1 || d_step == -1);

      // aliases
      const std::vector<uint32_t>& cut_ptr = gmat.cut->row_ptr;
      const std::vector<bst_float>& cut_val = gmat.cut->cut;

      // statistics on both sides of split
      TStats c(param);
      TStats e(param);
      // best split so far
      SplitEntry best;

      // bin boundaries
      CHECK_LE(cut_ptr[fid],
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
      CHECK_LE(cut_ptr[fid + 1],
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
      // imin: index (offset) of the minimum value for feature fid
      //       need this for backward enumeration
      const int32_t imin = static_cast<int32_t>(cut_ptr[fid]);
      // ibegin, iend: smallest/largest cut points for feature fid
      // use int to allow for value -1
      int32_t ibegin, iend;
      if (d_step > 0) {
        ibegin = static_cast<int32_t>(cut_ptr[fid]);
        iend = static_cast<int32_t>(cut_ptr[fid + 1]);
      } else {
        ibegin = static_cast<int32_t>(cut_ptr[fid + 1]) - 1;
        iend = static_cast<int32_t>(cut_ptr[fid]) - 1;
      }

      for (int32_t i = ibegin; i != iend; i += d_step) {
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
                split_pt = cut_val[i - 1];
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
      unsigned timestamp;
      ExpandEntry(int nid, int depth, bst_float loss_chg, unsigned tstmp)
        : nid(nid), depth(depth), loss_chg(loss_chg), timestamp(tstmp) {}
    };
    inline static bool depth_wise(ExpandEntry lhs, ExpandEntry rhs) {
      if (lhs.depth == rhs.depth) {
        return lhs.timestamp > rhs.timestamp;  // favor small timestamp
      } else {
        return lhs.depth > rhs.depth;  // favor small depth
      }
    }
    inline static bool loss_guide(ExpandEntry lhs, ExpandEntry rhs) {
      if (lhs.loss_chg == rhs.loss_chg) {
        return lhs.timestamp > rhs.timestamp;  // favor small timestamp
      } else {
        return lhs.loss_chg < rhs.loss_chg;  // favor large loss_chg
      }
    }

    //  --data fields--
    const TrainParam& param;
    const FastHistParam& fhparam;
    // number of omp thread used during training
    int nthread;
    // Per feature: shuffle index of each feature index
    std::vector<bst_uint> feat_index;
    // the internal row sets
    RowSetCollection row_set_collection_;
    // the temp space for split
    std::vector<RowSetCollection::Split> row_split_tloc_;
    std::vector<SplitEntry> best_split_tloc_;
    /*! \brief TreeNode Data: statistics for each constructed node */
    std::vector<NodeEntry> snode;
    /*! \brief culmulative histogram of gradients. */
    HistCollection hist_;
    /*! \brief feature with least # of bins. to be used for dense specialization
               of InitNewNode() */
    uint32_t fid_least_bins_;
    /*! \brief local prediction cache; maps node id to leaf value */
    std::vector<float> leaf_value_cache_;

    GHistBuilder hist_builder_;
    std::unique_ptr<TreeUpdater> pruner_;

    // back pointers to tree and data matrix
    const RegTree* p_last_tree_;
    const DMatrix* p_last_fmat_;

    // constraint value
    std::vector<TConstraint> constraints_;

    typedef std::priority_queue<ExpandEntry,
                            std::vector<ExpandEntry>,
                            std::function<bool(ExpandEntry, ExpandEntry)>> ExpandQueue;
    std::unique_ptr<ExpandQueue> qexpand_;

    enum DataLayout { kDenseDataZeroBased, kDenseDataOneBased, kSparseData };
    DataLayout data_layout_;
  };

  std::unique_ptr<Builder> builder_;
  std::unique_ptr<TreeUpdater> pruner_;
};

XGBOOST_REGISTER_TREE_UPDATER(FastHistMaker, "grow_fast_histmaker")
.describe("Grow tree using quantized histogram.")
.set_body([]() {
    return new FastHistMaker<GradStats, NoConstraint>();
  });

}  // namespace tree
}  // namespace xgboost
