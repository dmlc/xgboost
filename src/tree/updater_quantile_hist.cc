/*!
 * Copyright 2017-2019 by Contributors
 * \file updater_quantile_hist.cc
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Checn, Egor Smirnov
 */
#include <dmlc/timer.h>
#include <rabit/rabit.h>
#include <xgboost/logging.h>
#include <xgboost/tree_updater.h>

#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>
#include <queue>
#include <iomanip>
#include <numeric>
#include <string>
#include <utility>

#include "./param.h"
#include "./updater_quantile_hist.h"
#include "./split_evaluator.h"
#include "../common/random.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/column_matrix.h"
#include <immintrin.h>
namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_quantile_hist);

void QuantileHistMaker::Init(const std::vector<std::pair<std::string, std::string> >& args) {
  // initialize pruner
  if (!pruner_) {
    pruner_.reset(TreeUpdater::Create("prune"));
  }
  pruner_->Init(args);
  param_.InitAllowUnknown(args);
  is_gmat_initialized_ = false;

  // initialise the split evaluator
  if (!spliteval_) {
    spliteval_.reset(SplitEvaluator::Create(param_.split_evaluator));
  }

  spliteval_->Init(args);
}

void QuantileHistMaker::Update(HostDeviceVector<GradientPair> *gpair,
                               DMatrix *dmat,
                               const std::vector<RegTree *> &trees) {
  if (is_gmat_initialized_ == false) {
    double tstart = dmlc::GetTime();
    gmat_.Init(dmat, static_cast<uint32_t>(param_.max_bin));
    column_matrix_.Init(gmat_, param_.sparse_threshold);
    if (param_.enable_feature_grouping > 0) {
      gmatb_.Init(gmat_, column_matrix_, param_);
    }
    is_gmat_initialized_ = true;
    LOG(INFO) << "Generating gmat: " << dmlc::GetTime() - tstart << " sec";
  }
  // rescale learning rate according to size of trees
  float lr = param_.learning_rate;
  param_.learning_rate = lr / trees.size();
  // build tree
  if (!builder_) {
    builder_.reset(new Builder(
        param_,
        std::move(pruner_),
        std::unique_ptr<SplitEvaluator>(spliteval_->GetHostClone())));
  }
  for (auto tree : trees) {
    builder_->Update(gmat_, gmatb_, column_matrix_, gpair, dmat, tree);
  }
  param_.learning_rate = lr;
}

bool QuantileHistMaker::UpdatePredictionCache(
    const DMatrix* data,
    HostDeviceVector<bst_float>* out_preds) {
  if (!builder_ || param_.subsample < 1.0f) {
    return false;
  } else {
    return builder_->UpdatePredictionCache(data, out_preds);
  }
}

void QuantileHistMaker::Builder::SyncHistograms(
    int starting_index,
    int sync_count,
    RegTreeThreadSafe *p_tree) {
  perf_monitor.TickStart();
  this->histred_.Allreduce(hist_[starting_index].data(), hist_builder_.GetNumBins() * sync_count);
  // use Subtraction Trick
  for (auto local_it = nodes_for_subtraction_trick_.begin();
    local_it != nodes_for_subtraction_trick_.end(); local_it++) {
    hist_.AddHistRow(local_it->first);
    SubtractionTrick(hist_[local_it->first], hist_[local_it->second],
                     hist_[(*p_tree)[local_it->first].Parent()]);
  }
  perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::BUILD_HIST);
}

void QuantileHistMaker::Builder::BuildNodeStat(
    const GHistIndexMatrix &gmat,
    DMatrix *p_fmat,
    RegTreeThreadSafe *p_tree,
    const std::vector<GradientPair> &gpair_h,
    int32_t nid) {
  this->InitNewNode(nid, gmat, gpair_h, *p_fmat, *p_tree);
  // add constraints
  if (!(*p_tree)[nid].IsLeftChild() && !(*p_tree)[nid].IsRoot()) {
    // it's a right child
    auto parent_id = (*p_tree)[nid].Parent();
    auto left_sibling_id = (*p_tree)[parent_id].LeftChild();
    auto parent_split_feature_id = (*p_tree).Snode(parent_id).best.SplitIndex();
    {
      std::lock_guard<std::mutex> lock(spliteval_->mutex);
      spliteval_->AddSplit(parent_id, left_sibling_id, nid, parent_split_feature_id,
          p_tree->Snode(left_sibling_id).weight, p_tree->Snode(nid).weight);
    }
  }
}

void QuantileHistMaker::Builder::CreateNewNodes(
    const GHistIndexMatrix &gmat,
    const ColumnMatrix &column_matrix,
    DMatrix *p_fmat,
    RegTreeThreadSafe *p_tree,
    int *num_leaves,
    int depth,
    unsigned *timestamp,
    std::vector<ExpandEntry> *temp_qexpand_depth,
    int32_t nid,
    std::mutex& mutex_add_nodes) {
  if (p_tree->Snode(nid).best.loss_chg < kRtEps ||
      (param_.max_depth > 0 && depth == param_.max_depth) ||
      (param_.max_leaves > 0 && (*num_leaves) == param_.max_leaves)) {
    (*p_tree)[nid].SetLeaf(p_tree->Snode(nid).weight * param_.learning_rate);
  } else {
    this->ApplySplit(nid, gmat, column_matrix, hist_, *p_fmat, p_tree);
    int left_id = (*p_tree)[nid].LeftChild();
    int right_id = (*p_tree)[nid].RightChild();
    {
      std::lock_guard<std::mutex> lock(mutex_add_nodes);
      temp_qexpand_depth->push_back(ExpandEntry(left_id,
              p_tree->GetDepth(left_id), 0.0, (*timestamp)++));
      temp_qexpand_depth->push_back(ExpandEntry(right_id,
              p_tree->GetDepth(right_id), 0.0, (*timestamp)++));
      (*num_leaves)++;
    }
  }
}

void QuantileHistMaker::Builder::ExpandWithDepthWidthDistributed(
      const GHistIndexMatrix &gmat,
      const GHistIndexBlockMatrix &gmatb,
      const ColumnMatrix &column_matrix,
      DMatrix *p_fmat,
      RegTreeThreadSafe *p_tree,
      const std::vector<GradientPair> &gpair_h) {
  unsigned timestamp = 0;
  int num_leaves = 0;

  // in depth_wise growing, we feed loss_chg with 0.0 since it is not used anyway
  qexpand_depth_wise_.push_back(ExpandEntry(0, p_tree->GetDepth(0), 0.0, timestamp++));
  ++num_leaves;

  std::mutex mutex_add_nodes;

  #pragma omp parallel num_threads(this->nthread_)
  #pragma omp master
  {
    for (int depth = 0; depth < param_.max_depth + 1; depth++) {
      int starting_index = std::numeric_limits<int>::max();
      int sync_count = 0;
      std::vector<ExpandEntry> temp_qexpand_depth;

      perf_monitor.TickStart();
      SeqFor(qexpand_depth_wise_.size(), [&](size_t k) {
        int nid = qexpand_depth_wise_[k].nid;
        auto& node = (*p_tree)[nid];
        if (node.IsRoot() || node.IsLeftChild()) {
          hist_.AddHistRow(nid);
          // in distributed setting, we always calculate from left child or root node
          BuildHist(gpair_h, row_set_collection_[nid], gmat, gmatb, hist_[nid], *p_tree, nid,
              hist_[nid], hist_[nid], nid, -1, false);
          if (!node.IsRoot()) {
            nodes_for_subtraction_trick_[(*p_tree)[node.Parent()].RightChild()] = nid;
          }
          sync_count++;
          starting_index = std::min(starting_index, nid);
        }
      });
      SyncHistograms(starting_index, sync_count, p_tree);
      perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::BUILD_HIST);

      perf_monitor.TickStart();
      SeqFor(qexpand_depth_wise_.size(), [&](size_t k) {
        BuildNodeStat(gmat, p_fmat, p_tree, gpair_h, qexpand_depth_wise_[k].nid);
      });
      perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::INIT_NEW_NODE);

      SeqFor(qexpand_depth_wise_.size(), [&](size_t k) {
        int nid = qexpand_depth_wise_[k].nid;

        perf_monitor.TickStart();
        EvaluateSplit(nid, gmat, hist_, *p_fmat, *p_tree);
        perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::EVALUATE_SPLIT);

        perf_monitor.TickStart();
        CreateNewNodes(gmat, column_matrix, p_fmat, p_tree, &num_leaves, depth, &timestamp,
            &temp_qexpand_depth, nid, mutex_add_nodes);
        perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::APPLY_SPLIT);
      });

      // clean up
      qexpand_depth_wise_.clear();
      nodes_for_subtraction_trick_.clear();
      if (temp_qexpand_depth.empty()) {
        break;
      } else {
        qexpand_depth_wise_ = temp_qexpand_depth;
        temp_qexpand_depth.clear();
      }
    }
  }
}


void QuantileHistMaker::Builder::ExpandWithDepthWidth(
  const GHistIndexMatrix &gmat,
  const GHistIndexBlockMatrix &gmatb,
  const ColumnMatrix &column_matrix,
  DMatrix *p_fmat,
  RegTreeThreadSafe *p_tree,
  const std::vector<GradientPair> &gpair_h) {

  if(rabit::IsDistributed())
    ExpandWithDepthWidthDistributed(gmat, gmatb, column_matrix, p_fmat, p_tree, gpair_h);

  unsigned timestamp = 0;
  int num_leaves = 0;

  // in depth_wise growing, we feed loss_chg with 0.0 since it is not used anyway
  qexpand_depth_wise_.push_back(ExpandEntry(0, p_tree->GetDepth(0), 0.0, timestamp++));
  ++num_leaves;

  std::mutex mutex_add_nodes;

  #pragma omp parallel num_threads(this->nthread_)
  #pragma omp master
  {
    for (int depth = 0; depth < param_.max_depth + 1; depth++) {
      std::vector<ExpandEntry> temp_qexpand_depth;

      double time_apply_split = 0;
      double time_evaluate_split = 0;
      double time_build_hist = 0;
      double time_init_new_node = 0;
      double tstart = dmlc::GetTime();

      ParallelFor(qexpand_depth_wise_.size(), [&](size_t k) {
        int nid = qexpand_depth_wise_[k].nid;
        auto& node = (*p_tree)[nid];
        int32_t parent_nid = node.Parent();
        int32_t another_nid = node.IsRoot() ? nid : (node.IsLeftChild() ?
            (*p_tree)[parent_nid].RightChild() : (*p_tree)[parent_nid].LeftChild());

        auto BuildNodeByHist = [&](int32_t this_nid) {
          double t1 = dmlc::GetTime();
          EvaluateSplit(this_nid, gmat, hist_, *p_fmat, *p_tree);
          double t2 = dmlc::GetTime();
          time_evaluate_split += t2-t1;
          CreateNewNodes(gmat, column_matrix, p_fmat, p_tree, &num_leaves, depth, &timestamp,
              &temp_qexpand_depth, this_nid, mutex_add_nodes);
          time_apply_split += dmlc::GetTime() - t2;
        };

        size_t this_size = row_set_collection_[nid].Size();
        size_t another_size = row_set_collection_[another_nid].Size();

        if (node.IsRoot()) {
          double t1 = dmlc::GetTime();
          hist_.AddHistRow(nid);
          BuildHist(gpair_h, row_set_collection_[nid], gmat, gmatb, hist_[nid], *p_tree,
              nid, hist_[nid], hist_[nid], nid, -1, false);
          double t2 = dmlc::GetTime();
          time_build_hist += t2-t1;

          BuildNodeStat(gmat, p_fmat, p_tree, gpair_h, nid);
          time_init_new_node += dmlc::GetTime()-t2;

          BuildNodeByHist(nid);
        } else if (this_size < another_size || (nid < another_nid && this_size == another_size)) {
          double t1 = dmlc::GetTime();
          hist_.AddHistRow(nid, another_nid);
          BuildHist(gpair_h, row_set_collection_[nid], gmat, gmatb, hist_[nid], *p_tree,
              parent_nid, hist_[another_nid], hist_[parent_nid], nid, another_nid,  false);
          double t2 = dmlc::GetTime();
          time_build_hist += t2-t1;

          BuildNodeStat(gmat, p_fmat, p_tree, gpair_h, nid);
          BuildNodeStat(gmat, p_fmat, p_tree, gpair_h, another_nid);
          time_init_new_node += dmlc::GetTime()-t2;

          #pragma omp taskgroup
          {
            #pragma omp task
            BuildNodeByHist(nid);
            #pragma omp task
            BuildNodeByHist(another_nid);
          }
        } else {
          // nothing
        }
      });

      double ttotal = dmlc::GetTime()-tstart;

      double ttotal_by_threads = time_apply_split + time_evaluate_split + time_build_hist +
          time_init_new_node;

      // aproximate time for each kernel
      perf_monitor.time_init_new_node  += time_init_new_node * (ttotal/ttotal_by_threads);
      perf_monitor.time_build_hist     += time_build_hist * (ttotal/ttotal_by_threads);
      perf_monitor.time_evaluate_split += time_evaluate_split * (ttotal/ttotal_by_threads);
      perf_monitor.time_apply_split    += time_apply_split * (ttotal/ttotal_by_threads);

      // clean up
      qexpand_depth_wise_.clear();
      nodes_for_subtraction_trick_.clear();
      if (temp_qexpand_depth.empty()) {
        break;
      } else {
        qexpand_depth_wise_ = temp_qexpand_depth;
        temp_qexpand_depth.clear();
      }
    }
  }
}

void QuantileHistMaker::Builder::ExpandWithLossGuide(
    const GHistIndexMatrix& gmat,
    const GHistIndexBlockMatrix& gmatb,
    const ColumnMatrix& column_matrix,
    DMatrix* p_fmat,
    RegTreeThreadSafe* p_tree,
    const std::vector<GradientPair>& gpair_h) {
  unsigned timestamp = 0;
  int num_leaves = 0;

  for (int nid = 0; nid < p_tree->Param().num_roots; ++nid) {
    perf_monitor.TickStart();
    hist_.AddHistRow(nid);
    BuildHist(gpair_h, row_set_collection_[nid], gmat, gmatb, hist_[nid], *p_tree, nid,
        hist_[nid], hist_[nid], nid, -1, false);
    perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::BUILD_HIST);

    perf_monitor.TickStart();
    this->InitNewNode(nid, gmat, gpair_h, *p_fmat, *p_tree);
    perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::INIT_NEW_NODE);

    perf_monitor.TickStart();
    this->EvaluateSplit(nid, gmat, hist_, *p_fmat, *p_tree);
    perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::EVALUATE_SPLIT);
    qexpand_loss_guided_->push(ExpandEntry(nid, p_tree->GetDepth(nid),
                               p_tree->Snode(nid).best.loss_chg,
                               timestamp++));
    ++num_leaves;
  }

  while (!qexpand_loss_guided_->empty()) {
    const ExpandEntry candidate = qexpand_loss_guided_->top();
    const int nid = candidate.nid;
    qexpand_loss_guided_->pop();
    if (candidate.loss_chg <= kRtEps
        || (param_.max_depth > 0 && candidate.depth == param_.max_depth)
        || (param_.max_leaves > 0 && num_leaves == param_.max_leaves) ) {
      (*p_tree)[nid].SetLeaf(p_tree->Snode(nid).weight * param_.learning_rate);
    } else {
      perf_monitor.TickStart();
      this->ApplySplit(nid, gmat, column_matrix, hist_, *p_fmat, p_tree);
      perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::APPLY_SPLIT);

      const int cleft = (*p_tree)[nid].LeftChild();
      const int cright = (*p_tree)[nid].RightChild();
      hist_.AddHistRow(cleft, cright);

      perf_monitor.TickStart();
      if (rabit::IsDistributed()) {
        // in distributed mode, we need to keep consistent across workers
        BuildHist(gpair_h, row_set_collection_[cleft], gmat, gmatb, hist_[cleft], *p_tree, nid,
            hist_[cright], hist_[nid], cleft, -1, true);
        SubtractionTrick(hist_[cright], hist_[cleft], hist_[nid]);
      } else {
        if (row_set_collection_[cleft].Size() < row_set_collection_[cright].Size()) {
          BuildHist(gpair_h, row_set_collection_[cleft], gmat, gmatb, hist_[cleft], *p_tree, nid,
              hist_[cright], hist_[nid], cleft, cright, false);
        } else {
          BuildHist(gpair_h, row_set_collection_[cright], gmat, gmatb, hist_[cright], *p_tree, nid,
              hist_[cleft], hist_[nid], cright, cleft, false);
        }
      }
      perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::BUILD_HIST);

      perf_monitor.TickStart();
      this->InitNewNode(cleft, gmat, gpair_h, *p_fmat, *p_tree);
      this->InitNewNode(cright, gmat, gpair_h, *p_fmat, *p_tree);
      bst_uint featureid = p_tree->Snode(nid).best.SplitIndex();
      spliteval_->AddSplit(nid, cleft, cright, featureid,
                           p_tree->Snode(cleft).weight, p_tree->Snode(cright).weight);
      perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::INIT_NEW_NODE);

      perf_monitor.TickStart();
      auto evaluate_fun = [&](size_t this_nid) {
        this->EvaluateSplit(this_nid, gmat, hist_, *p_fmat, *p_tree);
      };
      #pragma omp taskgroup
      {
        #pragma omp task
        evaluate_fun(cleft);
        #pragma omp task
        evaluate_fun(cright);
      }
      perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::EVALUATE_SPLIT);

      qexpand_loss_guided_->push(ExpandEntry(cleft, p_tree->GetDepth(cleft),
                                 p_tree->Snode(cleft).best.loss_chg,
                                 timestamp++));
      qexpand_loss_guided_->push(ExpandEntry(cright, p_tree->GetDepth(cright),
                                 p_tree->Snode(cright).best.loss_chg,
                                 timestamp++));

      ++num_leaves;  // give two and take one, as parent is no longer a leaf
    }
  }
}

void QuantileHistMaker::Builder::Update(const GHistIndexMatrix& gmat,
                                        const GHistIndexBlockMatrix& gmatb,
                                        const ColumnMatrix& column_matrix,
                                        HostDeviceVector<GradientPair>* gpair,
                                        DMatrix* p_fmat,
                                        RegTree* p_tree) {
  perf_monitor.StartPerfMonitor();

  const std::vector<GradientPair>& gpair_h = gpair->ConstHostVector();
  spliteval_->Reset();

  RegTreeThreadSafe safe_tree(*p_tree, snode_, param_);
  safe_tree.ResizeSnode(param_);

  perf_monitor.TickStart();
  this->InitData(gmat, gpair_h, *p_fmat, safe_tree);
  perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::INIT_DATA);

  if (param_.grow_policy == TrainParam::kLossGuide) {
    ExpandWithLossGuide(gmat, gmatb, column_matrix, p_fmat, &safe_tree, gpair_h);
  } else {
    ExpandWithDepthWidth(gmat, gmatb, column_matrix, p_fmat, &safe_tree, gpair_h);
  }

  for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
    p_tree->Stat(nid).loss_chg = safe_tree.Snode(nid).best.loss_chg;
    p_tree->Stat(nid).base_weight = safe_tree.Snode(nid).weight;
    p_tree->Stat(nid).sum_hess = static_cast<float>(safe_tree.Snode(nid).stats.sum_hess);
  }

  pruner_->Update(gpair, p_fmat, std::vector<RegTree*>{p_tree});

  perf_monitor.EndPerfMonitor();
}

bool QuantileHistMaker::Builder::UpdatePredictionCache(
      const DMatrix* data,
      HostDeviceVector<bst_float>* p_out_preds) {
  std::vector<bst_float>& out_preds = p_out_preds->HostVector();

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

  const size_t n_nodes = row_set_collection_.end() - row_set_collection_.begin();

  #pragma omp parallel for
  for (size_t k = 0; k < n_nodes; ++k) {
    const RowSetCollection::Elem rowset = row_set_collection_[k];
    if (rowset.begin != nullptr && rowset.end != nullptr && rowset.node_id != -1) {
      int nid = rowset.node_id;
      bst_float leaf_value;
      // if a node is marked as deleted by the pruner, traverse upward to locate
      // a non-deleted leaf.
      if ((*p_last_tree_)[nid].IsDeleted()) {
        while ((*p_last_tree_)[nid].IsDeleted()) {
          nid = (*p_last_tree_)[nid].Parent();
        }
        CHECK((*p_last_tree_)[nid].IsLeaf());
      }
      leaf_value = (*p_last_tree_)[nid].LeafValue();

      size_t size = rowset.end - rowset.begin;
      for (size_t it = 0; it < size; ++it) {
        out_preds[rowset.begin[it]] += leaf_value;
      }
    }
  }

  return true;
}

void QuantileHistMaker::Builder::InitData(const GHistIndexMatrix& gmat,
                                          const std::vector<GradientPair>& gpair,
                                          const DMatrix& fmat,
                                          const RegTreeThreadSafe& tree) {
  CHECK_EQ(tree.Param().num_nodes, tree.Param().num_roots)
      << "ColMakerHist: can only grow new tree";
  CHECK((param_.max_depth > 0 || param_.max_leaves > 0))
      << "max_depth or max_leaves cannot be both 0 (unlimited); "
      << "at least one should be a positive quantity.";
  if (param_.grow_policy == TrainParam::kDepthWise) {
    CHECK(param_.max_depth > 0) << "max_depth cannot be 0 (unlimited) "
                                << "when grow_policy is depthwise.";
  }
  const auto& info = fmat.Info();

  {
    // initialize the row set
    row_set_collection_.Clear();
    // clear local prediction cache
    leaf_value_cache_.clear();
    // initialize histogram collection
    uint32_t nbins = gmat.cut.row_ptr.back();
    hist_.Init(nbins);

    // initialize histogram builder
    #pragma omp parallel
    {
      this->nthread_ = omp_get_num_threads();
    }
    if (!prow_set_collection_tls_)
    {
      prow_set_collection_tls_.reset(new RowCollectionTLS(
        this->nthread_,
        [&](){ return new RowSetCollection::Split(); },
        [&](RowSetCollection::Split* ptr) { delete ptr; return; }));
    }

    if (!hist_tls_)
    {
      hist_tls_.reset(new HistTLS(
        this->nthread_,
        [&](){ return new tree::GradStats[gmat.cut.row_ptr.back()+4]; },
        [&](tree::GradStats* ptr) { delete[] ptr; return; }));
    }

    const auto nthread = static_cast<bst_omp_uint>(this->nthread_);
    row_split_tloc_.resize(nthread);
    hist_builder_.Init(this->nthread_, nbins);

    CHECK_EQ(info.root_index_.size(), 0U);
    std::vector<size_t>& row_indices = row_set_collection_.row_indices_;
    row_indices.resize(info.num_row_);
    auto* p_row_indices = row_indices.data();
    // mark subsample and build list of member rows

    if (param_.subsample < 1.0f) {
      std::bernoulli_distribution coin_flip(param_.subsample);
      auto& rnd = common::GlobalRandom();
      size_t j = 0;
      for (size_t i = 0; i < info.num_row_; ++i) {
        if (gpair[i].GetHess() >= 0.0f && coin_flip(rnd)) {
          p_row_indices[j++] = i;
        }
      }
      row_indices.resize(j);
    } else {
      size_t j = 0;
      for (size_t i = 0; i < info.num_row_; ++i) {
        if (gpair[i].GetHess() >= 0.0f) {
          p_row_indices[j++] = i;
        }
      }
      row_indices.resize(j);
    }
  }
  row_set_collection_.Init();

  {
    /* determine layout of data */
    const size_t nrow = info.num_row_;
    const size_t ncol = info.num_col_;
    const size_t nnz = info.num_nonzero_;
    // number of discrete bins for feature 0
    const uint32_t nbins_f0 = gmat.cut.row_ptr[1] - gmat.cut.row_ptr[0];
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
    p_last_tree_ = &tree.Get();
    // store a pointer to training data
    p_last_fmat_ = &fmat;
  }
  if (data_layout_ == kDenseDataOneBased) {
    column_sampler_.Init(info.num_col_, param_.colsample_bynode, param_.colsample_bylevel,
            param_.colsample_bytree, true);
  } else {
    column_sampler_.Init(info.num_col_, param_.colsample_bynode, param_.colsample_bylevel,
            param_.colsample_bytree,  false);
  }
  if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
    /* specialized code for dense data:
       choose the column that has a least positive number of discrete bins.
       For dense data (with no missing value),
       the sum of gradient histogram is equal to snode[nid] */
    const std::vector<uint32_t>& row_ptr = gmat.cut.row_ptr;
    const auto nfeature = static_cast<bst_uint>(row_ptr.size() - 1);
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
    snode_.reserve(256);
    snode_.clear();
  }
  {
    if (param_.grow_policy == TrainParam::kLossGuide) {
      qexpand_loss_guided_.reset(new ExpandQueue(LossGuide));
    } else {
      qexpand_depth_wise_.clear();
    }
  }
}

void QuantileHistMaker::Builder::EvaluateSplit(const int nid,
                                               const GHistIndexMatrix& gmat,
                                               const HistCollection& hist,
                                               const DMatrix& fmat,
                                               RegTreeThreadSafe& tree) {
  // start enumeration
  const MetaInfo& info = fmat.Info();
  auto p_feature_set = column_sampler_.GetFeatureSet(tree.GetDepth(nid));
  const auto& feature_set = *p_feature_set;
  const auto nfeature = static_cast<bst_uint>(feature_set.size());
  const auto nthread = static_cast<bst_omp_uint>(this->nthread_);

  MemStackAllocator<SplitEntry, 128> split_entries(nthread);
  SplitEntry* p_split_entries = split_entries.Get();

  SplitEntry sp = tree.Snode(nid).best;
  for(size_t i = 0; i < nthread; ++i) {
    p_split_entries[i] = sp;
  }

  ParallelFor(nfeature, [&](size_t i) {
    const bst_uint fid = feature_set[i];
    const unsigned tid = omp_get_thread_num();

    bool compute_backward = this->EnumerateSplit(+1, gmat, hist[nid], tree.Snode(nid),
      info, &p_split_entries[tid], fid, nid);

    if (compute_backward)
      this->EnumerateSplit(-1, gmat, hist[nid], tree.Snode(nid), info,
                         &p_split_entries[tid], fid, nid);
  });

  for (unsigned tid = 0; tid < nthread; ++tid) {
    tree.Snode(nid).best.Update(p_split_entries[tid]);
  }
}

void QuantileHistMaker::Builder::ApplySplit(int nid,
                                            const GHistIndexMatrix& gmat,
                                            const ColumnMatrix& column_matrix,
                                            const HistCollection& hist,
                                            const DMatrix& fmat,
                                            RegTreeThreadSafe* p_tree) {
  // TODO(hcho3): support feature sampling by levels

  /* 1. Create child nodes */
  const NodeEntry& e = p_tree->Snode(nid);

  bst_float left_leaf_weight, right_leaf_weight;

  {
    std::lock_guard<std::mutex> lock(spliteval_->mutex);
    left_leaf_weight  = spliteval_->ComputeWeight(nid, e.best.left_sum) * param_.learning_rate;
    right_leaf_weight = spliteval_->ComputeWeight(nid, e.best.right_sum) * param_.learning_rate;
  }

  {
    p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                    e.best.DefaultLeft(), e.weight, left_leaf_weight,
                    right_leaf_weight, e.best.loss_chg, e.stats.sum_hess);
  }

  /* 2. Categorize member rows */
  const bool default_left = (*p_tree)[nid].DefaultLeft();
  const bst_uint fid = (*p_tree)[nid].SplitIndex();
  const bst_float split_pt = (*p_tree)[nid].SplitCond();
  const uint32_t lower_bound = gmat.cut.row_ptr[fid];
  const uint32_t upper_bound = gmat.cut.row_ptr[fid + 1];
  int32_t split_cond = -1;
  // convert floating-point split_pt into corresponding bin_id
  // split_cond = -1 indicates that split_pt is less than all known cut points
  CHECK_LT(upper_bound,
           static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
  for (uint32_t i = lower_bound; i < upper_bound; ++i) {
    if (split_pt == gmat.cut.cut[i]) {
      split_cond = static_cast<int32_t>(i);
    }
  }
  const RowSetCollection::Elem rowset = row_set_collection_[nid];

  size_t nLeft;
  Column column = column_matrix.GetColumn(fid);
  if (column.GetType() == xgboost::common::kDenseColumn) {
    nLeft = ApplySplitDenseData(rowset, gmat, column, split_cond,
                        default_left);
  } else {

    nLeft = ApplySplitSparseData(rowset, gmat, column, lower_bound,
                         upper_bound, split_cond, default_left);
  }

  row_set_collection_.AddSplit(
      nid, nLeft, (*p_tree)[nid].LeftChild(), (*p_tree)[nid].RightChild());
}

size_t QuantileHistMaker::Builder::ApplySplitDenseData(
    const RowSetCollection::Elem rowset,
    const GHistIndexMatrix& gmat,
    const Column& column,
    bst_int split_cond,
    bool default_left) {
  const size_t nrows = rowset.end - rowset.begin;

  constexpr size_t MAX_BLOCKS = 56;
  constexpr size_t MIN_SIZE = 1024;

  const size_t nblocks = std::max(size_t(1), std::min(size_t(MAX_BLOCKS),
      nrows/MIN_SIZE));
  const size_t block_size = nrows / nblocks;

  const uint32_t* idx = column.GetIndex();
  std::pair<RowSetCollection::Split*, size_t> local_buff[MAX_BLOCKS] = {{nullptr, 0}};
  std::pair<size_t, size_t> sizes[MAX_BLOCKS] = {{0,0}};

  for(size_t i = 0; i < MAX_BLOCKS; ++i)
    sizes[i] = {0,0};

  if (default_left) {
    ParallelFor(nblocks, [&](size_t iblock) {
      const size_t istart = iblock*block_size;
      const size_t iend = (iblock == nblocks-1) ? nrows : istart + block_size;

      local_buff[iblock] = prow_set_collection_tls_->get();

      auto* ptr = (size_t*)rowset.begin;

      auto& left = local_buff[iblock].first->left;
      auto& right = local_buff[iblock].first->right;

      left.reserve(iend-istart);
      right.reserve(iend-istart);

      size_t* p_left = left.data();
      size_t* p_right = right.data();

      size_t ileft = 0;
      size_t iright = 0;

      for (size_t i = istart; i < iend; i++) {
        if (idx[ptr[i]] == std::numeric_limits<uint32_t>::max()) {
          p_left[ileft++] = ptr[i];
        } else if ( static_cast<int32_t>(idx[ptr[i]] + column.GetBaseIdx()) <=
              split_cond) {
          p_left[ileft++] = ptr[i];
        } else {
          p_right[iright++] = ptr[i];
        }
      }
      sizes[iblock].first = ileft;
      sizes[iblock].second = iright;
    });
  } else {
    ParallelFor(nblocks, [&](size_t iblock) {
      const size_t istart = iblock*block_size;
      const size_t iend = (iblock == nblocks-1) ? nrows : istart + block_size;

      local_buff[iblock] = prow_set_collection_tls_->get();

      auto* ptr = (size_t*)rowset.begin;

      auto& left = local_buff[iblock].first->left;
      auto& right = local_buff[iblock].first->right;

      left.reserve(iend-istart);
      right.reserve(iend-istart);

      size_t* p_left = left.data();
      size_t* p_right = right.data();

      size_t ileft = 0;
      size_t iright = 0;

      for (size_t i = istart; i < iend; i++) {
        if (idx[ptr[i]] == std::numeric_limits<uint32_t>::max()) {
          p_right[iright++] = ptr[i];
        }
        else if ( static_cast<int32_t>(idx[ptr[i]] + column.GetBaseIdx())
                  <= split_cond) {
          p_left[ileft++] = ptr[i];
        } else {
          p_right[iright++] = ptr[i];
        }
      }
      sizes[iblock].first = ileft;
      sizes[iblock].second = iright;

    });
  }

  const size_t nLeft = MergeSplit(local_buff, sizes, nblocks,
      const_cast<size_t*>(rowset.begin));

  for(size_t i = 0; i < nblocks; ++i) {
    if(local_buff[i].first){
      prow_set_collection_tls_->release(local_buff[i]);
    }
  }
  return nLeft;
}

size_t QuantileHistMaker::Builder::ApplySplitSparseData(
    const RowSetCollection::Elem rowset,
    const GHistIndexMatrix& gmat,
    const Column& column,
    bst_uint lower_bound,
    bst_uint upper_bound,
    bst_int split_cond,
    bool default_left) {
  const size_t nrows = rowset.end - rowset.begin;

  constexpr size_t MAX_BLOCKS = 56;
  constexpr size_t MIN_SIZE = 1024;

  const size_t nblocks = std::max(size_t(1), std::min(size_t(MAX_BLOCKS),
      nrows/MIN_SIZE));
  const size_t block_size = nrows / nblocks;

  std::pair<RowSetCollection::Split*, size_t> local_buff[MAX_BLOCKS] = {{nullptr, 0}};
  std::pair<size_t, size_t> sizes[MAX_BLOCKS] = {{0,0}};

  ParallelFor(nblocks, [&](size_t iblock) {
    const size_t ibegin = iblock*block_size;
    const size_t iend = (iblock == nblocks-1) ? nrows : ibegin + block_size;

    if (ibegin < iend) {  // ensure that [ibegin, iend) is nonempty range
      // search first nonzero row with index >= rowset[ibegin]
      const size_t* p = std::lower_bound(column.GetRowData(),
                                         column.GetRowData() + column.Size(),
                                         rowset.begin[ibegin]);

      local_buff[iblock] = prow_set_collection_tls_->get();

      auto& left = local_buff[iblock].first->left;
      auto& right = local_buff[iblock].first->right;

      left.reserve(iend-ibegin);
      right.reserve(iend-ibegin);

      size_t* p_left = left.data();
      size_t* p_right = right.data();

      size_t ileft = 0;
      size_t iright = 0;

      if (p != column.GetRowData() + column.Size() && *p <= rowset.begin[iend - 1]) {
        size_t cursor = p - column.GetRowData();

        for (size_t i = ibegin; i < iend; ++i) {
          const size_t rid = rowset.begin[i];
          while (cursor < column.Size()
                 && column.GetRowIdx(cursor) < rid
                 && column.GetRowIdx(cursor) <= rowset.begin[iend - 1]) {
            ++cursor;
          }
          if (cursor < column.Size() && column.GetRowIdx(cursor) == rid) {
            const uint32_t rbin = column.GetFeatureBinIdx(cursor);
            if (static_cast<int32_t>(rbin + column.GetBaseIdx()) <= split_cond) {
              p_left[ileft++] = rid;
            } else {
              p_right[iright++] = rid;
            }
            ++cursor;
          } else {
            // missing value
            if (default_left) {
              p_left[ileft++] = rid;
            } else {
              p_right[iright++] = rid;
            }
          }
        }
      } else {  // all rows in [ibegin, iend) have missing values
        if (default_left) {
          for (size_t i = ibegin; i < iend; ++i) {
            const size_t rid = rowset.begin[i];
            p_left[ileft++] = rid;
          }
        } else {
          for (size_t i = ibegin; i < iend; ++i) {
            const size_t rid = rowset.begin[i];
            p_right[iright++] = rid;
          }
        }
      }
      sizes[iblock].first = ileft;
      sizes[iblock].second = iright;
    }
  });

  const size_t nLeft = MergeSplit(local_buff, sizes, nblocks,
      const_cast<size_t*>(rowset.begin));

  for(size_t i = 0; i < nblocks; ++i) {
    if(local_buff[i].first) {
      prow_set_collection_tls_->release(local_buff[i]);
    }
  }
  return nLeft;
}

size_t QuantileHistMaker::Builder::MergeSplit(std::pair<RowSetCollection::Split*,
    size_t>* local_buff, std::pair<size_t, size_t>* sizes, size_t nblocks,
    size_t* rowset_begin) {
  size_t nLeft = 0;
  for(size_t i = 0; i < nblocks; ++i) {
    if (local_buff[i].first) {
      nLeft += sizes[i].first;
    }
  }

  ParallelFor(nblocks, [&](size_t iblock) {
    size_t iLeft = 0;
    size_t iRight = 0;

    for(size_t i = 0; i < iblock; ++i) {
      iLeft += sizes[i].first;
      iRight += sizes[i].second;
    }

    if(local_buff[iblock].first) {
      memcpy(rowset_begin + iLeft, local_buff[iblock].first->left.data(), sizes[iblock].first * sizeof(rowset_begin[0]));
      memcpy(rowset_begin + nLeft + iRight, local_buff[iblock].first->right.data(), sizes[iblock].second * sizeof(rowset_begin[0]));
    }
  });

  // auto merge_left = [&]() {
  //   size_t iLeft = 0;
  //   for(size_t i = 0; i < nblocks; ++i) {
  //     if(local_buff[i].first) {
  //       auto st = rowset_begin + iLeft;
  //       const size_t size = sizes[i].first;
  //       memcpy(st, local_buff[i].first->left.data(), size * sizeof(st[0]));
  //       iLeft += size;
  //     }
  //   }
  // };

  // auto merge_right = [&]() {
  //   size_t iRight = 0;
  //   for(size_t i = 0; i < nblocks; ++i) {
  //     if(local_buff[i].first) {
  //       auto st = rowset_begin + iRight + nLeft;
  //       const size_t size = sizes[i].second;
  //       memcpy(st, local_buff[i].first->right.data(), size * sizeof(st[0]));
  //       iRight += size;
  //     }
  //   }
  // };

  // #pragma omp taskgroup
  // {
  //   #pragma omp task
  //   merge_left();
  //   #pragma omp task
  //   merge_right();
  // }

  return nLeft;
}

void QuantileHistMaker::Builder::InitNewNode(int nid,
                                             const GHistIndexMatrix& gmat,
                                             const std::vector<GradientPair>& gpair,
                                             const DMatrix& fmat,
                                             RegTreeThreadSafe& tree) {
  if (param_.enable_feature_grouping > 0 || rabit::IsDistributed()) {
    auto& stats = tree.Snode(nid).stats;
    if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
      /* specialized code for dense data
         For dense data (with no missing value),
         the sum of gradient histogram is equal to snode[nid] */
      GHistRow hist = hist_[nid];
      const std::vector<uint32_t>& row_ptr = gmat.cut.row_ptr;

      const uint32_t ibegin = row_ptr[fid_least_bins_];
      const uint32_t iend = row_ptr[fid_least_bins_ + 1];
      for (uint32_t i = ibegin; i < iend; ++i) {
        const tree::GradStats et = hist[i];
        stats.Add(et.sum_grad, et.sum_hess);
      }
    } else {
      const RowSetCollection::Elem e = row_set_collection_[nid];
      for (const size_t* it = e.begin; it < e.end; ++it) {
        stats.Add(gpair[*it]);
      }
    }
  }

  // calculating the weights
  {
    bst_uint parentid = tree[nid].Parent();

    {
      std::lock_guard<std::mutex> lock(spliteval_->mutex);
      tree.Snode(nid).weight = static_cast<float>(
        spliteval_->ComputeWeight(parentid, tree.Snode(nid).stats));
      tree.Snode(nid).root_gain = static_cast<float>(
        spliteval_->ComputeScore(parentid, tree.Snode(nid).stats,
        tree.Snode(nid).weight));
    }
  }
}

// enumerate the split values of specific feature
bool QuantileHistMaker::Builder::EnumerateSplit(int d_step,
                                                const GHistIndexMatrix& gmat,
                                                const GHistRow& hist,
                                                const NodeEntry& snode,
                                                const MetaInfo& info,
                                                SplitEntry* p_best,
                                                bst_uint fid,
                                                bst_uint nodeID) {
  CHECK(d_step == +1 || d_step == -1);

  // aliases
  const std::vector<uint32_t>& cut_ptr = gmat.cut.row_ptr;
  const std::vector<bst_float>& cut_val = gmat.cut.cut;

  // statistics on both sides of split
  GradStats c;
  GradStats e;
  // best split so far
  SplitEntry best;

  // bin boundaries
  CHECK_LE(cut_ptr[fid],
           static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
  CHECK_LE(cut_ptr[fid + 1],
           static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
  // imin: index (offset) of the minimum value for feature fid
  //       need this for backward enumeration
  const auto imin = static_cast<int32_t>(cut_ptr[fid]);
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

  if(d_step == 1) {
    int32_t i = ibegin;
    while(e.sum_hess < param_.min_child_weight && i < iend) {
      e.Add(hist[i].GetGrad(), hist[i].GetHess());
      i++;
    }

    // forward enumeration: split at right bound of each bin
    for (; i < iend; i++) {
      e.Add(hist[i].GetGrad(), hist[i].GetHess());
      c.SetSubstract(snode.stats, e);
      if (c.sum_hess >= param_.min_child_weight) {
        bst_float loss_chg;
        bst_float split_pt;
        {
          loss_chg = static_cast<bst_float>(spliteval_->ComputeSplitScore(nodeID,
            fid, e, c) - snode.root_gain);
        }

        split_pt = cut_val[i];
        best.Update(loss_chg, fid, split_pt, d_step == -1, e, c);
      }
    }
      p_best->Update(best);
      if (e.GetGrad() == snode.stats.GetGrad() && e.GetHess() == snode.stats.GetHess())
        return false;
  } else {

    int32_t i = ibegin;
    while(e.sum_hess < param_.min_child_weight && i != iend) {
      e.Add(hist[i].GetGrad(), hist[i].GetHess());
      i--;
    }

    for (; i != iend; i--) {
      e.Add(hist[i].GetGrad(), hist[i].GetHess());
      c.SetSubstract(snode.stats, e);
      if (c.sum_hess >= param_.min_child_weight) {
        bst_float loss_chg;
        bst_float split_pt;

        // backward enumeration: split at left bound of each bin
        {
          loss_chg = static_cast<bst_float>(
              spliteval_->ComputeSplitScore(nodeID, fid, c, e) -
              snode.root_gain);
        }

        if (i == imin) {
          // for leftmost bin, left bound is the smallest feature value
          split_pt = gmat.cut.min_val[fid];
        } else {
          split_pt = cut_val[i - 1];
        }
        best.Update(loss_chg, fid, split_pt, d_step == -1, c, e);
      }
    }
    p_best->Update(best);
  }

  return true;
}

XGBOOST_REGISTER_TREE_UPDATER(FastHistMaker, "grow_fast_histmaker")
.describe("(Deprecated, use grow_quantile_histmaker instead.)"
          " Grow tree using quantized histogram.")
.set_body(
    []() {
      LOG(WARNING) << "grow_fast_histmaker is deprecated, "
                   << "use grow_quantile_histmaker instead.";
      return new QuantileHistMaker();
    });

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMaker, "grow_quantile_histmaker")
.describe("Grow tree using quantized histogram.")
.set_body(
    []() {
      return new QuantileHistMaker();
    });

}  // namespace tree
}  // namespace xgboost
