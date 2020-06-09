/*!
 * Copyright 2017-2018 by Contributors
 * \file updater_quantile_hist.cc
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Checn, Egor Smirnov
 */
#include <dmlc/timer.h>
#include <rabit/rabit.h>

#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>
#include <queue>
#include <iomanip>
#include <numeric>
#include <string>
#include <utility>

#include "xgboost/logging.h"
#include "xgboost/tree_updater.h"

#include "constraints.h"
#include "param.h"
#include "./updater_quantile_hist.h"
#include "./split_evaluator.h"
#include "../common/random.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/column_matrix.h"
#include "../common/threading_utils.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_quantile_hist);

DMLC_REGISTER_PARAMETER(CPUHistMakerTrainParam);

void QuantileHistMaker::Configure(const Args& args) {
  // initialize pruner
  if (!pruner_) {
    pruner_.reset(TreeUpdater::Create("prune", tparam_));
  }
  pruner_->Configure(args);
  param_.UpdateAllowUnknown(args);
  hist_maker_param_.UpdateAllowUnknown(args);
  // initialize the split evaluator
  if (!spliteval_) {
    spliteval_.reset(SplitEvaluator::Create(param_.split_evaluator));
  }

  spliteval_->Init(&param_);
}

template<typename GradientSumT>
void QuantileHistMaker::SetBuilder(std::unique_ptr<Builder<GradientSumT>>* builder,
                                   DMatrix *dmat) {
  builder->reset(new Builder<GradientSumT>(
                param_,
                std::move(pruner_),
                std::unique_ptr<SplitEvaluator>(spliteval_->GetHostClone()),
                int_constraint_, dmat));
  if (rabit::IsDistributed()) {
    (*builder)->SetHistSynchronizer(new DistributedHistSynchronizer<GradientSumT>());
    (*builder)->SetHistRowsAdder(new DistributedHistRowsAdder<GradientSumT>());
  } else {
    (*builder)->SetHistSynchronizer(new BatchHistSynchronizer<GradientSumT>());
    (*builder)->SetHistRowsAdder(new BatchHistRowsAdder<GradientSumT>());
  }
}

template<typename GradientSumT>
void QuantileHistMaker::CallBuilderUpdate(const std::unique_ptr<Builder<GradientSumT>>& builder,
                                          HostDeviceVector<GradientPair> *gpair,
                                          DMatrix *dmat,
                                          const std::vector<RegTree *> &trees) {
  for (auto tree : trees) {
    builder->Update(gmat_, gmatb_, column_matrix_, gpair, dmat, tree);
  }
}
void QuantileHistMaker::Update(HostDeviceVector<GradientPair> *gpair,
                               DMatrix *dmat,
                               const std::vector<RegTree *> &trees) {
  if (dmat != p_last_dmat_ || is_gmat_initialized_ == false) {
    updater_monitor_.Start("GmatInitialization");
    gmat_.Init(dmat, static_cast<uint32_t>(param_.max_bin));
    column_matrix_.Init(gmat_, param_.sparse_threshold);
    if (param_.enable_feature_grouping > 0) {
      gmatb_.Init(gmat_, column_matrix_, param_);
    }
    updater_monitor_.Stop("GmatInitialization");
    // A proper solution is puting cut matrix in DMatrix, see:
    // https://github.com/dmlc/xgboost/issues/5143
    is_gmat_initialized_ = true;
  }
  // rescale learning rate according to size of trees
  float lr = param_.learning_rate;
  param_.learning_rate = lr / trees.size();
  int_constraint_.Configure(param_, dmat->Info().num_col_);
  // build tree
  if (hist_maker_param_.single_precision_histogram) {
    if (!float_builder_) {
      SetBuilder(&float_builder_, dmat);
    }
    CallBuilderUpdate(float_builder_, gpair, dmat, trees);
  } else {
    if (!double_builder_) {
      SetBuilder(&double_builder_, dmat);
    }
    CallBuilderUpdate(double_builder_, gpair, dmat, trees);
  }

  param_.learning_rate = lr;

  p_last_dmat_ = dmat;
}

bool QuantileHistMaker::UpdatePredictionCache(
    const DMatrix* data,
    HostDeviceVector<bst_float>* out_preds) {
  if (param_.subsample < 1.0f) {
    return false;
  } else {
    if (hist_maker_param_.single_precision_histogram && float_builder_) {
        return float_builder_->UpdatePredictionCache(data, out_preds);
    } else if (double_builder_) {
        return double_builder_->UpdatePredictionCache(data, out_preds);
    } else {
       return false;
    }
  }
}

template <typename GradientSumT>
void BatchHistSynchronizer<GradientSumT>::SyncHistograms(BuilderT* builder,
                                           int starting_index,
                                           int sync_count,
                                           RegTree *p_tree) {
  builder->builder_monitor_.Start("SyncHistograms");
  const size_t nbins = builder->hist_builder_.GetNumBins();
  common::BlockedSpace2d space(builder->nodes_for_explicit_hist_build_.size(), [&](size_t node) {
    return nbins;
  }, 1024);

  common::ParallelFor2d(space, builder->nthread_, [&](size_t node, common::Range1d r) {
    const auto entry = builder->nodes_for_explicit_hist_build_[node];
    auto this_hist = builder->hist_[entry.nid];
    // Merging histograms from each thread into once
    builder->hist_buffer_.ReduceHist(node, r.begin(), r.end());

    if (!(*p_tree)[entry.nid].IsRoot() && entry.sibling_nid > -1) {
      const size_t parent_id = (*p_tree)[entry.nid].Parent();
      auto parent_hist = builder->hist_[parent_id];
      auto sibling_hist = builder->hist_[entry.sibling_nid];
      SubtractionHist(sibling_hist, parent_hist, this_hist, r.begin(), r.end());
    }
  });
  builder->builder_monitor_.Stop("SyncHistograms");
}

template <typename GradientSumT>
void DistributedHistSynchronizer<GradientSumT>::SyncHistograms(BuilderT* builder,
                                                 int starting_index,
                                                 int sync_count,
                                                 RegTree *p_tree) {
  builder->builder_monitor_.Start("SyncHistograms");
  const size_t nbins = builder->hist_builder_.GetNumBins();
  common::BlockedSpace2d space(builder->nodes_for_explicit_hist_build_.size(), [&](size_t node) {
    return nbins;
  }, 1024);
  common::ParallelFor2d(space, builder->nthread_, [&](size_t node, common::Range1d r) {
    const auto entry = builder->nodes_for_explicit_hist_build_[node];
    auto this_hist = builder->hist_[entry.nid];
    // Merging histograms from each thread into once
    builder->hist_buffer_.ReduceHist(node, r.begin(), r.end());
    // Store posible parent node
    auto this_local = builder->hist_local_worker_[entry.nid];
    CopyHist(this_local, this_hist, r.begin(), r.end());

    if (!(*p_tree)[entry.nid].IsRoot() && entry.sibling_nid > -1) {
      const size_t parent_id = (*p_tree)[entry.nid].Parent();
      auto parent_hist = builder->hist_local_worker_[parent_id];
      auto sibling_hist = builder->hist_[entry.sibling_nid];
      SubtractionHist(sibling_hist, parent_hist, this_hist, r.begin(), r.end());
      // Store posible parent node
      auto sibling_local = builder->hist_local_worker_[entry.sibling_nid];
      CopyHist(sibling_local, sibling_hist, r.begin(), r.end());
    }
  });
  builder->builder_monitor_.Start("SyncHistogramsAllreduce");
  builder->histred_.Allreduce(builder->hist_[starting_index].data(),
                                    builder->hist_builder_.GetNumBins() * sync_count);
  builder->builder_monitor_.Stop("SyncHistogramsAllreduce");

  ParallelSubtractionHist(builder, space, builder->nodes_for_explicit_hist_build_, p_tree);

  common::BlockedSpace2d space2(builder->nodes_for_subtraction_trick_.size(), [&](size_t node) {
    return nbins;
  }, 1024);
  ParallelSubtractionHist(builder, space2, builder->nodes_for_subtraction_trick_, p_tree);
  builder->builder_monitor_.Stop("SyncHistograms");
}

template <typename GradientSumT>
void DistributedHistSynchronizer<GradientSumT>::ParallelSubtractionHist(
                                  BuilderT* builder,
                                  const common::BlockedSpace2d& space,
                                  const std::vector<ExpandEntryT>& nodes,
                                  const RegTree * p_tree) {
  common::ParallelFor2d(space, builder->nthread_, [&](size_t node, common::Range1d r) {
    const auto entry = nodes[node];
    if (!((*p_tree)[entry.nid].IsLeftChild())) {
      auto this_hist = builder->hist_[entry.nid];

      if (!(*p_tree)[entry.nid].IsRoot() && entry.sibling_nid > -1) {
        auto parent_hist = builder->hist_[(*p_tree)[entry.nid].Parent()];
        auto sibling_hist = builder->hist_[entry.sibling_nid];
        SubtractionHist(this_hist, parent_hist, sibling_hist, r.begin(), r.end());
      }
    }
  });
}

template <typename GradientSumT>
void BatchHistRowsAdder<GradientSumT>::AddHistRows(BuilderT* builder,
                                     int *starting_index, int *sync_count,
                                     RegTree *p_tree) {
  builder->builder_monitor_.Start("AddHistRows");

  for (auto const& entry : builder->nodes_for_explicit_hist_build_) {
    int nid = entry.nid;
    builder->hist_.AddHistRow(nid);
    (*starting_index) = std::min(nid, (*starting_index));
  }
  (*sync_count) = builder->nodes_for_explicit_hist_build_.size();

  for (auto const& node : builder->nodes_for_subtraction_trick_) {
    builder->hist_.AddHistRow(node.nid);
  }

  builder->builder_monitor_.Stop("AddHistRows");
}

template <typename GradientSumT>
void DistributedHistRowsAdder<GradientSumT>::AddHistRows(BuilderT* builder,
                                           int *starting_index, int *sync_count,
                                           RegTree *p_tree) {
  builder->builder_monitor_.Start("AddHistRows");
  const size_t explicit_size = builder->nodes_for_explicit_hist_build_.size();
  const size_t subtaction_size = builder->nodes_for_subtraction_trick_.size();
  std::vector<int> merged_node_ids(explicit_size + subtaction_size);
  for (size_t i = 0; i < explicit_size; ++i) {
    merged_node_ids[i] = builder->nodes_for_explicit_hist_build_[i].nid;
  }
  for (size_t i = 0; i < subtaction_size; ++i) {
    merged_node_ids[explicit_size + i] =
    builder->nodes_for_subtraction_trick_[i].nid;
  }
  std::sort(merged_node_ids.begin(), merged_node_ids.end());
  int n_left = 0;
  for (auto const& nid : merged_node_ids) {
    if ((*p_tree)[nid].IsLeftChild()) {
      builder->hist_.AddHistRow(nid);
      (*starting_index) = std::min(nid, (*starting_index));
      n_left++;
      builder->hist_local_worker_.AddHistRow(nid);
    }
  }
  for (auto const& nid : merged_node_ids) {
    if (!((*p_tree)[nid].IsLeftChild())) {
      builder->hist_.AddHistRow(nid);
      builder->hist_local_worker_.AddHistRow(nid);
    }
  }
  (*sync_count) = std::max(1, n_left);
  builder->builder_monitor_.Stop("AddHistRows");
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::SetHistSynchronizer(
                                               HistSynchronizer<GradientSumT>* sync) {
  hist_synchronizer_.reset(sync);
}
template void QuantileHistMaker::Builder<double>::SetHistSynchronizer(
                                                  HistSynchronizer<double>* sync);
template void QuantileHistMaker::Builder<float>::SetHistSynchronizer(
                                                  HistSynchronizer<float>* sync);

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::SetHistRowsAdder(
                                               HistRowsAdder<GradientSumT>* adder) {
  hist_rows_adder_.reset(adder);
}
template void QuantileHistMaker::Builder<double>::SetHistRowsAdder(
                                                  HistRowsAdder<double>* sync);
template void QuantileHistMaker::Builder<float>::SetHistRowsAdder(
                                                 HistRowsAdder<float>* sync);

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::BuildHistogramsLossGuide(
                        ExpandEntry entry,
                        const GHistIndexMatrix &gmat,
                        const GHistIndexBlockMatrix &gmatb,
                        RegTree *p_tree,
                        const std::vector<GradientPair> &gpair_h) {
  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();
  nodes_for_explicit_hist_build_.push_back(entry);

  if (entry.sibling_nid > -1) {
    nodes_for_subtraction_trick_.emplace_back(entry.sibling_nid, entry.nid,
        p_tree->GetDepth(entry.sibling_nid), 0.0f, 0);
  }

  int starting_index = std::numeric_limits<int>::max();
  int sync_count = 0;

  hist_rows_adder_->AddHistRows(this, &starting_index, &sync_count, p_tree);
  BuildLocalHistograms(gmat, gmatb, p_tree, gpair_h);
  hist_synchronizer_->SyncHistograms(this, starting_index, sync_count, p_tree);
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::BuildLocalHistograms(
    const GHistIndexMatrix &gmat,
    const GHistIndexBlockMatrix &gmatb,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h) {
  builder_monitor_.Start("BuildLocalHistograms");

  const size_t n_nodes = nodes_for_explicit_hist_build_.size();

  // create space of size (# rows in each node)
  common::BlockedSpace2d space(n_nodes, [&](size_t node) {
    const int32_t nid = nodes_for_explicit_hist_build_[node].nid;
    return row_set_collection_[nid].Size();
  }, 256);

  std::vector<GHistRowT> target_hists(n_nodes);
  for (size_t i = 0; i < n_nodes; ++i) {
    const int32_t nid = nodes_for_explicit_hist_build_[i].nid;
    target_hists[i] = hist_[nid];
  }

  hist_buffer_.Reset(this->nthread_, n_nodes, space, target_hists);

  // Parallel processing by nodes and data in each node
  common::ParallelFor2d(space, this->nthread_, [&](size_t nid_in_set, common::Range1d r) {
    const auto tid = static_cast<unsigned>(omp_get_thread_num());
    const int32_t nid = nodes_for_explicit_hist_build_[nid_in_set].nid;

    auto start_of_row_set = row_set_collection_[nid].begin;
    auto rid_set = RowSetCollection::Elem(start_of_row_set + r.begin(),
                                      start_of_row_set + r.end(),
                                      nid);
    BuildHist(gpair_h, rid_set, gmat, gmatb, hist_buffer_.GetInitializedHist(tid, nid_in_set));
  });

  builder_monitor_.Stop("BuildLocalHistograms");
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::BuildNodeStats(
    const GHistIndexMatrix &gmat,
    DMatrix *p_fmat,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h) {
  builder_monitor_.Start("BuildNodeStats");
  for (auto const& entry : qexpand_depth_wise_) {
    int nid = entry.nid;
    this->InitNewNode(nid, gmat, gpair_h, *p_fmat, *p_tree);
    // add constraints
    if (!(*p_tree)[nid].IsLeftChild() && !(*p_tree)[nid].IsRoot()) {
      // it's a right child
      auto parent_id = (*p_tree)[nid].Parent();
      auto left_sibling_id = (*p_tree)[parent_id].LeftChild();
      auto parent_split_feature_id = snode_[parent_id].best.SplitIndex();
      spliteval_->AddSplit(parent_id, left_sibling_id, nid, parent_split_feature_id,
                           snode_[left_sibling_id].weight, snode_[nid].weight);
      interaction_constraints_.Split(parent_id, parent_split_feature_id,
                                     left_sibling_id, nid);
    }
  }
  builder_monitor_.Stop("BuildNodeStats");
}
template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::AddSplitsToTree(
          const GHistIndexMatrix &gmat,
          RegTree *p_tree,
          int *num_leaves,
          int depth,
          unsigned *timestamp,
          std::vector<ExpandEntry>* nodes_for_apply_split,
          std::vector<ExpandEntry>* temp_qexpand_depth) {
  for (auto const& entry : qexpand_depth_wise_) {
    int nid = entry.nid;

    if (snode_[nid].best.loss_chg < kRtEps ||
        (param_.max_depth > 0 && depth == param_.max_depth) ||
        (param_.max_leaves > 0 && (*num_leaves) == param_.max_leaves)) {
      (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
    } else {
      nodes_for_apply_split->push_back(entry);

      NodeEntry& e = snode_[nid];
      bst_float left_leaf_weight =
          spliteval_->ComputeWeight(nid, e.best.left_sum) * param_.learning_rate;
      bst_float right_leaf_weight =
          spliteval_->ComputeWeight(nid, e.best.right_sum) * param_.learning_rate;
      p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                         e.best.DefaultLeft(), e.weight, left_leaf_weight,
                         right_leaf_weight, e.best.loss_chg, e.stats.sum_hess,
                         e.best.left_sum.GetHess(), e.best.right_sum.GetHess());

      int left_id = (*p_tree)[nid].LeftChild();
      int right_id = (*p_tree)[nid].RightChild();
      temp_qexpand_depth->push_back(ExpandEntry(left_id, right_id,
                                                p_tree->GetDepth(left_id), 0.0, (*timestamp)++));
      temp_qexpand_depth->push_back(ExpandEntry(right_id, left_id,
                                                p_tree->GetDepth(right_id), 0.0, (*timestamp)++));
      // - 1 parent + 2 new children
      (*num_leaves)++;
    }
  }
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::EvaluateAndApplySplits(
    const GHistIndexMatrix &gmat,
    const ColumnMatrix &column_matrix,
    RegTree *p_tree,
    int *num_leaves,
    int depth,
    unsigned *timestamp,
    std::vector<ExpandEntry> *temp_qexpand_depth) {
  EvaluateSplits(qexpand_depth_wise_, gmat, hist_, *p_tree);

  std::vector<ExpandEntry> nodes_for_apply_split;
  AddSplitsToTree(gmat, p_tree, num_leaves, depth, timestamp,
                  &nodes_for_apply_split, temp_qexpand_depth);
  ApplySplit(nodes_for_apply_split, gmat, column_matrix, hist_, p_tree);
}

// Split nodes to 2 sets depending on amount of rows in each node
// Histograms for small nodes will be built explicitly
// Histograms for big nodes will be built by 'Subtraction Trick'
// Exception: in distributed setting, we always build the histogram for the left child node
//    and use 'Subtraction Trick' to built the histogram for the right child node.
//    This ensures that the workers operate on the same set of tree nodes.
template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::SplitSiblings(const std::vector<ExpandEntry>& nodes,
                   std::vector<ExpandEntry>* small_siblings,
                   std::vector<ExpandEntry>* big_siblings,
                   RegTree *p_tree) {
  builder_monitor_.Start("SplitSiblings");
  for (auto const& entry : nodes) {
    int nid = entry.nid;
    RegTree::Node &node = (*p_tree)[nid];
    if (node.IsRoot()) {
      small_siblings->push_back(entry);
    } else {
      const int32_t left_id = (*p_tree)[node.Parent()].LeftChild();
      const int32_t right_id = (*p_tree)[node.Parent()].RightChild();

      if (nid == left_id && row_set_collection_[left_id ].Size() <
                            row_set_collection_[right_id].Size()) {
        small_siblings->push_back(entry);
      } else if (nid == right_id && row_set_collection_[right_id].Size() <=
                                    row_set_collection_[left_id ].Size()) {
        small_siblings->push_back(entry);
      } else {
        big_siblings->push_back(entry);
      }
    }
  }
  builder_monitor_.Stop("SplitSiblings");
}
template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::ExpandWithDepthWise(
  const GHistIndexMatrix &gmat,
  const GHistIndexBlockMatrix &gmatb,
  const ColumnMatrix &column_matrix,
  DMatrix *p_fmat,
  RegTree *p_tree,
  const std::vector<GradientPair> &gpair_h) {
  unsigned timestamp = 0;
  int num_leaves = 0;

  // in depth_wise growing, we feed loss_chg with 0.0 since it is not used anyway
  qexpand_depth_wise_.emplace_back(ExpandEntry(ExpandEntry::kRootNid, ExpandEntry::kEmptyNid,
      p_tree->GetDepth(ExpandEntry::kRootNid), 0.0, timestamp++));
  ++num_leaves;
  for (int depth = 0; depth < param_.max_depth + 1; depth++) {
    int starting_index = std::numeric_limits<int>::max();
    int sync_count = 0;
    std::vector<ExpandEntry> temp_qexpand_depth;
    SplitSiblings(qexpand_depth_wise_, &nodes_for_explicit_hist_build_,
                  &nodes_for_subtraction_trick_, p_tree);
    hist_rows_adder_->AddHistRows(this, &starting_index, &sync_count, p_tree);
    BuildLocalHistograms(gmat, gmatb, p_tree, gpair_h);
    hist_synchronizer_->SyncHistograms(this, starting_index, sync_count, p_tree);
    BuildNodeStats(gmat, p_fmat, p_tree, gpair_h);

    EvaluateAndApplySplits(gmat, column_matrix, p_tree, &num_leaves, depth, &timestamp,
                   &temp_qexpand_depth);

    // clean up
    qexpand_depth_wise_.clear();
    nodes_for_subtraction_trick_.clear();
    nodes_for_explicit_hist_build_.clear();
    if (temp_qexpand_depth.empty()) {
      break;
    } else {
      qexpand_depth_wise_ = temp_qexpand_depth;
      temp_qexpand_depth.clear();
    }
  }
}
template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::ExpandWithLossGuide(
    const GHistIndexMatrix& gmat,
    const GHistIndexBlockMatrix& gmatb,
    const ColumnMatrix& column_matrix,
    DMatrix* p_fmat,
    RegTree* p_tree,
    const std::vector<GradientPair>& gpair_h) {
  builder_monitor_.Start("ExpandWithLossGuide");
  unsigned timestamp = 0;
  int num_leaves = 0;

  ExpandEntry node(ExpandEntry::kRootNid, ExpandEntry::kEmptyNid,
      p_tree->GetDepth(0), 0.0f, timestamp++);
  BuildHistogramsLossGuide(node, gmat, gmatb, p_tree, gpair_h);

  this->InitNewNode(ExpandEntry::kRootNid, gmat, gpair_h, *p_fmat, *p_tree);

  this->EvaluateSplits({node}, gmat, hist_, *p_tree);
  node.loss_chg = snode_[ExpandEntry::kRootNid].best.loss_chg;

  qexpand_loss_guided_->push(node);
  ++num_leaves;

  while (!qexpand_loss_guided_->empty()) {
    const ExpandEntry candidate = qexpand_loss_guided_->top();
    const int nid = candidate.nid;
    qexpand_loss_guided_->pop();
    if (candidate.IsValid(param_, num_leaves)) {
      (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
    } else {
      NodeEntry& e = snode_[nid];
      bst_float left_leaf_weight =
          spliteval_->ComputeWeight(nid, e.best.left_sum) * param_.learning_rate;
      bst_float right_leaf_weight =
          spliteval_->ComputeWeight(nid, e.best.right_sum) * param_.learning_rate;
      p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                         e.best.DefaultLeft(), e.weight, left_leaf_weight,
                         right_leaf_weight, e.best.loss_chg, e.stats.sum_hess,
                         e.best.left_sum.GetHess(), e.best.right_sum.GetHess());

      this->ApplySplit({candidate}, gmat, column_matrix, hist_, p_tree);

      const int cleft = (*p_tree)[nid].LeftChild();
      const int cright = (*p_tree)[nid].RightChild();

      ExpandEntry left_node(cleft, cright, p_tree->GetDepth(cleft),
                            0.0f, timestamp++);
      ExpandEntry right_node(cright, cleft, p_tree->GetDepth(cright),
                            0.0f, timestamp++);

      if (row_set_collection_[cleft].Size() < row_set_collection_[cright].Size()) {
        BuildHistogramsLossGuide(left_node, gmat, gmatb, p_tree, gpair_h);
      } else {
        BuildHistogramsLossGuide(right_node, gmat, gmatb, p_tree, gpair_h);
      }

      this->InitNewNode(cleft, gmat, gpair_h, *p_fmat, *p_tree);
      this->InitNewNode(cright, gmat, gpair_h, *p_fmat, *p_tree);
      bst_uint featureid = snode_[nid].best.SplitIndex();
      spliteval_->AddSplit(nid, cleft, cright, featureid,
                           snode_[cleft].weight, snode_[cright].weight);
      interaction_constraints_.Split(nid, featureid, cleft, cright);

      this->EvaluateSplits({left_node, right_node}, gmat, hist_, *p_tree);
      left_node.loss_chg = snode_[cleft].best.loss_chg;
      right_node.loss_chg = snode_[cright].best.loss_chg;

      qexpand_loss_guided_->push(left_node);
      qexpand_loss_guided_->push(right_node);

      ++num_leaves;  // give two and take one, as parent is no longer a leaf
    }
  }
  builder_monitor_.Stop("ExpandWithLossGuide");
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::Update(const GHistIndexMatrix& gmat,
                                        const GHistIndexBlockMatrix& gmatb,
                                        const ColumnMatrix& column_matrix,
                                        HostDeviceVector<GradientPair>* gpair,
                                        DMatrix* p_fmat,
                                        RegTree* p_tree) {
  builder_monitor_.Start("Update");

  const std::vector<GradientPair>& gpair_h = gpair->ConstHostVector();

  spliteval_->Reset();
  interaction_constraints_.Reset();

  this->InitData(gmat, gpair_h, *p_fmat, *p_tree);
  if (param_.grow_policy == TrainParam::kLossGuide) {
    ExpandWithLossGuide(gmat, gmatb, column_matrix, p_fmat, p_tree, gpair_h);
  } else {
    ExpandWithDepthWise(gmat, gmatb, column_matrix, p_fmat, p_tree, gpair_h);
  }

  for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
    p_tree->Stat(nid).loss_chg = snode_[nid].best.loss_chg;
    p_tree->Stat(nid).base_weight = snode_[nid].weight;
    p_tree->Stat(nid).sum_hess = static_cast<float>(snode_[nid].stats.sum_hess);
  }
  pruner_->Update(gpair, p_fmat, std::vector<RegTree*>{p_tree});

  builder_monitor_.Stop("Update");
}
template<typename GradientSumT>
bool QuantileHistMaker::Builder<GradientSumT>::UpdatePredictionCache(
    const DMatrix* data,
    HostDeviceVector<bst_float>* p_out_preds) {
  // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
  // conjunction with Update().
  if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_) {
    return false;
  }
  builder_monitor_.Start("UpdatePredictionCache");

  std::vector<bst_float>& out_preds = p_out_preds->HostVector();

  if (leaf_value_cache_.empty()) {
    leaf_value_cache_.resize(p_last_tree_->param.num_nodes,
                             std::numeric_limits<float>::infinity());
  }

  CHECK_GT(out_preds.size(), 0U);

  size_t n_nodes = row_set_collection_.end() - row_set_collection_.begin();

  common::BlockedSpace2d space(n_nodes, [&](size_t node) {
    return row_set_collection_[node].Size();
  }, 1024);

  common::ParallelFor2d(space, this->nthread_, [&](size_t node, common::Range1d r) {
    const RowSetCollection::Elem rowset = row_set_collection_[node];
    if (rowset.begin != nullptr && rowset.end != nullptr) {
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

      for (const size_t* it = rowset.begin + r.begin(); it < rowset.begin + r.end(); ++it) {
        out_preds[*it] += leaf_value;
      }
    }
  });

  builder_monitor_.Stop("UpdatePredictionCache");
  return true;
}
template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitSampling(const std::vector<GradientPair>& gpair,
                                                const DMatrix& fmat,
                                                std::vector<size_t>* row_indices) {
  const auto& info = fmat.Info();
  auto& rnd = common::GlobalRandom();
  std::vector<size_t>& row_indices_local = *row_indices;
  size_t* p_row_indices = row_indices_local.data();
#if XGBOOST_CUSTOMIZE_GLOBAL_PRNG
  std::bernoulli_distribution coin_flip(param_.subsample);
  size_t j = 0;
  for (size_t i = 0; i < info.num_row_; ++i) {
    if (gpair[i].GetHess() >= 0.0f && coin_flip(rnd)) {
      p_row_indices[j++] = i;
    }
  }
  /* resize row_indices to reduce memory */
  row_indices_local.resize(j);
#else
  const size_t nthread = this->nthread_;
  std::vector<size_t> row_offsets(nthread, 0);
  /* usage of mt19937_64 give 2x speed up for subsampling */
  std::vector<std::mt19937> rnds(nthread);
  /* create engine for each thread */
  for (std::mt19937& r : rnds) {
    r = rnd;
  }
  const size_t discard_size = info.num_row_ / nthread;
  #pragma omp parallel num_threads(nthread)
  {
    const size_t tid = omp_get_thread_num();
    const size_t ibegin = tid * discard_size;
    const size_t iend = (tid == (nthread - 1)) ?
                        info.num_row_ : ibegin + discard_size;
    std::bernoulli_distribution coin_flip(param_.subsample);

    rnds[tid].discard(2*discard_size * tid);
    for (size_t i = ibegin; i < iend; ++i) {
      if (gpair[i].GetHess() >= 0.0f && coin_flip(rnds[tid])) {
        p_row_indices[ibegin + row_offsets[tid]++] = i;
      }
    }
  }
  /* discard global engine */
  rnd = rnds[nthread - 1];
  size_t prefix_sum = row_offsets[0];
  for (size_t i = 1; i < nthread; ++i) {
    const size_t ibegin = i * discard_size;

    for (size_t k = 0; k < row_offsets[i]; ++k) {
      row_indices_local[prefix_sum + k] = row_indices_local[ibegin + k];
    }
    prefix_sum += row_offsets[i];
  }
  /* resize row_indices to reduce memory */
  row_indices_local.resize(prefix_sum);
#endif  // XGBOOST_CUSTOMIZE_GLOBAL_PRNG
}
template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitData(const GHistIndexMatrix& gmat,
                                          const std::vector<GradientPair>& gpair,
                                          const DMatrix& fmat,
                                          const RegTree& tree) {
  CHECK((param_.max_depth > 0 || param_.max_leaves > 0))
      << "max_depth or max_leaves cannot be both 0 (unlimited); "
      << "at least one should be a positive quantity.";
  if (param_.grow_policy == TrainParam::kDepthWise) {
    CHECK(param_.max_depth > 0) << "max_depth cannot be 0 (unlimited) "
                                << "when grow_policy is depthwise.";
  }
  builder_monitor_.Start("InitData");
  const auto& info = fmat.Info();

  {
    // initialize the row set
    row_set_collection_.Clear();
    // clear local prediction cache
    leaf_value_cache_.clear();
    // initialize histogram collection
    uint32_t nbins = gmat.cut.Ptrs().back();
    hist_.Init(nbins);
    hist_local_worker_.Init(nbins);
    hist_buffer_.Init(nbins);

    // initialize histogram builder
#pragma omp parallel
    {
      this->nthread_ = omp_get_num_threads();
    }
    hist_builder_ = GHistBuilder<GradientSumT>(this->nthread_, nbins);

    std::vector<size_t>& row_indices = *row_set_collection_.Data();
    row_indices.resize(info.num_row_);
    size_t* p_row_indices = row_indices.data();
    // mark subsample and build list of member rows

    if (param_.subsample < 1.0f) {
      CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
        << "Only uniform sampling is supported, "
        << "gradient-based sampling is only support by GPU Hist.";
      InitSampling(gpair, fmat, &row_indices);
    } else {
      MemStackAllocator<bool, 128> buff(this->nthread_);
      bool* p_buff = buff.Get();
      std::fill(p_buff, p_buff + this->nthread_, false);

      const size_t block_size = info.num_row_ / this->nthread_ + !!(info.num_row_ % this->nthread_);

      #pragma omp parallel num_threads(this->nthread_)
      {
        const size_t tid = omp_get_thread_num();
        const size_t ibegin = tid * block_size;
        const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
            static_cast<size_t>(info.num_row_));

        for (size_t i = ibegin; i < iend; ++i) {
          if (gpair[i].GetHess() < 0.0f) {
            p_buff[tid] = true;
            break;
          }
        }
      }

      bool has_neg_hess = false;
      for (int32_t tid = 0; tid < this->nthread_; ++tid) {
        if (p_buff[tid]) {
          has_neg_hess = true;
        }
      }

      if (has_neg_hess) {
        size_t j = 0;
        for (size_t i = 0; i < info.num_row_; ++i) {
          if (gpair[i].GetHess() >= 0.0f) {
            p_row_indices[j++] = i;
          }
        }
        row_indices.resize(j);
      } else {
        #pragma omp parallel num_threads(this->nthread_)
        {
          const size_t tid = omp_get_thread_num();
          const size_t ibegin = tid * block_size;
          const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
              static_cast<size_t>(info.num_row_));
          for (size_t i = ibegin; i < iend; ++i) {
           p_row_indices[i] = i;
          }
        }
      }
    }
  }

  row_set_collection_.Init();

  {
    /* determine layout of data */
    const size_t nrow = info.num_row_;
    const size_t ncol = info.num_col_;
    const size_t nnz = info.num_nonzero_;
    // number of discrete bins for feature 0
    const uint32_t nbins_f0 = gmat.cut.Ptrs()[1] - gmat.cut.Ptrs()[0];
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
  // store a pointer to the tree
  p_last_tree_ = &tree;
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
    const std::vector<uint32_t>& row_ptr = gmat.cut.Ptrs();
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
  builder_monitor_.Stop("InitData");
}

// if sum of statistics for non-missing values in the node
// is equal to sum of statistics for all values:
// then - there are no missing values
// else - there are missing values
template<typename GradientSumT>
bool QuantileHistMaker::Builder<GradientSumT>::SplitContainsMissingValues(const GradStats e,
                                                            const NodeEntry& snode) {
  if (e.GetGrad() == snode.stats.GetGrad() && e.GetHess() == snode.stats.GetHess()) {
    return false;
  } else {
    return true;
  }
}

// nodes_set - set of nodes to be processed in parallel
template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::EvaluateSplits(
                                               const std::vector<ExpandEntry>& nodes_set,
                                               const GHistIndexMatrix& gmat,
                                               const HistCollection<GradientSumT>& hist,
                                               const RegTree& tree) {
  builder_monitor_.Start("EvaluateSplits");

  const size_t n_nodes_in_set = nodes_set.size();
  const size_t nthread = std::max(1, this->nthread_);

  using FeatureSetType = std::shared_ptr<HostDeviceVector<bst_feature_t>>;
  std::vector<FeatureSetType> features_sets(n_nodes_in_set);
  best_split_tloc_.resize(nthread * n_nodes_in_set);

  // Generate feature set for each tree node
  for (size_t nid_in_set = 0; nid_in_set < n_nodes_in_set; ++nid_in_set) {
    const int32_t nid = nodes_set[nid_in_set].nid;
    features_sets[nid_in_set] = column_sampler_.GetFeatureSet(tree.GetDepth(nid));

    for (unsigned tid = 0; tid < nthread; ++tid) {
      best_split_tloc_[nthread*nid_in_set + tid] = snode_[nid].best;
    }
  }

  // Create 2D space (# of nodes to process x # of features to process)
  // to process them in parallel
  const size_t grain_size = std::max<size_t>(1, features_sets[0]->Size() / nthread);
  common::BlockedSpace2d space(n_nodes_in_set, [&](size_t nid_in_set) {
      return features_sets[nid_in_set]->Size();
  }, grain_size);

  // Start parallel enumeration for all tree nodes in the set and all features
  common::ParallelFor2d(space, this->nthread_, [&](size_t nid_in_set, common::Range1d r) {
    const int32_t nid = nodes_set[nid_in_set].nid;
    const auto tid = static_cast<unsigned>(omp_get_thread_num());
    GHistRowT node_hist = hist[nid];

    for (auto idx_in_feature_set = r.begin(); idx_in_feature_set < r.end(); ++idx_in_feature_set) {
      const auto fid = features_sets[nid_in_set]->ConstHostVector()[idx_in_feature_set];
      if (interaction_constraints_.Query(nid, fid)) {
        auto grad_stats = this->EnumerateSplit<+1>(gmat, node_hist, snode_[nid],
            &best_split_tloc_[nthread*nid_in_set + tid], fid, nid);
        if (SplitContainsMissingValues(grad_stats, snode_[nid])) {
          this->EnumerateSplit<-1>(gmat, node_hist, snode_[nid],
              &best_split_tloc_[nthread*nid_in_set + tid], fid, nid);
        }
      }
    }
  });

  // Find Best Split across threads for each node in nodes set
  for (unsigned nid_in_set = 0; nid_in_set < n_nodes_in_set; ++nid_in_set) {
    const int32_t nid = nodes_set[nid_in_set].nid;
    for (unsigned tid = 0; tid < nthread; ++tid) {
      snode_[nid].best.Update(best_split_tloc_[nthread*nid_in_set + tid]);
    }
  }

  builder_monitor_.Stop("EvaluateSplits");
}

// split row indexes (rid_span) to 2 parts (left_part, right_part) depending
// on comparison of indexes values (idx_span) and split point (split_cond)
// Handle dense columns
// Analog of std::stable_partition, but in no-inplace manner
template <bool default_left, bool any_missing, typename BinIdxType>
inline std::pair<size_t, size_t> PartitionDenseKernel(const common::DenseColumn<BinIdxType>& column,
      common::Span<const size_t> rid_span, const int32_t split_cond,
      common::Span<size_t> left_part, common::Span<size_t> right_part) {
  const int32_t offset = column.GetBaseIdx();
  const BinIdxType* idx = column.GetFeatureBinIdxPtr().data();
  size_t* p_left_part = left_part.data();
  size_t* p_right_part = right_part.data();
  size_t nleft_elems = 0;
  size_t nright_elems = 0;

  if (any_missing) {
    for (auto rid : rid_span) {
      if (column.IsMissing(rid)) {
        if (default_left) {
          p_left_part[nleft_elems++] = rid;
        } else {
          p_right_part[nright_elems++] = rid;
        }
      } else {
        if ((static_cast<int32_t>(idx[rid]) + offset) <= split_cond) {
          p_left_part[nleft_elems++] = rid;
        } else {
          p_right_part[nright_elems++] = rid;
        }
      }
    }
  } else {
    for (auto rid : rid_span)  {
      if ((static_cast<int32_t>(idx[rid]) + offset) <= split_cond) {
        p_left_part[nleft_elems++] = rid;
      } else {
        p_right_part[nright_elems++] = rid;
      }
    }
  }
  return {nleft_elems, nright_elems};
}

// Split row indexes (rid_span) to 2 parts (left_part, right_part) depending
// on comparison of indexes values (idx_span) and split point (split_cond).
// Handle sparse columns
template<bool default_left, typename BinIdxType>
inline std::pair<size_t, size_t> PartitionSparseKernel(
  common::Span<const size_t> rid_span, const int32_t split_cond,
  const common::SparseColumn<BinIdxType>& column, common::Span<size_t> left_part,
  common::Span<size_t> right_part) {
  size_t* p_left_part  = left_part.data();
  size_t* p_right_part = right_part.data();

  size_t nleft_elems = 0;
  size_t nright_elems = 0;
  const size_t* row_data = column.GetRowData();
  const size_t column_size = column.Size();
  if (rid_span.size()) {  // ensure that rid_span is nonempty range
    // search first nonzero row with index >= rid_span.front()
    const size_t* p = std::lower_bound(row_data, row_data + column_size,
                                       rid_span.front());

    if (p != row_data + column_size && *p <= rid_span.back()) {
      size_t cursor = p - row_data;

      for (auto rid : rid_span) {
        while (cursor < column_size
               && column.GetRowIdx(cursor) < rid
               && column.GetRowIdx(cursor) <= rid_span.back()) {
          ++cursor;
        }
        if (cursor < column_size && column.GetRowIdx(cursor) == rid) {
          if (static_cast<int32_t>(column.GetGlobalBinIdx(cursor)) <= split_cond) {
            p_left_part[nleft_elems++] = rid;
          } else {
            p_right_part[nright_elems++] = rid;
          }
          ++cursor;
        } else {
          // missing value
          if (default_left) {
            p_left_part[nleft_elems++] = rid;
          } else {
            p_right_part[nright_elems++] = rid;
          }
        }
      }
    } else {  // all rows in rid_span have missing values
      if (default_left) {
        std::copy(rid_span.begin(), rid_span.end(), p_left_part);
        nleft_elems = rid_span.size();
      } else {
        std::copy(rid_span.begin(), rid_span.end(), p_right_part);
        nright_elems = rid_span.size();
      }
    }
  }

  return {nleft_elems, nright_elems};
}

template <typename GradientSumT>
template <typename BinIdxType>
void QuantileHistMaker::Builder<GradientSumT>::PartitionKernel(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const ColumnMatrix& column_matrix, const RegTree& tree) {
  const size_t* rid = row_set_collection_[nid].begin;

  common::Span<const size_t> rid_span(rid + range.begin(), rid + range.end());
  common::Span<size_t> left  = partition_builder_.GetLeftBuffer(node_in_set,
                                                                range.begin(), range.end());
  common::Span<size_t> right = partition_builder_.GetRightBuffer(node_in_set,
                                                                 range.begin(), range.end());
  const bst_uint fid = tree[nid].SplitIndex();
  const bool default_left = tree[nid].DefaultLeft();
  const auto column_ptr = column_matrix.GetColumn<BinIdxType>(fid);

  std::pair<size_t, size_t> child_nodes_sizes;

  if (column_ptr->GetType() == xgboost::common::kDenseColumn) {
    const common::DenseColumn<BinIdxType>& column =
          static_cast<const common::DenseColumn<BinIdxType>& >(*(column_ptr.get()));
    if (default_left) {
      if (column_matrix.AnyMissing()) {
        child_nodes_sizes = PartitionDenseKernel<true, true>(column, rid_span, split_cond,
                                                             left, right);
      } else {
        child_nodes_sizes = PartitionDenseKernel<true, false>(column, rid_span, split_cond,
                                                              left, right);
      }
    } else {
      if (column_matrix.AnyMissing()) {
        child_nodes_sizes = PartitionDenseKernel<false, true>(column, rid_span, split_cond,
                                                              left, right);
      } else {
        child_nodes_sizes = PartitionDenseKernel<false, false>(column, rid_span, split_cond,
                                                               left, right);
      }
    }
  } else {
    const common::SparseColumn<BinIdxType>& column
      = static_cast<const common::SparseColumn<BinIdxType>& >(*(column_ptr.get()));
    if (default_left) {
      child_nodes_sizes = PartitionSparseKernel<true>(rid_span, split_cond, column, left, right);
    } else {
      child_nodes_sizes = PartitionSparseKernel<false>(rid_span, split_cond, column, left, right);
    }
  }

  const size_t n_left  = child_nodes_sizes.first;
  const size_t n_right = child_nodes_sizes.second;

  partition_builder_.SetNLeftElems(node_in_set, range.begin(), range.end(), n_left);
  partition_builder_.SetNRightElems(node_in_set, range.begin(), range.end(), n_right);
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::FindSplitConditions(
                                                     const std::vector<ExpandEntry>& nodes,
                                                     const RegTree& tree,
                                                     const GHistIndexMatrix& gmat,
                                                     std::vector<int32_t>* split_conditions) {
  const size_t n_nodes = nodes.size();
  split_conditions->resize(n_nodes);

  for (size_t i = 0; i < nodes.size(); ++i) {
    const int32_t nid = nodes[i].nid;
    const bst_uint fid = tree[nid].SplitIndex();
    const bst_float split_pt = tree[nid].SplitCond();
    const uint32_t lower_bound = gmat.cut.Ptrs()[fid];
    const uint32_t upper_bound = gmat.cut.Ptrs()[fid + 1];
    int32_t split_cond = -1;
    // convert floating-point split_pt into corresponding bin_id
    // split_cond = -1 indicates that split_pt is less than all known cut points
    CHECK_LT(upper_bound,
             static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
    for (uint32_t i = lower_bound; i < upper_bound; ++i) {
      if (split_pt == gmat.cut.Values()[i]) {
        split_cond = static_cast<int32_t>(i);
      }
    }
    (*split_conditions)[i] = split_cond;
  }
}
template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::AddSplitsToRowSet(
                                               const std::vector<ExpandEntry>& nodes,
                                               RegTree* p_tree) {
  const size_t n_nodes = nodes.size();
  for (size_t i = 0; i < n_nodes; ++i) {
    const int32_t nid = nodes[i].nid;
    const size_t n_left = partition_builder_.GetNLeftElems(i);
    const size_t n_right = partition_builder_.GetNRightElems(i);

    row_set_collection_.AddSplit(nid, (*p_tree)[nid].LeftChild(),
        (*p_tree)[nid].RightChild(), n_left, n_right);
  }
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::ApplySplit(const std::vector<ExpandEntry> nodes,
                                            const GHistIndexMatrix& gmat,
                                            const ColumnMatrix& column_matrix,
                                            const HistCollection<GradientSumT>& hist,
                                            RegTree* p_tree) {
  builder_monitor_.Start("ApplySplit");
  // 1. Find split condition for each split
  const size_t n_nodes = nodes.size();
  std::vector<int32_t> split_conditions;
  FindSplitConditions(nodes, *p_tree, gmat, &split_conditions);
  // 2.1 Create a blocked space of size SUM(samples in each node)
  common::BlockedSpace2d space(n_nodes, [&](size_t node_in_set) {
    int32_t nid = nodes[node_in_set].nid;
    return row_set_collection_[nid].Size();
  }, kPartitionBlockSize);
  // 2.2 Initialize the partition builder
  // allocate buffers for storage intermediate results by each thread
  partition_builder_.Init(space.Size(), n_nodes, [&](size_t node_in_set) {
    const int32_t nid = nodes[node_in_set].nid;
    const size_t size = row_set_collection_[nid].Size();
    const size_t n_tasks = size / kPartitionBlockSize + !!(size % kPartitionBlockSize);
    return n_tasks;
  });
  // 2.3 Split elements of row_set_collection_ to left and right child-nodes for each node
  // Store results in intermediate buffers from partition_builder_
  common::ParallelFor2d(space, this->nthread_, [&](size_t node_in_set, common::Range1d r) {
    const int32_t nid = nodes[node_in_set].nid;
      switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        PartitionKernel<uint8_t>(node_in_set, nid, r,
                  split_conditions[node_in_set], column_matrix, *p_tree);
        break;
      case common::kUint16BinsTypeSize:
        PartitionKernel<uint16_t>(node_in_set, nid, r,
                  split_conditions[node_in_set], column_matrix, *p_tree);
        break;
      case common::kUint32BinsTypeSize:
        PartitionKernel<uint32_t>(node_in_set, nid, r,
                  split_conditions[node_in_set], column_matrix, *p_tree);
        break;
      default:
        CHECK(false);  // no default behavior
    }
    });
  // 3. Compute offsets to copy blocks of row-indexes
  // from partition_builder_ to row_set_collection_
  partition_builder_.CalculateRowOffsets();

  // 4. Copy elements from partition_builder_ to row_set_collection_ back
  // with updated row-indexes for each tree-node
  common::ParallelFor2d(space, this->nthread_, [&](size_t node_in_set, common::Range1d r) {
    const int32_t nid = nodes[node_in_set].nid;
    partition_builder_.MergeToArray(node_in_set, r.begin(),
        const_cast<size_t*>(row_set_collection_[nid].begin));
  });
  // 5. Add info about splits into row_set_collection_
  AddSplitsToRowSet(nodes, p_tree);
  builder_monitor_.Stop("ApplySplit");
}
template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitNewNode(int nid,
                                             const GHistIndexMatrix& gmat,
                                             const std::vector<GradientPair>& gpair,
                                             const DMatrix& fmat,
                                             const RegTree& tree) {
  builder_monitor_.Start("InitNewNode");
  {
    snode_.resize(tree.param.num_nodes, NodeEntry(param_));
  }

  {
    GHistRowT hist = hist_[nid];
    GradientPairT grad_stat;
    if (tree[nid].IsRoot()) {
      if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
        const std::vector<uint32_t>& row_ptr = gmat.cut.Ptrs();
        const uint32_t ibegin = row_ptr[fid_least_bins_];
        const uint32_t iend = row_ptr[fid_least_bins_ + 1];
        auto begin = hist.data();
        for (uint32_t i = ibegin; i < iend; ++i) {
          const GradientPairT et = begin[i];
          grad_stat.Add(et.GetGrad(), et.GetHess());
        }
      } else {
        const RowSetCollection::Elem e = row_set_collection_[nid];
        for (const size_t* it = e.begin; it < e.end; ++it) {
          grad_stat.Add(gpair[*it].GetGrad(), gpair[*it].GetHess());
        }
      }
      histred_.Allreduce(&grad_stat, 1);
      snode_[nid].stats = tree::GradStats(grad_stat.GetGrad(), grad_stat.GetHess());
    } else {
      int parent_id = tree[nid].Parent();
      if (tree[nid].IsLeftChild()) {
        snode_[nid].stats = snode_[parent_id].best.left_sum;
      } else {
        snode_[nid].stats = snode_[parent_id].best.right_sum;
      }
    }
  }

  // calculating the weights
  {
    bst_uint parentid = tree[nid].Parent();
    snode_[nid].weight = static_cast<float>(
            spliteval_->ComputeWeight(parentid, snode_[nid].stats));
    snode_[nid].root_gain = static_cast<float>(
            spliteval_->ComputeScore(parentid, snode_[nid].stats, snode_[nid].weight));
  }
  builder_monitor_.Stop("InitNewNode");
}


// Enumerate the split values of specific feature.
// Returns the sum of gradients corresponding to the data points that contains a non-missing value
// for the particular feature fid.
template<typename GradientSumT>
template <int d_step>
GradStats QuantileHistMaker::Builder<GradientSumT>::EnumerateSplit(
    const GHistIndexMatrix &gmat, const GHistRowT &hist, const NodeEntry &snode,
    SplitEntry *p_best, bst_uint fid, bst_uint nodeID) const {
  CHECK(d_step == +1 || d_step == -1);

  // aliases
  const std::vector<uint32_t>& cut_ptr = gmat.cut.Ptrs();
  const std::vector<bst_float>& cut_val = gmat.cut.Values();

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

  for (int32_t i = ibegin; i != iend; i += d_step) {
    // start working
    // try to find a split
    e.Add(hist[i].GetGrad(), hist[i].GetHess());
    if (e.sum_hess >= param_.min_child_weight) {
      c.SetSubstract(snode.stats, e);
      if (c.sum_hess >= param_.min_child_weight) {
        bst_float loss_chg;
        bst_float split_pt;
        if (d_step > 0) {
          // forward enumeration: split at right bound of each bin
          loss_chg = static_cast<bst_float>(
              spliteval_->ComputeSplitScore(nodeID, fid, e, c) -
              snode.root_gain);
          split_pt = cut_val[i];
          best.Update(loss_chg, fid, split_pt, d_step == -1, e, c);
        } else {
          // backward enumeration: split at left bound of each bin
          loss_chg = static_cast<bst_float>(
              spliteval_->ComputeSplitScore(nodeID, fid, c, e) -
              snode.root_gain);
          if (i == imin) {
            // for leftmost bin, left bound is the smallest feature value
            split_pt = gmat.cut.MinValues()[fid];
          } else {
            split_pt = cut_val[i - 1];
          }
          best.Update(loss_chg, fid, split_pt, d_step == -1, c, e);
        }
      }
    }
  }
  p_best->Update(best);

  return e;
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
