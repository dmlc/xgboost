/*!
 * Copyright 2017-2021 by Contributors
 * \file updater_quantile_hist.cc
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Checn, Egor Smirnov
 */
#include <dmlc/timer.h>
#include <rabit/rabit.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>

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
}

template<typename GradientSumT>
void QuantileHistMaker::SetBuilder(const size_t n_trees,
                                   std::unique_ptr<Builder<GradientSumT>>* builder,
                                   DMatrix *dmat) {
  builder->reset(new Builder<GradientSumT>(
                n_trees,
                param_,
                std::move(pruner_),
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
                                          GHistIndexMatrix const& gmat,
                                          const std::vector<RegTree *> &trees) {
  for (auto tree : trees) {
    builder->Update(gmat, column_matrix_, gpair, dmat, tree);
  }
}
void QuantileHistMaker::Update(HostDeviceVector<GradientPair> *gpair,
                               DMatrix *dmat,
                               const std::vector<RegTree *> &trees) {
  auto const &gmat =
      *(dmat->GetBatches<GHistIndexMatrix>(
                BatchParam{GenericParameter::kCpuId, param_.max_bin})
            .begin());
  if (dmat != p_last_dmat_ || is_gmat_initialized_ == false) {
    updater_monitor_.Start("GmatInitialization");
    column_matrix_.Init(gmat, param_.sparse_threshold);
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
  const size_t n_trees = trees.size();
  if (hist_maker_param_.single_precision_histogram) {
    if (!float_builder_) {
      SetBuilder(n_trees, &float_builder_, dmat);
    }
    CallBuilderUpdate(float_builder_, gpair, dmat, gmat, trees);
  } else {
    if (!double_builder_) {
      SetBuilder(n_trees, &double_builder_, dmat);
    }
    CallBuilderUpdate(double_builder_, gpair, dmat, gmat, trees);
  }

  param_.learning_rate = lr;

  p_last_dmat_ = dmat;
}

bool QuantileHistMaker::UpdatePredictionCache(
    const DMatrix* data, VectorView<float> out_preds) {
  if (hist_maker_param_.single_precision_histogram && float_builder_) {
      return float_builder_->UpdatePredictionCache(data, out_preds);
  } else if (double_builder_) {
      return double_builder_->UpdatePredictionCache(data, out_preds);
  } else {
      return false;
  }
}

template <typename GradientSumT>
void BatchHistSynchronizer<GradientSumT>::SyncHistograms(BuilderT *builder,
                                                         int,
                                                         int,
                                                         RegTree *p_tree) {
  builder->builder_monitor_.Start("SyncHistograms");
  const size_t nbins = builder->hist_builder_.GetNumBins();
  common::BlockedSpace2d space(builder->nodes_for_explicit_hist_build_.size(), [&](size_t) {
    return nbins;
  }, 1024);

  common::ParallelFor2d(space, builder->nthread_, [&](size_t node, common::Range1d r) {
    const auto& entry = builder->nodes_for_explicit_hist_build_[node];
    auto this_hist = builder->hist_[entry.nid];
    // Merging histograms from each thread into once
    builder->hist_buffer_.ReduceHist(node, r.begin(), r.end());

    if (!(*p_tree)[entry.nid].IsRoot()) {
      const size_t parent_id = (*p_tree)[entry.nid].Parent();
      const int subtraction_node_id = builder->nodes_for_subtraction_trick_[node].nid;
      auto parent_hist = builder->hist_[parent_id];
      auto sibling_hist = builder->hist_[subtraction_node_id];
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
  common::BlockedSpace2d space(builder->nodes_for_explicit_hist_build_.size(), [&](size_t) {
    return nbins;
  }, 1024);
  common::ParallelFor2d(space, builder->nthread_, [&](size_t node, common::Range1d r) {
    const auto& entry = builder->nodes_for_explicit_hist_build_[node];
    auto this_hist = builder->hist_[entry.nid];
    // Merging histograms from each thread into once
    builder->hist_buffer_.ReduceHist(node, r.begin(), r.end());
    // Store posible parent node
    auto this_local = builder->hist_local_worker_[entry.nid];
    CopyHist(this_local, this_hist, r.begin(), r.end());

    if (!(*p_tree)[entry.nid].IsRoot()) {
      const size_t parent_id = (*p_tree)[entry.nid].Parent();
      const int subtraction_node_id = builder->nodes_for_subtraction_trick_[node].nid;
      auto parent_hist = builder->hist_local_worker_[parent_id];
      auto sibling_hist = builder->hist_[subtraction_node_id];
      SubtractionHist(sibling_hist, parent_hist, this_hist, r.begin(), r.end());
      // Store posible parent node
      auto sibling_local = builder->hist_local_worker_[subtraction_node_id];
      CopyHist(sibling_local, sibling_hist, r.begin(), r.end());
    }
  });
  builder->builder_monitor_.Start("SyncHistogramsAllreduce");

  builder->histred_.Allreduce(builder->hist_[starting_index].data(),
                                    builder->hist_builder_.GetNumBins() * sync_count);

  builder->builder_monitor_.Stop("SyncHistogramsAllreduce");

  ParallelSubtractionHist(builder, space, builder->nodes_for_explicit_hist_build_,
                          builder->nodes_for_subtraction_trick_, p_tree);

  common::BlockedSpace2d space2(builder->nodes_for_subtraction_trick_.size(), [&](size_t) {
    return nbins;
  }, 1024);
  ParallelSubtractionHist(builder, space2, builder->nodes_for_subtraction_trick_,
                          builder->nodes_for_explicit_hist_build_, p_tree);
  builder->builder_monitor_.Stop("SyncHistograms");
}

template <typename GradientSumT>
void DistributedHistSynchronizer<GradientSumT>::ParallelSubtractionHist(
                                  BuilderT* builder,
                                  const common::BlockedSpace2d& space,
                                  const std::vector<CPUExpandEntry>& nodes,
                                  const std::vector<CPUExpandEntry>& subtraction_nodes,
                                  const RegTree * p_tree) {
  common::ParallelFor2d(space, builder->nthread_, [&](size_t node, common::Range1d r) {
    const auto& entry = nodes[node];
    if (!((*p_tree)[entry.nid].IsLeftChild())) {
      auto this_hist = builder->hist_[entry.nid];

      if (!(*p_tree)[entry.nid].IsRoot()) {
        const int subtraction_node_id = subtraction_nodes[node].nid;
        auto parent_hist = builder->hist_[(*p_tree)[entry.nid].Parent()];
        auto sibling_hist = builder->hist_[subtraction_node_id];
        SubtractionHist(this_hist, parent_hist, sibling_hist, r.begin(), r.end());
      }
    }
  });
}

template <typename GradientSumT>
void BatchHistRowsAdder<GradientSumT>::AddHistRows(BuilderT *builder,
                                                   int *starting_index,
                                                   int *sync_count,
                                                   RegTree *) {
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
  builder->hist_.AllocateAllData();
  builder->builder_monitor_.Stop("AddHistRows");
}

template <typename GradientSumT>
void DistributedHistRowsAdder<GradientSumT>::AddHistRows(BuilderT *builder,
                                                         int *starting_index,
                                                         int *sync_count,
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
  builder->hist_.AllocateAllData();
  builder->hist_local_worker_.AllocateAllData();
  (*sync_count) = std::max(1, n_left);
  builder->builder_monitor_.Stop("AddHistRows");
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::SetHistSynchronizer(
    HistSynchronizer<GradientSumT> *sync) {
  hist_synchronizer_.reset(sync);
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::SetHistRowsAdder(
    HistRowsAdder<GradientSumT> *adder) {
  hist_rows_adder_.reset(adder);
}

template <typename GradientSumT>
template <bool any_missing>
void QuantileHistMaker::Builder<GradientSumT>::InitRoot(
    const GHistIndexMatrix &gmat,
    const DMatrix& fmat,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h,
    int *num_leaves, std::vector<CPUExpandEntry> *expand) {

  CPUExpandEntry node(CPUExpandEntry::kRootNid, p_tree->GetDepth(0), 0.0f);

  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();
  nodes_for_explicit_hist_build_.push_back(node);

  int starting_index = std::numeric_limits<int>::max();
  int sync_count = 0;

  hist_rows_adder_->AddHistRows(this, &starting_index, &sync_count, p_tree);
  BuildLocalHistograms<any_missing>(gmat, p_tree, gpair_h);
  hist_synchronizer_->SyncHistograms(this, starting_index, sync_count, p_tree);

  this->InitNewNode(CPUExpandEntry::kRootNid, gmat, gpair_h, fmat, *p_tree);

  this->EvaluateSplits({node}, gmat, hist_, *p_tree);
  node.loss_chg = snode_[CPUExpandEntry::kRootNid].best.loss_chg;
  expand->push_back(node);
  ++(*num_leaves);
}

template<typename GradientSumT>
template <bool any_missing>
void QuantileHistMaker::Builder<GradientSumT>::BuildLocalHistograms(
    const GHistIndexMatrix &gmat,
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
    hist_builder_.template BuildHist<any_missing>(gpair_h, rid_set, gmat,
                                                  hist_buffer_.GetInitializedHist(tid, nid_in_set));
  });

  builder_monitor_.Stop("BuildLocalHistograms");
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::AddSplitsToTree(
          const std::vector<CPUExpandEntry>& expand,
          RegTree *p_tree,
          int *num_leaves,
          std::vector<CPUExpandEntry>* nodes_for_apply_split) {
  auto evaluator = tree_evaluator_.GetEvaluator();
  for (auto const& entry : expand) {
    int nid = entry.nid;

    if (entry.IsValid(param_, *num_leaves)) {
      (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
    } else {
      nodes_for_apply_split->push_back(entry);

      NodeEntry& e = snode_[nid];
      bst_float left_leaf_weight =
          evaluator.CalcWeight(nid, param_, GradStats{e.best.left_sum}) * param_.learning_rate;
      bst_float right_leaf_weight =
          evaluator.CalcWeight(nid, param_, GradStats{e.best.right_sum}) * param_.learning_rate;
      p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                         e.best.DefaultLeft(), e.weight, left_leaf_weight,
                         right_leaf_weight, e.best.loss_chg, e.stats.GetHess(),
                         e.best.left_sum.GetHess(), e.best.right_sum.GetHess());
      // - 1 parent + 2 new children
      (*num_leaves)++;
    }
  }
}

// Split nodes to 2 sets depending on amount of rows in each node
// Histograms for small nodes will be built explicitly
// Histograms for big nodes will be built by 'Subtraction Trick'
// Exception: in distributed setting, we always build the histogram for the left child node
//    and use 'Subtraction Trick' to built the histogram for the right child node.
//    This ensures that the workers operate on the same set of tree nodes.
template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::SplitSiblings(
    const std::vector<CPUExpandEntry> &nodes_for_apply_split,
    std::vector<CPUExpandEntry> *nodes_to_evaluate, RegTree *p_tree) {
  builder_monitor_.Start("SplitSiblings");
  for (auto const& entry : nodes_for_apply_split) {
    int nid = entry.nid;

    const int cleft = (*p_tree)[nid].LeftChild();
    const int cright = (*p_tree)[nid].RightChild();
    const CPUExpandEntry left_node = CPUExpandEntry(cleft, p_tree->GetDepth(cleft), 0.0);
    const CPUExpandEntry right_node = CPUExpandEntry(cright, p_tree->GetDepth(cright), 0.0);
    nodes_to_evaluate->push_back(left_node);
    nodes_to_evaluate->push_back(right_node);
    if (row_set_collection_[cleft].Size() < row_set_collection_[cright].Size()) {
      nodes_for_explicit_hist_build_.push_back(left_node);
      nodes_for_subtraction_trick_.push_back(right_node);
    } else {
      nodes_for_explicit_hist_build_.push_back(right_node);
      nodes_for_subtraction_trick_.push_back(left_node);
    }
  }
  CHECK_EQ(nodes_for_subtraction_trick_.size(), nodes_for_explicit_hist_build_.size());
  builder_monitor_.Stop("SplitSiblings");
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::BuildNodeStats(
  const GHistIndexMatrix &gmat,
  const DMatrix& fmat,
  const std::vector<GradientPair> &gpair_h,
  const std::vector<CPUExpandEntry>& nodes_for_apply_split, RegTree *p_tree) {
  for (auto const& candidate : nodes_for_apply_split) {
    const int nid = candidate.nid;
    const int cleft = (*p_tree)[nid].LeftChild();
    const int cright = (*p_tree)[nid].RightChild();

    InitNewNode(cleft, gmat, gpair_h, fmat, *p_tree);
    InitNewNode(cright, gmat, gpair_h, fmat, *p_tree);
    bst_uint featureid = snode_[nid].best.SplitIndex();
    tree_evaluator_.AddSplit(nid, cleft, cright, featureid,
                            snode_[cleft].weight, snode_[cright].weight);
    interaction_constraints_.Split(nid, featureid, cleft, cright);
  }
}

template<typename GradientSumT>
template <bool any_missing>
void QuantileHistMaker::Builder<GradientSumT>::ExpandTree(
    const GHistIndexMatrix& gmat,
    const ColumnMatrix& column_matrix,
    DMatrix* p_fmat,
    RegTree* p_tree,
    const std::vector<GradientPair>& gpair_h) {
  builder_monitor_.Start("ExpandTree");
  int num_leaves = 0;

  Driver<CPUExpandEntry> driver(static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy));
  std::vector<CPUExpandEntry> expand;
  InitRoot<any_missing>(gmat, *p_fmat, p_tree, gpair_h, &num_leaves, &expand);
  driver.Push(expand[0]);

  int depth = 0;
  while (!driver.IsEmpty()) {
    expand = driver.Pop();
    depth = expand[0].depth + 1;
    std::vector<CPUExpandEntry> nodes_for_apply_split;
    std::vector<CPUExpandEntry> nodes_to_evaluate;
    nodes_for_explicit_hist_build_.clear();
    nodes_for_subtraction_trick_.clear();

    AddSplitsToTree(expand, p_tree, &num_leaves, &nodes_for_apply_split);

    if (nodes_for_apply_split.size() != 0) {
      ApplySplit<any_missing>(nodes_for_apply_split, gmat, column_matrix, hist_, p_tree);
      SplitSiblings(nodes_for_apply_split, &nodes_to_evaluate, p_tree);

      int starting_index = std::numeric_limits<int>::max();
      int sync_count = 0;
      hist_rows_adder_->AddHistRows(this, &starting_index, &sync_count, p_tree);
      if (depth < param_.max_depth) {
        BuildLocalHistograms<any_missing>(gmat, p_tree, gpair_h);
        hist_synchronizer_->SyncHistograms(this, starting_index, sync_count, p_tree);
      }

      BuildNodeStats(gmat, *p_fmat, gpair_h, nodes_for_apply_split, p_tree);
      EvaluateSplits(nodes_to_evaluate, gmat, hist_, *p_tree);

      for (size_t i = 0; i < nodes_for_apply_split.size(); ++i) {
        const CPUExpandEntry candidate = nodes_for_apply_split[i];
        const int nid = candidate.nid;
        const int cleft = (*p_tree)[nid].LeftChild();
        const int cright = (*p_tree)[nid].RightChild();
        CPUExpandEntry left_node = nodes_to_evaluate[i*2 + 0];
        CPUExpandEntry right_node = nodes_to_evaluate[i*2 + 1];

        left_node.loss_chg = snode_[cleft].best.loss_chg;
        right_node.loss_chg = snode_[cright].best.loss_chg;

        driver.Push(left_node);
        driver.Push(right_node);
      }
    }
  }
  builder_monitor_.Stop("ExpandTree");
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::Update(
    const GHistIndexMatrix &gmat,
    const ColumnMatrix &column_matrix,
    HostDeviceVector<GradientPair> *gpair,
    DMatrix *p_fmat, RegTree *p_tree) {
  builder_monitor_.Start("Update");

  std::vector<GradientPair>* gpair_ptr = &(gpair->HostVector());
  // in case 'num_parallel_trees != 1' no posibility to change initial gpair
  if (GetNumberOfTrees() != 1) {
    gpair_local_.resize(gpair_ptr->size());
    gpair_local_ = *gpair_ptr;
    gpair_ptr = &gpair_local_;
  }
  tree_evaluator_ =
      TreeEvaluator(param_, p_fmat->Info().num_col_, GenericParameter::kCpuId);
  interaction_constraints_.Reset();
  p_last_fmat_mutable_ = p_fmat;

  this->InitData(gmat, *p_fmat, *p_tree, gpair_ptr);

  if (column_matrix.AnyMissing()) {
    ExpandTree<true>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
  } else {
    ExpandTree<false>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
  }
  for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
    p_tree->Stat(nid).loss_chg = snode_[nid].best.loss_chg;
    p_tree->Stat(nid).base_weight = snode_[nid].weight;
    p_tree->Stat(nid).sum_hess = static_cast<float>(snode_[nid].stats.GetHess());
  }
  pruner_->Update(gpair, p_fmat, std::vector<RegTree*>{p_tree});

  builder_monitor_.Stop("Update");
}

template<typename GradientSumT>
bool QuantileHistMaker::Builder<GradientSumT>::UpdatePredictionCache(
    const DMatrix* data,
    VectorView<float> out_preds) {
  // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
  // conjunction with Update().
  if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_ ||
      p_last_fmat_ != p_last_fmat_mutable_) {
    return false;
  }
  builder_monitor_.Start("UpdatePredictionCache");

  CHECK_GT(out_preds.Size(), 0U);

  size_t n_nodes = row_set_collection_.end() - row_set_collection_.begin();

  common::BlockedSpace2d space(n_nodes, [&](size_t node) {
    return row_set_collection_[node].Size();
  }, 1024);
  CHECK_EQ(out_preds.DeviceIdx(), GenericParameter::kCpuId);
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
void QuantileHistMaker::Builder<GradientSumT>::InitSampling(const DMatrix& fmat,
                                                std::vector<GradientPair>* gpair,
                                                std::vector<size_t>* row_indices) {
  const auto& info = fmat.Info();
  auto& rnd = common::GlobalRandom();
  std::vector<GradientPair>& gpair_ref = *gpair;

#if XGBOOST_CUSTOMIZE_GLOBAL_PRNG
  std::bernoulli_distribution coin_flip(param_.subsample);
  for (size_t i = 0; i < info.num_row_; ++i) {
    if (!(gpair_ref[i].GetHess() >= 0.0f && coin_flip(rnd)) || gpair_ref[i].GetGrad() == 0.0f) {
      gpair_ref[i] = GradientPair(0);
    }
  }
#else
  const size_t nthread = this->nthread_;
  uint64_t initial_seed = rnd();

  const size_t discard_size = info.num_row_ / nthread;
  std::bernoulli_distribution coin_flip(param_.subsample);

  dmlc::OMPException exc;
  #pragma omp parallel num_threads(nthread)
  {
    exc.Run([&]() {
      const size_t tid = omp_get_thread_num();
      const size_t ibegin = tid * discard_size;
      const size_t iend = (tid == (nthread - 1)) ?
                          info.num_row_ : ibegin + discard_size;
      RandomReplace::MakeIf([&](size_t i, RandomReplace::EngineT& eng) {
        return !(gpair_ref[i].GetHess() >= 0.0f && coin_flip(eng));
      }, GradientPair(0), initial_seed, ibegin, iend, &gpair_ref);
    });
  }
  exc.Rethrow();
#endif  // XGBOOST_CUSTOMIZE_GLOBAL_PRNG
}
template<typename GradientSumT>
size_t QuantileHistMaker::Builder<GradientSumT>::GetNumberOfTrees() {
  return n_trees_;
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitData(const GHistIndexMatrix& gmat,
                                          const DMatrix& fmat,
                                          const RegTree& tree,
                                          std::vector<GradientPair>* gpair) {
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
    // initialize histogram collection
    uint32_t nbins = gmat.cut.Ptrs().back();
    hist_.Init(nbins);
    hist_local_worker_.Init(nbins);
    hist_buffer_.Init(nbins);

    // initialize histogram builder
    dmlc::OMPException exc;
#pragma omp parallel
    {
      exc.Run([&]() {
        this->nthread_ = omp_get_num_threads();
      });
    }
    exc.Rethrow();
    hist_builder_ = GHistBuilder<GradientSumT>(this->nthread_, nbins);

    std::vector<size_t>& row_indices = *row_set_collection_.Data();
    row_indices.resize(info.num_row_);
    size_t* p_row_indices = row_indices.data();
    // mark subsample and build list of member rows

    if (param_.subsample < 1.0f) {
      CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
        << "Only uniform sampling is supported, "
        << "gradient-based sampling is only support by GPU Hist.";
      builder_monitor_.Start("InitSampling");
      InitSampling(fmat, gpair, &row_indices);
      builder_monitor_.Stop("InitSampling");
      CHECK_EQ(row_indices.size(), info.num_row_);
      // We should check that the partitioning was done correctly
      // and each row of the dataset fell into exactly one of the categories
    }
    common::MemStackAllocator<bool, 128> buff(this->nthread_);
    bool* p_buff = buff.Get();
    std::fill(p_buff, p_buff + this->nthread_, false);

    const size_t block_size = info.num_row_ / this->nthread_ + !!(info.num_row_ % this->nthread_);

    #pragma omp parallel num_threads(this->nthread_)
    {
      exc.Run([&]() {
        const size_t tid = omp_get_thread_num();
        const size_t ibegin = tid * block_size;
        const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
            static_cast<size_t>(info.num_row_));

        for (size_t i = ibegin; i < iend; ++i) {
          if ((*gpair)[i].GetHess() < 0.0f) {
            p_buff[tid] = true;
            break;
          }
        }
      });
    }
    exc.Rethrow();

    bool has_neg_hess = false;
    for (int32_t tid = 0; tid < this->nthread_; ++tid) {
      if (p_buff[tid]) {
        has_neg_hess = true;
      }
    }

    if (has_neg_hess) {
      size_t j = 0;
      for (size_t i = 0; i < info.num_row_; ++i) {
        if ((*gpair)[i].GetHess() >= 0.0f) {
          p_row_indices[j++] = i;
        }
      }
      row_indices.resize(j);
    } else {
      #pragma omp parallel num_threads(this->nthread_)
      {
        exc.Run([&]() {
          const size_t tid = omp_get_thread_num();
          const size_t ibegin = tid * block_size;
          const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
              static_cast<size_t>(info.num_row_));
          for (size_t i = ibegin; i < iend; ++i) {
            p_row_indices[i] = i;
          }
        });
      }
      exc.Rethrow();
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
      data_layout_ = DataLayout::kDenseDataZeroBased;
    } else if (nbins_f0 == 0 && nrow * (ncol - 1) == nnz) {
      // dense data with one-based indexing
      data_layout_ = DataLayout::kDenseDataOneBased;
    } else {
      // sparse data
      data_layout_ = DataLayout::kSparseData;
    }
  }
  // store a pointer to the tree
  p_last_tree_ = &tree;
  if (data_layout_ == DataLayout::kDenseDataOneBased) {
    column_sampler_.Init(info.num_col_, info.feature_weigths.ConstHostVector(),
                         param_.colsample_bynode, param_.colsample_bylevel,
                         param_.colsample_bytree, true);
  } else {
    column_sampler_.Init(info.num_col_, info.feature_weigths.ConstHostVector(),
                         param_.colsample_bynode, param_.colsample_bylevel,
                         param_.colsample_bytree, false);
  }
  if (data_layout_ == DataLayout::kDenseDataZeroBased
      || data_layout_ == DataLayout::kDenseDataOneBased) {
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

  builder_monitor_.Stop("InitData");
}

// if sum of statistics for non-missing values in the node
// is equal to sum of statistics for all values:
// then - there are no missing values
// else - there are missing values
template <typename GradientSumT>
bool QuantileHistMaker::Builder<GradientSumT>::SplitContainsMissingValues(
    const GradStats e, const NodeEntry &snode) {
  if (e.GetGrad() == snode.stats.GetGrad() && e.GetHess() == snode.stats.GetHess()) {
    return false;
  } else {
    return true;
  }
}

// nodes_set - set of nodes to be processed in parallel
template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::EvaluateSplits(
                                               const std::vector<CPUExpandEntry>& nodes_set,
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

  auto evaluator = tree_evaluator_.GetEvaluator();
  // Start parallel enumeration for all tree nodes in the set and all features
  common::ParallelFor2d(space, this->nthread_, [&](size_t nid_in_set, common::Range1d r) {
    const int32_t nid = nodes_set[nid_in_set].nid;
    const auto tid = static_cast<unsigned>(omp_get_thread_num());
    GHistRowT node_hist = hist[nid];

    for (auto idx_in_feature_set = r.begin(); idx_in_feature_set < r.end(); ++idx_in_feature_set) {
      const auto fid = features_sets[nid_in_set]->ConstHostVector()[idx_in_feature_set];
      if (interaction_constraints_.Query(nid, fid)) {
        auto grad_stats = this->EnumerateSplit<+1>(
            gmat, node_hist, snode_[nid],
            &best_split_tloc_[nthread * nid_in_set + tid], fid, nid, evaluator);
        if (SplitContainsMissingValues(grad_stats, snode_[nid])) {
          this->EnumerateSplit<-1>(
              gmat, node_hist, snode_[nid],
              &best_split_tloc_[nthread * nid_in_set + tid], fid, nid,
              evaluator);
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

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::FindSplitConditions(
                                                     const std::vector<CPUExpandEntry>& nodes,
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
    for (uint32_t bound = lower_bound; bound < upper_bound; ++bound) {
      if (split_pt == gmat.cut.Values()[bound]) {
        split_cond = static_cast<int32_t>(bound);
      }
    }
    (*split_conditions)[i] = split_cond;
  }
}
template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::AddSplitsToRowSet(
                                               const std::vector<CPUExpandEntry>& nodes,
                                               RegTree* p_tree) {
  const size_t n_nodes = nodes.size();
  for (unsigned int i = 0; i < n_nodes; ++i) {
    const int32_t nid = nodes[i].nid;
    const size_t n_left = partition_builder_.GetNLeftElems(i);
    const size_t n_right = partition_builder_.GetNRightElems(i);
    CHECK_EQ((*p_tree)[nid].LeftChild() + 1, (*p_tree)[nid].RightChild());
    row_set_collection_.AddSplit(nid, (*p_tree)[nid].LeftChild(),
        (*p_tree)[nid].RightChild(), n_left, n_right);
  }
}

template <typename GradientSumT>
template <bool any_missing>
void QuantileHistMaker::Builder<GradientSumT>::ApplySplit(const std::vector<CPUExpandEntry> nodes,
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
    size_t begin = r.begin();
    const int32_t nid = nodes[node_in_set].nid;
    const size_t task_id = partition_builder_.GetTaskIdx(node_in_set, begin);
    partition_builder_.AllocateForTask(task_id);
      switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        partition_builder_.template Partition<uint8_t, any_missing>(node_in_set, nid, r,
                  split_conditions[node_in_set], column_matrix,
                  *p_tree, row_set_collection_[nid].begin);
        break;
      case common::kUint16BinsTypeSize:
        partition_builder_.template Partition<uint16_t, any_missing>(node_in_set, nid, r,
                  split_conditions[node_in_set], column_matrix,
                  *p_tree, row_set_collection_[nid].begin);
        break;
      case common::kUint32BinsTypeSize:
        partition_builder_.template Partition<uint32_t, any_missing>(node_in_set, nid, r,
                  split_conditions[node_in_set], column_matrix,
                  *p_tree, row_set_collection_[nid].begin);
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
      if (data_layout_ == DataLayout::kDenseDataZeroBased
          || data_layout_ == DataLayout::kDenseDataOneBased) {
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
    auto evaluator = tree_evaluator_.GetEvaluator();
    bst_uint parentid = tree[nid].Parent();
    snode_[nid].weight = static_cast<float>(
        evaluator.CalcWeight(parentid, param_, GradStats{snode_[nid].stats}));
    snode_[nid].root_gain = static_cast<float>(
        evaluator.CalcGain(parentid, param_, GradStats{snode_[nid].stats}));
  }
  builder_monitor_.Stop("InitNewNode");
}

// Enumerate the split values of specific feature.
// Returns the sum of gradients corresponding to the data points that contains a non-missing value
// for the particular feature fid.
template <typename GradientSumT>
template <int d_step>
GradStats QuantileHistMaker::Builder<GradientSumT>::EnumerateSplit(
    const GHistIndexMatrix &gmat, const GHistRowT &hist, const NodeEntry &snode,
    SplitEntry *p_best, bst_uint fid, bst_uint nodeID,
    TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator) const {
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
    if (e.GetHess() >= param_.min_child_weight) {
      c.SetSubstract(snode.stats, e);
      if (c.GetHess() >= param_.min_child_weight) {
        bst_float loss_chg;
        bst_float split_pt;
        if (d_step > 0) {
          // forward enumeration: split at right bound of each bin
          loss_chg = static_cast<bst_float>(
              evaluator.CalcSplitGain(param_, nodeID, fid, GradStats{e},
                                      GradStats{c}) -
              snode.root_gain);
          split_pt = cut_val[i];
          best.Update(loss_chg, fid, split_pt, d_step == -1, e, c);
        } else {
          // backward enumeration: split at left bound of each bin
          loss_chg = static_cast<bst_float>(
              evaluator.CalcSplitGain(param_, nodeID, fid, GradStats{c},
                                      GradStats{e}) -
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

template struct QuantileHistMaker::Builder<float>;
template struct QuantileHistMaker::Builder<double>;

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
