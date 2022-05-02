/*!
 * Copyright 2017-2022 by XGBoost Contributors
 * \file updater_quantile_hist.cc
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Checn, Egor Smirnov
 */
#include "./updater_quantile_hist.h"

#include <rabit/rabit.h>

#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "../common/column_matrix.h"
#include "../common/hist_util.h"
#include "../common/random.h"
#include "../common/threading_utils.h"
#include "constraints.h"
#include "hist/evaluate_splits.h"
#include "param.h"
#include "xgboost/logging.h"
#include "xgboost/tree_updater.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_quantile_hist);

void QuantileHistMaker::Configure(const Args &args) {
  param_.UpdateAllowUnknown(args);
}

void QuantileHistMaker::Update(HostDeviceVector<GradientPair> *gpair, DMatrix *dmat,
                               common::Span<HostDeviceVector<bst_node_t>> out_position,
                               const std::vector<RegTree *> &trees) {
  // rescale learning rate according to size of trees
  float lr = param_.learning_rate;
  param_.learning_rate = lr / trees.size();

  // build tree
  const size_t n_trees = trees.size();
  if (!pimpl_) {
    pimpl_.reset(new Builder(n_trees, param_, dmat, task_, ctx_));
  }

  size_t t_idx{0};
  for (auto p_tree : trees) {
    auto &t_row_position = out_position[t_idx];
    this->pimpl_->UpdateTree(gpair, dmat, p_tree, &t_row_position);
    ++t_idx;
  }

  param_.learning_rate = lr;
}

bool QuantileHistMaker::UpdatePredictionCache(const DMatrix *data,
                                              linalg::VectorView<float> out_preds) {
  if (pimpl_) {
    return pimpl_->UpdatePredictionCache(data, out_preds);
  } else {
    return false;
  }
}

template <typename BinIdxType, bool any_missing>
void QuantileHistMaker::Builder::InitRoot(
    const GHistIndexMatrix &gmat,
    DMatrix *p_fmat, RegTree *p_tree, const std::vector<GradientPair> &gpair_h,
    int *num_leaves, std::vector<CPUExpandEntry> *expand) {
  CPUExpandEntry node(RegTree::kRoot, p_tree->GetDepth(0), 0.0f);

  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();
  nodes_for_explicit_hist_build_.push_back(node);

    std::vector<std::set<uint16_t>> merged_thread_ids_set(nodes_for_explicit_hist_build_.size());
    std::vector<std::vector<uint16_t>> merged_thread_ids(nodes_for_explicit_hist_build_.size());
    for (size_t nid = 0; nid < nodes_for_explicit_hist_build_.size(); ++nid) {
      const auto &entry = nodes_for_explicit_hist_build_[nid];
      for (size_t partition_id = 0; partition_id < partitioner_.size(); ++partition_id) {
        for (size_t tid = 0; tid <
             partitioner_[partition_id].GetOptPartition().
             GetThreadIdsForNode(entry.nid).size(); ++tid) {
          merged_thread_ids_set[nid].insert(
            partitioner_[partition_id].GetOptPartition().
            GetThreadIdsForNode(entry.nid)[tid]);
        }
      }
      merged_thread_ids[nid].resize(merged_thread_ids_set[nid].size());
      std::copy(merged_thread_ids_set[nid].begin(),
                merged_thread_ids_set[nid].end(), merged_thread_ids[nid].begin());
    }


  size_t page_id = 0;

  for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
      RowPartitioner &partitioner = this->partitioner_.at(page_id);

    this->histogram_builder_->template BuildHist<BinIdxType, true>(
        page_id, gidx, p_tree,
        nodes_for_explicit_hist_build_, nodes_for_subtraction_trick_, gpair_h,
        &(partitioner.GetOptPartition()),
        &(partitioner.GetNodeAssignments()), &merged_thread_ids);
// >>>>>>> fb16e1ca... partition optimizations
    ++page_id;
  }
  {
    GradientPairPrecise grad_stat;
    if (p_fmat->IsDense()) {
      /**
       * Specialized code for dense data: For dense data (with no missing value), the sum
       * of gradient histogram is equal to snode[nid]
       */
      auto const &gmat = *(p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_)).begin());
      std::vector<uint32_t> const &row_ptr = gmat.cut.Ptrs();
      CHECK_GE(row_ptr.size(), 2);
      uint32_t const ibegin = row_ptr[0];
      uint32_t const iend = row_ptr[1];
      auto hist = this->histogram_builder_->Histogram()[RegTree::kRoot];
      auto begin = hist.data();
      for (uint32_t i = ibegin; i < iend; ++i) {
        GradientPairPrecise const &et = begin[i];
        grad_stat.Add(et.GetGrad(), et.GetHess());
      }
    } else {
      for (const GradientPair& gh : gpair_h) {
        grad_stat.Add(gh.GetGrad(), gh.GetHess());
      }
      rabit::Allreduce<rabit::op::Sum, double>(reinterpret_cast<double *>(&grad_stat), 2);
    }

    auto weight = evaluator_->InitRoot(GradStats{grad_stat});
    p_tree->Stat(RegTree::kRoot).sum_hess = grad_stat.GetHess();
    p_tree->Stat(RegTree::kRoot).base_weight = weight;
    (*p_tree)[RegTree::kRoot].SetLeaf(param_.learning_rate * weight);

    std::vector<CPUExpandEntry> entries{node};
    monitor_->Start("EvaluateSplits");
    auto ft = p_fmat->Info().feature_types.ConstHostSpan();
    for (auto const &gmat : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
      evaluator_->EvaluateSplits(histogram_builder_->Histogram(), gmat.cut, ft, *p_tree, &entries);
      break;
    }
    monitor_->Stop("EvaluateSplits");
    node = entries.front();
  }
  expand->push_back(node);
  ++(*num_leaves);
  // return node;
}

// <<<<<<< HEAD
// template <typename GradientSumT>
// void QuantileHistMaker::Builder<GradientSumT>::BuildHistogram(
//     DMatrix *p_fmat, RegTree *p_tree, std::vector<CPUExpandEntry> const &valid_candidates,
//     std::vector<GradientPair> const &gpair) {
//   std::vector<CPUExpandEntry> nodes_to_build(valid_candidates.size());
//   std::vector<CPUExpandEntry> nodes_to_sub(valid_candidates.size());

//   size_t n_idx = 0;
//   for (auto const &c : valid_candidates) {
//     auto left_nidx = (*p_tree)[c.nid].LeftChild();
//     auto right_nidx = (*p_tree)[c.nid].RightChild();
//     auto fewer_right = c.split.right_sum.GetHess() < c.split.left_sum.GetHess();

//     auto build_nidx = left_nidx;
//     auto subtract_nidx = right_nidx;
//     if (fewer_right) {
//       std::swap(build_nidx, subtract_nidx);
// =======
// template<typename GradientSumT>
void QuantileHistMaker::Builder::AddSplitsToTree(
          const std::vector<CPUExpandEntry>& expand,
          RegTree *p_tree,
          int *num_leaves,
          std::vector<CPUExpandEntry>* nodes_for_apply_split,
          std::unordered_map<uint32_t, bool>* smalest_nodes_mask_ptr,
          size_t depth, bool * is_left_small) {
  std::unordered_map<uint32_t, bool>& smalest_nodes_mask = *smalest_nodes_mask_ptr;
  const bool is_loss_guided = static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy)
                              != TrainParam::kDepthWise;
  std::vector<uint16_t> complete_node_ids;
  if (param_.max_depth == 0) {
    size_t max_nid = 0;
    int max_nid_child = 0;
    size_t it = 0;
    for (auto const& entry : expand) {
      max_nid = std::max(max_nid, static_cast<size_t>(2*entry.nid + 2));
      if (entry.IsValid(param_, *num_leaves)) {
        nodes_for_apply_split->push_back(entry);
        evaluator_->ApplyTreeSplit(entry, p_tree);
        ++(*num_leaves);
        ++it;
        max_nid_child = std::max(max_nid_child,
                                static_cast<int>(std::max((*p_tree)[entry.nid].LeftChild(),
                                (*p_tree)[entry.nid].RightChild())));
      }
    }
    (*num_leaves) -= it;
    for (auto const& entry : expand) {
      if (entry.IsValid(param_, *num_leaves)) {
        (*num_leaves)++;
        complete_node_ids.push_back((*p_tree)[entry.nid].LeftChild());
        complete_node_ids.push_back((*p_tree)[entry.nid].RightChild());
        *is_left_small = entry.split.left_sum.GetHess() <= entry.split.right_sum.GetHess();
        if (entry.split.left_sum.GetHess() <= entry.split.right_sum.GetHess() || is_loss_guided) {
          smalest_nodes_mask[(*p_tree)[entry.nid].LeftChild()] = true;
        } else {
          smalest_nodes_mask[(*p_tree)[entry.nid].RightChild()] = true;
        }
      }
    }

  } else {
    for (auto const& entry : expand) {
      if (entry.IsValid(param_, *num_leaves)) {
        nodes_for_apply_split->push_back(entry);
        evaluator_->ApplyTreeSplit(entry, p_tree);
        (*num_leaves)++;
        complete_node_ids.push_back((*p_tree)[entry.nid].LeftChild());
        complete_node_ids.push_back((*p_tree)[entry.nid].RightChild());
        *is_left_small = entry.split.left_sum.GetHess() <= entry.split.right_sum.GetHess();
        if (entry.split.left_sum.GetHess() <= entry.split.right_sum.GetHess() || is_loss_guided) {
          smalest_nodes_mask[(*p_tree)[entry.nid].LeftChild()] = true;
          smalest_nodes_mask[(*p_tree)[entry.nid].RightChild()] = false;
        } else {
          smalest_nodes_mask[(*p_tree)[entry.nid].RightChild()] = true;
          smalest_nodes_mask[ (*p_tree)[entry.nid].LeftChild()] = false;
        }
      }
// >>>>>>> fb16e1ca... partition optimizations
    }
    // nodes_to_build[n_idx] = CPUExpandEntry{build_nidx, p_tree->GetDepth(build_nidx), {}};
    // nodes_to_sub[n_idx] = CPUExpandEntry{subtract_nidx, p_tree->GetDepth(subtract_nidx), {}};
    // n_idx++;
  }
// <<<<<<< HEAD

//   size_t page_id{0};
//   auto space = ConstructHistSpace(partitioner_, nodes_to_build);
//   for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
//     histogram_builder_->BuildHist(page_id, space, gidx, p_tree,
//                                   partitioner_.at(page_id).Partitions(), nodes_to_build,
//                                   nodes_to_sub, gpair);
//     ++page_id;
//   }
// }

// template <typename GradientSumT>
// void QuantileHistMaker::Builder<GradientSumT>::ExpandTree(
//     DMatrix *p_fmat, RegTree *p_tree, const std::vector<GradientPair> &gpair_h) {
//   monitor_->Start(__func__);

//   Driver<CPUExpandEntry> driver(static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy));
//   driver.Push(this->InitRoot(p_fmat, p_tree, gpair_h));
//   bst_node_t num_leaves{1};
//   auto expand_set = driver.Pop();

//   while (!expand_set.empty()) {
//     // candidates that can be further splited.
//     std::vector<CPUExpandEntry> valid_candidates;
//     // candidaates that can be applied.
//     std::vector<CPUExpandEntry> applied;
//     int32_t depth = expand_set.front().depth + 1;
//     for (auto const& candidate : expand_set) {
//       if (!candidate.IsValid(param_, num_leaves)) {
//         continue;
//       }
//       evaluator_->ApplyTreeSplit(candidate, p_tree);
//       applied.push_back(candidate);
//       num_leaves++;
//       if (CPUExpandEntry::ChildIsValid(param_, depth, num_leaves)) {
//         valid_candidates.emplace_back(candidate);
//       }
//     }

//     monitor_->Start("UpdatePosition");
//     size_t page_id{0};
//     for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
//       partitioner_.at(page_id).UpdatePosition(ctx_, page, applied, p_tree);
//       ++page_id;
//     }
//     monitor_->Stop("UpdatePosition");

//     std::vector<CPUExpandEntry> best_splits;
//     if (!valid_candidates.empty()) {
//       this->BuildHistogram(p_fmat, p_tree, valid_candidates, gpair_h);
//       auto const &tree = *p_tree;
//       for (auto const &candidate : valid_candidates) {
//         int left_child_nidx = tree[candidate.nid].LeftChild();
//         int right_child_nidx = tree[candidate.nid].RightChild();
//         CPUExpandEntry l_best{left_child_nidx, depth, 0.0};
//         CPUExpandEntry r_best{right_child_nidx, depth, 0.0};
//         best_splits.push_back(l_best);
//         best_splits.push_back(r_best);
//       }
//       auto const &histograms = histogram_builder_->Histogram();
//       auto ft = p_fmat->Info().feature_types.ConstHostSpan();
//       for (auto const &gmat : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
//         evaluator_->EvaluateSplits(histograms, gmat.cut, ft, *p_tree, &best_splits);
//         break;
// =======
  child_node_ids_ = complete_node_ids;
}

// Split nodes to 2 sets depending on amount of rows in each node
// Histograms for small nodes will be built explicitly
// Histograms for big nodes will be built by 'Subtraction Trick'
// Exception: in distributed setting, we always build the histogram for the left child node
//    and use 'Subtraction Trick' to built the histogram for the right child node.
//    This ensures that the workers operate on the same set of tree nodes.
// template <typename GradientSumT>
// <<<<<<< HEAD
void QuantileHistMaker::Builder::LeafPartition(
    RegTree const &tree,  // common::Span<GradientPair const> gpair,
    std::vector<bst_node_t> *p_out_position) {
  monitor_->Start(__func__);
  if (!task_.UpdateTreeLeaf()) {
    return;
  }
  for (auto const &part : partitioner_) {
    part.LeafPartition(ctx_, tree, p_out_position);
  }
  monitor_->Stop(__func__);
}

// template <typename GradientSumT>
// void QuantileHistMaker::Builder<GradientSumT>::ExpandTree(
//     DMatrix *p_fmat, RegTree *p_tree, const std::vector<GradientPair> &gpair_h,
//     HostDeviceVector<bst_node_t> *p_out_position) {
//   monitor_->Start(__func__);
// =======
void QuantileHistMaker::Builder::SplitSiblings(
    const std::vector<CPUExpandEntry> &nodes_for_apply_split,
    std::vector<CPUExpandEntry> *nodes_to_evaluate, RegTree *p_tree) {
  monitor_->Start("SplitSiblings");
  RowPartitioner &partitioner = this->partitioner_.front();

  // auto const& row_set_collection = this->partitioner_.front().Partitions();
  for (auto const& entry : nodes_for_apply_split) {
    int nid = entry.nid;

    const int cleft = (*p_tree)[nid].LeftChild();
    const int cright = (*p_tree)[nid].RightChild();
    const CPUExpandEntry left_node = CPUExpandEntry(cleft, p_tree->GetDepth(cleft), 0.0);
    const CPUExpandEntry right_node = CPUExpandEntry(cright, p_tree->GetDepth(cright), 0.0);
    nodes_to_evaluate->push_back(left_node);
    nodes_to_evaluate->push_back(right_node);
    bool is_loss_guide =  static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy) ==
                          TrainParam::kDepthWise ? false : true;
    // if (is_loss_guide) {
    //   if (partitioner.GetOptPartition().GetPartitionSize(cleft) <=
    //       partitioner.GetOptPartition().GetPartitionSize(cright)) {
    //     nodes_for_explicit_hist_build_.push_back(left_node);
    //     nodes_for_subtraction_trick_.push_back(right_node);
    //   } else {
    //     nodes_for_explicit_hist_build_.push_back(right_node);
    //     nodes_for_subtraction_trick_.push_back(left_node);
    //   }
    // } else {
      if (entry.split.left_sum.GetHess() <= entry.split.right_sum.GetHess()) {
        nodes_for_explicit_hist_build_.push_back(left_node);
        nodes_for_subtraction_trick_.push_back(right_node);
        // *is_left_small = true;
      } else {
        nodes_for_explicit_hist_build_.push_back(right_node);
        nodes_for_subtraction_trick_.push_back(left_node);
        // *is_left_small = false;
      }
    // }
  }
  monitor_->Stop("SplitSiblings");
}
// >>>>>>> 0755d8b2... partition optimizations

// template<typename GradientSumT>
template <typename BinIdxType, bool any_missing>
void QuantileHistMaker::Builder::ExpandTree(
    const GHistIndexMatrix& gmat,
    const common::ColumnMatrix& column_matrix,
    DMatrix* p_fmat,
    RegTree* p_tree,
    const std::vector<GradientPair>& gpair_h,
    HostDeviceVector<bst_node_t> *p_out_position) {
  monitor_->Start("ExpandTree");
  int num_leaves = 0;
  split_conditions_.clear();
  split_ind_.clear();
  Driver<CPUExpandEntry> driver(static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy));
// <<<<<<< HEAD
//   driver.Push(this->InitRoot(p_fmat, p_tree, gpair_h));
//   auto const &tree = *p_tree;
//   bst_node_t num_leaves{1};
//   auto expand_set = driver.Pop();

//   while (!expand_set.empty()) {
//     // candidates that can be further splited.
//     std::vector<CPUExpandEntry> valid_candidates;
//     // candidaates that can be applied.
//     std::vector<CPUExpandEntry> applied;
//     int32_t depth = expand_set.front().depth + 1;
//     for (auto const& candidate : expand_set) {
//       if (!candidate.IsValid(param_, num_leaves)) {
//         continue;
//       }
//       evaluator_->ApplyTreeSplit(candidate, p_tree);
//       applied.push_back(candidate);
//       num_leaves++;
//       if (CPUExpandEntry::ChildIsValid(param_, depth, num_leaves)) {
//         valid_candidates.emplace_back(candidate);
//       }
// =======
  std::vector<CPUExpandEntry> expand;
    size_t page_id{0};
  std::vector<size_t>& row_indices = *row_set_collection_.Data();
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
      // partitioner_.at(page_id).UpdatePosition(ctx_, page, applied, p_tree);
    const size_t size_threads = row_indices.size() == 0 ?
                              (page.row_ptr.size() - 1) : row_indices.size();
    RowPartitioner &partitioner = this->partitioner_.at(page_id);
    const_cast<common::OptPartitionBuilder&>(partitioner.GetOptPartition()).SetSlice(0,
                                           0, size_threads);
      ++page_id;
// >>>>>>> 0755d8b2... partition optimizations
    }

  // node_ids_.resize(size_threads, 0);
  bool is_loss_guide = static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy) ==
                       TrainParam::kDepthWise ? false : true;

  InitRoot<BinIdxType, any_missing>(gmat, p_fmat, p_tree, gpair_h, &num_leaves, &expand);
  driver.Push(expand[0]);
  child_node_ids_.clear();
  child_node_ids_.emplace_back(0);
  int32_t depth = 0;
  while (!driver.IsEmpty()) {
    std::unordered_map<uint32_t, bool> smalest_nodes_mask;
    expand = driver.Pop();
    depth = expand[0].depth + 1;
    std::vector<CPUExpandEntry> nodes_for_apply_split;
    std::vector<CPUExpandEntry> nodes_to_evaluate;
    nodes_for_explicit_hist_build_.clear();
    nodes_for_subtraction_trick_.clear();
    bool is_left_small = false;
    AddSplitsToTree(expand, p_tree, &num_leaves, &nodes_for_apply_split,
                    &smalest_nodes_mask, depth, &is_left_small);

    if (nodes_for_apply_split.size() != 0) {
      monitor_->Start("ApplySplit");
    size_t page_id{0};
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
//       partitioner_.at(page_id).UpdatePosition(ctx_, page, applied, p_tree);
      RowPartitioner &partitioner = this->partitioner_.at(page_id);
      if (is_loss_guide) {
        if (page.cut.HasCategorical()) {
          partitioner.UpdatePosition<any_missing, BinIdxType, true, true>(this->ctx_, page,
                      // column_matrix,
                      nodes_for_apply_split, p_tree,
                      depth,
                      &smalest_nodes_mask,
                      is_loss_guide,
                      &split_conditions_,
                      &split_ind_, param_.max_depth,
                      &child_node_ids_, is_left_small, true);
        } else {
          partitioner.UpdatePosition<any_missing, BinIdxType, true, false>(this->ctx_, page,
                      // column_matrix,
                      nodes_for_apply_split, p_tree,
                      depth,
                      &smalest_nodes_mask,
                      is_loss_guide,
                      &split_conditions_,
                      &split_ind_, param_.max_depth,
                      &child_node_ids_, is_left_small, true);
        }
      } else {
        if (page.cut.HasCategorical()) {
          partitioner.UpdatePosition<any_missing, BinIdxType, false, true>(this->ctx_, page,
                      // column_matrix,
                      nodes_for_apply_split, p_tree,
                      depth,
                      &smalest_nodes_mask,
                      is_loss_guide,
                      &split_conditions_,
                      &split_ind_, param_.max_depth,
                      &child_node_ids_, is_left_small, true);
        } else {
          partitioner.UpdatePosition<any_missing, BinIdxType, false, false>(this->ctx_, page,
                      // column_matrix,
                      nodes_for_apply_split, p_tree,
                      depth,
                      &smalest_nodes_mask,
                      is_loss_guide,
                      &split_conditions_,
                      &split_ind_, param_.max_depth,
                      &child_node_ids_, is_left_small, true);
        }
      }
        ++page_id;
    }
// <<<<<<< HEAD
//     monitor_->Stop("UpdatePosition");

//     std::vector<CPUExpandEntry> best_splits;
//     if (!valid_candidates.empty()) {
//       this->BuildHistogram(p_fmat, p_tree, valid_candidates, gpair_h);
//       for (auto const &candidate : valid_candidates) {
//         int left_child_nidx = tree[candidate.nid].LeftChild();
//         int right_child_nidx = tree[candidate.nid].RightChild();
//         CPUExpandEntry l_best{left_child_nidx, depth, 0.0};
//         CPUExpandEntry r_best{right_child_nidx, depth, 0.0};
//         best_splits.push_back(l_best);
//         best_splits.push_back(r_best);
// =======

      monitor_->Stop("ApplySplit");
      // bool is_left_small = false;
      SplitSiblings(nodes_for_apply_split, &nodes_to_evaluate, p_tree);
      if (param_.max_depth == 0 || depth < param_.max_depth) {
        size_t i = 0;
        monitor_->Start("BuildHist");
    std::vector<std::set<uint16_t>> merged_thread_ids_set(nodes_for_explicit_hist_build_.size());
    std::vector<std::vector<uint16_t>> merged_thread_ids(nodes_for_explicit_hist_build_.size());
    for (size_t nid = 0; nid < nodes_for_explicit_hist_build_.size(); ++nid) {
      const auto &entry = nodes_for_explicit_hist_build_[nid];
      for (size_t partition_id = 0; partition_id < partitioner_.size(); ++partition_id) {
        for (size_t tid = 0; tid <
             partitioner_[partition_id].GetOptPartition().
             GetThreadIdsForNode(entry.nid).size(); ++tid) {
          merged_thread_ids_set[nid].insert(
            partitioner_[partition_id].GetOptPartition().
            GetThreadIdsForNode(entry.nid)[tid]);
        }
      }
      merged_thread_ids[nid].resize(merged_thread_ids_set[nid].size());
      std::copy(merged_thread_ids_set[nid].begin(),
                merged_thread_ids_set[nid].end(), merged_thread_ids[nid].begin());
    }


        for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
        // feature_values_ = gidx.cut;

          RowPartitioner &partitioner = this->partitioner_.at(i);
          this->histogram_builder_->template BuildHist<BinIdxType, false>(
              i, gidx, p_tree,
              nodes_for_explicit_hist_build_, nodes_for_subtraction_trick_,
              gpair_h, &(partitioner.GetOptPartition()),
              &(partitioner.GetNodeAssignments()), &merged_thread_ids);
          ++i;
        }
        monitor_->Stop("BuildHist");
        monitor_->Start("EvaluateSplits");
        auto ft = p_fmat->Info().feature_types.ConstHostSpan();
        evaluator_->EvaluateSplits(this->histogram_builder_->Histogram(),
                                  feature_values_, ft, *p_tree, &nodes_to_evaluate);
        monitor_->Stop("EvaluateSplits");
// >>>>>>> 0755d8b2... partition optimizations
      }
      for (size_t i = 0; i < nodes_for_apply_split.size(); ++i) {
        CPUExpandEntry left_node = nodes_to_evaluate.at(i * 2 + 0);
        CPUExpandEntry right_node = nodes_to_evaluate.at(i * 2 + 1);
        driver.Push(left_node);
        driver.Push(right_node);
// >>>>>>> fb16e1ca... partition optimizations
      }
    }
    // driver.Push(best_splits.begin(), best_splits.end());
    // expand_set = driver.Pop();
  }

  auto &h_out_position = p_out_position->HostVector();
  this->LeafPartition(*p_tree, &h_out_position);
  monitor_->Stop(__func__);
}

void QuantileHistMaker::Builder::UpdateTree(HostDeviceVector<GradientPair> *gpair, DMatrix *p_fmat,
                                            RegTree *p_tree,
                                            HostDeviceVector<bst_node_t> *p_out_position) {
  monitor_->Start(__func__);

  std::vector<GradientPair> *gpair_ptr = &(gpair->HostVector());
  // in case 'num_parallel_trees != 1' no posibility to change initial gpair
  if (GetNumberOfTrees() != 1) {
    gpair_local_.resize(gpair_ptr->size());
    gpair_local_ = *gpair_ptr;
    gpair_ptr = &gpair_local_;
  }
  auto it = p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_)).begin();
  const  GHistIndexMatrix& gmat = *(it.Page());
  const common::ColumnMatrix& column_matrix = gmat.Transpose();
// <<<<<<< HEAD
//   this->InitData(p_fmat, *p_tree, gpair_ptr);

//   ExpandTree(p_fmat, p_tree, *gpair_ptr);
// =======
const bool any_missing = column_matrix.AnyMissing();
  switch (column_matrix.GetTypeSize()) {
    case common::kUint8BinsTypeSize:
      this->InitData<uint8_t>(gmat, column_matrix, *p_fmat, *p_tree, gpair_ptr);
      if (any_missing) {
        ExpandTree<uint8_t, true>(gmat, column_matrix, p_fmat, p_tree,
                                  *gpair_ptr, p_out_position);
      } else {
        ExpandTree<uint8_t, false>(gmat, column_matrix, p_fmat, p_tree,
                                   *gpair_ptr, p_out_position);
      }
      break;
    case common::kUint16BinsTypeSize:
      this->InitData<uint16_t>(gmat, column_matrix, *p_fmat, *p_tree, gpair_ptr);
      if (any_missing) {
        ExpandTree<uint16_t, true>(gmat, column_matrix, p_fmat, p_tree,
                                   *gpair_ptr, p_out_position);
      } else {
        ExpandTree<uint16_t, false>(gmat, column_matrix, p_fmat, p_tree,
                                    *gpair_ptr, p_out_position);
      }
      break;
    case common::kUint32BinsTypeSize:
      this->InitData<uint32_t>(gmat, column_matrix, *p_fmat, *p_tree, gpair_ptr);
      if (any_missing) {
        ExpandTree<uint32_t, true>(gmat, column_matrix, p_fmat, p_tree,
                                   *gpair_ptr, p_out_position);
      } else {
        ExpandTree<uint32_t, false>(gmat, column_matrix, p_fmat, p_tree,
                                    *gpair_ptr, p_out_position);
      }
      break;
    default:
      CHECK(false);  // no default behavior
  }

// <<<<<<< HEAD
//   this->InitData(p_fmat, *p_tree, gpair_ptr);

//   ExpandTree(p_fmat, p_tree, *gpair_ptr, p_out_position);
// =======
  // pruner_->Update(gpair, p_fmat, std::vector<RegTree*>{p_tree});
// >>>>>>> fb16e1ca... partition optimizations

// >>>>>>> 0755d8b2... partition optimizations
  monitor_->Stop(__func__);
}

bool QuantileHistMaker::Builder::UpdatePredictionCache(DMatrix const *data,
                                                       linalg::VectorView<float> out_preds) const {
  // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
  // conjunction with Update().
  if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_) {
    return false;
  }
// <<<<<<< HEAD
//   monitor_->Start(__func__);
//   CHECK_EQ(out_preds.Size(), data->Info().num_row_);
//   UpdatePredictionCacheImpl(ctx_, p_last_tree_, partitioner_, *evaluator_, param_, out_preds);
//   monitor_->Stop(__func__);
// =======
  monitor_->Start("UpdatePredictionCache");
  CHECK_GT(out_preds.Size(), 0U);
    size_t page_id{0};
    size_t page_disp = 0;
    for (auto const &page :
         const_cast<DMatrix*>(data)->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
      RowPartitioner &partitioner =
      const_cast<RowPartitioner&>(this->partitioner_.at(page_id));
    ++page_id;
  // RowPartitioner &partitioner = const_cast<RowPartitioner&>(this->partitioner_.front());
  common::BlockedSpace2d space(1, [&](size_t node) {
    return partitioner.GetNodeAssignments().size();
  }, 1024);
    common::ParallelFor2d(space, this->ctx_->Threads(), [&](size_t node, common::Range1d r) {
      int tid = omp_get_thread_num();
      for (size_t it = r.begin(); it <  r.end(); ++it) {
        bst_float leaf_value;
        // if a node is marked as deleted by the pruner, traverse upward to locate
        // a non-deleted leaf.
        int nid = partitioner.GetNodeAssignments()[it];
        if ((*p_last_tree_)[nid].IsDeleted()) {
          while ((*p_last_tree_)[nid].IsDeleted()) {
            nid = (*p_last_tree_)[nid].Parent();
          }
          CHECK((*p_last_tree_)[nid].IsLeaf());
        }
        leaf_value = (*p_last_tree_)[nid].LeafValue();
        out_preds(it + page_disp) += leaf_value;
      }
    });
    page_disp += partitioner.GetNodeAssignments().size();
    }
  monitor_->Stop("UpdatePredictionCache");
// >>>>>>> fb16e1ca... partition optimizations

      //     out_preds(it + page_disp) += leaf_value;
      //   }
      // });
      // page_disp += prt.GetNodeAssignments().size();


  return true;
}

void QuantileHistMaker::Builder::InitSampling(const DMatrix &fmat,
                                              std::vector<GradientPair> *gpair) {
  monitor_->Start(__func__);
  const auto &info = fmat.Info();
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
  uint64_t initial_seed = rnd();

  auto n_threads = static_cast<size_t>(ctx_->Threads());
  const size_t discard_size = info.num_row_ / n_threads;
  std::bernoulli_distribution coin_flip(param_.subsample);

  dmlc::OMPException exc;
  #pragma omp parallel num_threads(n_threads)
  {
    exc.Run([&]() {
      const size_t tid = omp_get_thread_num();
      const size_t ibegin = tid * discard_size;
      const size_t iend = (tid == (n_threads - 1)) ? info.num_row_ : ibegin + discard_size;
      RandomReplace::MakeIf([&](size_t i, RandomReplace::EngineT& eng) {
        return !(gpair_ref[i].GetHess() >= 0.0f && coin_flip(eng));
      }, GradientPair(0), initial_seed, ibegin, iend, &gpair_ref);
    });
  }
  exc.Rethrow();
#endif  // XGBOOST_CUSTOMIZE_GLOBAL_PRNG
  monitor_->Stop(__func__);
}
size_t QuantileHistMaker::Builder::GetNumberOfTrees() { return n_trees_; }

// template <typename GradientSumT>
// <<<<<<< HEAD
// void QuantileHistMaker::Builder<GradientSumT>::InitData(DMatrix *fmat, const RegTree &tree,
//                                                         std::vector<GradientPair> *gpair) {
//   monitor_->Start(__func__);
//   const auto& info = fmat->Info();

//   {
//     size_t page_id{0};
//     int32_t n_total_bins{0};
//     partitioner_.clear();
//     for (auto const &page : fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
//       if (n_total_bins == 0) {
//         n_total_bins = page.cut.TotalBins();
//       } else {
//         CHECK_EQ(n_total_bins, page.cut.TotalBins());
//       }
//       partitioner_.emplace_back(page.Size(), page.base_rowid, this->ctx_->Threads());
//       ++page_id;
//     }
//     histogram_builder_->Reset(n_total_bins, HistBatch(param_), ctx_->Threads(), page_id,
//                               rabit::IsDistributed());
// =======
template <typename BinIdxType>
void QuantileHistMaker::Builder::InitData(const GHistIndexMatrix& gmat,
                                          const common::ColumnMatrix& column_matrix,
                                          const DMatrix& fmat,
                                          const RegTree& tree,
                                          std::vector<GradientPair>* gpair) {
  monitor_->Start("InitData");
  const auto& info = fmat.Info();

  {
    // initialize histogram collection
    // uint32_t nbins = gmat.cut.Ptrs().back();
    // initialize histogram builder
    dmlc::OMPException exc;
    exc.Rethrow();
// >>>>>>> fb16e1ca... partition optimizations

    if (param_.subsample < 1.0f) {
      CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
          << "Only uniform sampling is supported, "
          << "gradient-based sampling is only support by GPU Hist.";
      InitSampling(fmat, gpair);
    }
// <<<<<<< HEAD
//   }

// =======
    const bool is_lossguide = static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy) !=
                              TrainParam::kDepthWise;

    size_t page_id{0};
    int32_t n_total_bins{0};
    // partitioner_.clear();
    if (!partition_is_initiated_) {
      partitioner_.clear();
    }
  // for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {

    for (auto const &page :
      const_cast<DMatrix&>(fmat).GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
      if (n_total_bins == 0) {
        n_total_bins = page.cut.TotalBins();
        feature_values_ = page.cut;
      } else {
        CHECK_EQ(n_total_bins, page.cut.TotalBins());
      }
      // partitioner_.emplace_back(page.Size(), page.base_rowid, this->ctx_->Threads());
      if (!partition_is_initiated_) {
        // partitioner_.clear();
        partitioner_.emplace_back(this->ctx_, page,
                                  &tree, param_.max_depth, is_lossguide);
      } else {
        partitioner_[page_id].Reset(this->ctx_, page,
                                   &tree,
                                   param_.max_depth, is_lossguide);
      }
      ++page_id;
    }
    partition_is_initiated_ = true;
    this->histogram_builder_->Reset(n_total_bins, param_.max_bin, this->ctx_->Threads(), page_id,
                                    param_.max_depth, rabit::IsDistributed());



    const size_t block_size = common::GetBlockSize(info.num_row_, this->ctx_->Threads());
  }
  // {
  //   /* determine layout of data */
  //   const size_t nrow = info.num_row_;
  //   const size_t ncol = info.num_col_;
  //   const size_t nnz = info.num_nonzero_;
  //   // number of discrete bins for feature 0
  //   const uint32_t nbins_f0 = gmat.cut.Ptrs()[1] - gmat.cut.Ptrs()[0];
  //   // if (nrow * ncol == nnz) {
  //   //   // dense data with zero-based indexing
  //   //   data_layout_ = DataLayout::kDenseDataZeroBased;
  //   // } else if (nbins_f0 == 0 && nrow * (ncol - 1) == nnz) {
  //   //   // dense data with one-based indexing
  //   //   data_layout_ = DataLayout::kDenseDataOneBased;
  //   // } else {
  //   //   // sparse data
  //   //   data_layout_ = DataLayout::kSparseData;
  //   // }
  // }
// >>>>>>> fb16e1ca... partition optimizations
  // store a pointer to the tree
  p_last_tree_ = &tree;
  evaluator_.reset(
      new HistEvaluator<CPUExpandEntry>{param_, info, this->ctx_->Threads(), column_sampler_});

  monitor_->Stop(__func__);
}

// template struct QuantileHistMaker::Builder<float>;
// template struct QuantileHistMaker::Builder<double>;

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMaker, "grow_quantile_histmaker")
    .describe("Grow tree using quantized histogram.")
    .set_body([](GenericParameter const *ctx, ObjInfo task) {
      return new QuantileHistMaker(ctx, task);
    });
}  // namespace tree
}  // namespace xgboost
