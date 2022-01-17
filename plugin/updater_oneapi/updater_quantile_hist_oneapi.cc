/*!
 * Copyright 2017-2021 by Contributors
 * \file updater_quantile_hist_oneapi.cc
 */
#include <dmlc/timer.h>
#include <rabit/rabit.h>

#include "xgboost/logging.h"
#include "xgboost/tree_updater.h"

#include "./updater_quantile_hist_oneapi.h"
#include "data_oneapi.h"

namespace xgboost {
namespace tree {

using sycl::ext::oneapi::plus;
using sycl::ext::oneapi::minimum;
using sycl::ext::oneapi::maximum;

DMLC_REGISTRY_FILE_TAG(updater_quantile_hist_oneapi);

DMLC_REGISTER_PARAMETER(OneAPIHistMakerTrainParam);

void QuantileHistMakerOneAPI::Configure(const Args& args) {
  GenericParameter param;
  param.UpdateAllowUnknown(args);

  bool is_cpu = false;
  std::vector<sycl::device> devices = sycl::device::get_devices();
  for (size_t i = 0; i < devices.size(); i++)
  {
    LOG(INFO) << "device_id = " << i << ", name = " << devices[i].get_info<sycl::info::device::name>();
  }
  if (param.device_id != GenericParameter::kDefaultId) {
    int n_devices = (int)devices.size();
    CHECK_LT(param.device_id, n_devices);
    is_cpu = devices[param.device_id].is_cpu() | devices[param.device_id].is_host();
  }
  LOG(INFO) << "device_id = " << param.device_id << ", is_cpu = " << int(is_cpu);

  if (is_cpu)
  {
    updater_backend_.reset(TreeUpdater::Create("grow_quantile_histmaker", tparam_));
    updater_backend_->Configure(args);
  }
  else
  {
    updater_backend_.reset(TreeUpdater::Create("grow_quantile_histmaker_oneapi_gpu", tparam_));
    updater_backend_->Configure(args);
  }
}

void QuantileHistMakerOneAPI::Update(HostDeviceVector<GradientPair> *gpair,
                                     DMatrix *dmat,
                                     const std::vector<RegTree *> &trees) {
  updater_backend_->Update(gpair, dmat, trees);
}

bool QuantileHistMakerOneAPI::UpdatePredictionCache(
    const DMatrix* data,
    HostDeviceVector<bst_float>* out_preds) {
  return updater_backend_->UpdatePredictionCache(data, out_preds);
}

void GPUQuantileHistMakerOneAPI::Configure(const Args& args) {
  GenericParameter param;
  param.UpdateAllowUnknown(args);

  std::vector<sycl::device> devices = sycl::device::get_devices();

  if (param.device_id != GenericParameter::kDefaultId) {
    qu_ = sycl::queue(devices[param.device_id]);
  } else {
    sycl::default_selector selector;
    qu_ = sycl::queue(selector);
  }

  // initialize pruner
  if (!pruner_) {
    pruner_.reset(TreeUpdater::Create("prune", tparam_));
  }
  pruner_->Configure(args);
  param_.UpdateAllowUnknown(args);
  hist_maker_param_.UpdateAllowUnknown(args);
}

template<typename GradientSumT>
void GPUQuantileHistMakerOneAPI::SetBuilder(std::unique_ptr<Builder<GradientSumT>>* builder,
                                            DMatrix *dmat) {
  builder->reset(new Builder<GradientSumT>(
                qu_,
                param_,
                std::move(pruner_),
                int_constraint_, dmat));
  if (rabit::IsDistributed()) {
    (*builder)->SetHistSynchronizer(new DistributedHistSynchronizerOneAPI<GradientSumT>());
    (*builder)->SetHistRowsAdder(new DistributedHistRowsAdderOneAPI<GradientSumT>());
  } else {
    (*builder)->SetHistSynchronizer(new BatchHistSynchronizerOneAPI<GradientSumT>());
    (*builder)->SetHistRowsAdder(new BatchHistRowsAdderOneAPI<GradientSumT>());
  }
}

template<typename GradientSumT>
void GPUQuantileHistMakerOneAPI::CallBuilderUpdate(const std::unique_ptr<Builder<GradientSumT>>& builder,
                                                   HostDeviceVector<GradientPair> *gpair,
                                                   DMatrix *dmat,
                                                   const std::vector<RegTree *> &trees) {
  for (auto tree : trees) {
    builder->Update(gmat_, gpair, dmat, tree);
  }
}
void GPUQuantileHistMakerOneAPI::Update(HostDeviceVector<GradientPair> *gpair,
                                        DMatrix *dmat,
                                        const std::vector<RegTree *> &trees) {
  if (dmat != p_last_dmat_ || is_gmat_initialized_ == false) {
    updater_monitor_.Start("GmatInitialization");
    DeviceMatrixOneAPI dmat_device(qu_, dmat);
    gmat_.Init(qu_, dmat_device, static_cast<uint32_t>(param_.max_bin));
    updater_monitor_.Stop("GmatInitialization");
    is_gmat_initialized_ = true;
  }
  // rescale learning rate according to size of trees
  float lr = param_.learning_rate;
  param_.learning_rate = lr / trees.size();
  int_constraint_.Configure(param_, dmat->Info().num_col_);
  // build tree
  bool has_double_support = qu_.get_device().has(sycl::aspect::fp64);
  if (hist_maker_param_.single_precision_histogram || !has_double_support) {
    if (!hist_maker_param_.single_precision_histogram) {
      LOG(WARNING) << "Target device doesn't support fp64, using single_precision_histogram=True";
    }
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

bool GPUQuantileHistMakerOneAPI::UpdatePredictionCache(const DMatrix* data,
                                                       HostDeviceVector<bst_float>* out_preds) {
  if (param_.subsample < 1.0f) {
    return false;
  } else {
    bool has_double_support = qu_.get_device().has(sycl::aspect::fp64);
    if ((hist_maker_param_.single_precision_histogram || !has_double_support) && float_builder_) {
        return float_builder_->UpdatePredictionCache(data, out_preds);
    } else if (double_builder_) {
        return double_builder_->UpdatePredictionCache(data, out_preds);
    } else {
       return false;
    }
  }
}

template <typename GradientSumT>
void BatchHistSynchronizerOneAPI<GradientSumT>::SyncHistograms(BuilderT *builder,
                                                               std::vector<int>& sync_ids,
                                                               RegTree *p_tree) {
  builder->builder_monitor_.Start("SyncHistograms");
  const size_t nbins = builder->hist_builder_.GetNumBins();

  for (int i = 0; i < builder->nodes_for_explicit_hist_build_.size(); i++) {
    const auto entry = builder->nodes_for_explicit_hist_build_[i];
    auto this_hist = builder->hist_[entry.nid];

    if (!(*p_tree)[entry.nid].IsRoot() && entry.sibling_nid > -1) {
      const size_t parent_id = (*p_tree)[entry.nid].Parent();
      auto parent_hist = builder->hist_[parent_id];
      auto sibling_hist = builder->hist_[entry.sibling_nid];
      common::SubtractionHist(builder->qu_, sibling_hist, parent_hist, this_hist, nbins);
    }
  }
  builder->builder_monitor_.Stop("SyncHistograms");
}

template <typename GradientSumT>
void DistributedHistSynchronizerOneAPI<GradientSumT>::SyncHistograms(BuilderT* builder,
                                                                     std::vector<int>& sync_ids,
                                                                     RegTree *p_tree) {
  builder->builder_monitor_.Start("SyncHistograms");
  const size_t nbins = builder->hist_builder_.GetNumBins();
  for (int node = 0; node < builder->nodes_for_explicit_hist_build_.size(); node++) {
    const auto entry = builder->nodes_for_explicit_hist_build_[node];
    auto this_hist = builder->hist_[entry.nid];
    // Store posible parent node
    auto this_local = builder->hist_local_worker_[entry.nid];
    common::CopyHist(builder->qu_, this_local, this_hist, nbins);

    if (!(*p_tree)[entry.nid].IsRoot() && entry.sibling_nid > -1) {
      const size_t parent_id = (*p_tree)[entry.nid].Parent();
      auto parent_hist = builder->hist_local_worker_[parent_id];
      auto sibling_hist = builder->hist_[entry.sibling_nid];
      common::SubtractionHist(builder->qu_, sibling_hist, parent_hist, this_hist, nbins);
      // Store posible parent node
      auto sibling_local = builder->hist_local_worker_[entry.sibling_nid];
      common::CopyHist(builder->qu_, sibling_local, sibling_hist, nbins);
    }
  }
  builder->ReduceHists(sync_ids, nbins);

  ParallelSubtractionHist(builder, builder->nodes_for_explicit_hist_build_, p_tree);
  ParallelSubtractionHist(builder, builder->nodes_for_subtraction_trick_, p_tree);

  builder->builder_monitor_.Stop("SyncHistograms");
}

template <typename GradientSumT>
void DistributedHistSynchronizerOneAPI<GradientSumT>::ParallelSubtractionHist(
    BuilderT* builder,
    const std::vector<ExpandEntryT>& nodes,
    const RegTree * p_tree) {
  const size_t nbins = builder->hist_builder_.GetNumBins();
  for (int node = 0; node < nodes.size(); node++) {
    const auto entry = nodes[node];
    if (!((*p_tree)[entry.nid].IsLeftChild())) {
      auto this_hist = builder->hist_[entry.nid];

      if (!(*p_tree)[entry.nid].IsRoot() && entry.sibling_nid > -1) {
        auto parent_hist = builder->hist_[(*p_tree)[entry.nid].Parent()];
        auto sibling_hist = builder->hist_[entry.sibling_nid];
        common::SubtractionHist(builder->qu_, this_hist, parent_hist, sibling_hist, nbins);
      }
    }
  }
}

template <typename GradientSumT>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::ReduceHists(std::vector<int>& sync_ids, size_t nbins) {
  std::vector<GradientPairT> reduce_buffer(sync_ids.size() * nbins);
  for (size_t i = 0; i < sync_ids.size(); i++) {
    auto this_hist = hist_[sync_ids[i]];
    const GradientPairT* psrc = reinterpret_cast<const GradientPairT*>(this_hist.DataConst());
    std::copy(psrc, psrc + nbins, reduce_buffer.begin() + i * nbins);
  }
  histred_.Allreduce(reduce_buffer.data(), nbins * sync_ids.size());
  for (size_t i = 0; i < sync_ids.size(); i++) {
    auto this_hist = hist_[sync_ids[i]];
    GradientPairT* psrc = reinterpret_cast<GradientPairT*>(this_hist.Data());
    std::copy(reduce_buffer.begin() + i * nbins, reduce_buffer.begin() + (i + 1) * nbins, psrc);
  }
}

template <typename GradientSumT>
void BatchHistRowsAdderOneAPI<GradientSumT>::AddHistRows(BuilderT *builder,
                                                         std::vector<int>& sync_ids,
                                                         RegTree *p_tree) {
  builder->builder_monitor_.Start("AddHistRows");

  for (auto const& entry : builder->nodes_for_explicit_hist_build_) {
    int nid = entry.nid;
    builder->hist_.AddHistRow(nid);
  }

  for (auto const& node : builder->nodes_for_subtraction_trick_) {
    builder->hist_.AddHistRow(node.nid);
  }

  builder->builder_monitor_.Stop("AddHistRows");
}

template <typename GradientSumT>
void DistributedHistRowsAdderOneAPI<GradientSumT>::AddHistRows(BuilderT *builder,
                                                               std::vector<int>& sync_ids,
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
  sync_ids.clear();
  for (auto const& nid : merged_node_ids) {
    if ((*p_tree)[nid].IsLeftChild()) {
      builder->hist_.AddHistRow(nid);
      builder->hist_local_worker_.AddHistRow(nid);
      sync_ids.push_back(nid);
    }
  }
  for (auto const& nid : merged_node_ids) {
    if (!((*p_tree)[nid].IsLeftChild())) {
      builder->hist_.AddHistRow(nid);
      builder->hist_local_worker_.AddHistRow(nid);
    }
  }
  builder->builder_monitor_.Stop("AddHistRows");
}

template <typename GradientSumT>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::SetHistSynchronizer(
    HistSynchronizerOneAPI<GradientSumT> *sync) {
  hist_synchronizer_.reset(sync);
}

template <typename GradientSumT>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::SetHistRowsAdder(
    HistRowsAdderOneAPI<GradientSumT> *adder) {
  hist_rows_adder_.reset(adder);
}

template <typename GradientSumT>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::BuildHistogramsLossGuide(
    ExpandEntry entry,
    const GHistIndexMatrixOneAPI &gmat,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h,
    const USMVector<GradientPair> &gpair_device) {
  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();
  nodes_for_explicit_hist_build_.push_back(entry);

  if (entry.sibling_nid > -1) {
    nodes_for_subtraction_trick_.emplace_back(entry.sibling_nid, entry.nid,
        p_tree->GetDepth(entry.sibling_nid), 0.0f, 0);
  }

  std::vector<int> sync_ids;

  hist_rows_adder_->AddHistRows(this, sync_ids, p_tree);
  BuildLocalHistograms(gmat, p_tree, gpair_h, gpair_device);
  hist_synchronizer_->SyncHistograms(this, sync_ids, p_tree);
}

template<typename GradientSumT>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::BuildLocalHistograms(
    const GHistIndexMatrixOneAPI &gmat,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h,
    const USMVector<GradientPair> &gpair_device) {
  builder_monitor_.Start("BuildLocalHistogramsOneAPI");
  const size_t n_nodes = nodes_for_explicit_hist_build_.size();
  hist_buffer_.Reset(64);
  for (size_t i = 0; i < n_nodes; i++) {
    const int32_t nid = nodes_for_explicit_hist_build_[i].nid;

    if (row_set_collection_[nid].Size() > 0) {
      BuildHist(gpair_h, gpair_device, row_set_collection_[nid], gmat, hist_[nid], hist_buffer_.GetDeviceBuffer());
    } else {
      common::InitHist(qu_, hist_[nid], hist_[nid].Size());
    }
  }
  builder_monitor_.Stop("BuildLocalHistogramsOneAPI");
}

template<typename GradientSumT>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::BuildNodeStats(
    const GHistIndexMatrixOneAPI &gmat,
    DMatrix *p_fmat,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h,
    const USMVector<GradientPair> &gpair_device) {
  builder_monitor_.Start("BuildNodeStats");
  for (auto const& entry : qexpand_depth_wise_) {
    int nid = entry.nid;
    this->InitNewNode(nid, gmat, gpair_h, gpair_device, *p_fmat, *p_tree);
    // add constraints
    if (!(*p_tree)[nid].IsLeftChild() && !(*p_tree)[nid].IsRoot()) {
      // it's a right child
      auto parent_id = (*p_tree)[nid].Parent();
      auto left_sibling_id = (*p_tree)[parent_id].LeftChild();
      auto parent_split_feature_id = snode_[parent_id].best.SplitIndex();
      tree_evaluator_.AddSplit(
          parent_id, left_sibling_id, nid, parent_split_feature_id,
          snode_[left_sibling_id].weight, snode_[nid].weight);
      interaction_constraints_.Split(parent_id, parent_split_feature_id,
                                     left_sibling_id, nid);
    }
  }
  builder_monitor_.Stop("BuildNodeStats");
}

template<typename GradientSumT>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::AddSplitsToTree(
    const GHistIndexMatrixOneAPI &gmat,
    RegTree *p_tree,
    int *num_leaves,
    int depth,
    unsigned *timestamp,
    std::vector<ExpandEntry>* nodes_for_apply_split,
    std::vector<ExpandEntry>* temp_qexpand_depth) {
  auto evaluator = tree_evaluator_.GetEvaluator();
  for (auto const& entry : qexpand_depth_wise_) {
    int nid = entry.nid;

    if (snode_[nid].best.loss_chg < kRtEps ||
        (param_.max_depth > 0 && depth == param_.max_depth) ||
        (param_.max_leaves > 0 && (*num_leaves) == param_.max_leaves)) {
      (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
    } else {
      nodes_for_apply_split->push_back(entry);

      NodeEntry<GradientSumT>& e = snode_[nid];
      bst_float left_leaf_weight =
          evaluator.CalcWeight(nid, GradStatsOneAPI<GradientSumT>{e.best.left_sum}) * param_.learning_rate;
      bst_float right_leaf_weight =
          evaluator.CalcWeight(nid, GradStatsOneAPI<GradientSumT>{e.best.right_sum}) * param_.learning_rate;
      p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                         e.best.DefaultLeft(), e.weight, left_leaf_weight,
                         right_leaf_weight, e.best.loss_chg, e.stats.GetHess(),
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
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::EvaluateAndApplySplits(
    const GHistIndexMatrixOneAPI &gmat,
    RegTree *p_tree,
    int *num_leaves,
    int depth,
    unsigned *timestamp,
    std::vector<ExpandEntry> *temp_qexpand_depth) {
  EvaluateSplits(qexpand_depth_wise_, gmat, hist_, *p_tree);

  std::vector<ExpandEntry> nodes_for_apply_split;
  AddSplitsToTree(gmat, p_tree, num_leaves, depth, timestamp,
                  &nodes_for_apply_split, temp_qexpand_depth);
  ApplySplit(nodes_for_apply_split, gmat, hist_, p_tree);
}

// Split nodes to 2 sets depending on amount of rows in each node
// Histograms for small nodes will be built explicitly
// Histograms for big nodes will be built by 'Subtraction Trick'
// Exception: in distributed setting, we always build the histogram for the left child node
//    and use 'Subtraction Trick' to built the histogram for the right child node.
//    This ensures that the workers operate on the same set of tree nodes.
template <typename GradientSumT>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::SplitSiblings(
    const std::vector<ExpandEntry> &nodes,
    std::vector<ExpandEntry> *small_siblings,
    std::vector<ExpandEntry> *big_siblings,
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
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::ExpandWithDepthWise(
    const GHistIndexMatrixOneAPI &gmat,
    DMatrix *p_fmat,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h,
    const USMVector<GradientPair> &gpair_device) {
  unsigned timestamp = 0;
  int num_leaves = 0;

  // in depth_wise growing, we feed loss_chg with 0.0 since it is not used anyway
  qexpand_depth_wise_.emplace_back(ExpandEntry(ExpandEntry::kRootNid, ExpandEntry::kEmptyNid,
      p_tree->GetDepth(ExpandEntry::kRootNid), 0.0, timestamp++));
  ++num_leaves;
  for (int depth = 0; depth < param_.max_depth + 1; depth++) {
    std::vector<int> sync_ids;
    std::vector<ExpandEntry> temp_qexpand_depth;
    SplitSiblings(qexpand_depth_wise_, &nodes_for_explicit_hist_build_,
                  &nodes_for_subtraction_trick_, p_tree);
    hist_rows_adder_->AddHistRows(this, sync_ids, p_tree);
    BuildLocalHistograms(gmat, p_tree, gpair_h, gpair_device);
    hist_synchronizer_->SyncHistograms(this, sync_ids, p_tree);
    BuildNodeStats(gmat, p_fmat, p_tree, gpair_h, gpair_device);

    EvaluateAndApplySplits(gmat, p_tree, &num_leaves, depth, &timestamp,
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
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::ExpandWithLossGuide(
    const GHistIndexMatrixOneAPI& gmat,
    DMatrix* p_fmat,
    RegTree* p_tree,
    const std::vector<GradientPair>& gpair_h,
    const USMVector<GradientPair> &gpair_device) {
  builder_monitor_.Start("ExpandWithLossGuide");
  unsigned timestamp = 0;
  int num_leaves = 0;

  ExpandEntry node(ExpandEntry::kRootNid, ExpandEntry::kEmptyNid,
      p_tree->GetDepth(0), 0.0f, timestamp++);
  BuildHistogramsLossGuide(node, gmat, p_tree, gpair_h, gpair_device);

  this->InitNewNode(ExpandEntry::kRootNid, gmat, gpair_h, gpair_device, *p_fmat, *p_tree);

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
      auto evaluator = tree_evaluator_.GetEvaluator();
      NodeEntry<GradientSumT>& e = snode_[nid];
      bst_float left_leaf_weight =
          evaluator.CalcWeight(nid, GradStatsOneAPI<GradientSumT>{e.best.left_sum}) * param_.learning_rate;
      bst_float right_leaf_weight =
          evaluator.CalcWeight(nid, GradStatsOneAPI<GradientSumT>{e.best.right_sum}) * param_.learning_rate;
      p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                         e.best.DefaultLeft(), e.weight, left_leaf_weight,
                         right_leaf_weight, e.best.loss_chg, e.stats.GetHess(),
                         e.best.left_sum.GetHess(), e.best.right_sum.GetHess());

      this->ApplySplit({candidate}, gmat, hist_, p_tree);

      const int cleft = (*p_tree)[nid].LeftChild();
      const int cright = (*p_tree)[nid].RightChild();

      ExpandEntry left_node(cleft, cright, p_tree->GetDepth(cleft),
                            0.0f, timestamp++);
      ExpandEntry right_node(cright, cleft, p_tree->GetDepth(cright),
                            0.0f, timestamp++);

      if (row_set_collection_[cleft].Size() < row_set_collection_[cright].Size()) {
        BuildHistogramsLossGuide(left_node, gmat, p_tree, gpair_h, gpair_device);
      } else {
        BuildHistogramsLossGuide(right_node, gmat, p_tree, gpair_h, gpair_device);
      }

      this->InitNewNode(cleft, gmat, gpair_h, gpair_device, *p_fmat, *p_tree);
      this->InitNewNode(cright, gmat, gpair_h, gpair_device, *p_fmat, *p_tree);
      bst_uint featureid = snode_[nid].best.SplitIndex();
      tree_evaluator_.AddSplit(nid, cleft, cright, featureid,
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

template <typename GradientSumT>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::Update(
    const GHistIndexMatrixOneAPI &gmat,
    HostDeviceVector<GradientPair> *gpair,
    DMatrix *p_fmat,
    RegTree *p_tree) {
  builder_monitor_.Start("Update");

  const std::vector<GradientPair>& gpair_h = gpair->ConstHostVector();

  USMVector<GradientPair> gpair_device(qu_, gpair_h);

  tree_evaluator_ = TreeEvaluatorOneAPI<GradientSumT>(qu_, param_, p_fmat->Info().num_col_);
  interaction_constraints_.Reset();

  this->InitData(gmat, gpair_h, gpair_device, *p_fmat, *p_tree);
  if (param_.grow_policy == TrainParam::kLossGuide) {
    ExpandWithLossGuide(gmat, p_fmat, p_tree, gpair_h, gpair_device);
  } else {
    ExpandWithDepthWise(gmat, p_fmat, p_tree, gpair_h, gpair_device);
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
bool GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::UpdatePredictionCache(
    const DMatrix* data,
    HostDeviceVector<bst_float>* p_out_preds) {
  // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
  // conjunction with Update().
  if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_) {
    return false;
  }
  builder_monitor_.Start("UpdatePredictionCache");

  std::vector<bst_float>& out_preds = p_out_preds->HostVector();
  sycl::buffer<float, 1> out_preds_buf(out_preds.data(), out_preds.size());

  CHECK_GT(out_preds.size(), 0U);

  size_t n_nodes = row_set_collection_.Size();
  for (size_t node = 0; node < n_nodes; node++) {
    const RowSetCollectionOneAPI::Elem& rowset = row_set_collection_[node];
    if (rowset.begin != nullptr && rowset.end != nullptr && rowset.Size() != 0) {
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

      const size_t* rid = rowset.begin;
      const size_t num_rows = rowset.Size();

      qu_.submit([&](sycl::handler& cgh) {
        auto out_predictions = out_preds_buf.template get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<>(sycl::range<1>(num_rows), [=](sycl::item<1> pid) {
          out_predictions[rid[pid.get_id(0)]] += leaf_value;
        });
      }).wait();
    }
  }

  builder_monitor_.Stop("UpdatePredictionCache");
  return true;
}
template<typename GradientSumT>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::InitSampling(const std::vector<GradientPair>& gpair,
                                                const USMVector<GradientPair>& gpair_device,
                                                const DMatrix& fmat,
                                                USMVector<size_t>& row_indices) {
  const auto& info = fmat.Info();
  auto& rnd = common::GlobalRandom();
#if XGBOOST_CUSTOMIZE_GLOBAL_PRNG
  std::bernoulli_distribution coin_flip(param_.subsample);
  size_t j = 0;
  for (size_t i = 0; i < info.num_row_; ++i) {
    if (gpair[i].GetHess() >= 0.0f && coin_flip(rnd)) {
      row_indices[j++] = i;
    }
  }
  /* resize row_indices to reduce memory */
  row_indices.Resize(qu_, j);
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
        row_indices[ibegin + row_offsets[tid]++] = i;
      }
    }
  }
  /* discard global engine */
  rnd = rnds[nthread - 1];
  size_t prefix_sum = row_offsets[0];
  for (size_t i = 1; i < nthread; ++i) {
    const size_t ibegin = i * discard_size;

    for (size_t k = 0; k < row_offsets[i]; ++k) {
      row_indices[prefix_sum + k] = row_indices[ibegin + k];
    }
    prefix_sum += row_offsets[i];
  }
  /* resize row_indices to reduce memory */
  row_indices.Resize(qu_, prefix_sum);
#endif  // XGBOOST_CUSTOMIZE_GLOBAL_PRNG
}
template<typename GradientSumT>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::InitData(const GHistIndexMatrixOneAPI& gmat,
                                          const std::vector<GradientPair>& gpair,
                                          const USMVector<GradientPair>& gpair_device,
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
    // initialize histogram collection
    uint32_t nbins = gmat.cut.Ptrs().back();
    hist_.Init(qu_, nbins);
    hist_local_worker_.Init(qu_, nbins);
    hist_buffer_.Init(qu_, nbins);

    // initialize histogram builder
#pragma omp parallel
    {
      this->nthread_ = omp_get_num_threads();
    }
    hist_builder_ = GHistBuilderOneAPI<GradientSumT>(qu_, nbins);

    USMVector<size_t>& row_indices = row_set_collection_.Data();
    row_indices.Resize(qu_, info.num_row_);
    size_t* p_row_indices = row_indices.Data();
    // mark subsample and build list of member rows

    if (param_.subsample < 1.0f) {
      CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
        << "Only uniform sampling is supported, "
        << "gradient-based sampling is only support by GPU Hist.";
      InitSampling(gpair, gpair_device, fmat, row_indices);
    } else {
      MemStackAllocatorOneAPI<bool, 128> buff(this->nthread_);
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
        row_indices.Resize(qu_, j);
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
    column_sampler_.Init(info.num_col_, info.feature_weigths.ConstHostVector(),
                         param_.colsample_bynode, param_.colsample_bylevel,
                         param_.colsample_bytree, true);
  } else {
    column_sampler_.Init(info.num_col_, info.feature_weigths.ConstHostVector(),
                         param_.colsample_bynode, param_.colsample_bylevel,
                         param_.colsample_bytree, false);
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
    snode_.Clear();
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
template <typename GradientSumT>
bool GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::SplitContainsMissingValues(
    const GradStatsOneAPI<GradientSumT>& e, const NodeEntry<GradientSumT>& snode) {
  if (e.GetGrad() == snode.stats.GetGrad() && e.GetHess() == snode.stats.GetHess()) {
    return false;
  } else {
    return true;
  }
}

// nodes_set - set of nodes to be processed in parallel
template<typename GradientSumT>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::EvaluateSplits(
                                               const std::vector<ExpandEntry>& nodes_set,
                                               const GHistIndexMatrixOneAPI& gmat,
                                               const HistCollectionOneAPI<GradientSumT>& hist,
                                               const RegTree& tree) {
  builder_monitor_.Start("EvaluateSplits");

  const size_t n_nodes_in_set = nodes_set.size();

  using FeatureSetType = std::shared_ptr<HostDeviceVector<bst_feature_t>>;
  std::vector<FeatureSetType> features_sets(n_nodes_in_set);

  // Generate feature set for each tree node
  size_t total_features = 0;
  for (size_t nid_in_set = 0; nid_in_set < n_nodes_in_set; ++nid_in_set) {
    const int32_t nid = nodes_set[nid_in_set].nid;
    features_sets[nid_in_set] = column_sampler_.GetFeatureSet(tree.GetDepth(nid));
    for (size_t idx_in_feature_set = 0; idx_in_feature_set < features_sets[nid_in_set]->Size(); idx_in_feature_set++) {
      const auto fid = features_sets[nid_in_set]->ConstHostVector()[idx_in_feature_set];
      if (interaction_constraints_.Query(nid, fid)) {
        total_features++;
      }
    }
  }

  split_queries_device_.Clear();
  split_queries_device_.Resize(qu_, total_features);

  size_t pos = 0;

  const size_t local_size = 16;

  for (size_t nid_in_set = 0; nid_in_set < n_nodes_in_set; ++nid_in_set) {
    const size_t nid = nodes_set[nid_in_set].nid;

    for (size_t idx_in_feature_set = 0; idx_in_feature_set < features_sets[nid_in_set]->Size(); idx_in_feature_set++) {
      const auto fid = features_sets[nid_in_set]->ConstHostVector()[idx_in_feature_set];
      if (interaction_constraints_.Query(nid, fid)) {
        split_queries_device_[pos].nid = nid;
        split_queries_device_[pos].fid = fid;
        split_queries_device_[pos].hist = hist[nid].DataConst();
        split_queries_device_[pos].best = snode_[nid].best;
        pos++;
      }
    }
  }

  auto evaluator = tree_evaluator_.GetEvaluator();
  SplitQuery* split_queries_device = split_queries_device_.Data();
  const uint32_t* cut_ptr = gmat.cut_device.Ptrs().DataConst();
  const bst_float* cut_val = gmat.cut_device.Values().DataConst();
  const bst_float* cut_minval = gmat.cut_device.MinValues().DataConst();
  const NodeEntry<GradientSumT>* snode = snode_.DataConst();

  TrainParamOneAPI param(param_);

  qu_.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<>(sycl::nd_range<2>(sycl::range<2>(total_features, local_size),
                                         sycl::range<2>(1, local_size)), [=](sycl::nd_item<2> pid) [[intel::reqd_sub_group_size(16)]] {
      TrainParamOneAPI param_device(param);
      typename TreeEvaluatorOneAPI<GradientSumT>::SplitEvaluator evaluator_device = evaluator;
      int i = pid.get_global_id(0);
      auto sg = pid.get_sub_group();
      int nid = split_queries_device[i].nid;
      int fid = split_queries_device[i].fid;
      const GradientPairT* hist_data = split_queries_device[i].hist;
      auto grad_stats = EnumerateSplit(sg, cut_ptr, cut_val, hist_data, snode[nid],
              split_queries_device[i].best, fid, nid, evaluator_device, param_device);
    });
  }).wait();

  for (size_t i = 0; i < total_features; i++) {
    int nid = split_queries_device[i].nid;
    snode_[nid].best.Update(split_queries_device[i].best);
  }

  builder_monitor_.Stop("EvaluateSplits");
}

// Enumerate the split values of specific feature.
// Returns the sum of gradients corresponding to the data points that contains a non-missing value
// for the particular feature fid.
template <typename GradientSumT>
template <int d_step>
GradStatsOneAPI<GradientSumT> GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::EnumerateSplit(
    const uint32_t* cut_ptr,
    const bst_float* cut_val,
    const bst_float* cut_minval,
    const GradientPairT* hist_data,
    const NodeEntry<GradientSumT>& snode,
    SplitEntryOneAPI<GradientSumT>& p_best,
    bst_uint fid,
    bst_uint nodeID,
    typename TreeEvaluatorOneAPI<GradientSumT>::SplitEvaluator const &evaluator_device,
    const TrainParamOneAPI& param) {
  GradStatsOneAPI<GradientSumT> c;
  GradStatsOneAPI<GradientSumT> e;
  // best split so far
  SplitEntryOneAPI<GradientSumT> best;

  // bin boundaries
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
    e.Add(hist_data[i].GetGrad(), hist_data[i].GetHess());
    if (e.GetHess() >= param.min_child_weight) {
      c.SetSubstract(snode.stats, e);
      if (c.GetHess() >= param.min_child_weight) {
        bst_float loss_chg;
        bst_float split_pt;
        if (d_step > 0) {
          loss_chg = static_cast<bst_float>(
              evaluator_device.CalcSplitGain(nodeID, fid, e, c) - snode.root_gain);
          split_pt = cut_val[i];
          best.Update(loss_chg, fid, split_pt, d_step == -1, e, c);
        } else {
          loss_chg = static_cast<bst_float>(
              evaluator_device.CalcSplitGain(nodeID, fid, GradStatsOneAPI<GradientSumT>{c}, GradStatsOneAPI<GradientSumT>{e}) - snode.root_gain);
          if (i == imin) {
            split_pt = cut_minval[fid];
          } else {
            split_pt = cut_val[i - 1];
          }
          best.Update(loss_chg, fid, split_pt, d_step == -1, c, e);
        }
      }
    }
  }
  p_best.Update(best);
  return e;
}

// Enumerate the split values of specific feature.
// Returns the sum of gradients corresponding to the data points that contains a non-missing value
// for the particular feature fid.
template <typename GradientSumT>
GradStatsOneAPI<GradientSumT> GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::EnumerateSplit(
    sycl::ext::oneapi::sub_group& sg,
    const uint32_t* cut_ptr,
    const bst_float* cut_val,
    const GradientPairT* hist_data,
    const NodeEntry<GradientSumT>& snode,
    SplitEntryOneAPI<GradientSumT>& p_best,
    bst_uint fid,
    bst_uint nodeID,
    typename TreeEvaluatorOneAPI<GradientSumT>::SplitEvaluator const &evaluator_device,
    const TrainParamOneAPI& param) {
  SplitEntryOneAPI<GradientSumT> best;

  int32_t ibegin = static_cast<int32_t>(cut_ptr[fid]);
  int32_t iend = static_cast<int32_t>(cut_ptr[fid + 1]);

  GradientSumT tot_grad = snode.stats.GetGrad();
  GradientSumT tot_hess = snode.stats.GetHess();

  GradientSumT sum_grad = 0.0f;
  GradientSumT sum_hess = 0.0f;

  int32_t local_size = sg.get_local_range().size();

  for (int32_t i = ibegin + sg.get_local_id(); i < iend; i += local_size) {
    GradientSumT e_grad = sum_grad + sycl::inclusive_scan_over_group(sg, hist_data[i].GetGrad(), std::plus<>());
    GradientSumT e_hess = sum_hess + sycl::inclusive_scan_over_group(sg, hist_data[i].GetHess(), std::plus<>());
    if (e_hess >= param.min_child_weight) {
      GradientSumT c_grad = tot_grad - e_grad;
      GradientSumT c_hess = tot_hess - e_hess;
      if (c_hess >= param.min_child_weight) {
        GradStatsOneAPI<GradientSumT> e(e_grad, e_hess);
        GradStatsOneAPI<GradientSumT> c(c_grad, c_hess);
        bst_float loss_chg;
        bst_float split_pt;
        loss_chg = static_cast<bst_float>(
            evaluator_device.CalcSplitGain(nodeID, fid, e, c) - snode.root_gain);
        split_pt = cut_val[i];
        best.Update(loss_chg, fid, split_pt, false, e, c);
      }
    }
    sum_grad += sycl::reduce_over_group(sg, hist_data[i].GetGrad(), std::plus<>());
    sum_hess += sycl::reduce_over_group(sg, hist_data[i].GetHess(), std::plus<>());
  }

  bst_float total_loss_chg = sycl::reduce_over_group(sg, best.loss_chg, maximum<>());
  bst_feature_t total_split_index = sycl::reduce_over_group(sg, best.loss_chg == total_loss_chg ? best.SplitIndex() : (1U << 31) - 1U, minimum<>());
  if (best.loss_chg == total_loss_chg && best.SplitIndex() == total_split_index) p_best.Update(best);
  return GradStatsOneAPI<GradientSumT>(sum_grad, sum_hess);
}

// split row indexes (rid_span) to 2 parts (both stored in rid_buf) depending
// on comparison of indexes values (idx_span) and split point (split_cond)
// Handle dense columns
template <bool default_left, typename BinIdxType>
inline void PartitionDenseKernel(sycl::queue& qu,
                                 const size_t node_in_set,
                                 const GHistIndexMatrixOneAPI& gmat,
                                 common::Span<const size_t>& rid_span,
                                 const size_t fid,
                                 const int32_t split_cond,
                                 common::Span<size_t>& rid_buf,
                                 sycl::buffer<size_t, 1>& parts_size) {
  const size_t row_stride = gmat.row_stride;
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const size_t* rid = &rid_span[0];
  const size_t range_size = rid_span.size();
  const size_t offset = gmat.cut.Ptrs()[fid];

  size_t* p_rid_buf = rid_buf.data();

  qu.submit([&](sycl::handler& cgh) {
    auto parts_size_acc = parts_size.template get_access<sycl::access::mode::atomic>(cgh);
    cgh.parallel_for<>(sycl::range<1>(range_size), [=](sycl::item<1> nid) {
      const size_t id = rid[nid.get_id(0)];
      const int32_t value = static_cast<int32_t>(gradient_index[id * row_stride + fid] + offset);
      const bool is_left = value <= split_cond;
      if (is_left) {
        p_rid_buf[sycl::atomic_fetch_add<size_t>(parts_size_acc[node_in_set * 2], 1)] = id;
      } else {
        p_rid_buf[range_size - sycl::atomic_fetch_add<size_t>(parts_size_acc[node_in_set * 2 + 1], 1) - 1] = id;
      }
    });
  }).wait();
}

// split row indexes (rid_span) to 2 parts (both stored in rid_buf) depending
// on comparison of indexes values (idx_span) and split point (split_cond)
// Handle dense columns
template <bool default_left, typename BinIdxType>
inline void PartitionSparseKernel(sycl::queue& qu,
                                  const size_t node_in_set,
                                  const GHistIndexMatrixOneAPI& gmat,
                                  common::Span<const size_t>& rid_span,
                                  const size_t fid,
                                  const int32_t split_cond,
                                  common::Span<size_t>& rid_buf,
                                  sycl::buffer<size_t, 1>& parts_size) {
  const size_t row_stride = gmat.row_stride;
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const size_t* rid = &rid_span[0];
  const size_t range_size = rid_span.size();
  const uint32_t* cut_ptrs = gmat.cut_device.Ptrs().DataConst();
  const bst_float* cut_vals = gmat.cut_device.Values().DataConst();

  size_t* p_rid_buf = rid_buf.data();
  qu.submit([&](sycl::handler& cgh) {
    auto parts_size_acc = parts_size.template get_access<sycl::access::mode::atomic>(cgh);
    cgh.parallel_for<>(sycl::range<1>(range_size), [=](sycl::item<1> nid) {
      const size_t id = rid[nid.get_id(0)];
      const BinIdxType* gr_index_local = gradient_index + row_stride * id;
      const int32_t fid_local = std::lower_bound(gr_index_local, gr_index_local + row_stride, cut_ptrs[fid]) - gr_index_local;
      const bool is_left = (fid_local >= row_stride || gr_index_local[fid_local] >= cut_ptrs[fid + 1]) ? default_left : gr_index_local[fid_local] <= split_cond;
      if (is_left) {
        p_rid_buf[sycl::atomic_fetch_add<size_t>(parts_size_acc[node_in_set * 2], 1)] = id;
      } else {
        p_rid_buf[range_size - sycl::atomic_fetch_add<size_t>(parts_size_acc[node_in_set * 2 + 1], 1) - 1] = id;
      }
    });
  }).wait();
}

template <typename GradientSumT>
template <typename BinIdxType>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::PartitionKernel(
    const size_t node_in_set,
    const size_t nid,
    common::Range1d range,
    const int32_t split_cond,
    const GHistIndexMatrixOneAPI& gmat,
    const RegTree& tree,
    sycl::buffer<size_t, 1>& parts_size) {
  const size_t* rid = row_set_collection_[nid].begin;

  common::Span<const size_t> rid_span(rid + range.begin(), rid + range.end());
  common::Span<size_t> rid_buf = partition_builder_.GetData(node_in_set);
  const bst_uint fid = tree[nid].SplitIndex();
  const bool default_left = tree[nid].DefaultLeft();

  if (gmat.IsDense()) {
    if (default_left) {
      PartitionDenseKernel<true, BinIdxType>(qu_, node_in_set, gmat, rid_span, fid, split_cond, rid_buf, parts_size);
    } else {
      PartitionDenseKernel<false, BinIdxType>(qu_, node_in_set, gmat, rid_span, fid, split_cond, rid_buf, parts_size);
    }
  } else {
    if (default_left) {
      PartitionSparseKernel<true, BinIdxType>(qu_, node_in_set, gmat, rid_span, fid, split_cond, rid_buf, parts_size);
    } else {
      PartitionSparseKernel<false, BinIdxType>(qu_, node_in_set, gmat, rid_span, fid, split_cond, rid_buf, parts_size);
    }
  }
}

template <typename GradientSumT>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::FindSplitConditions(
    const std::vector<ExpandEntry>& nodes,
    const RegTree& tree,
    const GHistIndexMatrixOneAPI& gmat,
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
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::AddSplitsToRowSet(const std::vector<ExpandEntry>& nodes,
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
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::ApplySplit(const std::vector<ExpandEntry> nodes,
                                                                   const GHistIndexMatrixOneAPI& gmat,
                                                                   const HistCollectionOneAPI<GradientSumT>& hist,
                                                                   RegTree* p_tree) {
  builder_monitor_.Start("ApplySplit");

  const size_t n_nodes = nodes.size();
  std::vector<int32_t> split_conditions;
  FindSplitConditions(nodes, *p_tree, gmat, &split_conditions);

  partition_builder_.Init(qu_, n_nodes, [&](size_t node_in_set) {
    const int32_t nid = nodes[node_in_set].nid;
    return row_set_collection_[nid].Size();
  });

  sycl::buffer<size_t, 1> parts_size(n_nodes * 2);
  {
    auto parts_size_acc = parts_size.template get_access<sycl::access::mode::write>();
    for (size_t node_in_set = 0; node_in_set < n_nodes; node_in_set++) {
      parts_size_acc[node_in_set * 2] = 0;
      parts_size_acc[node_in_set * 2 + 1] = 0;
    }
  }
  for (size_t node_in_set = 0; node_in_set < n_nodes; node_in_set++) {
    const int32_t nid = nodes[node_in_set].nid;
    if (row_set_collection_[nid].Size() > 0) {
      switch (gmat.index.GetBinTypeSize()) {
        case common::kUint8BinsTypeSize:
          PartitionKernel<uint8_t>(node_in_set, nid, common::Range1d(0, row_set_collection_[nid].Size()),
                    split_conditions[node_in_set], gmat, *p_tree, parts_size);
          break;
        case common::kUint16BinsTypeSize:
          PartitionKernel<uint16_t>(node_in_set, nid, common::Range1d(0, row_set_collection_[nid].Size()),
                    split_conditions[node_in_set], gmat, *p_tree, parts_size);
          break;
        case common::kUint32BinsTypeSize:
          PartitionKernel<uint32_t>(node_in_set, nid, common::Range1d(0, row_set_collection_[nid].Size()),
                    split_conditions[node_in_set], gmat, *p_tree, parts_size);
          break;
        default:
          CHECK(false);  // no default behavior
      }
    }
  }

  {
    auto parts_size_acc = parts_size.template get_access<sycl::access::mode::write>();
    for (size_t node_in_set = 0; node_in_set < n_nodes; node_in_set++) {
      partition_builder_.SetNLeftElems(node_in_set, parts_size_acc[node_in_set * 2]);
      partition_builder_.SetNRightElems(node_in_set, parts_size_acc[node_in_set * 2 + 1]);
    }
  }
  partition_builder_.CalculateRowOffsets();

  for (size_t node_in_set = 0; node_in_set < n_nodes; node_in_set++) {
    const int32_t nid = nodes[node_in_set].nid;
    partition_builder_.MergeToArray(node_in_set, const_cast<size_t*>(row_set_collection_[nid].begin));
  }

  qu_.wait();

  AddSplitsToRowSet(nodes, p_tree);

  builder_monitor_.Stop("ApplySplit");
}

template <typename GradientSumT>
void GPUQuantileHistMakerOneAPI::Builder<GradientSumT>::InitNewNode(int nid,
                                                                    const GHistIndexMatrixOneAPI& gmat,
                                                                    const std::vector<GradientPair>& gpair,
                                                                    const USMVector<GradientPair>& gpair_device,
                                                                    const DMatrix& fmat,
                                                                    const RegTree& tree) {
  builder_monitor_.Start("InitNewNode");
  {
    snode_.Resize(qu_, tree.param.num_nodes, NodeEntry<GradientSumT>(param_));
  }

  {
    auto hist = hist_[nid];
    GradientPairT grad_stat;
    if (tree[nid].IsRoot()) {
      if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
        const std::vector<uint32_t>& row_ptr = gmat.cut.Ptrs();
        const uint32_t ibegin = row_ptr[fid_least_bins_];
        const uint32_t iend = row_ptr[fid_least_bins_ + 1];
        xgboost::detail::GradientPairInternal<GradientSumT>* begin =
          reinterpret_cast<xgboost::detail::GradientPairInternal<GradientSumT>*>(hist.Data());
        for (uint32_t i = ibegin; i < iend; ++i) {
          const GradientPairT et = begin[i];
          grad_stat.Add(et.GetGrad(), et.GetHess());
        }
      } else {
        const RowSetCollectionOneAPI::Elem e = row_set_collection_[nid];
        for (const size_t* it = e.begin; it < e.end; ++it) {
          grad_stat.Add(gpair[*it].GetGrad(), gpair[*it].GetHess());
        }
      }
      histred_.Allreduce(&grad_stat, 1);
      snode_[nid].stats = GradStatsOneAPI<GradientSumT>(grad_stat.GetGrad(), grad_stat.GetHess());
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
        evaluator.CalcWeight(parentid, GradStatsOneAPI<GradientSumT>{snode_[nid].stats}));
    snode_[nid].root_gain = static_cast<float>(
        evaluator.CalcGain(parentid, GradStatsOneAPI<GradientSumT>{snode_[nid].stats}));
  }
  builder_monitor_.Stop("InitNewNode");
}

template struct GPUQuantileHistMakerOneAPI::Builder<float>;
template struct GPUQuantileHistMakerOneAPI::Builder<double>;
template void GPUQuantileHistMakerOneAPI::Builder<float>::PartitionKernel<uint8_t>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const GHistIndexMatrixOneAPI &gmat, const RegTree& tree, sycl::buffer<size_t, 1>& parts_size);
template void GPUQuantileHistMakerOneAPI::Builder<float>::PartitionKernel<uint16_t>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const GHistIndexMatrixOneAPI &gmat, const RegTree& tree, sycl::buffer<size_t, 1>& parts_size);
template void GPUQuantileHistMakerOneAPI::Builder<float>::PartitionKernel<uint32_t>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const GHistIndexMatrixOneAPI &gmat, const RegTree& tree, sycl::buffer<size_t, 1>& parts_size);
template void GPUQuantileHistMakerOneAPI::Builder<double>::PartitionKernel<uint8_t>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const GHistIndexMatrixOneAPI &gmat, const RegTree& tree, sycl::buffer<size_t, 1>& parts_size);
template void GPUQuantileHistMakerOneAPI::Builder<double>::PartitionKernel<uint16_t>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const GHistIndexMatrixOneAPI &gmat, const RegTree& tree, sycl::buffer<size_t, 1>& parts_size);
template void GPUQuantileHistMakerOneAPI::Builder<double>::PartitionKernel<uint32_t>(
    const size_t node_in_set, const size_t nid, common::Range1d range,
    const int32_t split_cond, const GHistIndexMatrixOneAPI &gmat, const RegTree& tree, sycl::buffer<size_t, 1>& parts_size);

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMakerOneAPI, "grow_quantile_histmaker_oneapi")
.describe("Grow tree using quantized histogram with dpc++.")
.set_body(
    []() {
      return new QuantileHistMakerOneAPI();
    });

XGBOOST_REGISTER_TREE_UPDATER(GPUQuantileHistMakerOneAPI, "grow_quantile_histmaker_oneapi_gpu")
.describe("Grow tree using quantized histogram with dpc++ on GPU.")
.set_body(
    []() {
      return new GPUQuantileHistMakerOneAPI();
    });
}  // namespace tree
}  // namespace xgboost