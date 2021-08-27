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
  builder->reset(
      new Builder<GradientSumT>(n_trees, param_, std::move(pruner_), dmat));
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

  // build tree
  const size_t n_trees = trees.size();
  if (hist_maker_param_.single_precision_histogram) {
    if (!float_builder_) {
      this->SetBuilder(n_trees, &float_builder_, dmat);
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
QuantileHistMaker::Builder<GradientSumT>::~Builder() = default;

template <typename GradientSumT>
template <typename BinIdxType, bool any_missing, bool hist_fit_to_l2>
void QuantileHistMaker::Builder<GradientSumT>::InitRoot(
    const GHistIndexMatrix &gmat, DMatrix* p_fmat, RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h, int *num_leaves,
    std::vector<CPUExpandEntry> *expand, common::BlockedSpace2d* space_ptr,
    const ColumnMatrix& column_matrix, bool is_loss_guide) {
  CPUExpandEntry node(CPUExpandEntry::kRootNid, p_tree->GetDepth(0), 0.0f);

  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();
  nodes_for_explicit_hist_build_.push_back(node);

  int starting_index = std::numeric_limits<int>::max();
  int sync_count = 0;
  opt_partition_builder_.UpdateRootThreadWork(gmat.IsDense());

  this->histogram_builder_->template BuildHist<BinIdxType,
                                               any_missing,
                                               hist_fit_to_l2>(p_fmat, gmat, p_tree,
                                                               gpair_h, 0, column_matrix,
                                                               nodes_for_explicit_hist_build_,
                                                               nodes_for_subtraction_trick_,
                                                               &opt_partition_builder_,
                                                               &node_ids_);

  {
    auto nid = CPUExpandEntry::kRootNid;
    GHistRowT hist = this->histogram_builder_->Histogram()[nid];
    GradientPairT grad_stat;
    if (data_layout_ == DataLayout::kDenseDataZeroBased ||
        data_layout_ == DataLayout::kDenseDataOneBased) {
      const std::vector<uint32_t> &row_ptr = gmat.cut.Ptrs();
      const uint32_t ibegin = row_ptr[fid_least_bins_];
      const uint32_t iend = row_ptr[fid_least_bins_ + 1];
      auto begin = hist.data();
      for (uint32_t i = ibegin; i < iend; ++i) {
        const GradientPairT et = begin[i];
        grad_stat.Add(et.GetGrad(), et.GetHess());
      }
    } else {
      builder_monitor_.Start("RootStatsCalculation");
      for (const GradientPair& gh : gpair_h) {
        grad_stat.Add(gh.GetGrad(), gh.GetHess());
      }
      rabit::Allreduce<rabit::op::Sum, GradientSumT>(
          reinterpret_cast<GradientSumT *>(&grad_stat), 2);
      builder_monitor_.Stop("RootStatsCalculation");
    }

    auto weight = evaluator_->InitRoot(GradStats{grad_stat});
    p_tree->Stat(RegTree::kRoot).sum_hess = grad_stat.GetHess();
    p_tree->Stat(RegTree::kRoot).base_weight = weight;
    (*p_tree)[RegTree::kRoot].SetLeaf(param_.learning_rate * weight);

    std::vector<CPUExpandEntry> entries{node};
    builder_monitor_.Start("EvaluateSplits");
    for (auto const &gmat_local : p_fmat->GetBatches<GHistIndexMatrix>(
             BatchParam{GenericParameter::kCpuId, param_.max_bin})) {
      evaluator_->EvaluateSplits(histogram_builder_->Histogram(), gmat_local,
                                *p_tree, &nodes_for_explicit_hist_build_,
                                &nodes_for_subtraction_trick_, &entries,
                                histogram_builder_->GetHistBuffer(),
                                histogram_builder_->GetLocalThreadsMapping(),
                                &opt_partition_builder_, p_tree,
                                param_.colsample_bylevel != 1 ||
                                param_.colsample_bynode  != 1 ||
                                param_.colsample_bytree  != 1);
    }
    builder_monitor_.Stop("EvaluateSplits");
    node = entries.front();
  }

  expand->push_back(node);
  ++(*num_leaves);
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::AddSplitsToTree(
          const std::vector<CPUExpandEntry>& expand,
          RegTree *p_tree,
          int *num_leaves,
          std::vector<CPUExpandEntry>* nodes_for_apply_split,
          std::vector<bool>* smalest_nodes_mask_ptr) {
  std::vector<bool>& smalest_nodes_mask = *smalest_nodes_mask_ptr;
  const bool is_loss_guided = static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy)
                              != TrainParam::kDepthWise;
  std::vector<uint16_t> compleate_node_ids;
  for (auto const& entry : expand) {
    if (entry.IsValid(param_, *num_leaves)) {
      nodes_for_apply_split->push_back(entry);
      evaluator_->ApplyTreeSplit(entry, p_tree);
      (*num_leaves)++;
      curr_level_nodes_[2*entry.nid] = (*p_tree)[entry.nid].LeftChild();
      curr_level_nodes_[2*entry.nid + 1] = (*p_tree)[entry.nid].RightChild();
      compleate_node_ids.push_back((*p_tree)[entry.nid].LeftChild());
      compleate_node_ids.push_back((*p_tree)[entry.nid].RightChild());
      if (entry.split.left_sum.GetHess() <= entry.split.right_sum.GetHess() || is_loss_guided) {
        smalest_nodes_mask[curr_level_nodes_[2*entry.nid]] = true;
      } else {
        smalest_nodes_mask[curr_level_nodes_[2*entry.nid + 1]] = true;
      }
    } else {
      is_compleate_tree_ = false;
      curr_level_nodes_[2*entry.nid] = static_cast<uint16_t>(1) << 15 |
                                       static_cast<uint16_t>(entry.nid);
      curr_level_nodes_[2*entry.nid + 1] = curr_level_nodes_[2*entry.nid];
    }
  }
  compleate_trees_depth_wise_ = compleate_node_ids;
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
    bool is_loss_guide =  static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy) ==
                          TrainParam::kDepthWise ? false : true;
    if (is_loss_guide) {
      if (opt_partition_builder_.partitions[cleft].Size() <=
          opt_partition_builder_.partitions[cright].Size()) {
        nodes_for_explicit_hist_build_.push_back(left_node);
        nodes_for_subtraction_trick_.push_back(right_node);
      } else {
        nodes_for_explicit_hist_build_.push_back(right_node);
        nodes_for_subtraction_trick_.push_back(left_node);
      }
    } else {
      if (entry.split.left_sum.GetHess() <= entry.split.right_sum.GetHess()) {
        nodes_for_explicit_hist_build_.push_back(left_node);
        nodes_for_subtraction_trick_.push_back(right_node);
      } else {
        nodes_for_explicit_hist_build_.push_back(right_node);
        nodes_for_subtraction_trick_.push_back(left_node);
      }
    }
  }
  builder_monitor_.Stop("SplitSiblings");
}

template<typename GradientSumT>
template <typename BinIdxType, bool any_missing, bool hist_fit_to_l2>
void QuantileHistMaker::Builder<GradientSumT>::ExpandTree(
    const GHistIndexMatrix& gmat,
    const ColumnMatrix& column_matrix,
    DMatrix* p_fmat,
    RegTree* p_tree,
    const std::vector<GradientPair>& gpair_h) {
  builder_monitor_.Start("ExpandTree");
  int num_leaves = 0;
  saved_split_ind_.clear();
  saved_split_ind_.resize(1 << (param_.max_depth + 1), 0);
  split_conditions_.clear();
  split_ind_.clear();
  is_compleate_tree_ = true;

  Driver<CPUExpandEntry> driver(static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy));
  std::vector<CPUExpandEntry> expand;

  std::vector<size_t>& row_indices = *row_set_collection_.Data();
  const size_t size_threads = row_indices.size() == 0 ?
                              (gmat.row_ptr.size() - 1) : row_indices.size();

  opt_partition_builder_.SetSlice(0, 0, size_threads);
  const size_t n_threads = omp_get_max_threads();
  opt_partition_builder_.n_threads = n_threads;
  node_ids_.resize(size_threads, 0);
  bool is_loss_guide = static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy) ==
                       TrainParam::kDepthWise ? false : true;
  int depth = 0;
  curr_level_nodes_.clear();
  curr_level_nodes_.resize(1 << (param_.max_depth + 2), 0);
  InitRoot<BinIdxType, any_missing, hist_fit_to_l2>(gmat, p_fmat, p_tree, gpair_h,
                                                    &num_leaves, &expand, nullptr,
                                                    column_matrix, is_loss_guide);
  driver.Push(expand[0]);
  compleate_trees_depth_wise_.clear();
  compleate_trees_depth_wise_.emplace_back(0);
  while (!driver.IsEmpty()) {
    std::vector<bool> smalest_nodes_mask(1 << (param_.max_depth + 2), false);
    expand = driver.Pop();
    depth = expand[0].depth + 1;

    std::vector<CPUExpandEntry> nodes_for_apply_split;
    std::vector<CPUExpandEntry> nodes_to_evaluate;
    nodes_for_explicit_hist_build_.clear();
    nodes_for_subtraction_trick_.clear();

    AddSplitsToTree(expand, p_tree, &num_leaves, &nodes_for_apply_split, &smalest_nodes_mask);
    if (nodes_for_apply_split.size() != 0) {
      if (is_loss_guide) {
        ApplySplit<any_missing, BinIdxType, true>(nodes_for_apply_split,
                                                  gmat, column_matrix,
                                                  this->histogram_builder_->Histogram(),
                                                  p_tree, depth, &smalest_nodes_mask,
                                                  gpair_h, is_loss_guide);
      } else {
        ApplySplit<any_missing, BinIdxType, false>(nodes_for_apply_split,
                                                   gmat, column_matrix,
                                                   this->histogram_builder_->Histogram(),
                                                   p_tree, depth, &smalest_nodes_mask,
                                                   gpair_h, is_loss_guide);
      }

      SplitSiblings(nodes_for_apply_split, &nodes_to_evaluate, p_tree);
      int starting_index = std::numeric_limits<int>::max();
      int sync_count = 0;
      if (depth < param_.max_depth) {
        this->histogram_builder_->template BuildHist<BinIdxType,
                                                     any_missing,
                                                     hist_fit_to_l2>(p_fmat, gmat, p_tree,
                                                                     gpair_h, depth,
                                                                     column_matrix,
                                                                     nodes_for_explicit_hist_build_,
                                                                     nodes_for_subtraction_trick_,
                                                                     &opt_partition_builder_,
                                                                     &node_ids_);
        builder_monitor_.Start("EvaluateSplits");
        for (auto const &gmat_local : p_fmat->GetBatches<GHistIndexMatrix>(
             BatchParam{GenericParameter::kCpuId, param_.max_bin})) {
          evaluator_->EvaluateSplits(this->histogram_builder_->Histogram(),
                                    gmat_local, *p_tree, &nodes_for_explicit_hist_build_,
                                    &nodes_for_subtraction_trick_, &nodes_to_evaluate,
                                    histogram_builder_->GetHistBuffer(),
                                    histogram_builder_->GetLocalThreadsMapping(),
                                    &opt_partition_builder_, p_tree,
                                    param_.colsample_bylevel != 1 ||
                                    param_.colsample_bynode  != 1 ||
                                    param_.colsample_bytree  != 1);
        }
        builder_monitor_.Stop("EvaluateSplits");
      }
      for (size_t i = 0; i < nodes_for_apply_split.size(); ++i) {
        CPUExpandEntry left_node = nodes_to_evaluate.at(i * 2 + 0);
        CPUExpandEntry right_node = nodes_to_evaluate.at(i * 2 + 1);
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
  p_last_fmat_mutable_ = p_fmat;

  this->InitData(gmat, column_matrix, *p_fmat, *p_tree, gpair_ptr);
  CHECK_EQ(!column_matrix.AnyMissing(), gmat.IsDense());
  const bool hist_fit_to_l2 = 1024*1024*0.8 > 16*gmat.cut.Ptrs().back();
  switch (column_matrix.GetTypeSize()) {
    case common::kUint8BinsTypeSize:
      if (column_matrix.AnyMissing()) {
        if (hist_fit_to_l2) {
          ExpandTree<uint8_t, true, true>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
        } else {
          ExpandTree<uint8_t, true, false>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
        }
      } else {
        if (hist_fit_to_l2) {
          ExpandTree<uint8_t, false, true>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
        } else {
          ExpandTree<uint8_t, false, false>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
        }
      }
      break;
    case common::kUint16BinsTypeSize:
      if (column_matrix.AnyMissing()) {
        if (hist_fit_to_l2) {
          ExpandTree<uint16_t, true, true>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
        } else {
          ExpandTree<uint16_t, true, false>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
        }
      } else {
        if (hist_fit_to_l2) {
          ExpandTree<uint16_t, false, true>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
        } else {
          ExpandTree<uint16_t, false, false>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
        }
      }
      break;
    case common::kUint32BinsTypeSize:
      if (column_matrix.AnyMissing()) {
        if (hist_fit_to_l2) {
          ExpandTree<uint32_t, true, true>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
        } else {
          ExpandTree<uint32_t, true, false>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
        }
      } else {
        if (hist_fit_to_l2) {
          ExpandTree<uint32_t, false, true>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
        } else {
          ExpandTree<uint32_t, false, false>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
        }
      }
      break;
    default:
      CHECK(false);  // no default behavior
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

    common::BlockedSpace2d space(1, [&](size_t node) {
      return node_ids_.size();
    }, 1024);
    common::ParallelFor2d(space, this->nthread_, [&](size_t node, common::Range1d r) {
      int tid = omp_get_thread_num();
      for (size_t it = r.begin(); it <  r.end(); ++it) {
        bst_float leaf_value;
        // if a node is marked as deleted by the pruner, traverse upward to locate
        // a non-deleted leaf.
        int nid = (~(static_cast<uint16_t>(1) << 15)) & node_ids_[it];
        if ((*p_last_tree_)[nid].IsDeleted()) {
          while ((*p_last_tree_)[nid].IsDeleted()) {
            nid = (*p_last_tree_)[nid].Parent();
          }
          CHECK((*p_last_tree_)[nid].IsLeaf());
        }
        leaf_value = (*p_last_tree_)[nid].LeafValue();
        out_preds[it] += leaf_value;
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
                                          const ColumnMatrix& column_matrix,
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
    // row_set_collection_.Clear();
    // initialize histogram collection
    uint32_t nbins = gmat.cut.Ptrs().back();
    // initialize histogram builder
    dmlc::OMPException exc;
#pragma omp parallel
    {
      exc.Run([&]() {
        this->nthread_ = omp_get_num_threads();
      });
    }
    exc.Rethrow();
    this->histogram_builder_->Reset(nbins, param_.max_bin, this->nthread_,
                                    gmat.cut.Ptrs().size() - 1, param_.max_depth,
                                    param_.colsample_bytree, param_.colsample_bylevel,
                                    param_.colsample_bynode,
                                    column_sampler_, gmat.index.Offset(), gmat.IsDense());
    if (param_.subsample < 1.0f) {
      CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
        << "Only uniform sampling is supported, "
        << "gradient-based sampling is only support by GPU Hist.";
      builder_monitor_.Start("InitSampling");
      InitSampling(fmat, gpair, nullptr);
      builder_monitor_.Stop("InitSampling");
      // We should check that the partitioning was done correctly
      // and each row of the dataset fell into exactly one of the categories
    }
    const bool is_lossguide = static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy) !=
                              TrainParam::kDepthWise;

    switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        opt_partition_builder_.Init<uint8_t>(gmat, column_matrix, &tree,
                                             this->nthread_, param_.max_depth,
                                             info.num_row_, is_lossguide);

        break;
      case common::kUint16BinsTypeSize:
        opt_partition_builder_.Init<uint16_t>(gmat, column_matrix, &tree,
                                              this->nthread_, param_.max_depth,
                                              info.num_row_, is_lossguide);

        break;
      case common::kUint32BinsTypeSize:
        opt_partition_builder_.Init<uint32_t>(gmat, column_matrix, &tree,
                                              this->nthread_, param_.max_depth,
                                              info.num_row_, is_lossguide);
        break;
      default:
        CHECK(false);  // no default behavior
    }

    // mark subsample and build list of member rows
    const size_t block_size = info.num_row_ / this->nthread_ + !!(info.num_row_ % this->nthread_);

    // #pragma omp parallel num_threads(this->nthread_)
    // {
    //   exc.Run([&]() {
    //     const size_t tid = omp_get_thread_num();
    //     const size_t ibegin = tid * block_size;
    //     const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
    //         static_cast<size_t>(info.num_row_));

    //     for (size_t i = ibegin; i < iend; ++i) {
    //       if ((*gpair)[i].GetHess() < 0.0f) {
    //         (*gpair)[i] = GradientPair(0);
    //       }
    //     }
    //   });
    // }
    // exc.Rethrow();

    if (is_lossguide) {
      opt_partition_builder_.ResizeRowsBuffer(info.num_row_);
      uint32_t* row_set_collection_vec_p = opt_partition_builder_.GetRowsBuffer();
      #pragma omp parallel num_threads(this->nthread_)
      {
        exc.Run([&]() {
          const size_t tid = omp_get_thread_num();
          const size_t ibegin = tid * block_size;
          const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
              static_cast<size_t>(info.num_row_));
          for (size_t i = ibegin; i < iend; ++i) {
            row_set_collection_vec_p[i] = i;
          }
        });
      }
      exc.Rethrow();
    }
  }

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
    evaluator_.reset(new HistEvaluator<GradientSumT, CPUExpandEntry>{
        param_, info, this->nthread_, column_sampler_, true});
  } else {
    evaluator_.reset(new HistEvaluator<GradientSumT, CPUExpandEntry>{
        param_, info, this->nthread_, column_sampler_, false});
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

  builder_monitor_.Stop("InitData");
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::FindSplitConditions(
                                                     const std::vector<CPUExpandEntry>& nodes,
                                                     const RegTree& tree,
                                                     const GHistIndexMatrix& gmat,
                                                     std::vector<int32_t>* split_conditions) {
  for (const auto& node : nodes) {
    const int32_t nid = node.nid;
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

    (*split_conditions)[nid] = split_cond;
    saved_split_ind_[nid] = split_cond;
  }
}

template <typename GradientSumT>
template <bool any_missing, typename BinIdxType, bool is_loss_guided>
void QuantileHistMaker::Builder<GradientSumT>::ApplySplit(const std::vector<CPUExpandEntry> nodes,
                                                          const GHistIndexMatrix& gmat,
                                                          const ColumnMatrix& column_matrix,
                                                          const HistCollection<GradientSumT>& hist,
                                                          RegTree* p_tree, int depth,
                                                          std::vector<bool>* smalest_nodes_mask_ptr,
                                                          const std::vector<GradientPair> &gpair_h,
                                                          bool loss_guide) {
  builder_monitor_.Start("ApplySplit");
  // 1. Find split condition for each split
  const size_t n_nodes = nodes.size();
  split_conditions_.resize(1 << (param_.max_depth + 1), 0);
  std::vector<int32_t>& split_conditions = split_conditions_;
  FindSplitConditions(nodes, *p_tree, gmat, &split_conditions);
  // 2.1 Create a blocked space of size SUM(samples in each node)
  split_ind_.resize(split_conditions.size(), 0);
  std::vector<uint64_t>& split_ind = split_ind_;
  const uint32_t* offsets = gmat.index.Offset();
  const uint64_t rows_offset = gmat.row_ptr.size() - 1;
  std::vector<uint32_t> split_nodes(n_nodes, 0);
  for (size_t i = 0; i < n_nodes; ++i) {
      const int32_t nid = nodes[i].nid;
      split_nodes[i] = nid;
      const uint64_t fid = (*p_tree)[nid].SplitIndex();
      split_ind[nid] = fid*((gmat.IsDense() ? rows_offset : 1));
      split_conditions[nid] = split_conditions[nid] - gmat.cut.Ptrs()[fid];
  }
  const size_t max_depth = param_.max_depth;

  const size_t n_features = gmat.cut.Ptrs().size() - 1;
  int nthreads = this->nthread_;
  nthreads = std::min(nthreads, omp_get_max_threads());
  nthreads = std::max(nthreads, 1);

  const size_t depth_begin = opt_partition_builder_.DepthBegin(compleate_trees_depth_wise_,
                                                               p_tree, loss_guide);
  const size_t depth_size = opt_partition_builder_.DepthSize(gmat, compleate_trees_depth_wise_,
                                                             p_tree, loss_guide);
  std::vector<bool>& smalest_nodes_mask = *smalest_nodes_mask_ptr;
#pragma omp parallel num_threads(nthreads)
  {
    size_t tid = omp_get_thread_num();
    const BinIdxType* numa = tid < nthreads/2 ?
      reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData()) :
      reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexSecondData());
    size_t chunck_size = depth_size / nthreads + !!(depth_size % nthreads);
    size_t thread_size = chunck_size;
    size_t begin = thread_size * tid;
    size_t end = std::min(begin + thread_size, depth_size);
    begin += depth_begin;
    end += depth_begin;
    opt_partition_builder_.template CommonPartition<BinIdxType,
                                                    is_loss_guided,
                                                    !any_missing>(tid, begin, end, numa,
                                                                  node_ids_.data(),
                                                                  &split_conditions,
                                                                  &split_ind,
                                                                  smalest_nodes_mask, gpair_h,
                                                                  &curr_level_nodes_,
                                                                  column_matrix, split_nodes);
  }

  if (depth != max_depth || loss_guide) {
    builder_monitor_.Start("UpdateRowBuffer&UpdateThreadsWork");
    opt_partition_builder_.UpdateRowBuffer(compleate_trees_depth_wise_,
                                           p_tree, gmat, n_features, depth,
                                           node_ids_, is_loss_guided);
    opt_partition_builder_.UpdateThreadsWork(compleate_trees_depth_wise_, gmat,
                                             n_features, depth, is_loss_guided);
    builder_monitor_.Stop("UpdateRowBuffer&UpdateThreadsWork");
  }
  builder_monitor_.Stop("ApplySplit");
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
