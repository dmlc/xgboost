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
      for (auto & partitioner : partitioner_) {
        size_t n_threads = partitioner.GetOptPartition().GetThreadIdsForNode(entry.nid).size();
        for (size_t tid = 0; tid < n_threads; ++tid) {
          merged_thread_ids_set[nid].insert(
            partitioner.GetOptPartition().GetThreadIdsForNode(entry.nid)[tid]);
        }
      }
      merged_thread_ids[nid].resize(merged_thread_ids_set[nid].size());
      std::copy(merged_thread_ids_set[nid].begin(),
                merged_thread_ids_set[nid].end(), merged_thread_ids[nid].begin());
    }


  size_t page_id = 0;

  for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
      CommonRowPartitioner &partitioner = this->partitioner_.at(page_id);

    this->histogram_builder_->template BuildHist<BinIdxType, true>(
        page_id, gidx, p_tree,
        nodes_for_explicit_hist_build_, nodes_for_subtraction_trick_, gpair_h,
        &(partitioner.GetOptPartition()),
        &(partitioner.GetNodeAssignments()), &merged_thread_ids);
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
}

void QuantileHistMaker::Builder::AddSplitsToTree(
          const std::vector<CPUExpandEntry>& expand,
          Driver<CPUExpandEntry>* driver,
          RegTree *p_tree,
          int *num_leaves,
          std::vector<CPUExpandEntry>* nodes_for_apply_split,
          std::unordered_map<uint32_t, bool>* smalest_nodes_mask_ptr,
          size_t depth, bool * is_left_small) {
  std::unordered_map<uint32_t, bool>& smalest_nodes_mask = *smalest_nodes_mask_ptr;
  const bool is_loss_guided = static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy)
                              != TrainParam::kDepthWise;
  std::vector<uint16_t> complete_node_ids;
  for (auto const& entry : expand) {
      nodes_for_apply_split->push_back(entry);
      evaluator_->ApplyTreeSplit(entry, p_tree);
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
  child_node_ids_ = complete_node_ids;
}

void QuantileHistMaker::Builder::LeafPartition(
    RegTree const &tree,
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

void QuantileHistMaker::Builder::SplitSiblings(
    const std::vector<CPUExpandEntry> &nodes_for_apply_split,
    std::vector<CPUExpandEntry> *nodes_to_evaluate, RegTree *p_tree) {
  monitor_->Start("SplitSiblings");
  CommonRowPartitioner &partitioner = this->partitioner_.front();

  for (auto const& entry : nodes_for_apply_split) {
    int nid = entry.nid;

    const int cleft = (*p_tree)[nid].LeftChild();
    const int cright = (*p_tree)[nid].RightChild();
    const CPUExpandEntry left_node = CPUExpandEntry(cleft, p_tree->GetDepth(cleft), 0.0);
    const CPUExpandEntry right_node = CPUExpandEntry(cright, p_tree->GetDepth(cright), 0.0);
    nodes_to_evaluate->push_back(left_node);
    nodes_to_evaluate->push_back(right_node);
    if (entry.split.left_sum.GetHess() <= entry.split.right_sum.GetHess()) {
      nodes_for_explicit_hist_build_.push_back(left_node);
      nodes_for_subtraction_trick_.push_back(right_node);
    } else {
      nodes_for_explicit_hist_build_.push_back(right_node);
      nodes_for_subtraction_trick_.push_back(left_node);
    }
  }
  monitor_->Stop("SplitSiblings");
}

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
  Driver<CPUExpandEntry> driver(param_);
  std::vector<CPUExpandEntry> expand;
  size_t page_id{0};
  std::vector<size_t>& row_indices = *row_set_collection_.Data();
  for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
    const size_t size_threads = row_indices.size() == 0 ?
                                (page.row_ptr.size() - 1) : row_indices.size();
    CommonRowPartitioner &partitioner = this->partitioner_.at(page_id);
    const_cast<common::OptPartitionBuilder&>(partitioner.GetOptPartition()).SetSlice(0,
                                             0, size_threads);
    ++page_id;
  }
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
    if (expand.size()) {
      depth = expand[0].depth + 1;
    }
    std::vector<CPUExpandEntry> nodes_for_apply_split;
    std::vector<CPUExpandEntry> nodes_to_evaluate;
    nodes_for_explicit_hist_build_.clear();
    nodes_for_subtraction_trick_.clear();
    bool is_left_small = false;
    AddSplitsToTree(expand, &driver, p_tree, &num_leaves, &nodes_for_apply_split,
                    &smalest_nodes_mask, depth, &is_left_small);
    if (nodes_for_apply_split.size() != 0) {
      monitor_->Start("ApplySplit");
      size_t page_id{0};
      for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
        CommonRowPartitioner &partitioner = this->partitioner_.at(page_id);
        partitioner.UpdatePositionDispatched({any_missing,
          static_cast<common::BinTypeSize>(sizeof(BinIdxType)),
          is_loss_guide, page.cut.HasCategorical()},
          this->ctx_,
          page,
          nodes_for_apply_split,
          p_tree,
          depth,
          &smalest_nodes_mask,
          is_loss_guide,
          &split_conditions_,
          &split_ind_, param_.max_depth,
          &child_node_ids_, is_left_small,
          true);
        ++page_id;
      }

      monitor_->Stop("ApplySplit");
      SplitSiblings(nodes_for_apply_split, &nodes_to_evaluate, p_tree);
      if (param_.max_depth == 0 || depth < param_.max_depth) {
        size_t i = 0;
        monitor_->Start("BuildHist");
        size_t num_nodes = nodes_for_explicit_hist_build_.size();
        std::vector<std::set<uint16_t>> merged_thread_ids_set(num_nodes);
        std::vector<std::vector<uint16_t>> merged_thread_ids(num_nodes);
        for (size_t nid = 0; nid < num_nodes; ++nid) {
          const auto &entry = nodes_for_explicit_hist_build_[nid];
          for (auto & partitioner : partitioner_) {
            for (auto&  tid : partitioner.GetOptPartition().GetThreadIdsForNode(entry.nid)) {
              merged_thread_ids_set[nid].insert(tid);
            }
          }
          merged_thread_ids[nid].resize(merged_thread_ids_set[nid].size());
          std::copy(merged_thread_ids_set[nid].begin(),
                    merged_thread_ids_set[nid].end(), merged_thread_ids[nid].begin());
        }

        for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
          CommonRowPartitioner &partitioner = this->partitioner_.at(i);
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
      }
      for (size_t i = 0; i < nodes_for_apply_split.size(); ++i) {
        CPUExpandEntry left_node = nodes_to_evaluate.at(i * 2 + 0);
        CPUExpandEntry right_node = nodes_to_evaluate.at(i * 2 + 1);
        driver.Push(left_node);
        driver.Push(right_node);
      }
    }
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
  monitor_->Start("UpdatePredictionCache");
  CHECK_GT(out_preds.Size(), 0U);
  size_t page_id{0};
  size_t page_disp = 0;
  for (auto const &page :
    const_cast<DMatrix*>(data)->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
    CommonRowPartitioner &partitioner =
    const_cast<CommonRowPartitioner&>(this->partitioner_.at(page_id));
    ++page_id;
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
// =======
//   monitor_->Start(__func__);
//   CHECK_EQ(out_preds.Size(), data->Info().num_row_);
//   UpdatePredictionCacheImpl(ctx_, p_last_tree_, partitioner_, *evaluator_, out_preds);
//   monitor_->Stop(__func__);
// >>>>>>> 0725fd60819f9758fbed6ee54f34f3696a2fb2f8
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
    // initialize histogram builder
    dmlc::OMPException exc;
    exc.Rethrow();
    if (param_.subsample < 1.0f) {
      CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
          << "Only uniform sampling is supported, "
          << "gradient-based sampling is only support by GPU Hist.";
      InitSampling(fmat, gpair);
    }
    const bool is_lossguide = static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy) !=
                              TrainParam::kDepthWise;

    size_t page_id{0};
    int32_t n_total_bins{0};
    if (!partition_is_initiated_) {
      partitioner_.clear();
    }

    for (auto const &page :
      const_cast<DMatrix&>(fmat).GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
      if (n_total_bins == 0) {
        n_total_bins = page.cut.TotalBins();
        feature_values_ = page.cut;
      } else {
        CHECK_EQ(n_total_bins, page.cut.TotalBins());
      }
      if (!partition_is_initiated_) {
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
  p_last_tree_ = &tree;
  evaluator_.reset(
      new HistEvaluator<CPUExpandEntry>{param_, info, this->ctx_->Threads(), column_sampler_});

  monitor_->Stop(__func__);
}

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMaker, "grow_quantile_histmaker")
    .describe("Grow tree using quantized histogram.")
    .set_body([](GenericParameter const *ctx, ObjInfo task) {
      return new QuantileHistMaker(ctx, task);
    });
}  // namespace tree
}  // namespace xgboost
