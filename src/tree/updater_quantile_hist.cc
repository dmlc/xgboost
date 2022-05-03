/*!
 * Copyright 2017-2022 by XGBoost Contributors
 * \file updater_quantile_hist.cc
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Checn, Egor Smirnov
 */
#include "./updater_quantile_hist.h"

#include <rabit/rabit.h>

#include <algorithm>
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
  hist_maker_param_.UpdateAllowUnknown(args);
}

void QuantileHistMaker::Update(HostDeviceVector<GradientPair> *gpair, DMatrix *dmat,
                               const std::vector<RegTree *> &trees) {
  // rescale learning rate according to size of trees
  float lr = param_.learning_rate;
  param_.learning_rate = lr / trees.size();

  // build tree
  const size_t n_trees = trees.size();
  if (hist_maker_param_.single_precision_histogram) {
    if (!float_builder_) {
      float_builder_.reset(new Builder<float>(n_trees, param_, dmat, task_, ctx_));
    }
  } else {
    if (!double_builder_) {
      double_builder_.reset(new Builder<double>(n_trees, param_, dmat, task_, ctx_));
    }
  }

  for (auto p_tree : trees) {
    if (hist_maker_param_.single_precision_histogram) {
      this->float_builder_->UpdateTree(gpair, dmat, p_tree);
    } else {
      this->double_builder_->UpdateTree(gpair, dmat, p_tree);
    }
  }

  param_.learning_rate = lr;
}

bool QuantileHistMaker::UpdatePredictionCache(const DMatrix *data,
                                              linalg::VectorView<float> out_preds) {
  if (hist_maker_param_.single_precision_histogram && float_builder_) {
    return float_builder_->UpdatePredictionCache(data, out_preds);
  } else if (double_builder_) {
    return double_builder_->UpdatePredictionCache(data, out_preds);
  } else {
    return false;
  }
}

template <typename GradientSumT>
CPUExpandEntry QuantileHistMaker::Builder<GradientSumT>::InitRoot(
    DMatrix *p_fmat, RegTree *p_tree, const std::vector<GradientPair> &gpair_h) {
  CPUExpandEntry node(RegTree::kRoot, p_tree->GetDepth(0), 0.0f);

  size_t page_id = 0;
  auto space = ConstructHistSpace(partitioner_, {node});
  for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
    std::vector<CPUExpandEntry> nodes_to_build{node};
    std::vector<CPUExpandEntry> nodes_to_sub;
    this->histogram_builder_->BuildHist(page_id, space, gidx, p_tree,
                                        partitioner_.at(page_id).Partitions(), nodes_to_build,
                                        nodes_to_sub, gpair_h);
    ++page_id;
  }

  {
    GradientPairT grad_stat;
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
        GradientPairT const &et = begin[i];
        grad_stat.Add(et.GetGrad(), et.GetHess());
      }
    } else {
      for (auto const &grad : gpair_h) {
        grad_stat.Add(grad.GetGrad(), grad.GetHess());
      }
      rabit::Allreduce<rabit::op::Sum, GradientSumT>(reinterpret_cast<GradientSumT *>(&grad_stat),
                                                     2);
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

  return node;
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::BuildHistogram(
    DMatrix *p_fmat, RegTree *p_tree, std::vector<CPUExpandEntry> const &valid_candidates,
    std::vector<GradientPair> const &gpair) {
  std::vector<CPUExpandEntry> nodes_to_build(valid_candidates.size());
  std::vector<CPUExpandEntry> nodes_to_sub(valid_candidates.size());

  size_t n_idx = 0;
  for (auto const &c : valid_candidates) {
    auto left_nidx = (*p_tree)[c.nid].LeftChild();
    auto right_nidx = (*p_tree)[c.nid].RightChild();
    auto fewer_right = c.split.right_sum.GetHess() < c.split.left_sum.GetHess();

    auto build_nidx = left_nidx;
    auto subtract_nidx = right_nidx;
    if (fewer_right) {
      std::swap(build_nidx, subtract_nidx);
    }
    nodes_to_build[n_idx] = CPUExpandEntry{build_nidx, p_tree->GetDepth(build_nidx), {}};
    nodes_to_sub[n_idx] = CPUExpandEntry{subtract_nidx, p_tree->GetDepth(subtract_nidx), {}};
    n_idx++;
  }

  size_t page_id{0};
  auto space = ConstructHistSpace(partitioner_, nodes_to_build);
  for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
    histogram_builder_->BuildHist(page_id, space, gidx, p_tree,
                                  partitioner_.at(page_id).Partitions(), nodes_to_build,
                                  nodes_to_sub, gpair);
    ++page_id;
  }
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::ExpandTree(
    DMatrix *p_fmat, RegTree *p_tree, const std::vector<GradientPair> &gpair_h) {
  monitor_->Start(__func__);

  Driver<CPUExpandEntry> driver(static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy));
  driver.Push(this->InitRoot(p_fmat, p_tree, gpair_h));
  bst_node_t num_leaves{1};
  auto expand_set = driver.Pop();

  while (!expand_set.empty()) {
    // candidates that can be further splited.
    std::vector<CPUExpandEntry> valid_candidates;
    // candidaates that can be applied.
    std::vector<CPUExpandEntry> applied;
    int32_t depth = expand_set.front().depth + 1;
    for (auto const& candidate : expand_set) {
      if (!candidate.IsValid(param_, num_leaves)) {
        continue;
      }
      evaluator_->ApplyTreeSplit(candidate, p_tree);
      applied.push_back(candidate);
      num_leaves++;
      if (CPUExpandEntry::ChildIsValid(param_, depth, num_leaves)) {
        valid_candidates.emplace_back(candidate);
      }
    }

    monitor_->Start("UpdatePosition");
    size_t page_id{0};
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
      partitioner_.at(page_id).UpdatePosition(ctx_, page, applied, p_tree);
      ++page_id;
    }
    monitor_->Stop("UpdatePosition");

    std::vector<CPUExpandEntry> best_splits;
    if (!valid_candidates.empty()) {
      this->BuildHistogram(p_fmat, p_tree, valid_candidates, gpair_h);
      auto const &tree = *p_tree;
      for (auto const &candidate : valid_candidates) {
        int left_child_nidx = tree[candidate.nid].LeftChild();
        int right_child_nidx = tree[candidate.nid].RightChild();
        CPUExpandEntry l_best{left_child_nidx, depth, 0.0};
        CPUExpandEntry r_best{right_child_nidx, depth, 0.0};
        best_splits.push_back(l_best);
        best_splits.push_back(r_best);
      }
      auto const &histograms = histogram_builder_->Histogram();
      auto ft = p_fmat->Info().feature_types.ConstHostSpan();
      for (auto const &gmat : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
        evaluator_->EvaluateSplits(histograms, gmat.cut, ft, *p_tree, &best_splits);
        break;
      }
    }
    driver.Push(best_splits.begin(), best_splits.end());
    expand_set = driver.Pop();
  }

  monitor_->Stop(__func__);
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::UpdateTree(HostDeviceVector<GradientPair> *gpair,
                                                          DMatrix *p_fmat, RegTree *p_tree) {
  monitor_->Start(__func__);

  std::vector<GradientPair> *gpair_ptr = &(gpair->HostVector());
  // in case 'num_parallel_trees != 1' no posibility to change initial gpair
  if (GetNumberOfTrees() != 1) {
    gpair_local_.resize(gpair_ptr->size());
    gpair_local_ = *gpair_ptr;
    gpair_ptr = &gpair_local_;
  }

  this->InitData(p_fmat, *p_tree, gpair_ptr);

  ExpandTree(p_fmat, p_tree, *gpair_ptr);

  monitor_->Stop(__func__);
}

template <typename GradientSumT>
bool QuantileHistMaker::Builder<GradientSumT>::UpdatePredictionCache(
    DMatrix const *data, linalg::VectorView<float> out_preds) const {
  // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
  // conjunction with Update().
  if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_) {
    return false;
  }
  monitor_->Start(__func__);
  CHECK_EQ(out_preds.Size(), data->Info().num_row_);
  UpdatePredictionCacheImpl(ctx_, p_last_tree_, partitioner_, *evaluator_, param_, out_preds);
  monitor_->Stop(__func__);
  return true;
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitSampling(const DMatrix &fmat,
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
template<typename GradientSumT>
size_t QuantileHistMaker::Builder<GradientSumT>::GetNumberOfTrees() {
  return n_trees_;
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitData(DMatrix *fmat, const RegTree &tree,
                                                        std::vector<GradientPair> *gpair) {
  monitor_->Start(__func__);
  const auto& info = fmat->Info();

  {
    size_t page_id{0};
    int32_t n_total_bins{0};
    partitioner_.clear();
    for (auto const &page : fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
      if (n_total_bins == 0) {
        n_total_bins = page.cut.TotalBins();
      } else {
        CHECK_EQ(n_total_bins, page.cut.TotalBins());
      }
      partitioner_.emplace_back(page.Size(), page.base_rowid, this->ctx_->Threads());
      ++page_id;
    }
    histogram_builder_->Reset(n_total_bins, HistBatch(param_), ctx_->Threads(), page_id,
                              rabit::IsDistributed());

    if (param_.subsample < 1.0f) {
      CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
          << "Only uniform sampling is supported, "
          << "gradient-based sampling is only support by GPU Hist.";
      InitSampling(*fmat, gpair);
    }
  }

  // store a pointer to the tree
  p_last_tree_ = &tree;
  evaluator_.reset(new HistEvaluator<GradientSumT, CPUExpandEntry>{
      param_, info, this->ctx_->Threads(), column_sampler_});

  monitor_->Stop(__func__);
}

void HistRowPartitioner::FindSplitConditions(const std::vector<CPUExpandEntry> &nodes,
                                             const RegTree &tree, const GHistIndexMatrix &gmat,
                                             std::vector<int32_t> *split_conditions) {
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
    CHECK_LT(upper_bound, static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
    for (uint32_t bound = lower_bound; bound < upper_bound; ++bound) {
      if (split_pt == gmat.cut.Values()[bound]) {
        split_cond = static_cast<int32_t>(bound);
      }
    }
    (*split_conditions)[i] = split_cond;
  }
}

void HistRowPartitioner::AddSplitsToRowSet(const std::vector<CPUExpandEntry> &nodes,
                                           RegTree const *p_tree) {
  const size_t n_nodes = nodes.size();
  for (unsigned int i = 0; i < n_nodes; ++i) {
    const int32_t nid = nodes[i].nid;
    const size_t n_left = partition_builder_.GetNLeftElems(i);
    const size_t n_right = partition_builder_.GetNRightElems(i);
    CHECK_EQ((*p_tree)[nid].LeftChild() + 1, (*p_tree)[nid].RightChild());
    row_set_collection_.AddSplit(nid, (*p_tree)[nid].LeftChild(), (*p_tree)[nid].RightChild(),
                                 n_left, n_right);
  }
}

template struct QuantileHistMaker::Builder<float>;
template struct QuantileHistMaker::Builder<double>;

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMaker, "grow_quantile_histmaker")
    .describe("Grow tree using quantized histogram.")
    .set_body([](ObjInfo task) { return new QuantileHistMaker(task); });
}  // namespace tree
}  // namespace xgboost
