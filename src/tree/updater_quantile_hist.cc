/*!
 * Copyright 2017-2022 by XGBoost Contributors
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
    pruner_.reset(TreeUpdater::Create("prune", ctx_, task_));
  }
  pruner_->Configure(args);
  param_.UpdateAllowUnknown(args);
  hist_maker_param_.UpdateAllowUnknown(args);
}

template <typename GradientSumT>
void QuantileHistMaker::SetBuilder(const size_t n_trees,
                                   std::unique_ptr<Builder<GradientSumT>>* builder, DMatrix* dmat) {
  builder->reset(
      new Builder<GradientSumT>(n_trees, param_, std::move(pruner_), dmat, task_, ctx_));
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
  auto it = dmat->GetBatches<GHistIndexMatrix>(HistBatch(param_)).begin();
  auto p_gmat = it.Page();
  if (dmat != p_last_dmat_ || is_gmat_initialized_ == false) {
    updater_monitor_.Start("GmatInitialization");
    column_matrix_.Init(*p_gmat, param_.sparse_threshold, ctx_->Threads());
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
    CallBuilderUpdate(float_builder_, gpair, dmat, *p_gmat, trees);
  } else {
    if (!double_builder_) {
      SetBuilder(n_trees, &double_builder_, dmat);
    }
    CallBuilderUpdate(double_builder_, gpair, dmat, *p_gmat, trees);
  }

  param_.learning_rate = lr;

  p_last_dmat_ = dmat;
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
template <bool any_missing>
void QuantileHistMaker::Builder<GradientSumT>::InitRoot(
    DMatrix *p_fmat, RegTree *p_tree, const std::vector<GradientPair> &gpair_h,
    int *num_leaves, std::vector<CPUExpandEntry> *expand) {
  CPUExpandEntry node(RegTree::kRoot, p_tree->GetDepth(0), 0.0f);

  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();
  nodes_for_explicit_hist_build_.push_back(node);

  auto const& row_set_collection = partitioner_.front().Partitions();
  size_t page_id = 0;
  for (auto const& gidx :
       p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
    this->histogram_builder_->BuildHist(
        page_id, gidx, p_tree, row_set_collection,
        nodes_for_explicit_hist_build_, nodes_for_subtraction_trick_, gpair_h);
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
    builder_monitor_->Start("EvaluateSplits");
    auto ft = p_fmat->Info().feature_types.ConstHostSpan();
    for (auto const& gmat : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
      evaluator_->EvaluateSplits(histogram_builder_->Histogram(), gmat.cut, ft,
                                 *p_tree, &entries);
      break;
    }
    builder_monitor_->Stop("EvaluateSplits");
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
          std::vector<CPUExpandEntry>* nodes_for_apply_split) {
  for (auto const& entry : expand) {
    if (entry.IsValid(param_, *num_leaves)) {
      nodes_for_apply_split->push_back(entry);
      evaluator_->ApplyTreeSplit(entry, p_tree);
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
  builder_monitor_->Start("SplitSiblings");
  auto const& row_set_collection = this->partitioner_.front().Partitions();
  for (auto const& entry : nodes_for_apply_split) {
    int nid = entry.nid;

    const int cleft = (*p_tree)[nid].LeftChild();
    const int cright = (*p_tree)[nid].RightChild();
    const CPUExpandEntry left_node = CPUExpandEntry(cleft, p_tree->GetDepth(cleft), 0.0);
    const CPUExpandEntry right_node = CPUExpandEntry(cright, p_tree->GetDepth(cright), 0.0);
    nodes_to_evaluate->push_back(left_node);
    nodes_to_evaluate->push_back(right_node);
    if (row_set_collection[cleft].Size() < row_set_collection[cright].Size()) {
      nodes_for_explicit_hist_build_.push_back(left_node);
      nodes_for_subtraction_trick_.push_back(right_node);
    } else {
      nodes_for_explicit_hist_build_.push_back(right_node);
      nodes_for_subtraction_trick_.push_back(left_node);
    }
  }
  CHECK_EQ(nodes_for_subtraction_trick_.size(), nodes_for_explicit_hist_build_.size());
  builder_monitor_->Stop("SplitSiblings");
}

template<typename GradientSumT>
template <bool any_missing>
void QuantileHistMaker::Builder<GradientSumT>::ExpandTree(
    const GHistIndexMatrix& gmat,
    const common::ColumnMatrix& column_matrix,
    DMatrix* p_fmat,
    RegTree* p_tree,
    const std::vector<GradientPair>& gpair_h) {
  builder_monitor_->Start("ExpandTree");
  int num_leaves = 0;

  Driver<CPUExpandEntry> driver(static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy));
  std::vector<CPUExpandEntry> expand;
  InitRoot<any_missing>(p_fmat, p_tree, gpair_h, &num_leaves, &expand);
  driver.Push(expand[0]);

  int32_t depth = 0;
  while (!driver.IsEmpty()) {
    expand = driver.Pop();
    depth = expand[0].depth + 1;
    std::vector<CPUExpandEntry> nodes_for_apply_split;
    std::vector<CPUExpandEntry> nodes_to_evaluate;
    nodes_for_explicit_hist_build_.clear();
    nodes_for_subtraction_trick_.clear();

    AddSplitsToTree(expand, p_tree, &num_leaves, &nodes_for_apply_split);

    if (nodes_for_apply_split.size() != 0) {
      HistRowPartitioner &partitioner = this->partitioner_.front();
      if (gmat.cut.HasCategorical()) {
        partitioner.UpdatePosition<any_missing, true>(this->ctx_, gmat, column_matrix,
                                                      nodes_for_apply_split, p_tree);
      } else {
        partitioner.UpdatePosition<any_missing, false>(this->ctx_, gmat, column_matrix,
                                                       nodes_for_apply_split, p_tree);
      }

      SplitSiblings(nodes_for_apply_split, &nodes_to_evaluate, p_tree);

      if (param_.max_depth == 0 || depth < param_.max_depth) {
        size_t i = 0;
        for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
          this->histogram_builder_->BuildHist(i, gidx, p_tree, partitioner_.front().Partitions(),
                                              nodes_for_explicit_hist_build_,
                                              nodes_for_subtraction_trick_, gpair_h);
          ++i;
        }
      } else {
        int starting_index = std::numeric_limits<int>::max();
        int sync_count = 0;
        this->histogram_builder_->AddHistRows(
            &starting_index, &sync_count, nodes_for_explicit_hist_build_,
            nodes_for_subtraction_trick_, p_tree);
      }

      builder_monitor_->Start("EvaluateSplits");
      auto ft = p_fmat->Info().feature_types.ConstHostSpan();
      evaluator_->EvaluateSplits(this->histogram_builder_->Histogram(),
                                 gmat.cut, ft, *p_tree, &nodes_to_evaluate);
      builder_monitor_->Stop("EvaluateSplits");

      for (size_t i = 0; i < nodes_for_apply_split.size(); ++i) {
        CPUExpandEntry left_node = nodes_to_evaluate.at(i * 2 + 0);
        CPUExpandEntry right_node = nodes_to_evaluate.at(i * 2 + 1);
        driver.Push(left_node);
        driver.Push(right_node);
      }
    }
  }
  builder_monitor_->Stop("ExpandTree");
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::Update(
    const GHistIndexMatrix &gmat,
    const common::ColumnMatrix &column_matrix,
    HostDeviceVector<GradientPair> *gpair,
    DMatrix *p_fmat, RegTree *p_tree) {
  builder_monitor_->Start("Update");

  std::vector<GradientPair>* gpair_ptr = &(gpair->HostVector());
  // in case 'num_parallel_trees != 1' no posibility to change initial gpair
  if (GetNumberOfTrees() != 1) {
    gpair_local_.resize(gpair_ptr->size());
    gpair_local_ = *gpair_ptr;
    gpair_ptr = &gpair_local_;
  }
  p_last_fmat_mutable_ = p_fmat;

  this->InitData(gmat, p_fmat, *p_tree, gpair_ptr);

  if (column_matrix.AnyMissing()) {
    ExpandTree<true>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
  } else {
    ExpandTree<false>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
  }
  pruner_->Update(gpair, p_fmat, std::vector<RegTree*>{p_tree});

  builder_monitor_->Stop("Update");
}

template <typename GradientSumT>
bool QuantileHistMaker::Builder<GradientSumT>::UpdatePredictionCache(
    DMatrix const *data, linalg::VectorView<float> out_preds) const {
  // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
  // conjunction with Update().
  if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_ ||
      p_last_fmat_ != p_last_fmat_mutable_) {
    return false;
  }
  builder_monitor_->Start(__func__);
  CHECK_EQ(out_preds.Size(), data->Info().num_row_);
  UpdatePredictionCacheImpl(ctx_, p_last_tree_, partitioner_, *evaluator_, param_, out_preds);
  builder_monitor_->Stop(__func__);
  return true;
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitSampling(const DMatrix &fmat,
                                                            std::vector<GradientPair> *gpair) {
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
}
template<typename GradientSumT>
size_t QuantileHistMaker::Builder<GradientSumT>::GetNumberOfTrees() {
  return n_trees_;
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitData(const GHistIndexMatrix &gmat, DMatrix *fmat,
                                                        const RegTree &tree,
                                                        std::vector<GradientPair> *gpair) {
  builder_monitor_->Start("InitData");
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
    histogram_builder_->Reset(n_total_bins, BatchParam{param_.max_bin, param_.sparse_threshold},
                              ctx_->Threads(), page_id, rabit::IsDistributed());

    if (param_.subsample < 1.0f) {
      CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
        << "Only uniform sampling is supported, "
        << "gradient-based sampling is only support by GPU Hist.";
      builder_monitor_->Start("InitSampling");
      InitSampling(*fmat, gpair);
      builder_monitor_->Stop("InitSampling");
      // We should check that the partitioning was done correctly
      // and each row of the dataset fell into exactly one of the categories
    }
  }

  // store a pointer to the tree
  p_last_tree_ = &tree;
  evaluator_.reset(new HistEvaluator<GradientSumT, CPUExpandEntry>{
      param_, info, this->ctx_->Threads(), column_sampler_, task_});

  builder_monitor_->Stop("InitData");
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

XGBOOST_REGISTER_TREE_UPDATER(FastHistMaker, "grow_fast_histmaker")
.describe("(Deprecated, use grow_quantile_histmaker instead.)"
          " Grow tree using quantized histogram.")
.set_body(
    [](ObjInfo task) {
      LOG(WARNING) << "grow_fast_histmaker is deprecated, "
                   << "use grow_quantile_histmaker instead.";
      return new QuantileHistMaker(task);
    });

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMaker, "grow_quantile_histmaker")
.describe("Grow tree using quantized histogram.")
.set_body(
    [](ObjInfo task) {
      return new QuantileHistMaker(task);
    });
}  // namespace tree
}  // namespace xgboost
