/**
 * Copyright 2017-2023 by XGBoost Contributors
 * \file updater_quantile_hist.cc
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Checn, Egor Smirnov
 */
#include <algorithm>                         // for max, copy, transform
#include <cstddef>                           // for size_t
#include <cstdint>                           // for uint32_t, int32_t
#include <memory>                            // for unique_ptr, allocator, make_unique, shared_ptr
#include <numeric>                           // for accumulate
#include <ostream>                           // for basic_ostream, char_traits, operator<<
#include <utility>                           // for move, swap
#include <vector>                            // for vector

#include "../collective/aggregator.h"        // for GlobalSum
#include "../collective/communicator-inl.h"  // for Allreduce, IsDistributed
#include "../collective/communicator.h"      // for Operation
#include "../common/hist_util.h"             // for HistogramCuts, HistCollection
#include "../common/linalg_op.h"             // for begin, cbegin, cend
#include "../common/random.h"                // for ColumnSampler
#include "../common/threading_utils.h"       // for ParallelFor
#include "../common/timer.h"                 // for Monitor
#include "../common/transform_iterator.h"    // for IndexTransformIter, MakeIndexTransformIter
#include "../data/gradient_index.h"          // for GHistIndexMatrix
#include "common_row_partitioner.h"          // for CommonRowPartitioner
#include "dmlc/omp.h"                        // for omp_get_thread_num
#include "dmlc/registry.h"                   // for DMLC_REGISTRY_FILE_TAG
#include "driver.h"                          // for Driver
#include "hist/evaluate_splits.h"            // for HistEvaluator, HistMultiEvaluator, UpdatePre...
#include "hist/expand_entry.h"               // for MultiExpandEntry, CPUExpandEntry
#include "hist/histogram.h"                  // for HistogramBuilder, ConstructHistSpace
#include "hist/sampler.h"                    // for SampleGradient
#include "param.h"                           // for TrainParam, SplitEntryContainer, GradStats
#include "xgboost/base.h"                    // for GradientPairInternal, GradientPair, bst_targ...
#include "xgboost/context.h"                 // for Context
#include "xgboost/data.h"                    // for BatchIterator, BatchSet, DMatrix, MetaInfo
#include "xgboost/host_device_vector.h"      // for HostDeviceVector
#include "xgboost/linalg.h"                  // for All, MatrixView, TensorView, Matrix, Empty
#include "xgboost/logging.h"                 // for LogCheck_EQ, CHECK_EQ, CHECK, LogCheck_GE
#include "xgboost/span.h"                    // for Span, operator!=, SpanIterator
#include "xgboost/string_view.h"             // for operator<<
#include "xgboost/task.h"                    // for ObjInfo
#include "xgboost/tree_model.h"              // for RegTree, MTNotImplemented, RTreeNodeStat
#include "xgboost/tree_updater.h"            // for TreeUpdater, TreeUpdaterReg, XGBOOST_REGISTE...

namespace xgboost::tree {

DMLC_REGISTRY_FILE_TAG(updater_quantile_hist);

BatchParam HistBatch(TrainParam const *param) { return {param->max_bin, param->sparse_threshold}; }

template <typename ExpandEntry, typename Updater>
void UpdateTree(common::Monitor *monitor_, linalg::MatrixView<GradientPair const> gpair,
                Updater *updater, DMatrix *p_fmat, TrainParam const *param,
                HostDeviceVector<bst_node_t> *p_out_position, RegTree *p_tree) {
  monitor_->Start(__func__);
  updater->InitData(p_fmat, p_tree);

  Driver<ExpandEntry> driver{*param};
  auto const &tree = *p_tree;
  driver.Push(updater->InitRoot(p_fmat, gpair, p_tree));
  auto expand_set = driver.Pop();

  /**
   * Note for update position
   * Root:
   *   Not applied: No need to update position as initialization has got all the rows ordered.
   *   Applied: Update position is run on applied nodes so the rows are partitioned.
   * Non-root:
   *   Not applied: That node is root of the subtree, same rule as root.
   *   Applied: Ditto
   */
  while (!expand_set.empty()) {
    // candidates that can be further splited.
    std::vector<ExpandEntry> valid_candidates;
    // candidaates that can be applied.
    std::vector<ExpandEntry> applied;
    for (auto const &candidate : expand_set) {
      updater->ApplyTreeSplit(candidate, p_tree);
      CHECK_GT(p_tree->LeftChild(candidate.nid), candidate.nid);
      applied.push_back(candidate);
      if (driver.IsChildValid(candidate)) {
        valid_candidates.emplace_back(candidate);
      }
    }

    updater->UpdatePosition(p_fmat, p_tree, applied);

    std::vector<ExpandEntry> best_splits;
    if (!valid_candidates.empty()) {
      updater->BuildHistogram(p_fmat, p_tree, valid_candidates, gpair);
      for (auto const &candidate : valid_candidates) {
        auto left_child_nidx = tree.LeftChild(candidate.nid);
        auto right_child_nidx = tree.RightChild(candidate.nid);
        ExpandEntry l_best{left_child_nidx, tree.GetDepth(left_child_nidx)};
        ExpandEntry r_best{right_child_nidx, tree.GetDepth(right_child_nidx)};
        best_splits.push_back(l_best);
        best_splits.push_back(r_best);
      }
      updater->EvaluateSplits(p_fmat, p_tree, &best_splits);
    }
    driver.Push(best_splits.begin(), best_splits.end());
    expand_set = driver.Pop();
  }

  auto &h_out_position = p_out_position->HostVector();
  updater->LeafPartition(tree, gpair, &h_out_position);
  monitor_->Stop(__func__);
}

/**
 * \brief Updater for building multi-target trees. The implementation simply iterates over
 *        each target.
 */
class MultiTargetHistBuilder {
 private:
  common::Monitor *monitor_{nullptr};
  TrainParam const *param_{nullptr};
  std::shared_ptr<common::ColumnSampler> col_sampler_;
  std::unique_ptr<HistMultiEvaluator> evaluator_;
  // Histogram builder for each target.
  std::vector<HistogramBuilder<MultiExpandEntry>> histogram_builder_;
  Context const *ctx_{nullptr};
  // Partitioner for each data batch.
  std::vector<CommonRowPartitioner> partitioner_;
  // Pointer to last updated tree, used for update prediction cache.
  RegTree const *p_last_tree_{nullptr};
  DMatrix const * p_last_fmat_{nullptr};

  ObjInfo const *task_{nullptr};

 public:
  void UpdatePosition(DMatrix *p_fmat, RegTree const *p_tree,
                      std::vector<MultiExpandEntry> const &applied) {
    monitor_->Start(__func__);
    std::size_t page_id{0};
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(this->param_))) {
      this->partitioner_.at(page_id).UpdatePosition(this->ctx_, page, applied, p_tree);
      page_id++;
    }
    monitor_->Stop(__func__);
  }

  void ApplyTreeSplit(MultiExpandEntry const &candidate, RegTree *p_tree) {
    this->evaluator_->ApplyTreeSplit(candidate, p_tree);
  }

  void InitData(DMatrix *p_fmat, RegTree const *p_tree) {
    monitor_->Start(__func__);

    p_last_fmat_ = p_fmat;
    std::size_t page_id = 0;
    bst_bin_t n_total_bins = 0;
    partitioner_.clear();
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      if (n_total_bins == 0) {
        n_total_bins = page.cut.TotalBins();
      } else {
        CHECK_EQ(n_total_bins, page.cut.TotalBins());
      }
      partitioner_.emplace_back(ctx_, page.Size(), page.base_rowid, p_fmat->Info().IsColumnSplit());
      page_id++;
    }

    bst_target_t n_targets = p_tree->NumTargets();
    histogram_builder_.clear();
    for (std::size_t i = 0; i < n_targets; ++i) {
      histogram_builder_.emplace_back();
      histogram_builder_.back().Reset(n_total_bins, HistBatch(param_), ctx_->Threads(), page_id,
                                      collective::IsDistributed(), p_fmat->Info().IsColumnSplit());
    }

    evaluator_ = std::make_unique<HistMultiEvaluator>(ctx_, p_fmat->Info(), param_, col_sampler_);
    p_last_tree_ = p_tree;
    monitor_->Stop(__func__);
  }

  MultiExpandEntry InitRoot(DMatrix *p_fmat, linalg::MatrixView<GradientPair const> gpair,
                            RegTree *p_tree) {
    monitor_->Start(__func__);
    MultiExpandEntry best;
    best.nid = RegTree::kRoot;
    best.depth = 0;

    auto n_targets = p_tree->NumTargets();
    linalg::Matrix<GradientPairPrecise> root_sum_tloc =
        linalg::Empty<GradientPairPrecise>(ctx_, ctx_->Threads(), n_targets);
    CHECK_EQ(root_sum_tloc.Shape(1), gpair.Shape(1));
    auto h_root_sum_tloc = root_sum_tloc.HostView();
    common::ParallelFor(gpair.Shape(0), ctx_->Threads(), [&](auto i) {
      for (bst_target_t t{0}; t < n_targets; ++t) {
        h_root_sum_tloc(omp_get_thread_num(), t) += GradientPairPrecise{gpair(i, t)};
      }
    });
    // Aggregate to the first row.
    auto root_sum = h_root_sum_tloc.Slice(0, linalg::All());
    for (std::int32_t tidx{1}; tidx < ctx_->Threads(); ++tidx) {
      for (bst_target_t t{0}; t < n_targets; ++t) {
        root_sum(t) += h_root_sum_tloc(tidx, t);
      }
    }
    CHECK(root_sum.CContiguous());
    collective::GlobalSum(p_fmat->Info(), reinterpret_cast<double *>(root_sum.Values().data()),
                          root_sum.Size() * 2);

    std::vector<MultiExpandEntry> nodes{best};
    std::size_t i = 0;
    auto space = ConstructHistSpace(partitioner_, nodes);
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      for (bst_target_t t{0}; t < n_targets; ++t) {
        auto t_gpair = gpair.Slice(linalg::All(), t);
        histogram_builder_[t].BuildHist(i, space, page, p_tree, partitioner_.at(i).Partitions(),
                                        nodes, {}, t_gpair.Values());
      }
      i++;
    }

    auto weight = evaluator_->InitRoot(root_sum);
    auto weight_t = weight.HostView();
    std::transform(linalg::cbegin(weight_t), linalg::cend(weight_t), linalg::begin(weight_t),
                   [&](float w) { return w * param_->learning_rate; });

    p_tree->SetLeaf(RegTree::kRoot, weight_t);
    std::vector<common::HistCollection const *> hists;
    for (bst_target_t t{0}; t < p_tree->NumTargets(); ++t) {
      hists.push_back(&histogram_builder_[t].Histogram());
    }
    for (auto const &gmat : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      evaluator_->EvaluateSplits(*p_tree, hists, gmat.cut, &nodes);
      break;
    }
    monitor_->Stop(__func__);

    return nodes.front();
  }

  void BuildHistogram(DMatrix *p_fmat, RegTree const *p_tree,
                      std::vector<MultiExpandEntry> const &valid_candidates,
                      linalg::MatrixView<GradientPair const> gpair) {
    monitor_->Start(__func__);
    std::vector<MultiExpandEntry> nodes_to_build;
    std::vector<MultiExpandEntry> nodes_to_sub;

    for (auto const &c : valid_candidates) {
      auto left_nidx = p_tree->LeftChild(c.nid);
      auto right_nidx = p_tree->RightChild(c.nid);

      auto build_nidx = left_nidx;
      auto subtract_nidx = right_nidx;
      auto lit =
          common::MakeIndexTransformIter([&](auto i) { return c.split.left_sum[i].GetHess(); });
      auto left_sum = std::accumulate(lit, lit + c.split.left_sum.size(), .0);
      auto rit =
          common::MakeIndexTransformIter([&](auto i) { return c.split.right_sum[i].GetHess(); });
      auto right_sum = std::accumulate(rit, rit + c.split.right_sum.size(), .0);
      auto fewer_right = right_sum < left_sum;
      if (fewer_right) {
        std::swap(build_nidx, subtract_nidx);
      }
      nodes_to_build.emplace_back(build_nidx, p_tree->GetDepth(build_nidx));
      nodes_to_sub.emplace_back(subtract_nidx, p_tree->GetDepth(subtract_nidx));
    }

    std::size_t i = 0;
    auto space = ConstructHistSpace(partitioner_, nodes_to_build);
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      for (std::size_t t = 0; t < p_tree->NumTargets(); ++t) {
        auto t_gpair = gpair.Slice(linalg::All(), t);
        // Make sure the gradient matrix is f-order.
        CHECK(t_gpair.Contiguous());
        histogram_builder_[t].BuildHist(i, space, page, p_tree, partitioner_.at(i).Partitions(),
                                        nodes_to_build, nodes_to_sub, t_gpair.Values());
      }
      i++;
    }
    monitor_->Stop(__func__);
  }

  void EvaluateSplits(DMatrix *p_fmat, RegTree const *p_tree,
                      std::vector<MultiExpandEntry> *best_splits) {
    monitor_->Start(__func__);
    std::vector<common::HistCollection const *> hists;
    for (bst_target_t t{0}; t < p_tree->NumTargets(); ++t) {
      hists.push_back(&histogram_builder_[t].Histogram());
    }
    for (auto const &gmat : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      evaluator_->EvaluateSplits(*p_tree, hists, gmat.cut, best_splits);
      break;
    }
    monitor_->Stop(__func__);
  }

  void LeafPartition(RegTree const &tree, linalg::MatrixView<GradientPair const> gpair,
                     std::vector<bst_node_t> *p_out_position) {
    monitor_->Start(__func__);
    if (!task_->UpdateTreeLeaf()) {
      monitor_->Stop(__func__);
      return;
    }
    for (auto const &part : partitioner_) {
      part.LeafPartition(ctx_, tree, gpair, p_out_position);
    }
    monitor_->Stop(__func__);
  }

 public:
  explicit MultiTargetHistBuilder(Context const *ctx, MetaInfo const &info, TrainParam const *param,
                                  std::shared_ptr<common::ColumnSampler> column_sampler,
                                  ObjInfo const *task, common::Monitor *monitor)
      : monitor_{monitor},
        param_{param},
        col_sampler_{std::move(column_sampler)},
        evaluator_{std::make_unique<HistMultiEvaluator>(ctx, info, param, col_sampler_)},
        ctx_{ctx},
        task_{task} {
    monitor_->Init(__func__);
  }

  bool UpdatePredictionCache(DMatrix const *data, linalg::MatrixView<float> out_preds) const {
    // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
    // conjunction with Update().
    if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_) {
      return false;
    }
    monitor_->Start(__func__);
    CHECK_EQ(out_preds.Size(), data->Info().num_row_ * p_last_tree_->NumTargets());
    UpdatePredictionCacheImpl(ctx_, p_last_tree_, partitioner_, out_preds);
    monitor_->Stop(__func__);
    return true;
  }
};

class HistBuilder {
 private:
  common::Monitor *monitor_;
  TrainParam const *param_;
  std::shared_ptr<common::ColumnSampler> col_sampler_;
  std::unique_ptr<HistEvaluator> evaluator_;
  std::vector<CommonRowPartitioner> partitioner_;

  // back pointers to tree and data matrix
  const RegTree *p_last_tree_{nullptr};
  DMatrix const *const p_last_fmat_{nullptr};

  std::unique_ptr<HistogramBuilder<CPUExpandEntry>> histogram_builder_;
  ObjInfo const *task_{nullptr};
  // Context for number of threads
  Context const *ctx_{nullptr};

 public:
  explicit HistBuilder(Context const *ctx, std::shared_ptr<common::ColumnSampler> column_sampler,
                       TrainParam const *param, DMatrix const *fmat, ObjInfo const *task,
                       common::Monitor *monitor)
      : monitor_{monitor},
        param_{param},
        col_sampler_{std::move(column_sampler)},
        evaluator_{std::make_unique<HistEvaluator>(ctx, param, fmat->Info(),
                                                                   col_sampler_)},
        p_last_fmat_(fmat),
        histogram_builder_{new HistogramBuilder<CPUExpandEntry>},
        task_{task},
        ctx_{ctx} {
    monitor_->Init(__func__);
  }

  bool UpdatePredictionCache(DMatrix const *data, linalg::MatrixView<float> out_preds) const {
    // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
    // conjunction with Update().
    if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_) {
      return false;
    }
    monitor_->Start(__func__);
    CHECK_EQ(out_preds.Size(), data->Info().num_row_);
    UpdatePredictionCacheImpl(ctx_, p_last_tree_, partitioner_, out_preds);
    monitor_->Stop(__func__);
    return true;
  }

 public:
  // initialize temp data structure
  void InitData(DMatrix *fmat, RegTree const *p_tree) {
    monitor_->Start(__func__);
    std::size_t page_id{0};
    bst_bin_t n_total_bins{0};
    partitioner_.clear();
    for (auto const &page : fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      if (n_total_bins == 0) {
        n_total_bins = page.cut.TotalBins();
      } else {
        CHECK_EQ(n_total_bins, page.cut.TotalBins());
      }
      partitioner_.emplace_back(this->ctx_, page.Size(), page.base_rowid,
                                fmat->Info().IsColumnSplit());
      ++page_id;
    }
    histogram_builder_->Reset(n_total_bins, HistBatch(param_), ctx_->Threads(), page_id,
                              collective::IsDistributed(), fmat->Info().IsColumnSplit());
    evaluator_ = std::make_unique<HistEvaluator>(ctx_, this->param_, fmat->Info(), col_sampler_);
    p_last_tree_ = p_tree;
    monitor_->Stop(__func__);
  }

  void EvaluateSplits(DMatrix *p_fmat, RegTree const *p_tree,
                      std::vector<CPUExpandEntry> *best_splits) {
    monitor_->Start(__func__);
    auto const &histograms = histogram_builder_->Histogram();
    auto ft = p_fmat->Info().feature_types.ConstHostSpan();
    for (auto const &gmat : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      evaluator_->EvaluateSplits(histograms, gmat.cut, ft, *p_tree, best_splits);
      break;
    }
    monitor_->Stop(__func__);
  }

  void ApplyTreeSplit(CPUExpandEntry const &candidate, RegTree *p_tree) {
    this->evaluator_->ApplyTreeSplit(candidate, p_tree);
  }

  CPUExpandEntry InitRoot(DMatrix *p_fmat, linalg::MatrixView<GradientPair const> gpair,
                          RegTree *p_tree) {
    CPUExpandEntry node(RegTree::kRoot, p_tree->GetDepth(0));

    std::size_t page_id = 0;
    auto space = ConstructHistSpace(partitioner_, {node});
    for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      std::vector<CPUExpandEntry> nodes_to_build{node};
      std::vector<CPUExpandEntry> nodes_to_sub;
      this->histogram_builder_->BuildHist(page_id, space, gidx, p_tree,
                                          partitioner_.at(page_id).Partitions(), nodes_to_build,
                                          nodes_to_sub, gpair.Slice(linalg::All(), 0).Values());
      ++page_id;
    }

    {
      GradientPairPrecise grad_stat;
      if (p_fmat->IsDense()) {
        /**
         * Specialized code for dense data: For dense data (with no missing value), the sum
         * of gradient histogram is equal to snode[nid]
         */
        auto const &gmat = *(p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_)).begin());
        std::vector<std::uint32_t> const &row_ptr = gmat.cut.Ptrs();
        CHECK_GE(row_ptr.size(), 2);
        std::uint32_t const ibegin = row_ptr[0];
        std::uint32_t const iend = row_ptr[1];
        auto hist = this->histogram_builder_->Histogram()[RegTree::kRoot];
        auto begin = hist.data();
        for (std::uint32_t i = ibegin; i < iend; ++i) {
          GradientPairPrecise const &et = begin[i];
          grad_stat.Add(et.GetGrad(), et.GetHess());
        }
      } else {
        auto gpair_h = gpair.Slice(linalg::All(), 0).Values();
        for (auto const &grad : gpair_h) {
          grad_stat.Add(grad.GetGrad(), grad.GetHess());
        }
        collective::GlobalSum(p_fmat->Info(), reinterpret_cast<double *>(&grad_stat), 2);
      }

      auto weight = evaluator_->InitRoot(GradStats{grad_stat});
      p_tree->Stat(RegTree::kRoot).sum_hess = grad_stat.GetHess();
      p_tree->Stat(RegTree::kRoot).base_weight = weight;
      (*p_tree)[RegTree::kRoot].SetLeaf(param_->learning_rate * weight);

      std::vector<CPUExpandEntry> entries{node};
      monitor_->Start("EvaluateSplits");
      auto ft = p_fmat->Info().feature_types.ConstHostSpan();
      for (auto const &gmat : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
        evaluator_->EvaluateSplits(histogram_builder_->Histogram(), gmat.cut, ft, *p_tree,
                                   &entries);
        break;
      }
      monitor_->Stop("EvaluateSplits");
      node = entries.front();
    }

    return node;
  }

  void BuildHistogram(DMatrix *p_fmat, RegTree *p_tree,
                      std::vector<CPUExpandEntry> const &valid_candidates,
                      linalg::MatrixView<GradientPair const> gpair) {
    std::vector<CPUExpandEntry> nodes_to_build(valid_candidates.size());
    std::vector<CPUExpandEntry> nodes_to_sub(valid_candidates.size());

    std::size_t n_idx = 0;
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

    std::size_t page_id{0};
    auto space = ConstructHistSpace(partitioner_, nodes_to_build);
    for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      histogram_builder_->BuildHist(page_id, space, gidx, p_tree,
                                    partitioner_.at(page_id).Partitions(), nodes_to_build,
                                    nodes_to_sub, gpair.Values());
      ++page_id;
    }
  }

  void UpdatePosition(DMatrix *p_fmat, RegTree const *p_tree,
                      std::vector<CPUExpandEntry> const &applied) {
    monitor_->Start(__func__);
    std::size_t page_id{0};
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      this->partitioner_.at(page_id).UpdatePosition(this->ctx_, page, applied, p_tree);
      page_id++;
    }
    monitor_->Stop(__func__);
  }

  void LeafPartition(RegTree const &tree, linalg::MatrixView<GradientPair const> gpair,
                     std::vector<bst_node_t> *p_out_position) {
    monitor_->Start(__func__);
    if (!task_->UpdateTreeLeaf()) {
      return;
    }
    for (auto const &part : partitioner_) {
      part.LeafPartition(ctx_, tree, gpair, p_out_position);
    }
    monitor_->Stop(__func__);
  }
};

/*! \brief construct a tree using quantized feature values */
class QuantileHistMaker : public TreeUpdater {
  std::unique_ptr<HistBuilder> p_impl_{nullptr};
  std::unique_ptr<MultiTargetHistBuilder> p_mtimpl_{nullptr};
  std::shared_ptr<common::ColumnSampler> column_sampler_ =
      std::make_shared<common::ColumnSampler>();
  common::Monitor monitor_;
  ObjInfo const *task_{nullptr};

 public:
  explicit QuantileHistMaker(Context const *ctx, ObjInfo const *task)
      : TreeUpdater{ctx}, task_{task} {}
  void Configure(const Args &) override {}

  void LoadConfig(Json const &) override {}
  void SaveConfig(Json *) const override {}

  [[nodiscard]] char const *Name() const override { return "grow_quantile_histmaker"; }

  void Update(TrainParam const *param, HostDeviceVector<GradientPair> *gpair, DMatrix *p_fmat,
              common::Span<HostDeviceVector<bst_node_t>> out_position,
              const std::vector<RegTree *> &trees) override {
    if (trees.front()->IsMultiTarget()) {
      CHECK(param->monotone_constraints.empty()) << "monotone constraint" << MTNotImplemented();
      if (!p_mtimpl_) {
        this->p_mtimpl_ = std::make_unique<MultiTargetHistBuilder>(
            ctx_, p_fmat->Info(), param, column_sampler_, task_, &monitor_);
      }
    } else {
      if (!p_impl_) {
        p_impl_ =
            std::make_unique<HistBuilder>(ctx_, column_sampler_, param, p_fmat, task_, &monitor_);
      }
    }

    bst_target_t n_targets = trees.front()->NumTargets();
    auto h_gpair =
        linalg::MakeTensorView(ctx_, gpair->HostSpan(), p_fmat->Info().num_row_, n_targets);

    linalg::Matrix<GradientPair> sample_out;
    auto h_sample_out = h_gpair;
    auto need_copy = [&] { return trees.size() > 1 || n_targets > 1; };
    if (need_copy()) {
      // allocate buffer
      sample_out = decltype(sample_out){h_gpair.Shape(), ctx_->gpu_id, linalg::Order::kF};
      h_sample_out = sample_out.HostView();
    }

    for (auto tree_it = trees.begin(); tree_it != trees.end(); ++tree_it) {
      if (need_copy()) {
        // Copy gradient into buffer for sampling. This converts C-order to F-order.
        std::copy(linalg::cbegin(h_gpair), linalg::cend(h_gpair), linalg::begin(h_sample_out));
      }
      SampleGradient(ctx_, *param, h_sample_out);
      auto *h_out_position = &out_position[tree_it - trees.begin()];
      if ((*tree_it)->IsMultiTarget()) {
        UpdateTree<MultiExpandEntry>(&monitor_, h_sample_out, p_mtimpl_.get(), p_fmat, param,
                                     h_out_position, *tree_it);
      } else {
        UpdateTree<CPUExpandEntry>(&monitor_, h_sample_out, p_impl_.get(), p_fmat, param,
                                   h_out_position, *tree_it);
      }
    }
  }

  bool UpdatePredictionCache(const DMatrix *data, linalg::MatrixView<float> out_preds) override {
    if (p_impl_) {
      return p_impl_->UpdatePredictionCache(data, out_preds);
    } else if (p_mtimpl_) {
      return p_mtimpl_->UpdatePredictionCache(data, out_preds);
    } else {
      return false;
    }
  }

  [[nodiscard]] bool HasNodePosition() const override { return true; }
};

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMaker, "grow_quantile_histmaker")
    .describe("Grow tree using quantized histogram.")
    .set_body([](Context const *ctx, ObjInfo const *task) {
      return new QuantileHistMaker{ctx, task};
    });
}  // namespace xgboost::tree
