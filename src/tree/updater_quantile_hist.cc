/**
 * Copyright 2017-2023 by XGBoost Contributors
 * \file updater_quantile_hist.cc
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Checn, Egor Smirnov
 */
#include <algorithm>                         // for max
#include <cstddef>                           // for size_t
#include <cstdint>                           // for uint32_t
#include <memory>                            // for unique_ptr, allocator, make_unique, make_shared
#include <ostream>                           // for operator<<, char_traits, basic_ostream
#include <tuple>                             // for apply
#include <utility>                           // for move, swap
#include <vector>                            // for vector

#include "../collective/communicator-inl.h"  // for Allreduce, IsDistributed
#include "../collective/communicator.h"      // for Operation
#include "../common/hist_util.h"             // for HistogramCuts, HistCollection
#include "../common/random.h"                // for ColumnSampler
#include "../common/threading_utils.h"       // for ParallelFor
#include "../common/timer.h"                 // for Monitor
#include "../data/gradient_index.h"          // for GHistIndexMatrix
#include "common_row_partitioner.h"          // for CommonRowPartitioner
#include "dmlc/registry.h"                   // for DMLC_REGISTRY_FILE_TAG
#include "driver.h"                          // for Driver
#include "hist/evaluate_splits.h"            // for HistEvaluator, UpdatePredictionCacheImpl
#include "hist/expand_entry.h"               // for CPUExpandEntry
#include "hist/histogram.h"                  // for HistogramBuilder, ConstructHistSpace
#include "hist/sampler.h"                    // for SampleGradient
#include "param.h"                           // for TrainParam, GradStats
#include "xgboost/base.h"                    // for GradientPair, GradientPairInternal, bst_node_t
#include "xgboost/context.h"                 // for Context
#include "xgboost/data.h"                    // for BatchIterator, BatchSet, DMatrix, MetaInfo
#include "xgboost/host_device_vector.h"      // for HostDeviceVector
#include "xgboost/linalg.h"                  // for TensorView, MatrixView, UnravelIndex, All
#include "xgboost/logging.h"                 // for LogCheck_EQ, LogCheck_GE, CHECK_EQ, LOG, LOG...
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

class HistBuilder {
 private:
  common::Monitor *monitor_;
  TrainParam const *param_;
  std::shared_ptr<common::ColumnSampler> col_sampler_;
  std::unique_ptr<HistEvaluator<CPUExpandEntry>> evaluator_;
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
        evaluator_{std::make_unique<HistEvaluator<CPUExpandEntry>>(ctx, param, fmat->Info(),
                                                                   col_sampler_)},
        p_last_fmat_(fmat),
        histogram_builder_{new HistogramBuilder<CPUExpandEntry>},
        task_{task},
        ctx_{ctx} {
    monitor_->Init(__func__);
  }

  bool UpdatePredictionCache(DMatrix const *data, linalg::VectorView<float> out_preds) const {
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

    size_t page_id{0};
    bst_bin_t n_total_bins{0};
    partitioner_.clear();
    for (auto const &page : fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
      if (n_total_bins == 0) {
        n_total_bins = page.cut.TotalBins();
      } else {
        CHECK_EQ(n_total_bins, page.cut.TotalBins());
      }
      partitioner_.emplace_back(this->ctx_, page.Size(), page.base_rowid, fmat->IsColumnSplit());
      ++page_id;
    }
    histogram_builder_->Reset(n_total_bins, HistBatch(param_), ctx_->Threads(), page_id,
                              collective::IsDistributed(), fmat->IsColumnSplit());
    evaluator_ = std::make_unique<HistEvaluator<CPUExpandEntry>>(ctx_, this->param_, fmat->Info(),
                                                                 col_sampler_);
    p_last_tree_ = p_tree;
  }

  void EvaluateSplits(DMatrix *p_fmat, RegTree const *p_tree,
                      std::vector<CPUExpandEntry> *best_splits) {
    monitor_->Start(__func__);
    auto const &histograms = histogram_builder_->Histogram();
    auto ft = p_fmat->Info().feature_types.ConstHostSpan();
    for (auto const &gmat : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
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

    size_t page_id = 0;
    auto space = ConstructHistSpace(partitioner_, {node});
    for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
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
        auto gpair_h = gpair.Slice(linalg::All(), 0).Values();
        for (auto const &grad : gpair_h) {
          grad_stat.Add(grad.GetGrad(), grad.GetHess());
        }
        collective::Allreduce<collective::Operation::kSum>(reinterpret_cast<double *>(&grad_stat),
                                                           2);
      }

      auto weight = evaluator_->InitRoot(GradStats{grad_stat});
      p_tree->Stat(RegTree::kRoot).sum_hess = grad_stat.GetHess();
      p_tree->Stat(RegTree::kRoot).base_weight = weight;
      (*p_tree)[RegTree::kRoot].SetLeaf(param_->learning_rate * weight);

      std::vector<CPUExpandEntry> entries{node};
      monitor_->Start("EvaluateSplits");
      auto ft = p_fmat->Info().feature_types.ConstHostSpan();
      for (auto const &gmat : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(param_))) {
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
                                    nodes_to_sub, gpair.Values());
      ++page_id;
    }
  }

  void UpdatePosition(DMatrix *p_fmat, RegTree const *p_tree,
                      std::vector<CPUExpandEntry> const &applied) {
    monitor_->Start(__func__);
    std::size_t page_id{0};
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(HistBatch(this->param_))) {
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
  std::unique_ptr<HistBuilder> p_impl_;
  std::shared_ptr<common::ColumnSampler> column_sampler_ =
      std::make_shared<common::ColumnSampler>();
  common::Monitor monitor_;
  ObjInfo const *task_;

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
      LOG(FATAL) << "Not implemented.";
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
        // Copy gradient into buffer for sampling.
        common::ParallelFor(h_gpair.Size(), ctx_->Threads(), [&](auto i) {
          std::apply(h_sample_out, linalg::UnravelIndex(i, h_gpair.Shape())) =
              std::apply(h_gpair, linalg::UnravelIndex(i, h_gpair.Shape()));
        });
      }
      SampleGradient(ctx_, *param, h_sample_out);
      auto *h_out_position = &out_position[tree_it - trees.begin()];
      if ((*tree_it)->IsMultiTarget()) {
        LOG(FATAL) << "Not implemented.";
      } else {
        UpdateTree<CPUExpandEntry>(&monitor_, h_sample_out, p_impl_.get(), p_fmat, param,
                                   h_out_position, *tree_it);
      }
    }
  }

  bool UpdatePredictionCache(const DMatrix *data, linalg::VectorView<float> out_preds) override {
    if (p_impl_) {
      return p_impl_->UpdatePredictionCache(data, out_preds);
    } else {
      return false;
    }
  }

  [[nodiscard]] bool HasNodePosition() const override { return true; }
};

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMaker, "grow_quantile_histmaker")
    .describe("Grow tree using quantized histogram.")
    .set_body([](Context const *ctx, ObjInfo const *task) {
      return new QuantileHistMaker(ctx, task);
    });
}  // namespace xgboost::tree
