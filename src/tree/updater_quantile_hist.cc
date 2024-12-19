/**
 * Copyright 2017-2024, XGBoost Contributors
 * \file updater_quantile_hist.cc
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Checn, Egor Smirnov
 */
#include <algorithm>  // for max, copy, transform
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t, int32_t
#include <memory>     // for allocator, unique_ptr, make_unique, shared_ptr
#include <ostream>    // for operator<<, basic_ostream, char_traits
#include <utility>    // for move
#include <vector>     // for vector

#include "../collective/aggregator.h"        // for GlobalSum
#include "../collective/communicator-inl.h"  // for IsDistributed
#include "../common/hist_util.h"             // for HistogramCuts, GHistRow
#include "../common/linalg_op.h"             // for begin, cbegin, cend
#include "../common/random.h"                // for ColumnSampler
#include "../common/threading_utils.h"       // for ParallelFor
#include "../common/timer.h"                 // for Monitor
#include "../data/gradient_index.h"          // for GHistIndexMatrix
#include "common_row_partitioner.h"          // for CommonRowPartitioner
#include "dmlc/registry.h"                   // for DMLC_REGISTRY_FILE_TAG
#include "driver.h"                          // for Driver
#include "hist/evaluate_splits.h"            // for HistEvaluator, HistMultiEvaluator, UpdatePre...
#include "hist/expand_entry.h"               // for MultiExpandEntry, CPUExpandEntry
#include "hist/hist_cache.h"                 // for BoundedHistCollection
#include "hist/histogram.h"                  // for MultiHistogramBuilder
#include "hist/param.h"                      // for HistMakerTrainParam
#include "hist/sampler.h"                    // for SampleGradient
#include "param.h"                           // for TrainParam, GradStats
#include "xgboost/base.h"                    // for Args, GradientPairPrecise, GradientPair, Gra...
#include "xgboost/context.h"                 // for Context
#include "xgboost/data.h"                    // for BatchSet, DMatrix, BatchIterator, MetaInfo
#include "xgboost/host_device_vector.h"      // for HostDeviceVector
#include "xgboost/json.h"                    // for Object, Json, FromJson, ToJson, get
#include "xgboost/linalg.h"                  // for MatrixView, TensorView, All, Matrix, Empty
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
  HistMakerTrainParam const *hist_param_{nullptr};
  std::shared_ptr<common::ColumnSampler> col_sampler_;
  std::unique_ptr<HistMultiEvaluator> evaluator_;
  // Histogram builder for each target.
  std::unique_ptr<MultiHistogramBuilder> histogram_builder_;
  Context const *ctx_{nullptr};
  // Partitioner for each data batch.
  std::vector<CommonRowPartitioner> partitioner_;
  // Pointer to last updated tree, used for update prediction cache.
  RegTree const *p_last_tree_{nullptr};
  DMatrix const *p_last_fmat_{nullptr};

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
    bst_bin_t n_total_bins = 0;
    partitioner_.clear();
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      if (n_total_bins == 0) {
        n_total_bins = page.cut.TotalBins();
      } else {
        CHECK_EQ(n_total_bins, page.cut.TotalBins());
      }
      partitioner_.emplace_back(ctx_, page.Size(), page.base_rowid, p_fmat->Info().IsColumnSplit());
    }

    bst_target_t n_targets = p_tree->NumTargets();
    histogram_builder_ = std::make_unique<MultiHistogramBuilder>();
    histogram_builder_->Reset(ctx_, n_total_bins, n_targets, HistBatch(param_),
                              collective::IsDistributed(), p_fmat->Info().IsColumnSplit(),
                              hist_param_);

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
    auto rc = collective::GlobalSum(
        ctx_, p_fmat->Info(),
        linalg::MakeVec(reinterpret_cast<double *>(root_sum.Values().data()), root_sum.Size() * 2));
    collective::SafeColl(rc);

    histogram_builder_->BuildRootHist(p_fmat, p_tree, partitioner_, gpair, best, HistBatch(param_));

    auto weight = evaluator_->InitRoot(root_sum);
    auto weight_t = weight.HostView();
    std::transform(linalg::cbegin(weight_t), linalg::cend(weight_t), linalg::begin(weight_t),
                   [&](float w) { return w * param_->learning_rate; });

    p_tree->SetLeaf(RegTree::kRoot, weight_t);
    std::vector<BoundedHistCollection const *> hists;
    std::vector<MultiExpandEntry> nodes{{RegTree::kRoot, 0}};
    for (bst_target_t t{0}; t < p_tree->NumTargets(); ++t) {
      hists.push_back(&(*histogram_builder_).Histogram(t));
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
    histogram_builder_->BuildHistLeftRight(ctx_, p_fmat, p_tree, partitioner_, valid_candidates,
                                           gpair, HistBatch(param_));
    monitor_->Stop(__func__);
  }

  void EvaluateSplits(DMatrix *p_fmat, RegTree const *p_tree,
                      std::vector<MultiExpandEntry> *best_splits) {
    monitor_->Start(__func__);
    std::vector<BoundedHistCollection const *> hists;
    for (bst_target_t t{0}; t < p_tree->NumTargets(); ++t) {
      hists.push_back(&(*histogram_builder_).Histogram(t));
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
    p_out_position->resize(gpair.Shape(0));
    for (auto const &part : partitioner_) {
      part.LeafPartition(ctx_, tree, gpair,
                         common::Span{p_out_position->data(), p_out_position->size()});
    }
    monitor_->Stop(__func__);
  }

 public:
  explicit MultiTargetHistBuilder(Context const *ctx, MetaInfo const &info, TrainParam const *param,
                                  HistMakerTrainParam const *hist_param,
                                  std::shared_ptr<common::ColumnSampler> column_sampler,
                                  ObjInfo const *task, common::Monitor *monitor)
      : monitor_{monitor},
        param_{param},
        hist_param_{hist_param},
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

/**
 * @brief Tree updater for single-target trees.
 */
class HistUpdater {
 private:
  common::Monitor *monitor_;
  TrainParam const *param_;
  HistMakerTrainParam const *hist_param_{nullptr};
  std::shared_ptr<common::ColumnSampler> col_sampler_;
  std::unique_ptr<HistEvaluator> evaluator_;
  std::vector<CommonRowPartitioner> partitioner_;

  // back pointers to tree and data matrix
  const RegTree *p_last_tree_{nullptr};
  DMatrix const *const p_last_fmat_{nullptr};

  std::unique_ptr<MultiHistogramBuilder> histogram_builder_;
  ObjInfo const *task_{nullptr};
  // Context for number of threads
  Context const *ctx_{nullptr};

 public:
  explicit HistUpdater(Context const *ctx, std::shared_ptr<common::ColumnSampler> column_sampler,
                       TrainParam const *param, HistMakerTrainParam const *hist_param,
                       DMatrix const *fmat, ObjInfo const *task, common::Monitor *monitor)
      : monitor_{monitor},
        param_{param},
        hist_param_{hist_param},
        col_sampler_{std::move(column_sampler)},
        evaluator_{std::make_unique<HistEvaluator>(ctx, param, fmat->Info(), col_sampler_)},
        p_last_fmat_(fmat),
        histogram_builder_{new MultiHistogramBuilder},
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
    bst_bin_t n_total_bins{0};
    size_t page_idx = 0;
    for (auto const &page : fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      if (n_total_bins == 0) {
        n_total_bins = page.cut.TotalBins();
      } else {
        CHECK_EQ(n_total_bins, page.cut.TotalBins());
      }
      if (page_idx < partitioner_.size()) {
        partitioner_[page_idx].Reset(this->ctx_, page.Size(), page.base_rowid,
                                     fmat->Info().IsColumnSplit());
      } else {
        partitioner_.emplace_back(this->ctx_, page.Size(), page.base_rowid,
                                  fmat->Info().IsColumnSplit());
      }
      page_idx++;
    }
    histogram_builder_->Reset(ctx_, n_total_bins, 1, HistBatch(param_), collective::IsDistributed(),
                              fmat->Info().IsColumnSplit(), hist_param_);
    evaluator_ = std::make_unique<HistEvaluator>(ctx_, this->param_, fmat->Info(), col_sampler_);
    p_last_tree_ = p_tree;
    monitor_->Stop(__func__);
  }

  void EvaluateSplits(DMatrix *p_fmat, RegTree const *p_tree,
                      std::vector<CPUExpandEntry> *best_splits) {
    monitor_->Start(__func__);
    auto const &histograms = histogram_builder_->Histogram(0);
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
    monitor_->Start(__func__);
    CPUExpandEntry node(RegTree::kRoot, p_tree->GetDepth(0));

    this->histogram_builder_->BuildRootHist(p_fmat, p_tree, partitioner_, gpair, node,
                                            HistBatch(param_));

    {
      GradientPairPrecise grad_stat;
      if (p_fmat->IsDense() && !collective::IsDistributed()) {
        /**
         * Specialized code for dense data: For dense data (with no missing value), the sum
         * of gradient histogram is equal to snode[nid]
         */
        auto const &gmat = *(p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_)).begin());
        std::vector<std::uint32_t> const &row_ptr = gmat.cut.Ptrs();
        CHECK_GE(row_ptr.size(), 2);
        std::uint32_t const ibegin = row_ptr[0];
        std::uint32_t const iend = row_ptr[1];
        auto hist = this->histogram_builder_->Histogram(0)[RegTree::kRoot];
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
        auto rc = collective::GlobalSum(ctx_, p_fmat->Info(),
                                        linalg::MakeVec(reinterpret_cast<double *>(&grad_stat), 2));
        collective::SafeColl(rc);
      }

      auto weight = evaluator_->InitRoot(GradStats{grad_stat});
      p_tree->Stat(RegTree::kRoot).sum_hess = grad_stat.GetHess();
      p_tree->Stat(RegTree::kRoot).base_weight = weight;
      (*p_tree)[RegTree::kRoot].SetLeaf(param_->learning_rate * weight);

      std::vector<CPUExpandEntry> entries{node};
      monitor_->Start("EvaluateSplits");
      auto ft = p_fmat->Info().feature_types.ConstHostSpan();
      for (auto const &gmat : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
        evaluator_->EvaluateSplits(histogram_builder_->Histogram(0), gmat.cut, ft, *p_tree,
                                   &entries);
        break;
      }
      monitor_->Stop("EvaluateSplits");
      node = entries.front();
    }

    monitor_->Stop(__func__);
    return node;
  }

  void BuildHistogram(DMatrix *p_fmat, RegTree *p_tree,
                      std::vector<CPUExpandEntry> const &valid_candidates,
                      linalg::MatrixView<GradientPair const> gpair) {
    monitor_->Start(__func__);
    this->histogram_builder_->BuildHistLeftRight(ctx_, p_fmat, p_tree, partitioner_,
                                                 valid_candidates, gpair, HistBatch(param_));
    monitor_->Stop(__func__);
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
      monitor_->Stop(__func__);
      return;
    }
    p_out_position->resize(gpair.Shape(0));
    for (auto const &part : partitioner_) {
      part.LeafPartition(ctx_, tree, gpair,
                         common::Span{p_out_position->data(), p_out_position->size()});
    }
    monitor_->Stop(__func__);
  }
};

/*! \brief construct a tree using quantized feature values */
class QuantileHistMaker : public TreeUpdater {
  std::unique_ptr<HistUpdater> p_impl_{nullptr};
  std::unique_ptr<MultiTargetHistBuilder> p_mtimpl_{nullptr};
  std::shared_ptr<common::ColumnSampler> column_sampler_;

  common::Monitor monitor_;
  ObjInfo const *task_{nullptr};
  HistMakerTrainParam hist_param_;

 public:
  explicit QuantileHistMaker(Context const *ctx, ObjInfo const *task)
      : TreeUpdater{ctx}, task_{task} {}

  void Configure(Args const &args) override { hist_param_.UpdateAllowUnknown(args); }
  void LoadConfig(Json const &in) override {
    auto const &config = get<Object const>(in);
    FromJson(config.at("hist_train_param"), &hist_param_);
  }
  void SaveConfig(Json *p_out) const override {
    auto &out = *p_out;
    out["hist_train_param"] = ToJson(hist_param_);
  }

  [[nodiscard]] char const *Name() const override { return "grow_quantile_histmaker"; }

  void Update(TrainParam const *param, linalg::Matrix<GradientPair> *gpair, DMatrix *p_fmat,
              common::Span<HostDeviceVector<bst_node_t>> out_position,
              const std::vector<RegTree *> &trees) override {
    if (!column_sampler_) {
      column_sampler_ = common::MakeColumnSampler(ctx_);
    }

    if (trees.front()->IsMultiTarget()) {
      CHECK(hist_param_.GetInitialised());
      CHECK(param->monotone_constraints.empty()) << "monotone constraint" << MTNotImplemented();
      if (!p_mtimpl_) {
        this->p_mtimpl_ = std::make_unique<MultiTargetHistBuilder>(
            ctx_, p_fmat->Info(), param, &hist_param_, column_sampler_, task_, &monitor_);
      }
    } else {
      CHECK(hist_param_.GetInitialised());
      if (!p_impl_) {
        p_impl_ = std::make_unique<HistUpdater>(ctx_, column_sampler_, param, &hist_param_, p_fmat,
                                                task_, &monitor_);
      }
    }

    bst_target_t n_targets = trees.front()->NumTargets();
    auto h_gpair = gpair->HostView();

    linalg::Matrix<GradientPair> sample_out;
    auto h_sample_out = h_gpair;
    auto need_copy = [&] {
      return trees.size() > 1 || n_targets > 1;
    };
    if (need_copy()) {
      // allocate buffer
      sample_out = decltype(sample_out){h_gpair.Shape(), ctx_->Device(), linalg::Order::kF};
      h_sample_out = sample_out.HostView();
    }

    for (auto tree_it = trees.begin(); tree_it != trees.end(); ++tree_it) {
      if (need_copy()) {
        // Copy gradient into buffer for sampling. This converts C-order to F-order.
        std::copy(linalg::cbegin(h_gpair), linalg::cend(h_gpair), linalg::begin(h_sample_out));
      }
      error::NoPageConcat(this->hist_param_.extmem_single_page);
      SampleGradient(ctx_, *param, h_sample_out);
      auto *h_out_position = &out_position[tree_it - trees.begin()];
      if ((*tree_it)->IsMultiTarget()) {
        UpdateTree<MultiExpandEntry>(&monitor_, h_sample_out, p_mtimpl_.get(), p_fmat, param,
                                     h_out_position, *tree_it);
      } else {
        UpdateTree<CPUExpandEntry>(&monitor_, h_sample_out, p_impl_.get(), p_fmat, param,
                                   h_out_position, *tree_it);
      }

      hist_param_.CheckTreesSynchronized(ctx_, *tree_it);
    }
  }

  bool UpdatePredictionCache(const DMatrix *data, linalg::MatrixView<float> out_preds) override {
    if (out_preds.Shape(1) > 1) {
      CHECK(p_mtimpl_);
      return p_mtimpl_->UpdatePredictionCache(data, out_preds);
    } else {
      CHECK(p_impl_);
      return p_impl_->UpdatePredictionCache(data, out_preds);
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
