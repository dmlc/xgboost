/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <thrust/count.h>  // for count_if
#include <thrust/fill.h>   // for fill

#include <algorithm>  // for any_of, copy_if, max
#include <iterator>   // for back_inserter, size
#include <limits>     // for numeric_limits
#include <memory>     // for make_shared, make_unique, unique_ptr
#include <sstream>    // for ostringstream
#include <utility>    // for move
#include <vector>     // for vector

#include "../c_api/c_api_error.h"
#include "../c_api/c_api_utils.h"        // for CastDMatrixHandle
#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/device_helpers.cuh"  // for LaunchN
#include "../tree/updater_gpu_hist.cuh"  // for HistBatch, InitBatchCuts
#include "./gather.cuh"                  // for GatherRows
#include "./tree_method.cuh"             // for RouteHeldOut
#include "cross_validate.h"
#include "xgboost/json.h"  // for Json

namespace xgboost::cv {
namespace {
// Scatter the gradient of a batch back into the global gradient buffer of a fold.
void ScatterBatchGpair(Context const* ctx, linalg::Matrix<GradientPair> const& batch_gpair,
                       common::Span<bst_idx_t const> ridxs,
                       linalg::Matrix<GradientPair>* out_gpairs) {
  CHECK_EQ(batch_gpair.Shape(0), ridxs.size());
  CHECK_EQ(batch_gpair.Shape(1), out_gpairs->Shape(1));

  auto d_batch_gpair = batch_gpair.View(ctx->Device());
  auto d_out = out_gpairs->View(ctx->Device());
  dh::LaunchN(d_batch_gpair.Size(), ctx->CUDACtx()->Stream(),
              [=] XGBOOST_DEVICE(std::size_t i) mutable {
                auto [ridx, target_idx] = linalg::UnravelIndex(i, d_batch_gpair.Shape());
                d_out(ridxs[ridx], target_idx) = d_batch_gpair(ridx, target_idx);
              });
}

[[nodiscard]] Args JsonToArgs(Json const& config) {
  CHECK(config.GetValue().Type() == Value::ValueKind::kObject)
      << "CV tree method configuration must be a JSON object.";

  Args args;
  for (auto const& kv : get<Object const>(config)) {
    args.emplace_back(kv.first, JsonScalarToString(kv.second));
  }
  return args;
}

void CheckNoUnknownParams(Args const& unknown) {
  if (unknown.empty()) {
    return;
  }
  std::stringstream ss;
  ss << "Unknown CV tree method parameters: { ";
  for (std::size_t i = 0; i < unknown.size(); ++i) {
    ss << unknown[i].first;
    if (i + 1 != unknown.size()) {
      ss << ", ";
    }
  }
  ss << " }";
  LOG(FATAL) << ss.str();
}
}  // namespace

void FoldModels::GetGradient(Context const* ctx, MetaInfo const& info,
                             FoldPredictions const& predts, FoldInfoBatches const& finfo,
                             std::int32_t iter, FoldGpairs* out) const {
  CHECK(!finfo.Empty());
  CHECK(out);

  auto k_folds = finfo.KFolds();
  CHECK_EQ(this->KFolds(), k_folds);
  CheckLayout(this->Layout(), predts.layout, "prediction caches");

  auto n_units = this->NumUnits();
  auto& gpairs = out->gpairs;
  out->layout = this->Layout();
  gpairs.resize(n_units);

  // The gradient is indexed by the global row index. Zero out the buffer first, the
  // validation rows of a fold are never written to and must not contribute to its
  // histograms.
  for (std::size_t u = 0; u < n_units; ++u) {
    auto& unit_gpair = gpairs.at(u);
    unit_gpair.SetDevice(ctx->Device());
    unit_gpair.Reshape(info.num_row_, this->OutputLength(u));
    unit_gpair.Data()->Fill(GradientPair{});
  }

  if (this->HasRefit()) {
    // The refit unit trains on every row, so the global prediction cache and the unsliced
    // info are already what the objective expects: no gather, slice, or scatter.
    auto u = this->RefitIdx();
    auto const& predt = predts.Prediction(u);
    CHECK_EQ(predt.Size(), info.num_row_ * this->OutputLength(u));
    this->Objective(u)->GetGradient(predt, info, iter, &gpairs.at(u));
  }

  for (std::size_t i = 0, n = finfo.Size(); i < n; ++i) {
    auto const& batch = finfo.batches.at(i);
    CHECK_EQ(batch.KFolds(), k_folds);

    for (std::size_t k = 0; k < k_folds; ++k) {
      auto ridxs = batch.TrainingFold(k);

      constexpr std::size_t kNnz = 0;  // fixme
      auto fold_info = info.Slice(ctx, ridxs, kNnz);

      auto output_length = this->OutputLength(k);
      CHECK_EQ(fold_info.labels.Shape(1), output_length);
      CHECK_EQ(fold_info.labels.Size(), ridxs.size() * output_length);
      auto const& fold_preds = predts.Prediction(k);
      CHECK_EQ(fold_preds.Size(), info.num_row_ * output_length);
      HostDeviceVector<float> preds(ridxs.size() * output_length, 0.0f, ctx->Device());
      GatherRows(ctx, fold_preds.ConstDeviceSpan(), ridxs, 0, output_length, preds.DeviceSpan());

      linalg::Matrix<GradientPair> batch_gpair;
      this->Objective(k)->GetGradient(preds, fold_info, iter, &batch_gpair);
      ScatterBatchGpair(ctx, batch_gpair, ridxs, &gpairs.at(k));
    }
  }
}

// Copying a page to the device pays off once the kernels read enough of it. Same crossover
// as `GPUHistMakerDevice::NeedCopy`.
inline constexpr std::size_t kNeedCopyThreshold = 4;

class FoldTreeMethod {
  Context const* ctx_{nullptr};
  DMatrix const* p_last_fmat_{nullptr};
  tree::TrainParam param_;
  tree::HistMakerTrainParam hist_param_;
  bool initialized_{false};

  // FIXME(jiamingy): The columns_sampler_ cannot be shared between folds.
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  std::shared_ptr<common::HistogramCuts const> cuts_;
  std::unique_ptr<tree::FeatureGroups> feature_groups_;
  std::vector<bst_idx_t> batch_ptr_;
  common::Span<FeatureType const> feature_types_;
  // The sampled feature set of the round, held for the round so that the device span stays
  // valid. See `Reset` for why one set serves every node of every unit.
  std::shared_ptr<HostDeviceVector<bst_feature_t>> feature_set_;

  // Everything that is per-unit, and the layout that indexes it.
  FoldTreeState state_;

  // Fusion guard. The number of passes over the Ellpack pages must not depend on the number
  // of folds. Both are reset at the top of Update.
  std::size_t n_page_passes_{0};
  std::size_t n_levels_{0};

  // Reject the parameters that this prototype silently ignores instead of implementing.
  void CheckSupportedParams() const {
    auto check_default = [](float value, StringView name) {
      CHECK_EQ(value, 1.0f) << "`" << name << "` is not yet supported by the CV tree method.";
    };
    check_default(param_.subsample, "subsample");
    check_default(param_.colsample_bytree, "colsample_bytree");
    check_default(param_.colsample_bylevel, "colsample_bylevel");
    check_default(param_.colsample_bynode, "colsample_bynode");
    CHECK(param_.interaction_constraints.empty())
        << "`interaction_constraints` is not yet supported by the CV tree method.";
    CHECK(!this->cuts_->HasCategorical())
        << "Categorical features are not yet supported by the CV tree method.";
    // Loss-guided growth pops one node per iteration, so the number of page passes would
    // scale with the node count instead of with the depth, which is the cost this design
    // exists to avoid.
    CHECK_EQ(param_.grow_policy, tree::TrainParam::kDepthWise)
        << "Only the depthwise grow policy is supported by the CV tree method.";
  }

  void BuildHist(EllpackPage const& page, std::int32_t batch_idx, UnitState* p_unit,
                 std::vector<bst_node_t> const& build_nodes) {
    xgboost_NVTX_FN_RANGE();
    auto& unit = *p_unit;
    if (build_nodes.empty()) {
      return;
    }
    auto d_gpair = unit.quantized_gpair.View(this->ctx_->Device());
    auto acc = page.Impl()->GetDeviceEllpack(this->ctx_, {});

    std::vector<common::Span<tree::cuda_impl::RowIndexT const>> h_ridxs;
    std::vector<common::Span<GradientPairInt64>> h_hists;
    std::vector<std::size_t> h_sizes_csum{0};
    for (auto nidx : build_nodes) {
      auto d_ridx = unit.partitioners.At(batch_idx)->GetRows(nidx);
      if (d_ridx.empty()) {
        // A fold can have no training rows for a node in this batch.
        continue;
      }
      h_ridxs.push_back(d_ridx);
      h_hists.push_back(unit.histogram->GetNodeHistogram(nidx));
      h_sizes_csum.push_back(d_ridx.size() + h_sizes_csum.back());
    }
    if (h_ridxs.empty()) {
      return;
    }

    dh::device_vector<common::Span<GradientPairInt64>> hists{h_hists};
    dh::device_vector<common::Span<tree::cuda_impl::RowIndexT const>> ridxs{h_ridxs};
    unit.histogram->BuildHistogram(this->ctx_, acc,
                                   this->feature_groups_->DeviceAccessor(this->ctx_->Device()),
                                   d_gpair, dh::ToSpan(ridxs), dh::ToSpan(hists), h_sizes_csum);
  }

  // The feature set of every node of every unit, see `Reset`.
  [[nodiscard]] common::Span<bst_feature_t const> FeatureSet() const {
    CHECK(this->feature_set_);
    return this->feature_set_->ConstDeviceSpan();
  }

  [[nodiscard]] auto MakeSharedInputs(UnitState const& unit) const {
    std::size_t constexpr kCatStorageSize = 0;  // FIXME(jiamingy): Support categorical features.
    auto feature_set = this->FeatureSet();
    return tree::MultiEvaluateSplitSharedInputs{unit.quantizer->DeviceSpan(),
                                                this->cuts_->cut_ptrs_.ConstDeviceSpan(),
                                                this->cuts_->cut_values_.ConstDevicePointer(),
                                                this->feature_types_,
                                                kCatStorageSize,
                                                this->cuts_->TotalBins(),
                                                static_cast<bst_feature_t>(feature_set.size()),
                                                tree::EvalParam{this->param_}};
  }

 public:
  explicit FoldTreeMethod(std::shared_ptr<DMatrix> p_fmat)
      : ctx_{p_fmat->Ctx()},
        p_last_fmat_{p_fmat.get()},
        column_sampler_{std::make_shared<common::ColumnSampler>()} {}

  void Configure(Args const& args) {
    CHECK(ctx_->IsCUDA()) << "CV tree method `hist` requires a CUDA device.";

    auto unknown = param_.UpdateAllowUnknown(args);
    unknown = hist_param_.UpdateAllowUnknown(unknown);
    CheckNoUnknownParams(unknown);
  }

  void InitDataOnce(DMatrix* p_fmat) {
    xgboost_NVTX_FN_RANGE();
    CHECK(ctx_->IsCUDA()) << "CV tree method `hist` requires a CUDA device.";
    p_fmat->Info().feature_types.SetDevice(ctx_->Device());

    auto batch = tree::cuda_impl::HistBatch(param_);
    auto [cuts, dense_compressed] = tree::InitBatchCuts(ctx_, p_fmat, batch);
    this->cuts_ = std::move(cuts);
    this->batch_ptr_ = p_fmat->BatchPtr();
    this->feature_groups_ = std::make_unique<tree::FeatureGroups>(
        *this->cuts_, dense_compressed, tree::DftMtHistShmemBytes(ctx_->Ordinal()));

    this->CheckSupportedParams();
    initialized_ = true;
  }

  void Reset(FoldModels const& folds, DMatrix* p_fmat, FoldInfoBatches const& finfo,
             FoldGpairs const& gpairs) {
    xgboost_NVTX_FN_RANGE();
    CHECK(!collective::IsDistributed())
        << "Distributed training is not supported by the CV tree method.";
    CHECK(!finfo.Empty());
    CHECK(cuts_);
    // The page loops index the partitioners by the page counter, and the prediction cache
    // indexes `batch_ptr_` the same way.
    CHECK_EQ(p_fmat->NumBatches(), finfo.Size());
    CHECK_EQ(this->batch_ptr_.size(), finfo.Size() + 1);

    this->state_.layout = gpairs.layout;
    // The guards that keep the refit unit out of the out-of-fold path rely on this: a unit
    // index that is not the refit unit must be a valid index into the fold info.
    CHECK_EQ(finfo.KFolds(), this->state_.layout.k_folds);

    auto const& info = p_fmat->Info();
    info.feature_types.SetDevice(ctx_->Device());
    this->feature_types_ = info.feature_types.ConstDeviceSpan();
    // Once per round, not per unit. GetFeatureSet returns the tree-level feature set
    // unconditionally when colsample is disabled, and that is null until Init runs.
    this->column_sampler_->Init(ctx_, info.num_col_, info.feature_weights, param_.colsample_bynode,
                                param_.colsample_bylevel, param_.colsample_bytree);
    // Column sampling is rejected, so the tree-level set is the feature set of every node of
    // every unit and the depth argument is ignored. Fetched after `Init`, which resizes the
    // set to zero and back, and given a device here, because the short-circuit return is the
    // one exit of `GetFeatureSet` that does not set one.
    this->feature_set_ = this->column_sampler_->GetFeatureSet(ctx_, 0);
    CHECK(this->feature_set_);
    this->feature_set_->SetDevice(ctx_->Device());

    // Resizing keeps the caches of the units that already exist, which is the common case
    // across boosting rounds.
    auto n_units = this->state_.layout.NumUnits();
    CHECK_EQ(folds.NumUnits(), n_units);
    this->state_.Resize(n_units);

    this->state_.oof_position.resize(info.num_row_);
    thrust::fill_n(ctx_->CUDACtx()->CTP(), this->state_.oof_position.begin(),
                   this->state_.oof_position.size(), RegTree::kRoot);

    // Each unit owns a histogram cache, so the parameter is split between them: it bounds
    // the histogram memory of the run, not of one model. The clamp matters because the
    // division is the only route to a budget of zero, which the storage rejects.
    auto n_cached =
        std::max<std::size_t>(1, hist_param_.MaxCachedHistNodes(ctx_->Device()) / n_units);

    bst_target_t n_split_targets{0};
    for (std::size_t u = 0; u < n_units; ++u) {
      auto& unit = this->state_.At(u);
      auto const& unit_gpair = gpairs.gpairs.at(u);
      CHECK_EQ(info.num_row_, unit_gpair.Shape(0));
      auto n_train = this->state_.IsRefit(u) ? info.num_row_ : finfo.TrainFoldSize(u);
      CHECK_GT(n_train, 0) << "Every CV model must have at least one training row. `k_folds` must "
                              "be at least 2, a single fold holds out all of its rows.";
      CHECK_GT(unit_gpair.Shape(1), 0);

      auto in_gpair = unit_gpair.View(ctx_->Device());
      CHECK(in_gpair.CContiguous());
      if (u == 0) {
        n_split_targets = in_gpair.Shape(1);
      }
      CHECK_EQ(n_split_targets, in_gpair.Shape(1));

      // Only the training rows of the unit are accumulated, the rest of the buffer is zero.
      unit.quantizer = std::make_unique<tree::GradientQuantiserGroup>(ctx_, in_gpair, n_train);
      tree::CalcQuantizedGpairs(ctx_, in_gpair, unit.quantizer->DeviceSpan(),
                                &unit.quantized_gpair);

      auto n_total_bins = static_cast<bst_idx_t>(this->cuts_->TotalBins()) * n_split_targets;
      CHECK_LT(n_total_bins, std::numeric_limits<bst_bin_t>::max())
          << "Too many histogram bins: n_total_bins = total_bins * n_targets";
      bool force_global = false;
      if (!unit.histogram) {
        unit.histogram = std::make_unique<tree::DeviceHistogramBuilder>();
      }
      unit.histogram->Reset(ctx_, n_cached, n_total_bins, force_global);

      // A fold's partitioner holds only the training rows of that fold, so the histograms
      // and the final leaf positions never see a row held out by the fold. The refit unit
      // owns every row, which the batch pointer describes without an index array.
      if (this->state_.IsRefit(u)) {
        unit.partitioners.Reset(ctx_, this->batch_ptr_);
      } else {
        std::vector<common::Span<bst_idx_t const>> fold_ridxs;
        for (auto const& batch : finfo.batches) {
          fold_ridxs.emplace_back(batch.TrainingFold(u));
        }
        unit.partitioners.Reset(ctx_, fold_ridxs);
      }

      if (!unit.evaluator) {
        unit.evaluator = std::make_unique<tree::cuda_impl::MultiHistEvaluator>();
      }
      unit.evaluator->Reset(ctx_, this->cuts_->cut_ptrs_.ConstDeviceSpan(), this->feature_types_,
                            this->param_, n_split_targets);

      // The tree and the frontier of this round. The driver cannot be carried over from the
      // previous one: its leaf count is cumulative state.
      unit.tree = std::make_unique<RegTree>(folds.LeafLength(u), folds.NumFeatures(u), true);
      CHECK_EQ(unit.tree->NumTargets(), n_split_targets);
      unit.driver = std::make_unique<tree::Driver<MultiExpandEntry>>(
          this->param_, tree::cuda_impl::kMaxNodeBatchSize);
      unit.expand_set.clear();
      unit.candidates.clear();
    }
    this->state_.n_targets = n_split_targets;
  }

  // Build the root histogram of every unit, evaluate its split, and seed every frontier.
  void InitRoot(DMatrix* p_fmat) {
    xgboost_NVTX_FN_RANGE();
    auto n_units = this->state_.NumUnits();
    CHECK_GT(n_units, 0);
    auto n_targets = this->state_.n_targets;

    for (std::size_t u = 0; u < n_units; ++u) {
      auto& unit = this->state_.At(u);
      auto d_gpair = unit.quantized_gpair.View(ctx_->Device());
      CHECK_EQ(d_gpair.Shape(1), n_targets);

      unit.evaluator->AllocNodeSum(RegTree::kRoot, n_targets);
      tree::cuda_impl::CalcRootSum(ctx_, d_gpair,
                                   unit.evaluator->GetNodeSum(RegTree::kRoot, n_targets));
      unit.histogram->AllocateHistograms(ctx_, {RegTree::kRoot});
    }

    // Fused: every unit consumes the page before it is dropped.
    std::int32_t batch_idx = 0;
    for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx_, StaticBatch(true))) {
      for (std::size_t u = 0; u < n_units; ++u) {
        this->BuildHist(page, batch_idx, &this->state_.At(u), {RegTree::kRoot});
      }
      ++batch_idx;
    }
    ++this->n_page_passes_;

    auto feature_set = this->FeatureSet();
    auto root_weights = linalg::Empty<float>(ctx_, n_units, n_targets);
    auto d_root_weights = root_weights.View(ctx_->Device());
    auto eta = this->param_.learning_rate;

    for (std::size_t u = 0; u < n_units; ++u) {
      auto& unit = this->state_.At(u);
      // No histogram or root sum reduction: `Reset` rejects distributed training.
      tree::MultiEvaluateSplitInputs input{
          RegTree::kRoot, 0, unit.evaluator->GetNodeSum(RegTree::kRoot, n_targets), feature_set,
          unit.histogram->GetNodeHistogram(RegTree::kRoot)};
      auto entry = unit.evaluator->EvaluateSingleSplit(ctx_, input, this->MakeSharedInputs(unit));

      // The weight is owned by the evaluator, ApplySplit reads it back by node id.
      auto base_weight = unit.evaluator->GetNodeWeights(n_targets).Base(RegTree::kRoot);
      dh::LaunchN(n_targets, ctx_->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t t) mutable {
        d_root_weights(u, t) = base_weight[t] * eta;
      });
      // The root's coverage is the sum of the hessians of both children.
      auto root_sum_hess = static_cast<float>(entry.left_sum + entry.right_sum);
      unit.tree->SetRoot(d_root_weights.Slice(u, linalg::All()), root_sum_hess);

      unit.driver->Push(entry);
      unit.expand_set = unit.driver->Pop();
    }
  }

  [[nodiscard]] bool NeedCopy(DMatrix const* p_fmat, std::vector<LevelNodes> const& level) const {
    xgboost_NVTX_FN_RANGE();
    if (p_fmat->SingleColBlock()) {
      return true;  // Use the default for in-core data.
    }
    CHECK_EQ(level.size(), this->state_.NumUnits());
    bst_idx_t n_visits = 0;
    for (std::size_t u = 0; u < level.size(); ++u) {
      for (auto const& part : this->state_.At(u).partitioners) {
        for (auto nidx : level[u].partition.nidx) {
          n_visits += part->GetRows(nidx).size();
        }
      }
    }
    return n_visits * kNeedCopyThreshold > p_fmat->Info().num_row_;
  }

  // Decide how each child of this unit's candidates obtains its histogram. Subtraction
  // needs the parent histogram, which the allocation below can evict from the overflow
  // cache; the single-model maker discovers that afterwards and re-streams the pages, while
  // here the sibling is instead built by the page pass that is about to happen anyway. A
  // tight histogram cache then costs an extra kernel, never an extra page fetch.
  void AssignChildren(UnitState* p_unit, LevelNodes* out) {
    xgboost_NVTX_FN_RANGE();
    auto& unit = *p_unit;
    auto const& candidates = unit.candidates;
    CHECK(!candidates.empty());

    // Sized rather than reserved: `AssignNodes` writes through the pointer.
    std::vector<bst_node_t> nodes_to_build(candidates.size());
    std::vector<bst_node_t> nodes_to_sub(candidates.size());
    // Host code, hence the tree and not the device view of the level.
    tree::cuda_impl::AssignNodes(
        *unit.tree, candidates, nodes_to_build, nodes_to_sub,
        [](MultiExpandEntry const& e) { return e.right_sum < e.left_sum; });
    unit.histogram->AllocateHistograms(ctx_, nodes_to_build, nodes_to_sub);

    out->nodes_to_build = nodes_to_build;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (unit.histogram->CanSubtract(candidates[i].nidx, nodes_to_build[i])) {
        out->sub_parent.push_back(candidates[i]);
        out->sub_sibling.push_back(nodes_to_build[i]);
        out->sub.push_back(nodes_to_sub[i]);
      } else {
        out->nodes_to_build.push_back(nodes_to_sub[i]);
      }
    }
  }

  void PartitionAndBuildHist(DMatrix* p_fmat, FoldInfoBatches const& finfo,
                             std::vector<LevelNodes> const& level) {
    xgboost_NVTX_FN_RANGE();
    auto n_units = this->state_.NumUnits();
    CHECK_EQ(level.size(), n_units);
    // One decision for the whole pass, as the single-model maker makes it: copying the page
    // pays off only if there is a histogram to build from it.
    auto has_build = std::any_of(level.cbegin(), level.cend(), [](LevelNodes const& nodes) {
      return !nodes.nodes_to_build.empty();
    });
    auto prefetch_copy = has_build && this->NeedCopy(p_fmat, level);
    prefetch_copy = true;

    std::int32_t batch_idx = 0;
    for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx_, StaticBatch(prefetch_copy))) {
      page.Impl()->Visit(ctx_, {}, [&](auto&& d_acc) {
        using Acc = std::remove_reference_t<decltype(d_acc)>;
        for (std::size_t u = 0; u < n_units; ++u) {
          auto const& nodes = level[u];
          if (!nodes.Active()) {
            continue;
          }
          auto& unit = this->state_.At(u);
          auto tree = *nodes.tree_view;
          auto go_left = GoLeftOp<Acc>{d_acc, tree};
          unit.partitioners.UpdatePositionBatch(
              ctx_, batch_idx, nodes.partition.nidx, nodes.partition.left_nidx,
              nodes.partition.right_nidx, nodes.partition.split_data,
              tree::cuda_impl::GoLeftWrapperOp<GoLeftOp<Acc>>{go_left});

          // A fold-only concept: the refit unit holds nothing out, and the fold info has no
          // entry for it.
          if (!this->state_.IsRefit(u)) {
            auto valid_idx = finfo.batches[batch_idx].ValidationFold(u);
            RouteHeldOut(this->ctx_, valid_idx, tree, go_left,
                         dh::ToSpan(this->state_.oof_position));
          }

          // After the partitioner, which synchronizes, so the child row segments this reads
          // already exist.
          this->BuildHist(page, batch_idx, &unit, nodes.nodes_to_build);
        }
      });
      ++batch_idx;
    }
    ++this->n_page_passes_;
  }

  // Evaluate the children of every unit and advance the frontiers. Mirrors
  // `MultiTargetHistMaker::EvaluateSplits`, including the (left, right) interleaving, which
  // is what keeps the node ids of a fused tree equal to the single-model maker's.
  void EvaluateLevel() {
    xgboost_NVTX_FN_RANGE();
    auto n_units = this->state_.NumUnits();
    auto n_targets = this->state_.n_targets;
    auto feature_set = this->FeatureSet();
    // The staging buffer is grow-only, so the span is re-taken for every unit at every
    // level: a unit that had candidates last level and none now would otherwise hand the
    // driver those entries a second time.
    std::vector<common::Span<MultiExpandEntry>> staged(n_units);

    for (std::size_t u = 0; u < n_units; ++u) {
      auto& unit = this->state_.At(u);
      auto n_children = unit.candidates.size() * 2;
      staged[u] = unit.staged.GetSpan<MultiExpandEntry>(n_children, MultiExpandEntry{});
      if (unit.candidates.empty()) {
        continue;
      }

      auto const& tree = *unit.tree;
      std::vector<tree::MultiEvaluateSplitInputs> h_inputs(n_children);
      bst_node_t max_nidx = 0;
      for (std::size_t i = 0; i < unit.candidates.size(); ++i) {
        auto const& candidate = unit.candidates[i];
        bst_node_t children[]{tree.LeftChild(candidate.nidx), tree.RightChild(candidate.nidx)};
        for (std::size_t j = 0; j < std::size(children); ++j) {
          auto nidx = children[j];
          // No allocation here, the parent sum was calculated by the last ApplySplit.
          h_inputs[i * std::size(children) + j] = tree::MultiEvaluateSplitInputs{
              nidx, candidate.depth + 1, unit.evaluator->GetNodeSum(nidx, n_targets), feature_set,
              unit.histogram->GetNodeHistogram(nidx)};
          max_nidx = std::max(max_nidx, nidx);
        }
      }

      unit.eval_inputs.resize(n_children);
      unit.eval_outputs.resize(n_children);
      dh::safe_cuda(cudaMemcpyAsync(unit.eval_inputs.data(), h_inputs.data(),
                                    common::SizeBytes<tree::MultiEvaluateSplitInputs>(n_children),
                                    cudaMemcpyDefault, ctx_->CUDACtx()->Stream()));
      unit.evaluator->EvaluateSplits(ctx_, dh::ToSpan(unit.eval_inputs),
                                     this->MakeSharedInputs(unit), max_nidx,
                                     dh::ToSpan(unit.eval_outputs));
      dh::safe_cuda(cudaMemcpyAsync(staged[u].data(), unit.eval_outputs.data(),
                                    staged[u].size_bytes(), cudaMemcpyDefault,
                                    ctx_->CUDACtx()->Stream()));
    }

    ctx_->CUDACtx()->Stream().Sync();
    for (std::size_t u = 0; u < n_units; ++u) {
      auto& unit = this->state_.At(u);
      // A child with no split arrives with the default loss change, which `Push` drops; that
      // is what the staging initializer above is for.
      unit.driver->Push(staged[u].begin(), staged[u].end());
      unit.expand_set = unit.driver->Pop();
    }
  }

  void ApplySplit(UnitState* p_unit) {
    xgboost_NVTX_FN_RANGE();
    auto& unit = *p_unit;
    auto const& candidates = unit.expand_set;
    CHECK(!candidates.empty());
    auto n_targets = this->state_.n_targets;
    auto weights = unit.evaluator->GetNodeWeights(n_targets);

    tree::ExpandBatch batch{this->param_.learning_rate};
    for (auto const& candidate : candidates) {
      // Categorical splits are rejected by CheckSupportedParams.
      CHECK(!candidate.split.is_cat);
      batch.Push(candidate.nidx, candidate.split.findex, candidate.split.fvalue,
                 candidate.split.dir == tree::kLeftDir, weights.Base(candidate.nidx),
                 weights.Left(candidate.nidx), weights.Right(candidate.nidx),
                 candidate.split.loss_chg, candidate.left_sum, candidate.right_sum);
    }
    unit.tree->Expand(this->ctx_, batch);

    dh::device_vector<MultiExpandEntry> d_candidates{candidates};
    unit.evaluator->ApplyTreeSplit(this->ctx_, unit.tree.get(),
                                   common::Span<MultiExpandEntry const>{candidates},
                                   dh::ToSpan(d_candidates), n_targets);
  }

  // One level of every unit: apply the splits, stream the pages once to partition the
  // training rows, route the held-out rows and build the child histograms, obtain the
  // remaining children by subtraction, then evaluate them and advance the frontiers.
  void GrowLevel(DMatrix* p_fmat, FoldInfoBatches const& finfo) {
    xgboost_NVTX_FN_RANGE();
    auto n_units = this->state_.NumUnits();
    std::vector<LevelNodes> level(n_units);

    for (std::size_t u = 0; u < n_units; ++u) {
      auto& unit = this->state_.At(u);
      unit.candidates.clear();
      if (unit.expand_set.empty()) {
        continue;
      }
      this->ApplySplit(&unit);

      auto& nodes = level[u];
      // Pulled to the device once per level, not once per page: nothing mutates the tree
      // between the pages of a level, and each pull copies the tree's host arrays.
      nodes.tree_view.emplace(ctx_->Device(), false, unit.tree.get());
      nodes.partition = HistMaker::CreatePartitionNodes(unit.tree.get(), unit.expand_set);
      // The nodes whose children may split again. The children of the rest are leaves, so
      // no histogram is ever built for them.
      std::copy_if(unit.expand_set.cbegin(), unit.expand_set.cend(),
                   std::back_inserter(unit.candidates),
                   [&](MultiExpandEntry const& e) { return unit.driver->IsChildValid(e); });
      if (!unit.candidates.empty()) {
        this->AssignChildren(&unit, &nodes);
      }
    }

    // The pass runs whenever any unit is active, even at `max_depth` where no unit has a
    // candidate and nothing is built: the partitioner node count must keep up with the
    // tree's, and a held-out row advances one level per pass.
    this->PartitionAndBuildHist(p_fmat, finfo, level);

    for (std::size_t u = 0; u < n_units; ++u) {
      auto& unit = this->state_.At(u);
      if (unit.candidates.empty()) {
        continue;
      }
      auto const& nodes = level[u];
      // The root histogram is the first allocation of a round, so it stays in the resident
      // cache for the whole round and the children of the root are subtractable under any
      // budget. This is what fails loudly if subtraction stops being used at all.
      if (this->n_levels_ == 0) {
        CHECK_EQ(nodes.sub.size(), unit.candidates.size());
      }
      auto need_build =
          unit.histogram->SubtractHist(ctx_, nodes.sub_parent, nodes.sub_sibling, nodes.sub);
      // `CanSubtract` was asked before the page pass, so nothing can be left over.
      CHECK(need_build.empty());
    }

    this->EvaluateLevel();
  }

  void FinalizeTrees() {
    xgboost_NVTX_FN_RANGE();
    for (std::size_t u = 0, n_units = this->state_.NumUnits(); u < n_units; ++u) {
      auto* tree = this->state_.At(u).tree.get();
      tree->GetMultiTargetTree()->SetLeaves();
      hist_param_.CheckTreesSynchronized(ctx_, tree);
    }
  }

  // Add the leaf value of the newly grown tree to the training prediction of every row
  // that the unit owns.
  void UpdatePredictionCache(FoldInfoBatches const& finfo, MetaInfo const& info,
                             FoldPredictions* predts) {
    xgboost_NVTX_FN_RANGE();
    auto n_units = this->state_.NumUnits();
    auto n_samples = info.num_row_;

    // A single scratch buffer is enough, a unit is finished before the next one starts.
    // Scratch buffer for the leaf position of each row, reused by every unit.
    dh::DeviceUVector<bst_node_t> positions;
    positions.resize(n_samples);
    auto d_position = dh::ToSpan(positions);

    for (std::size_t u = 0; u < n_units; ++u) {
      auto& unit = this->state_.At(u);
      auto& tr_predt = predts->Training(u);
      tr_predt.predictions.SetDevice(ctx_->Device());
      auto output_length = predts->output_length;
      CHECK_EQ(output_length * n_samples, tr_predt.predictions.Size());
      auto d_tr_predt =
          linalg::MakeTensorView(ctx_, &tr_predt.predictions, n_samples, output_length);

      // Rows held out by this fold are not in the partitioner and keep the sentinel. The
      // refit unit owns every row, so none of them keeps it.
      thrust::fill(ctx_->CUDACtx()->CTP(), dh::tbegin(d_position), dh::tend(d_position),
                   RegTree::kInvalidNodeId);
      for (std::size_t i = 0, n = finfo.Size(); i < n; ++i) {
        auto base_ridx = this->batch_ptr_[i];
        auto n_batch_samples = this->batch_ptr_.at(i + 1) - base_ridx;
        // The partitioner and the tree must have grown in lockstep. With fewer nodes than
        // the tree, the partitioner returns a node the tree has since split, reading the
        // wrong leaf or past the weights. With more, it returns a node the tree does not
        // have.
        CHECK_EQ(unit.partitioners.At(i)->GetNumNodes(), unit.tree->NumNodes());
        unit.partitioners.At(i)->FinalisePosition(
            ctx_, d_position.subspan(base_ridx, n_batch_samples), base_ridx,
            [] XGBOOST_DEVICE(tree::cuda_impl::RowIndexT, bst_node_t nidx) { return nidx; });
      }

      auto tree = tree::MultiTargetTreeView{ctx_->Device(), false, unit.tree.get()};
      dh::LaunchN(d_tr_predt.Size(), ctx_->CUDACtx()->Stream(),
                  [=] XGBOOST_DEVICE(std::size_t i) mutable {
                    auto [ridx, t] = linalg::UnravelIndex(i, d_tr_predt.Shape());
                    auto nidx = d_position[ridx];
                    if (nidx == RegTree::kInvalidNodeId) {
                      return;  // Held out by this fold, the entry is unused padding.
                    }
                    d_tr_predt(ridx, t) += tree.LeafValue(nidx)(t);
                  });

      // Handle the held out prediction.
      if (!this->state_.IsRefit(u)) {
        auto d_oof_position = dh::ToSpan(this->state_.oof_position);
        auto& va_predt = predts->valid.predictions;
        va_predt.SetDevice(this->ctx_->Device());
        CHECK_EQ(va_predt.Size(), output_length * n_samples);
        auto d_va_predt =
            linalg::MakeTensorView(ctx_->Device(), va_predt.DeviceSpan(), n_samples, output_length);
        for (auto const& batch : finfo.batches) {
          auto valid_idx = batch.ValidationFold(u);
          dh::LaunchN(valid_idx.size() * output_length, this->ctx_->CUDACtx()->Stream(),
                      [=] XGBOOST_DEVICE(std::size_t i) mutable {
                        auto ridx_in_set = i / output_length;
                        auto target_idx = i % output_length;

                        auto ridx = valid_idx[ridx_in_set];
                        auto nidx = d_oof_position[ridx];
                        // `LeafValue` maps a node id to a leaf row without testing for a
                        // leaf, so an unfinished walk would read a wrong leaf, or past the
                        // weight matrix. Dereferencing `nidx` here is safe either way:
                        // `Reset` seeds every position with the root and `RouteHeldOut` only
                        // ever descends, so the sentinel the training kernel above guards
                        // against cannot reach this one.
                        KERNEL_CHECK(tree.IsLeaf(nidx));
                        d_va_predt(ridx, target_idx) += tree.LeafValue(nidx)(target_idx);
                      });
        }
      }

      if (this->hist_param_.debug_synchronize) {
        auto n_train = this->state_.IsRefit(u) ? n_samples : finfo.TrainFoldSize(u);
        DebugCheckValid(this->ctx_, n_train, d_position);
      }
      tr_predt.Update(1);
    }
    // Once per round, after every fold has written its contribution.
    predts->valid.Update(1);
  }

  void Update(FoldModels* folds, DMatrix* p_fmat, FoldInfoBatches const& finfo,
              FoldGpairs const& gpairs, FoldPredictions* predts) {
    xgboost_NVTX_FN_RANGE();
    CHECK(folds);
    CHECK(p_fmat);
    CHECK(predts);
    CHECK_EQ(p_fmat, p_last_fmat_)
        << "CV tree method update must use the training DMatrix supplied at construction.";
    CHECK_EQ(folds->KFolds(), finfo.KFolds());
    CheckLayout(folds->Layout(), gpairs.layout, "gradients");
    CheckLayout(folds->Layout(), predts->layout, "prediction caches");

    if (!initialized_) {
      this->InitDataOnce(p_fmat);
    }
    this->n_page_passes_ = this->n_levels_ = 0;

    this->Reset(*folds, p_fmat, finfo, gpairs);
    this->InitRoot(p_fmat);

    // Level-synchronous across units, so that the partition pass can serve every unit from a
    // single sweep over the pages.
    while (this->state_.Growing()) {
      this->GrowLevel(p_fmat, finfo);
      ++this->n_levels_;
    }

    this->FinalizeTrees();
    this->UpdatePredictionCache(finfo, p_fmat->Info(), predts);
    // One root build plus one partition pass per level, independent of the number of units
    // and of whether the histogram cache could serve a subtraction.
    CHECK_EQ(this->n_page_passes_, 1 + this->n_levels_);
    folds->CommitModel(this->state_.TakeTrees());
  }
};
}  // namespace xgboost::cv

using namespace xgboost;  // NOLINT

XGB_DLL int XGBCvFoldModelsGetGradient(FoldModelsHandle c_cv_folds, DMatrixHandle dtrain,
                                       FoldInfoBatchesHandle c_fold_info,
                                       FoldPredictionsHandle c_predt, FoldGpairsHandle hdl,
                                       int iter) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(c_cv_folds);
  xgboost_CHECK_C_ARG_PTR(c_fold_info);
  xgboost_CHECK_C_ARG_PTR(c_predt);
  xgboost_CHECK_C_ARG_PTR(hdl);
  auto p_fmat = CastDMatrixHandle(dtrain);
  auto cv_folds = static_cast<cv::FoldModels*>(c_cv_folds);
  auto fold_info = static_cast<cv::FoldInfoBatches*>(c_fold_info);
  auto predt = static_cast<cv::FoldPredictions*>(c_predt);
  auto const& info = p_fmat->Info();
  CHECK(!fold_info->batches.empty());
  CHECK_EQ(cv_folds->KFolds(), fold_info->KFolds());

  auto fold_gpairs = static_cast<cv::FoldGpairs*>(hdl);
  cv_folds->GetGradient(p_fmat->Ctx(), info, *predt, *fold_info, iter, fold_gpairs);

  API_END();
}

XGB_DLL int XGBCvFoldTreeMethodCreate(FoldModelsHandle c_cv_folds, DMatrixHandle dtrain,
                                      char const* c_config, TreeMethodHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(c_cv_folds);
  xgboost_CHECK_C_ARG_PTR(dtrain);
  xgboost_CHECK_C_ARG_PTR(c_config);
  xgboost_CHECK_C_ARG_PTR(out);
  auto p_fmat = CastDMatrixHandle(dtrain);
  Json config{Json::Load(StringView{c_config})};
  auto args = cv::JsonToArgs(config);
  auto ptr = std::make_unique<cv::FoldTreeMethod>(std::move(p_fmat));
  ptr->Configure(std::move(args));
  *out = ptr.release();
  API_END();
}

XGB_DLL int XGBCvFoldTreeMethodFree(TreeMethodHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<cv::FoldTreeMethod*>(hdl);
  API_END();
}

XGB_DLL int XGBCvFoldTreeMethodUpdate(TreeMethodHandle hdl, FoldModelsHandle c_cv_folds,
                                      DMatrixHandle dtrain, FoldInfoBatchesHandle c_fold_info,
                                      FoldGpairsHandle c_gpairs, FoldPredictionsHandle c_predt) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  xgboost_CHECK_C_ARG_PTR(c_cv_folds);
  xgboost_CHECK_C_ARG_PTR(c_fold_info);
  xgboost_CHECK_C_ARG_PTR(c_gpairs);
  xgboost_CHECK_C_ARG_PTR(c_predt);
  auto tree_method = static_cast<cv::FoldTreeMethod*>(hdl);
  auto cv_folds = static_cast<cv::FoldModels*>(c_cv_folds);
  auto p_fmat = CastDMatrixHandle(dtrain);
  auto fold_info = static_cast<cv::FoldInfoBatches*>(c_fold_info);
  auto gpairs = static_cast<cv::FoldGpairs*>(c_gpairs);
  auto predt = static_cast<cv::FoldPredictions*>(c_predt);
  tree_method->Update(cv_folds, p_fmat.get(), *fold_info, *gpairs, predt);
  API_END();
}
