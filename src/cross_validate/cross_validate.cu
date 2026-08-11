/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <thrust/count.h>  // for count_if
#include <thrust/fill.h>   // for fill

#include <algorithm>  // for any_of
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
#include "cross_validate.h"
#include "xgboost/json.h"  // for Json

namespace xgboost::cv {
namespace {
// Gather the predictions of the rows listed in `ridxs` into a dense buffer that the objective
// function can consume.
[[nodiscard]] HostDeviceVector<float> GatherPrediction(Context const* ctx,
                                                       HostDeviceVector<float> const& predt,
                                                       common::Span<bst_idx_t const> ridxs,
                                                       bst_target_t n_columns) {
  HostDeviceVector<float> out(ridxs.size() * n_columns, 0.0f, ctx->Device());
  auto d_predt = predt.ConstDeviceSpan();
  auto d_out = out.DeviceSpan();
  dh::LaunchN(d_out.size(), ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t i) {
    auto ridx = ridxs[i / n_columns];
    d_out[i] = d_predt[ridx * n_columns + (i % n_columns)];
  });
  return out;
}

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
  CHECK_EQ(finfo.n_samples, info.num_row_);

  auto k_folds = finfo.KFolds();
  CHECK_EQ(this->KFolds(), k_folds);
  CHECK_EQ(predts.KFolds(), k_folds);

  auto& gpairs = out->gpairs;
  if (gpairs.empty()) {
    gpairs.resize(k_folds);
  }
  CHECK_EQ(gpairs.size(), k_folds);

  // The gradient is indexed by the global row index. Zero out the buffer first, the
  // validation rows of a fold are never written to and must not contribute to its
  // histograms.
  for (std::size_t k = 0; k < k_folds; ++k) {
    auto& fold_gpair = gpairs.at(k);
    fold_gpair.SetDevice(ctx->Device());
    fold_gpair.Reshape(info.num_row_, this->OutputLength(k));
    fold_gpair.Data()->Fill(GradientPair{});
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
      auto preds = GatherPrediction(ctx, fold_preds, ridxs, output_length);

      linalg::Matrix<GradientPair> batch_gpair;
      this->Objective(k)->GetGradient(preds, fold_info, iter, &batch_gpair);
      ScatterBatchGpair(ctx, batch_gpair, ridxs, &gpairs.at(k));
    }
  }
}

using tree::cuda_impl::MultiExpandEntry;
using tree::cuda_impl::StaticBatch;
// The partitioning helpers are shared with the single-model GPU hist maker.
using HistMaker = tree::cuda_impl::MultiTargetHistMaker;
// Copying a page to the device pays off once the kernels read enough of it. Same crossover
// as `GPUHistMakerDevice::NeedCopy`.
inline constexpr std::size_t kNeedCopyThreshold = 4;
template <typename Accessor>
using GoLeftOp = HistMaker::GoLeftOp<Accessor>;
using PartitionNodes = HistMaker::PartitionNodes;

class FoldTreeMethod {
  std::shared_ptr<DMatrix> p_fmat_;
  Context const* ctx_{nullptr};
  tree::TrainParam param_;
  tree::HistMakerTrainParam hist_param_;
  bool initialized_{false};

  // FIXME(jiamingy): The columns_sampler_ cannot be shared between folds.
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  std::shared_ptr<common::HistogramCuts const> cuts_;
  std::unique_ptr<tree::FeatureGroups> feature_groups_;
  std::vector<bst_idx_t> batch_ptr_;
  common::Span<FeatureType const> feature_types_;

  // Per-fold state.
  std::vector<std::unique_ptr<tree::DeviceHistogramBuilder>> histogram_;
  std::vector<std::unique_ptr<tree::GradientQuantiserGroup>> quantizers_;
  std::vector<linalg::Matrix<GradientPairInt64>> quantized_gpairs_;
  std::vector<std::unique_ptr<tree::cuda_impl::MultiHistEvaluator>> evaluators_;
  std::vector<tree::RowPartitionerBatches> partitioners_;

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
  }

  void BuildHist(EllpackPage const& page, std::int32_t batch_idx, std::size_t k,
                 std::vector<bst_node_t> const& build_nodes) {
    auto d_gpair = this->quantized_gpairs_.at(k).View(this->ctx_->Device());
    auto acc = page.Impl()->GetDeviceEllpack(this->ctx_, {});

    std::vector<common::Span<tree::cuda_impl::RowIndexT const>> h_ridxs;
    std::vector<common::Span<GradientPairInt64>> h_hists;
    std::vector<std::size_t> h_sizes_csum{0};
    for (auto nidx : build_nodes) {
      auto d_ridx = this->partitioners_.at(k).At(batch_idx)->GetRows(nidx);
      if (d_ridx.empty()) {
        // A fold can have no training rows for a node in this batch.
        continue;
      }
      h_ridxs.push_back(d_ridx);
      h_hists.push_back(this->histogram_.at(k)->GetNodeHistogram(nidx));
      h_sizes_csum.push_back(d_ridx.size() + h_sizes_csum.back());
    }
    if (h_ridxs.empty()) {
      return;
    }

    dh::device_vector<common::Span<GradientPairInt64>> hists{h_hists};
    dh::device_vector<common::Span<tree::cuda_impl::RowIndexT const>> ridxs{h_ridxs};
    this->histogram_.at(k)->BuildHistogram(
        this->ctx_, acc, this->feature_groups_->DeviceAccessor(this->ctx_->Device()), d_gpair,
        dh::ToSpan(ridxs), dh::ToSpan(hists), h_sizes_csum);
  }

  [[nodiscard]] auto MakeSharedInputs(std::size_t k, bst_feature_t max_active_feature) const {
    std::size_t constexpr kCatStorageSize = 0;  // FIXME(jiamingy): Support categorical features.
    return tree::MultiEvaluateSplitSharedInputs{this->quantizers_.at(k)->DeviceSpan(),
                                                this->cuts_->cut_ptrs_.ConstDeviceSpan(),
                                                this->cuts_->cut_values_.ConstDevicePointer(),
                                                this->feature_types_,
                                                kCatStorageSize,
                                                this->cuts_->TotalBins(),
                                                max_active_feature,
                                                tree::EvalParam{this->param_}};
  }

 public:
  explicit FoldTreeMethod(std::shared_ptr<DMatrix> p_fmat)
      : p_fmat_{std::move(p_fmat)},
        ctx_{p_fmat_->Ctx()},
        column_sampler_{std::make_shared<common::ColumnSampler>()} {}

  void Configure(Args const& args) {
    CHECK(ctx_->IsCUDA()) << "CV tree method `hist` requires a CUDA device.";

    auto unknown = param_.UpdateAllowUnknown(args);
    unknown = hist_param_.UpdateAllowUnknown(unknown);
    CheckNoUnknownParams(unknown);
  }

  void InitDataOnce() {
    CHECK(ctx_->IsCUDA()) << "CV tree method `hist` requires a CUDA device.";
    auto* p_fmat = p_fmat_.get();
    CHECK(p_fmat);
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

  void Reset(Context const* ctx, DMatrix* p_fmat, FoldInfoBatches const& finfo,
             FoldGpairs const& gpairs) {
    CHECK(!collective::IsDistributed())
        << "Distributed training is not supported by the CV tree method.";
    CHECK(!finfo.Empty());
    CHECK_EQ(finfo.KFolds(), gpairs.KFolds());
    CHECK(cuts_);
    // The page loops index the partitioners by the page counter, and the prediction cache
    // indexes `batch_ptr_` the same way.
    CHECK_EQ(p_fmat->NumBatches(), finfo.Size());
    CHECK_EQ(this->batch_ptr_.size(), finfo.Size() + 1);

    auto const& info = p_fmat->Info();
    info.feature_types.SetDevice(ctx->Device());
    this->feature_types_ = info.feature_types.ConstDeviceSpan();
    // Once per round, not per fold. GetFeatureSet returns the tree-level feature set
    // unconditionally when colsample is disabled, and that is null until Init runs.
    this->column_sampler_->Init(ctx, info.num_col_, info.feature_weights, param_.colsample_bynode,
                                param_.colsample_bylevel, param_.colsample_bytree);

    auto k_folds = finfo.KFolds();
    if (this->histogram_.empty()) {
      this->histogram_.resize(k_folds);
    }
    if (this->quantizers_.empty()) {
      this->quantizers_.resize(k_folds);
    }
    if (this->quantized_gpairs_.empty()) {
      this->quantized_gpairs_.resize(k_folds);
    }
    if (this->evaluators_.empty()) {
      this->evaluators_.resize(k_folds);
    }
    if (this->partitioners_.empty()) {
      this->partitioners_.resize(k_folds);
    }
    CHECK_EQ(this->histogram_.size(), k_folds);
    CHECK_EQ(this->quantizers_.size(), k_folds);
    CHECK_EQ(this->quantized_gpairs_.size(), k_folds);
    CHECK_EQ(this->evaluators_.size(), k_folds);
    CHECK_EQ(this->partitioners_.size(), k_folds);

    bst_target_t n_split_targets{0};
    for (std::size_t k = 0; k < k_folds; ++k) {
      auto const& fold_gpair = gpairs.gpairs.at(k);
      CHECK_EQ(finfo.n_samples, fold_gpair.Shape(0));
      auto n_train = finfo.FoldSize(k);
      CHECK_GT(n_train, 0) << "Empty training folds are not supported.";
      CHECK_GT(fold_gpair.Shape(1), 0);

      auto in_gpair = fold_gpair.View(ctx->Device());
      CHECK(in_gpair.CContiguous());
      if (k == 0) {
        n_split_targets = in_gpair.Shape(1);
      }
      CHECK_EQ(n_split_targets, in_gpair.Shape(1));

      // Only the training rows of the fold are accumulated, the rest of the buffer is zero.
      this->quantizers_[k] = std::make_unique<tree::GradientQuantiserGroup>(ctx, in_gpair, n_train);
      tree::CalcQuantizedGpairs(ctx, in_gpair, this->quantizers_[k]->DeviceSpan(),
                                &this->quantized_gpairs_[k]);

      auto n_total_bins = static_cast<bst_idx_t>(this->cuts_->TotalBins()) * n_split_targets;
      CHECK_LT(n_total_bins, std::numeric_limits<bst_bin_t>::max())
          << "Too many histogram bins: n_total_bins = total_bins * n_targets";
      bool force_global = false;
      if (!this->histogram_[k]) {
        this->histogram_[k] = std::make_unique<tree::DeviceHistogramBuilder>();
      }
      this->histogram_[k]->Reset(ctx, this->hist_param_.MaxCachedHistNodes(ctx->Device()),
                                 n_total_bins, force_global);

      // The partitioner holds only the training rows of this fold, so the histograms and the
      // final leaf positions never see a row held out by the fold.
      std::vector<common::Span<bst_idx_t const>> fold_ridxs;
      fold_ridxs.reserve(finfo.Size());
      for (auto const& batch : finfo.batches) {
        fold_ridxs.emplace_back(batch.TrainingFold(k));
      }
      this->partitioners_[k].Reset(ctx, finfo.n_samples, fold_ridxs);

      if (!this->evaluators_[k]) {
        this->evaluators_[k] = std::make_unique<tree::cuda_impl::MultiHistEvaluator>();
      }
      this->evaluators_[k]->Reset(ctx, this->cuts_->cut_ptrs_.ConstDeviceSpan(),
                                  this->feature_types_, this->param_, n_split_targets);
    }
  }

  // Build the root histogram of every fold and evaluate its split. Returns one expand entry
  // per fold, to be pushed into that fold's driver.
  [[nodiscard]] std::vector<MultiExpandEntry> InitRoot(DMatrix* p_fmat,
                                                       std::vector<RegTree*> const& trees) {
    auto k_folds = trees.size();
    CHECK_GT(k_folds, 0);
    CHECK_EQ(this->quantized_gpairs_.size(), k_folds);

    auto n_targets = this->quantized_gpairs_.front().Shape(1);
    for (std::size_t k = 0; k < k_folds; ++k) {
      CHECK(trees.at(k));
      CHECK_EQ(trees[k]->NumTargets(), n_targets);
      auto d_gpair = this->quantized_gpairs_.at(k).View(ctx_->Device());
      CHECK_EQ(d_gpair.Shape(1), n_targets);

      this->evaluators_[k]->AllocNodeSum(RegTree::kRoot, n_targets);
      tree::cuda_impl::CalcRootSum(ctx_, d_gpair,
                                   this->evaluators_[k]->GetNodeSum(RegTree::kRoot, n_targets));
      this->histogram_[k]->AllocateHistograms(ctx_, {RegTree::kRoot});
    }

    // Fused: every fold consumes the page before it is dropped.
    std::int32_t batch_idx = 0;
    for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx_, StaticBatch(true))) {
      for (std::size_t k = 0; k < k_folds; ++k) {
        this->BuildHist(page, batch_idx, k, {RegTree::kRoot});
      }
      ++batch_idx;
    }
    ++this->n_page_passes_;

    auto sampled_features = this->column_sampler_->GetFeatureSet(ctx_, 0);
    CHECK(sampled_features);
    sampled_features->SetDevice(ctx_->Device());
    auto feature_set = sampled_features->ConstDeviceSpan();

    auto root_weights = linalg::Empty<float>(ctx_, k_folds, n_targets);
    auto d_root_weights = root_weights.View(ctx_->Device());
    auto eta = this->param_.learning_rate;

    std::vector<MultiExpandEntry> entries(k_folds);
    for (std::size_t k = 0; k < k_folds; ++k) {
      // No histogram or root sum reduction: `Reset` rejects distributed training.
      tree::MultiEvaluateSplitInputs input{
          RegTree::kRoot, 0, this->evaluators_[k]->GetNodeSum(RegTree::kRoot, n_targets),
          feature_set, this->histogram_[k]->GetNodeHistogram(RegTree::kRoot)};
      auto shared_inputs =
          this->MakeSharedInputs(k, static_cast<bst_feature_t>(feature_set.size()));
      entries[k] = this->evaluators_[k]->EvaluateSingleSplit(ctx_, input, shared_inputs);

      // The weight is owned by the evaluator, ApplySplit reads it back by node id.
      auto base_weight = this->evaluators_[k]->GetNodeWeights(n_targets).Base(RegTree::kRoot);
      dh::LaunchN(n_targets, ctx_->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t t) mutable {
        d_root_weights(k, t) = base_weight[t] * eta;
      });
      // The root's coverage is the sum of the hessians of both children.
      auto root_sum_hess = static_cast<float>(entries[k].left_sum + entries[k].right_sum);
      trees[k]->SetRoot(d_root_weights.Slice(k, linalg::All()), root_sum_hess);
    }

    return entries;
  }

  // Push the rows of every fold through the splits that were just applied. Fused: a single
  // sweep over the pages serves all folds.
  // Whether the pages of the partition pass are worth copying to the device. Mirrors
  // `GPUHistMakerDevice::NeedCopy`, except that the rows are counted across the folds too: the
  // pass reads every page once per active fold, so the copy amortizes that many times better
  // here than it does for a single model.
  [[nodiscard]] bool NeedCopy(std::vector<std::size_t> const& active,
                              std::vector<PartitionNodes> const& nodes) const {
    if (this->p_fmat_->SingleColBlock()) {
      return true;  // Use the default for in-core data.
    }
    bst_idx_t n_visits = 0;
    for (std::size_t i = 0; i < active.size(); ++i) {
      for (auto const& part : this->partitioners_[active[i]]) {
        for (auto nidx : nodes[i].nidx) {
          n_visits += part->GetRows(nidx).size();
        }
      }
    }
    return n_visits * kNeedCopyThreshold > this->p_fmat_->Info().num_row_;
  }

  void PartitionAndBuildHist(DMatrix* p_fmat,
                             std::vector<std::vector<MultiExpandEntry>> const& expand_sets,
                             std::vector<RegTree*> const& trees) {
    // A fold whose split had no gain has nothing to partition. Skipping it here also
    // skips the tree view, whose construction copies the tree to the device.
    std::vector<std::size_t> active;
    std::vector<PartitionNodes> nodes;
    std::vector<tree::MultiTargetTreeView> views;
    for (std::size_t k = 0, k_folds = trees.size(); k < k_folds; ++k) {
      if (expand_sets[k].empty()) {
        continue;
      }
      active.push_back(k);
      nodes.emplace_back(HistMaker::CreatePartitionNodes(trees[k], expand_sets[k]));
      views.emplace_back(ctx_->Device(), false, trees[k]);
    }

    std::int32_t batch_idx = 0;
    for (auto const& page :
         p_fmat->GetBatches<EllpackPage>(ctx_, StaticBatch(this->NeedCopy(active, nodes)))) {
      page.Impl()->Visit(ctx_, {}, [&](auto&& d_acc) {
        using Acc = std::remove_reference_t<decltype(d_acc)>;
        for (std::size_t i = 0; i < active.size(); ++i) {
          auto go_left = GoLeftOp<Acc>{d_acc, views[i]};
          this->partitioners_[active[i]].UpdatePositionBatch(
              ctx_, batch_idx, nodes[i].nidx, nodes[i].left_nidx, nodes[i].right_nidx,
              nodes[i].split_data, tree::cuda_impl::GoLeftWrapperOp<GoLeftOp<Acc>>{go_left});

          // FIXME(jiamingy): Build histogram here.
        }
      });
      ++batch_idx;
    }
    ++this->n_page_passes_;
  }

  // Add the leaf value of the newly grown tree to the training prediction of every row
  // that the fold owns.
  void UpdatePredictionCache(FoldInfoBatches const& finfo, FoldPredictions* predts,
                             std::vector<RegTree*> const& trees) {
    auto k_folds = trees.size();
    auto n_samples = finfo.n_samples;

    // A single scratch buffer is enough, a fold is finished before the next one starts.
    // Scratch buffer for the leaf position of each row, reused by every fold.
    dh::DeviceUVector<bst_node_t> positions;
    positions.resize(n_samples);
    auto d_pos = dh::ToSpan(positions);

    for (std::size_t k = 0; k < k_folds; ++k) {
      auto& predt = predts->Training(k);
      predt.predictions.SetDevice(ctx_->Device());
      auto n_columns = predts->output_length;
      CHECK_EQ(n_columns * n_samples, predt.predictions.Size());
      auto d_predt = linalg::MakeTensorView(ctx_, &predt.predictions, n_samples, n_columns);

      // Rows held out by this fold are not in the partitioner and keep the sentinel.
      thrust::fill(ctx_->CUDACtx()->CTP(), dh::tbegin(d_pos), dh::tend(d_pos),
                   RegTree::kInvalidNodeId);
      for (std::size_t i = 0, n = finfo.Size(); i < n; ++i) {
        auto base_ridx = this->batch_ptr_[i];
        auto n_batch_samples = this->batch_ptr_.at(i + 1) - base_ridx;
        // The partitioner and the tree must have grown in lockstep. With fewer nodes than
        // the tree, the partitioner returns a node the tree has since split, reading the
        // wrong leaf or past the weights. With more, it returns a node the tree does not
        // have.
        CHECK_EQ(this->partitioners_[k].At(i)->GetNumNodes(), trees[k]->NumNodes());
        this->partitioners_[k].At(i)->FinalisePosition(
            ctx_, d_pos.subspan(base_ridx, n_batch_samples), base_ridx,
            [] XGBOOST_DEVICE(tree::cuda_impl::RowIndexT, bst_node_t nidx) { return nidx; });
      }

      auto tree = tree::MultiTargetTreeView{ctx_->Device(), false, trees[k]};
      dh::LaunchN(d_predt.Size(), ctx_->CUDACtx()->Stream(),
                  [=] XGBOOST_DEVICE(std::size_t i) mutable {
                    auto [ridx, t] = linalg::UnravelIndex(i, d_predt.Shape());
                    auto nidx = d_pos[ridx];
                    if (nidx == RegTree::kInvalidNodeId) {
                      return;  // Held out by this fold, the entry is unused padding.
                    }
                    d_predt(ridx, t) += tree.LeafValue(nidx)(t);
                  });

      if (this->hist_param_.debug_synchronize) {
        // Every training row of the fold, and only those, must have received a position.
        auto n_valid = thrust::count_if(
            ctx_->CUDACtx()->CTP(), dh::tcbegin(d_pos), dh::tcend(d_pos),
            [] XGBOOST_DEVICE(bst_node_t nidx) { return nidx != RegTree::kInvalidNodeId; });
        CHECK_EQ(static_cast<std::size_t>(n_valid), finfo.FoldSize(k));
      }
      predt.Update(1);
    }
  }

  void ApplySplit(std::size_t k, std::vector<MultiExpandEntry> const& candidates, RegTree* p_tree) {
    CHECK(!candidates.empty());
    auto n_targets = this->quantized_gpairs_.at(k).Shape(1);
    auto weights = this->evaluators_[k]->GetNodeWeights(n_targets);

    tree::ExpandBatch batch{this->param_.learning_rate};
    for (auto const& candidate : candidates) {
      // Categorical splits are rejected by CheckSupportedParams.
      CHECK(!candidate.split.is_cat);
      batch.Push(candidate.nidx, candidate.split.findex, candidate.split.fvalue,
                 candidate.split.dir == tree::kLeftDir, weights.Base(candidate.nidx),
                 weights.Left(candidate.nidx), weights.Right(candidate.nidx),
                 candidate.split.loss_chg, candidate.left_sum, candidate.right_sum);
    }
    p_tree->Expand(this->ctx_, batch);

    dh::device_vector<MultiExpandEntry> d_candidates{candidates};
    this->evaluators_[k]->ApplyTreeSplit(this->ctx_, p_tree,
                                         common::Span<MultiExpandEntry const>{candidates},
                                         dh::ToSpan(d_candidates), n_targets);
  }

  void Update(FoldModels* folds, DMatrix* p_fmat, FoldInfoBatches const& finfo,
              FoldGpairs const& gpairs, FoldPredictions* predts) {
    CHECK(folds);
    CHECK(p_fmat);
    CHECK(predts);
    CHECK_EQ(p_fmat, p_fmat_.get())
        << "CV tree method update must use the training DMatrix supplied at construction.";
    auto k_folds = folds->KFolds();
    CHECK_EQ(k_folds, finfo.KFolds());
    CHECK_EQ(k_folds, gpairs.KFolds());
    CHECK_EQ(k_folds, predts->KFolds());

    if (!initialized_) {
      this->InitDataOnce();
    }
    this->n_page_passes_ = this->n_levels_ = 0;

    std::vector<gbm::TreesOneIter> new_trees(k_folds);
    std::vector<RegTree*> tree_ptrs;
    tree_ptrs.reserve(k_folds);
    for (std::size_t k = 0; k < k_folds; ++k) {
      new_trees[k].resize(1);
      auto tree = std::make_unique<RegTree>(folds->LeafLength(k), folds->NumFeatures(k), true);
      tree_ptrs.push_back(tree.get());
      new_trees[k].front().push_back(std::move(tree));
    }

    this->Reset(ctx_, p_fmat, finfo, gpairs);

    std::vector<tree::Driver<MultiExpandEntry>> drivers;
    drivers.reserve(k_folds);
    for (std::size_t k = 0; k < k_folds; ++k) {
      drivers.emplace_back(param_, tree::cuda_impl::kMaxNodeBatchSize);
    }

    auto roots = this->InitRoot(p_fmat, tree_ptrs);
    std::vector<std::vector<MultiExpandEntry>> expand_sets(k_folds);
    for (std::size_t k = 0; k < k_folds; ++k) {
      drivers[k].Push(roots[k]);
      expand_sets[k] = drivers[k].Pop();
    }

    // Level-synchronous across folds, so that the partition pass can serve every fold from a
    // single sweep over the pages.
    while (std::any_of(expand_sets.cbegin(), expand_sets.cend(),
                       [](auto const& set) { return !set.empty(); })) {
      for (std::size_t k = 0; k < k_folds; ++k) {
        if (!expand_sets[k].empty()) {
          this->ApplySplit(k, expand_sets[k], tree_ptrs[k]);
        }
      }
      this->PartitionAndBuildHist(p_fmat, expand_sets, tree_ptrs);
      // TODO(jiamingy): Build the child histograms, evaluate them, and push the resulting
      // candidates back into the drivers to grow beyond depth 1.
      ++this->n_levels_;
      for (std::size_t k = 0; k < k_folds; ++k) {
        expand_sets[k] = drivers[k].Pop();
      }
    }

    for (std::size_t k = 0; k < k_folds; ++k) {
      auto* tree = tree_ptrs.at(k);
      tree->GetMultiTargetTree()->SetLeaves();
      hist_param_.CheckTreesSynchronized(ctx_, tree);
    }
    this->UpdatePredictionCache(finfo, predts, tree_ptrs);
    // One root build plus one partition pass per level, independent of the fold count.
    CHECK_EQ(this->n_page_passes_, 1 + this->n_levels_);
    folds->CommitModel(std::move(new_trees));
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
