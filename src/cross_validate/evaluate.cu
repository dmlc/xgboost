/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <dmlc/thread_local.h>  // for ThreadLocalStore

#include <algorithm>  // for none_of
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t, uint32_t
#include <iterator>   // for cbegin, cend
#include <memory>     // for unique_ptr
#include <string>     // for string
#include <utility>    // for move
#include <vector>     // for vector

#include "../c_api/c_api_error.h"
#include "../c_api/c_api_utils.h"  // for CastDMatrixHandle
#include "../common/api_entry.h"   // for XGBAPIThreadLocalEntry
#include "../common/json_utils.h"  // for OptionalArg
#include "./gather.cuh"            // for GatherRows
#include "cross_validate.h"
#include "xgboost/metric.h"       // for Metric
#include "xgboost/string_view.h"  // for StringView

namespace xgboost::cv {
namespace {
// FIXME(jiamingy): Metrics that are *not* derived from `MetricNoCache` are not supported.
constexpr StringView kUnsupported[]{
    // Reads `MetaInfo::LabelAbsSort`, whose cache is keyed on label size alone.
    "cox-nloglik",
    // Read a three-dimensional prediction layout that a fold cache does not have.
    "quantile", "expectile",
    // Read censoring bounds, which the scratch `MetaInfo` does not gather.
    "aft-nloglik", "interval-regression-accuracy"};

[[nodiscard]] std::unique_ptr<MetricNoCache> CreateFoldMetric(std::string const& name,
                                                              Context const* ctx) {
  std::unique_ptr<Metric> metric{Metric::Create(name, ctx)};
  auto* p_metric = dynamic_cast<MetricNoCache*>(metric.get());
  CHECK(p_metric) << "The `" << name << "` metric is not supported by cross-validation yet.";
  // Exact against `Metric::Name()`; a name embedding a parameter needs a split on `@`.
  auto resolved = StringView{p_metric->Name()};
  CHECK(std::none_of(std::cbegin(kUnsupported), std::cend(kUnsupported),
                     [&](StringView unsupported) { return resolved == unsupported; }))
      << "The `" << resolved << "` metric is not supported by cross-validation yet.";
  static_cast<void>(metric.release());
  return std::unique_ptr<MetricNoCache>{p_metric};
}

// The sample-aligned fields a metric reads, into `out` at `row_offset`. Not `MetaInfo::Slice`:
// it takes one index span, forcing the concatenated index array the batch loop avoids.
void GatherSampleInfo(Context const* ctx, MetaInfo const& info, common::Span<bst_idx_t const> ridx,
                      bst_idx_t row_offset, MetaInfo* out) {
  auto device = ctx->Device();
  info.labels.SetDevice(device);
  GatherRows(ctx, info.labels.Data()->ConstDeviceSpan(), ridx, row_offset, info.labels.Shape(1),
             out->labels.Data()->DeviceSpan());
  if (!info.weights_.Empty()) {
    info.weights_.SetDevice(device);
    GatherRows(ctx, info.weights_.ConstDeviceSpan(), ridx, row_offset, 1,
               out->weights_.DeviceSpan());
  }
}

constexpr StringView kStale{
    "CV prediction caches are not from the round being evaluated. Evaluation must run after "
    "the tree method has updated the caches for this round."};
}  // namespace

FoldEvaluator::FoldEvaluator(FoldModels const& models, Json const& config) {
  this->ctx_.FromJson(models.Ctx()->ToJson());
  CHECK(this->Ctx()->IsCUDA()) << "Fused cross-validation requires CUDA.";
  this->eval_train_ = OptionalArg<Boolean>(config, "eval_train", true);

  // No metric parameter is supported yet.
  for (auto const& kv : get<Object const>(config)) {
    CHECK(kv.first == "eval_metric" || kv.first == "eval_train")
        << "Unknown CV evaluator parameter: `" << kv.first << "`.";
  }

  // Resolved here and not lazily, so an unusable metric fails before any round does work.
  auto const& obj = get<Object const>(config);
  auto it = obj.find("eval_metric");
  std::vector<std::string> requested;
  if (it == obj.cend() || IsA<Null>(it->second)) {
    requested.emplace_back(models.Objective(0)->DefaultEvalMetric());
  } else if (IsA<String>(it->second)) {
    requested.emplace_back(get<String const>(it->second));
  } else {
    for (auto const& name : get<Array const>(it->second)) {
      requested.emplace_back(get<String const>(name));
    }
  }
  // No `disable_default_eval_metric`: a driver wanting no evaluation builds no evaluator.
  CHECK(!requested.empty()) << "`eval_metric` must name at least one metric.";

  for (auto const& name : requested) {
    auto metric = CreateFoldMetric(name, this->Ctx());
    // FIXME(jiamingy): Support metric arguments.
    metric->Configure(Args{});
    // `Name()` and not the request: a parameterized metric renames itself.
    this->result_.names.emplace_back(metric->Name());
    this->metrics_.push_back(std::move(metric));
  }
}

void FoldEvaluator::ResizeScratch(MetaInfo const& info, bst_idx_t n_rows) {
  auto device = this->Ctx()->Device();
  auto n_columns = info.labels.Shape(1);

  // Before resizing, or `Resize` takes the host path and leaves the rows there for good.
  this->predt_.SetDevice(device);
  this->predt_.Resize(n_rows * n_columns);

  auto& out = this->info_;
  // `num_row_` is load-bearing: `metric::CheckRowWeights` asserts the weight count equals it.
  out.num_row_ = n_rows;
  out.num_col_ = info.num_col_;
  out.labels.SetDevice(device);
  out.labels.Reshape(n_rows, n_columns);

  // Empty, not zero-filled: `OptionalWeights` reads all-zero weights as a perfect score.
  if (info.weights_.Empty()) {
    out.weights_.Resize(0);
  } else {
    out.weights_.SetDevice(device);
    out.weights_.Resize(n_rows);
  }
}

void FoldEvaluator::Reset(MetaInfo const& info, FoldInfoBatches const& finfo,
                          FoldPredictions const& predts) {
  CHECK(!finfo.Empty());
  // What keeps the refit unit out of the fold path.
  CHECK_EQ(finfo.KFolds(), predts.layout.k_folds);

  // Fold indices are drawn per batch, so a group spanning batches would be split.
  CHECK(info.group_ptr_.empty())
      << "Cross-validation evaluation does not support ranking data with query groups.";

  // Caches of another matrix would overrun the gather, which the version check cannot catch.
  auto output_length = predts.output_length;
  CHECK_EQ(info.labels.Shape(1), output_length);
  CHECK_EQ(predts.Validation().predictions.Size(), info.num_row_ * output_length);

  // Global, not per batch, and unchecked it would report `sqrt(0) == 0`.
  for (std::size_t k = 0; k < predts.layout.k_folds; ++k) {
    CHECK_GT(finfo.ValidFoldSize(k), 0)
        << "Fold " << k << " holds out no row, so it cannot be evaluated. `k_folds` must not "
        << "exceed the number of rows in the data batch.";
  }

  this->result_.Reset(predts.layout.k_folds, this->eval_train_);
}

void FoldEvaluator::EvalFold(FoldModels const& models, MetaInfo const& info,
                             FoldInfoBatches const& finfo, FoldPredictions const& predts,
                             std::size_t k, Split split) {
  auto valid = split == Split::kValid;
  auto n_rows = valid ? finfo.ValidFoldSize(k) : finfo.TrainFoldSize(k);
  // One shared held-out cache; the training caches are per fold, padded at the held-out rows.
  auto const& src = valid ? predts.Validation().predictions : predts.Prediction(k);
  src.SetDevice(this->Ctx()->Device());

  this->ResizeScratch(info, n_rows);

  // Ascending within a batch, batches in order, so rows arrive in dataset row order.
  auto n_columns = info.labels.Shape(1);
  bst_idx_t offset = 0;
  for (auto const& batch : finfo.batches) {
    auto ridx = valid ? batch.ValidationFold(k) : batch.TrainingFold(k);
    GatherRows(this->Ctx(), src.ConstDeviceSpan(), ridx, offset, n_columns,
               this->predt_.DeviceSpan());
    GatherSampleInfo(this->Ctx(), info, ridx, offset, &this->info_);
    offset += ridx.size();
  }
  // No row of a larger fold survives into a metric.
  CHECK_EQ(offset, n_rows);

  // A copy, so no transform reaches a cache, and each split resizes because one may resize.
  models.EvalTransform(&this->predt_);
  for (std::size_t m = 0, n = this->metrics_.size(); m < n; ++m) {
    auto idx = valid ? this->result_.ValidIdx(m, k) : this->result_.TrainIdx(m, k);
    this->result_.values.at(idx) = this->metrics_[m]->Eval(this->predt_, this->info_);
  }
}

[[nodiscard]] FoldEvalResult const& FoldEvaluator::Eval(FoldModels const& models,
                                                        MetaInfo const& info,
                                                        FoldInfoBatches const& finfo,
                                                        FoldPredictions const& predts,
                                                        std::int32_t iter) {
  CheckLayout(models.Layout(), predts.layout, "prediction caches");
  this->Reset(info, finfo, predts);

  // The caches agree with each other before and after a round, so only the round number
  // tells "after the update" from "before it".
  CHECK_GE(iter, 0) << "Cross-validation evaluation needs a committed round.";
  auto expected = static_cast<std::uint32_t>(iter) + 1;
  for (auto const& train : predts.train) {
    CHECK_EQ(train.version, expected) << kStale;
  }
  CHECK_EQ(predts.Validation().version, expected) << kStale;

  if (this->eval_train_) {
    for (std::size_t k = 0; k < models.Layout().k_folds; ++k) {
      this->EvalFold(models, info, finfo, predts, k, Split::kTrain);
    }
  }
  for (std::size_t k = 0; k < models.Layout().k_folds; ++k) {
    this->EvalFold(models, info, finfo, predts, k, Split::kValid);
  }
  return this->result_;
}
}  // namespace xgboost::cv

using namespace xgboost;  // NOLINT

namespace {
using CvAPIThreadLocalStore = dmlc::ThreadLocalStore<XGBAPIThreadLocalEntry>;
}  // namespace

XGB_DLL int XGBCvFoldEvaluatorCreate(FoldModelsHandle c_cv_folds, char const* c_config,
                                     FoldEvaluatorHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(c_cv_folds);
  xgboost_CHECK_C_ARG_PTR(c_config);
  xgboost_CHECK_C_ARG_PTR(out);
  auto const* cv_folds = static_cast<cv::FoldModels const*>(c_cv_folds);
  Json config{Json::Load(StringView{c_config})};
  *out = new cv::FoldEvaluator{*cv_folds, config};
  API_END();
}

XGB_DLL int XGBCvFoldEvaluatorFree(FoldEvaluatorHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<cv::FoldEvaluator*>(hdl);
  API_END();
}

XGB_DLL int XGBCvFoldEvaluatorEval(FoldEvaluatorHandle hdl, FoldModelsHandle c_cv_folds,
                                   DMatrixHandle dtrain, FoldInfoBatchesHandle c_fold_info,
                                   FoldPredictionsHandle c_predt, int iter, char const** out_meta,
                                   double const** out_values) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  xgboost_CHECK_C_ARG_PTR(c_cv_folds);
  xgboost_CHECK_C_ARG_PTR(dtrain);
  xgboost_CHECK_C_ARG_PTR(c_fold_info);
  xgboost_CHECK_C_ARG_PTR(c_predt);
  xgboost_CHECK_C_ARG_PTR(out_meta);
  xgboost_CHECK_C_ARG_PTR(out_values);
  auto const* cv_folds = static_cast<cv::FoldModels const*>(c_cv_folds);
  auto p_fmat = CastDMatrixHandle(dtrain);
  auto const* fold_info = static_cast<cv::FoldInfoBatches const*>(c_fold_info);
  auto const* predt = static_cast<cv::FoldPredictions const*>(c_predt);
  auto const& result = static_cast<cv::FoldEvaluator*>(hdl)->Eval(*cv_folds, p_fmat->Info(),
                                                                  *fold_info, *predt, iter);

  // Values raw, because `JsonNumber` is a `float` and would round to seven digits. The shape
  // rides with the names because the buffer goes over as a bare pointer with no length.
  Json jnames{Array{}};
  for (auto const& name : result.names) {
    get<Array>(jnames).emplace_back(String{name});
  }
  Json out{Object{}};
  out["names"] = std::move(jnames);
  auto jint = [](std::size_t v) {
    return Json{Integer{static_cast<std::int64_t>(v)}};
  };
  // Sections, metrics, folds: the order `FoldEvalResult` indexes in.
  out["shape"] = Array{std::vector<Json>{jint(result.eval_train ? 2 : 1), jint(result.NumMetrics()),
                                         jint(result.k_folds)}};
  auto& ret_str = CvAPIThreadLocalStore::Get()->ret_str;
  Json::Dump(out, &ret_str);
  *out_meta = ret_str.c_str();
  *out_values = result.values.data();
  API_END();
}
