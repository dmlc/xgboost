/**
 * Copyright 2026, XGBoost Contributors
 * \file multiclass_obj.cc
 * \brief CPU implementations and registration of multiclass objectives.
 */
#include "multiclass_obj.h"

#include <dmlc/registry.h>

#include <algorithm>  // for max
#include <cassert>    // for assert
#include <cmath>      // for expf, fmaxf
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t, int64_t
#include <limits>     // for numeric_limits
#include <vector>     // for vector

#include "../collective/aggregator.h"   // for GlobalSum
#include "../common/kernel.h"           // for DispatchKernel, KernelRegistration
#include "../common/linalg_op.h"        // for SmallHistogram, vector operations
#include "../common/math.h"             // for FindMaxIndex, Softmax
#include "../common/optional_weight.h"  // for OptionalWeights
#include "../common/stats.h"            // for Mean
#include "../common/threading_utils.h"  // for ParallelFor
#include "multiclass_param.h"           // for SoftmaxMultiClassParam
#include "xgboost/json.h"               // for FromJson, Json, String, ToJson
#include "xgboost/logging.h"            // for CHECK
#include "xgboost/objective.h"          // for ObjFunction

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(multiclass_obj);
DMLC_REGISTER_PARAMETER(SoftmaxMultiClassParam);

namespace {
void MulticlassGradientCpu(Context const* ctx, HostDeviceVector<float> const& preds,
                           MetaInfo const& info, std::int64_t n_classes,
                           linalg::Matrix<GradientPair>* out_gpair) {
  auto n_samples = info.num_row_;
  auto predt =
      linalg::MakeTensorView(DeviceOrd::CPU(), preds.ConstHostSpan(), n_samples, n_classes);
  auto labels = info.labels.HostView();
  common::OptionalWeights weights{info.weights_.ConstHostSpan()};
  out_gpair->SetDevice(DeviceOrd::CPU());
  out_gpair->Reshape(n_samples, n_classes);
  auto gpair = out_gpair->HostView();

  common::ParallelFor(n_samples, ctx->Threads(), [&](std::size_t row) {
    auto point = predt.Slice(row, linalg::All());
    float wmax = std::numeric_limits<float>::min();
    for (std::size_t k{0}; k < point.Size(); ++k) {
      wmax = fmaxf(point(k), wmax);
    }
    double wsum{0.0};
    for (std::size_t k{0}; k < point.Size(); ++k) {
      wsum += expf(point(k) - wmax);
    }
    auto label = labels(row, 0);
    auto weight = weights[row];
    for (std::int64_t k{0}; k < n_classes; ++k) {
      auto probability = expf(point(k) - wmax) / static_cast<float>(wsum);
      auto hess = fmaxf(2.0f * probability * (1.0f - probability) * weight, 1e-16f);
      auto grad = label == k ? probability - 1.0f : probability;
      gpair(row, k) = {grad * weight, hess};
    }
  });
}

void MulticlassTransformCpu(Context const* ctx, HostDeviceVector<float>* predictions,
                            std::int32_t n_classes, bool probability) {
  auto values = predictions->HostSpan();
  auto n_samples = values.size() / n_classes;
  if (probability) {
    common::ParallelFor(n_samples, ctx->Threads(), [&](std::size_t row) {
      auto point = values.subspan(row * n_classes, n_classes);
      common::Softmax(point.begin(), point.end());
    });
  } else {
    std::vector<float> output(n_samples);
    common::ParallelFor(n_samples, ctx->Threads(), [&](std::size_t row) {
      auto point = common::Span<float const>{values.data() + row * n_classes,
                                             static_cast<std::size_t>(n_classes)};
      output[row] = common::FindMaxIndex(point.cbegin(), point.cend()) - point.cbegin();
    });
    predictions->HostVector() = std::move(output);
  }
}

void MulticlassInitEstimationCpu(Context const* ctx, MetaInfo const& info, std::int64_t n_classes,
                                 linalg::Vector<float>* base_score) {
  *base_score = linalg::Zeros<float>(ctx, n_classes);
  auto labels = info.labels.HostView();
  common::OptionalWeights weights{info.weights_.ConstHostSpan()};
  auto intercept = base_score->HostView();
  linalg::SmallHistogram(ctx, labels, weights, intercept);
  auto sum_weight = common::SumOptionalWeights(ctx, weights, info.labels.Size());
  collective::SafeColl(collective::GlobalSum(ctx, intercept, &sum_weight));
  CHECK_GE(sum_weight, kRtEps);
  linalg::VecScaDiv(ctx, intercept, sum_weight);
  linalg::LogE(ctx, intercept, kRtEps);
  linalg::Vector<float> mean;
  common::Mean(ctx, intercept, &mean);
  common::DispatchKernel<MulticlassCenterKernel>(ctx, base_score->Data(),
                                                 MulticlassCenter{mean.HostView()(0)});
}

auto const kRegisterMulticlassGradientCpu =
    common::KernelRegistration<MulticlassGradientKernel>{DeviceOrd::kCPU, &MulticlassGradientCpu};
auto const kRegisterMulticlassInitEstimationCpu =
    common::KernelRegistration<MulticlassInitEstimationKernel>{DeviceOrd::kCPU,
                                                               &MulticlassInitEstimationCpu};
auto const kRegisterMulticlassTransformCpu =
    common::KernelRegistration<MulticlassTransformKernel>{DeviceOrd::kCPU, &MulticlassTransformCpu};
auto const kRegisterMulticlassValidationCpu =
    elementwise::RegisterValidationCpu<MulticlassLabelCheck>();
auto const kRegisterMulticlassCenterCpu = elementwise::RegisterTransformCpu<MulticlassCenter>();
}  // namespace

class SoftmaxMultiClassObj : public ObjFunction {
 public:
  explicit SoftmaxMultiClassObj(bool output_prob) : output_prob_{output_prob} {}
  std::set<std::string> Configure(Args const& args) override {
    return UpdateAndGetUsedParameters(&param_, args);
  }
  ObjInfo Task() const override { return ObjInfo::kClassification; }

  void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info, std::int32_t iter,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    if (info.labels.Size() == 0) {
      return;
    }
    auto n_classes = static_cast<std::int64_t>(param_.num_class);
    CHECK_EQ(preds.Size(), static_cast<std::size_t>(n_classes) * info.labels.Size())
        << "SoftmaxMultiClassObj: label size and pred size does not match.";
    CHECK_EQ(preds.Size() / n_classes, info.num_row_);
    CHECK_LE(info.labels.Shape(1), 1) << "multi-class-multi-label is not yet supported.";
    if (!info.weights_.Empty()) {
      CHECK_EQ(info.weights_.Size(), info.num_row_)
          << "Number of weights should be equal to number of data points.";
    }

    auto device = ctx_->DeviceFP64();
    auto kernel_ctx = device.IsCPU() ? ctx_->MakeCPU() : *ctx_;
    if (iter == 0) {
      auto valid = common::DispatchKernel<MulticlassValidationKernel>(
          &kernel_ctx, info.labels, MulticlassLabelCheck{n_classes});
      CHECK(valid)
          << "SoftmaxMultiClassObj: label must be discrete values in the range of [0, num_class).";
    }
    common::DispatchKernel<MulticlassGradientKernel>(&kernel_ctx, preds, info, n_classes,
                                                     out_gpair);
  }

  void PredTransform(HostDeviceVector<float>* predictions) const override {
    this->Transform(predictions, output_prob_);
  }
  void EvalTransform(HostDeviceVector<float>* predictions) override {
    this->Transform(predictions, true);
  }
  char const* DefaultEvalMetric() const override { return "mlogloss"; }

  void Transform(HostDeviceVector<float>* predictions, bool probability) const {
    common::DispatchKernel<MulticlassTransformKernel>(ctx_, predictions, param_.num_class,
                                                      probability);
  }

  void SaveConfig(Json* out) const override {
    (*out)["name"] = String(output_prob_ ? "multi:softprob" : "multi:softmax");
    (*out)["softmax_multiclass_param"] = ToJson(param_);
  }
  void LoadConfig(Json const& in) override { FromJson(in["softmax_multiclass_param"], &param_); }

  void InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) const override {
    auto n_classes = static_cast<std::int64_t>(param_.num_class);
    CHECK_LE(info.labels.Shape(1), 1) << "multi-class-multi-label is not yet supported.";
    auto valid = common::DispatchKernel<MulticlassValidationKernel>(
        ctx_, info.labels, MulticlassLabelCheck{n_classes});
    CHECK(valid)
        << "SoftmaxMultiClassObj: label must be discrete values in the range of [0, num_class).";

    common::DispatchKernel<MulticlassInitEstimationKernel>(ctx_, info, n_classes, base_score);
  }

 private:
  bool const output_prob_;
  SoftmaxMultiClassParam param_;
};

XGBOOST_REGISTER_OBJECTIVE(SoftmaxMultiClass, "multi:softmax")
    .describe("Softmax for multi-class classification, output class index.")
    .set_body([]() { return new SoftmaxMultiClassObj(false); });
XGBOOST_REGISTER_OBJECTIVE(SoftprobMultiClass, "multi:softprob")
    .describe("Softmax for multi-class classification, output probability distribution.")
    .set_body([]() { return new SoftmaxMultiClassObj(true); });
}  // namespace xgboost::obj
