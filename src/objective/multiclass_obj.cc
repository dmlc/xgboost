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

#include "../collective/aggregator.h"                  // for GlobalSum
#include "../common/exact_multinomial/packed_stats.h"  // for PackedHessianAtRow
#include "../common/kernel.h"                          // for DispatchKernel, KernelRegistration
#include "../common/linalg_op.h"                       // for SmallHistogram, vector operations
#include "../common/math.h"                            // for FindMaxIndex, Softmax
#include "../common/optional_weight.h"                 // for OptionalWeights
#include "../common/stats.h"                           // for Mean
#include "../common/threading_utils.h"                 // for ParallelFor
#include "multiclass_param.h"                          // for SoftmaxMultiClassParam
#include "xgboost/json.h"                              // for FromJson, Json, String, ToJson
#include "xgboost/logging.h"                           // for CHECK
#include "xgboost/objective.h"                         // for ObjFunction

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(multiclass_obj);
DMLC_REGISTER_PARAMETER(SoftmaxMultiClassParam);

namespace {
/** @brief Softmax normalization terms for one row. */
struct MulticlassRowNorm {
  float wmax;
  double wsum;
};

/**
 * @brief Reduce one row to its softmax normalization terms.
 *
 *   Shared with the exact kernel so that neither producer can drift away from the other's
 *   floating point behaviour.
 */
MulticlassRowNorm MulticlassNormalize(linalg::VectorView<float const> point) {
  float wmax = std::numeric_limits<float>::min();
  for (std::size_t k{0}; k < point.Size(); ++k) {
    wmax = fmaxf(point(k), wmax);
  }
  double wsum{0.0};
  for (std::size_t k{0}; k < point.Size(); ++k) {
    wsum += expf(point(k) - wmax);
  }
  return {wmax, wsum};
}

/** @brief Probability of class `k` in an already normalized row. */
float MulticlassProbability(linalg::VectorView<float const> point, MulticlassRowNorm norm,
                            std::int64_t k) {
  return expf(point(k) - norm.wmax) / static_cast<float>(norm.wsum);
}

/** @brief Gradient pair of a single class. */
GradientPair MulticlassClassGradient(float probability, float label, std::int64_t k,
                                     float weight) {
  auto grad = label == k ? probability - 1.0f : probability;
  // Absolute-residual pseudo-Hessian |p - y|.  Because sum|g| >= |sum g| in every
  // leaf, the leaf value -G/H is bounded by 1, and the curvature never vanishes on a
  // mis-predicted class.  The diagonal-dominance bound 2p(1-p) used previously
  // collapses to ~0 for rare classes while the gradient stays O(1), producing
  // unbounded leaf values at small reg_lambda.
  auto hess = fmaxf(fabsf(grad) * weight, 1e-16f);
  return {grad * weight, hess};
}

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
    auto norm = MulticlassNormalize(point);
    auto label = labels(row, 0);
    auto weight = weights[row];
    for (std::int64_t k{0}; k < n_classes; ++k) {
      gpair(row, k) =
          MulticlassClassGradient(MulticlassProbability(point, norm, k), label, k, weight);
    }
  });
}

/**
 * @brief Gradient and exact packed Hessian from one softmax evaluation per row.
 *
 *   The gradient is produced by the same helpers as @ref MulticlassGradientCpu, so the
 *   `out_gpair` result is bit-for-bit what the gradient only path produces.
 *
 *   With the last class as the reference class and `i, j < n_classes - 1`, the exact
 *   multinomial Hessian is
 *
 *     H_ij = weight * p_i * (delta_ij - p_j)
 *
 *   which depends on the probabilities and the weight alone, never on the label.
 */
void MulticlassExactGradientCpu(Context const* ctx, HostDeviceVector<float> const& preds,
                                MetaInfo const& info, std::int64_t n_classes,
                                linalg::Matrix<GradientPair>* out_gpair,
                                ExactHessian* out_hessian) {
  CHECK(ctx->IsCPU()) << "multi_hessian=exact is implemented for the CPU only.";
  CHECK_GE(n_classes, 2) << "multi_hessian=exact requires num_class >= 2, got " << n_classes
                         << ".";
  auto n_samples = info.num_row_;
  auto predt =
      linalg::MakeTensorView(DeviceOrd::CPU(), preds.ConstHostSpan(), n_samples, n_classes);
  auto labels = info.labels.HostView();
  common::OptionalWeights weights{info.weights_.ConstHostSpan()};
  out_gpair->SetDevice(DeviceOrd::CPU());
  out_gpair->Reshape(n_samples, n_classes);
  auto gpair = out_gpair->HostView();

  // The last class is the reference class, which leaves `n_classes - 1` free coordinates.
  auto n_free = static_cast<bst_target_t>(n_classes - 1);
  out_hessian->Reshape(n_samples, n_free);
  auto h_hessian = out_hessian->HostValues();

  // One probability buffer per thread, reused by every row that thread handles.
  auto n_threads = ctx->Threads();
  std::vector<float> probability_tloc(static_cast<std::size_t>(n_threads) * n_classes);

  common::ParallelFor(n_samples, n_threads, [&](std::size_t row) {
    auto point = predt.Slice(row, linalg::All());
    auto norm = MulticlassNormalize(point);
    auto label = labels(row, 0);
    auto weight = weights[row];
    auto probability = common::Span<float>{
        probability_tloc.data() + static_cast<std::size_t>(omp_get_thread_num()) * n_classes,
        static_cast<std::size_t>(n_classes)};

    for (std::int64_t k{0}; k < n_classes; ++k) {
      probability[k] = MulticlassProbability(point, norm, k);
      gpair(row, k) = MulticlassClassGradient(probability[k], label, k, weight);
    }

    auto hessian = common::PackedHessianAtRow(h_hessian, n_free, row);
    for (std::size_t i = 0; i < n_free; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        hessian.Set(i, j, weight * probability[i] * ((i == j ? 1.0f : 0.0f) - probability[j]));
      }
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
auto const kRegisterMulticlassExactGradientCpu =
    common::KernelRegistration<MulticlassExactGradientKernel>{DeviceOrd::kCPU,
                                                              &MulticlassExactGradientCpu};
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
  // The objective implements an exact dense Hessian producer. Whether a given run can use
  // it (device, tree method) is decided by the training path, not by this flag.
  ObjInfo Task() const override { return {ObjInfo::kClassification, false, true}; }

  void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info, std::int32_t iter,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    if (info.labels.Size() == 0) {
      return;
    }
    auto n_classes = static_cast<std::int64_t>(param_.num_class);
    auto kernel_ctx = this->MakeKernelCtx(preds, info, iter, n_classes);
    common::DispatchKernel<MulticlassGradientKernel>(&kernel_ctx, preds, info, n_classes,
                                                     out_gpair);
  }

  void GetGradientAndExactHessian(HostDeviceVector<float> const& preds, MetaInfo const& info,
                                  std::int32_t iter, linalg::Matrix<GradientPair>* out_gpair,
                                  ExactHessian* out_hessian) override {
    // Device support is deliberately checked here rather than advertised through ObjInfo:
    // the objective has the producer, this build only implements it for the host.
    CHECK(ctx_->IsCPU())
        << "multi_hessian=exact is implemented for the CPU only, but the objective runs on "
        << ctx_->Device().Name() << ". Set device=cpu, or use multi_hessian=diagonal.";
    if (info.labels.Size() == 0) {
      return;
    }
    auto n_classes = static_cast<std::int64_t>(param_.num_class);
    auto kernel_ctx = this->MakeKernelCtx(preds, info, iter, n_classes);
    common::DispatchKernel<MulticlassExactGradientKernel>(&kernel_ctx, preds, info, n_classes,
                                                          out_gpair, out_hessian);
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
  /**
   * @brief Validate the input shared by both gradient entry points and pick the kernel
   *        context.
   */
  [[nodiscard]] Context MakeKernelCtx(HostDeviceVector<float> const& preds, MetaInfo const& info,
                                      std::int32_t iter, std::int64_t n_classes) const {
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
    return kernel_ctx;
  }

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
