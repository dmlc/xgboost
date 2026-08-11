/**
 * Copyright 2023-2026, XGBoost contributors
 */
#include <cmath>    // std::abs, std::sqrt
#include <cstddef>  // std::size_t
#include <cstdint>  // std::int32_t
#include <vector>   // std::vector

#include "../collective/aggregator.h"
#include "../collective/communicator-inl.h"
#include "../common/linalg_op.h"            // ElementWiseKernel,cbegin,cend
#include "../common/math.h"                 // CloseTo
#include "../common/numeric.h"              // Reduce
#include "../common/optional_weight.h"      // MakeOptionalWeights,SumOptionalWeights
#include "../common/quantile_loss_utils.h"  // QuantileLossParam
#include "../common/stats.h"                // Quantile,WeightedQuantile
#include "../common/transform.h"            // Transform
#include "init_estimation.h"                // CheckInitInputs
#include "xgboost/base.h"                   // GradientPair,XGBOOST_DEVICE,bst_target_t
#include "xgboost/data.h"                   // MetaInfo
#include "xgboost/host_device_vector.h"     // HostDeviceVector
#include "xgboost/json.h"                   // Json,String,ToJson,FromJson
#include "xgboost/linalg.h"                 // Tensor,MakeTensorView,MakeVec
#include "xgboost/objective.h"              // ObjFunction

#if defined(XGBOOST_USE_CUDA)
#include "../common/stats.cuh"  // SegmentedQuantile
#endif                          // defined(XGBOOST_USE_CUDA)

namespace xgboost::obj {
namespace {
// Fixed internal constants selected by the quantile experiments associated with
// https://github.com/dmlc/xgboost/issues/12351. Sweeps over quantile level, sample size,
// learning rate, and regularization did not find an alpha- or sample-size-dependent bandwidth
// rule that consistently improved c = 0.04. The relative 3e-4 curvature floor was the
// conservative safeguard for unregularized tail updates. Neither constant is user-facing.
constexpr float kSmoothingScale{0.04f};
constexpr float kMinSurrogateRatio{3.0e-4f};
}  // namespace

/**
 * @brief Automatically scaled logistic-smoothed quantile score with MM curvature.
 *
 * For residual r_ij = prediction_ij - label_i, compute a global scale independently for every
 * quantile:
 *
 *   S_j = (sum_i w_i * sqrt(abs(r_ij)) / sum_i w_i)^2,
 *   x_ij = r_ij / (c * S_j).
 *
 * Holding S_j fixed for the current boosting iteration, the gradient and supplied curvature are
 *
 *   g_ij = w_i * S_j / 2 * (tanh(x_ij) + 1 - 2 * alpha_j),
 *   h_ij = w_i / (2 * c) * max(tanh(x_ij) / x_ij, epsilon),
 *
 * where tanh(x) / x has the continuous value 1 at zero, c = 0.04, and epsilon = 3e-4.
 * Epsilon acts only as a relative Hessian floor; there is no beta*x or quadratic-tail term in
 * the gradient. When S_j is zero, including zero total weight, both output statistics are zero.
 *
 * Since (1 + tanh(x)) / 2 = logistic(2*x), g_ij / (w_i*S_j) is the score of a
 * logistic-convolution smoothing of pinball loss with bandwidth c*S_j/2. For fixed S_j, the exact
 * curvature of its even log-cosh component is w_i/(2*c) * sech(x_ij)^2; the alpha-dependent
 * linear tilt has zero curvature. The secant curvature w_i/(2*c) * tanh(x_ij)/x_ij gives a sharp
 * quadratic MM majorizer at the current residual. Applying the floor only increases this
 * curvature, preserving majorization while preventing unstable tail updates when ordinary tree
 * regularization is absent.
 *
 * Multiplication by S_j makes gradients scale with the response while leaving Hessians unchanged,
 * giving the standard leaf solve the same response-unit scaling as squared error. S_j is automatic
 * and uses globally reduced weighted statistics. Since it is recomputed per quantile and boosting
 * iteration, the supplied pairs define an iteration-dependent MM/IRLS surrogate rather than
 * derivatives of one fixed smooth loss.
 *
 * Logistic smoothing of quantile loss: https://doi.org/10.1016/j.jeconom.2021.07.010.
 * Quadratic majorization of log-cosh: https://doi.org/10.1016/j.csda.2009.01.002.
 */
class QuantileRegression : public ObjFunction {
  common::QuantileLossParam param_;
  HostDeviceVector<float> alpha_;

  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    auto const& alpha = param_.quantile_alpha.Get();
    CHECK_EQ(alpha.size(), alpha_.Size()) << "The objective is not yet configured.";
    CHECK_EQ(info.labels.Shape(1), 1) << "Multi-target is not yet supported by the quantile loss.";
    CHECK(!alpha.empty());
    // We have some placeholders for multi-target in the quantile loss. But it's not
    // supported as the gbtree doesn't know how to slice the gradient and there's no 3-dim
    // model shape in general.
    auto n_y = std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
    return alpha_.Size() * n_y;
  }

 public:
  void GetGradient(HostDeviceVector<float> const& preds, const MetaInfo& info, std::int32_t iter,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    if (iter == 0) {
      CheckInitInputs(info);
    }
    CHECK_EQ(param_.quantile_alpha.Get().size(), alpha_.Size());

    using SizeT = decltype(info.num_row_);
    SizeT n_targets = this->Targets(info);
    SizeT n_alphas = alpha_.Size();
    CHECK_NE(n_alphas, 0);
    CHECK_GE(n_targets, n_alphas);
    CHECK_EQ(preds.Size(), info.num_row_ * n_targets);

    auto labels = info.labels.View(ctx_->Device());

    out_gpair->SetDevice(ctx_->Device());
    CHECK_EQ(info.labels.Shape(1), 1)
        << "Multi-target for quantile regression is not yet supported.";
    out_gpair->Reshape(info.num_row_, n_targets);
    auto gpair = out_gpair->View(ctx_->Device());

    auto weight = common::MakeOptionalWeights(ctx_->Device(), info.weights_);

    preds.SetDevice(ctx_->Device());
    auto predt = linalg::MakeTensorView(ctx_, &preds, info.num_row_, n_targets);

    alpha_.SetDevice(ctx_->Device());
    auto alpha = ctx_->IsCPU() ? alpha_.ConstHostSpan() : alpha_.ConstDeviceSpan();

    HostDeviceVector<float> root_residual(info.num_row_, 0.0f, ctx_->Device());
    std::vector<double> scale_stats(n_targets + 1, 0.0);
    for (bst_target_t target{0}; target < n_targets; ++target) {
      if (info.num_row_ != 0) {
        common::Transform<>::Init(
            [target, labels, predt, weight] XGBOOST_DEVICE(std::size_t i,
                                                           common::Span<float> root_residual) {
              root_residual[i] = weight[i] * sqrtf(fabsf(predt(i, target) - labels(i, 0)));
            },
            common::Range{0, static_cast<std::int64_t>(info.num_row_)}, ctx_->Threads(),
            ctx_->Device())
            .Eval(&root_residual);
        scale_stats[target] = common::Reduce(ctx_, root_residual);
      }
    }
    scale_stats.back() = common::SumOptionalWeights(ctx_, weight, info.num_row_);

    auto cpu_ctx = ctx_->MakeCPU();
    collective::SafeColl(
        collective::GlobalSum(&cpu_ctx, linalg::MakeVec(scale_stats.data(), scale_stats.size())));

    HostDeviceVector<float> scale(n_targets, 0.0f, ctx_->Device());
    auto h_scale = scale.HostSpan();
    for (bst_target_t target{0}; target < n_targets; ++target) {
      if (common::CloseTo(scale_stats.back(), 0.0)) {
        h_scale[target] = 0.0f;
      } else {
        auto root_mean = scale_stats[target] / scale_stats.back();
        h_scale[target] = static_cast<float>(root_mean * root_mean);
      }
    }
    auto scale_view = ctx_->IsCPU() ? scale.ConstHostSpan() : scale.ConstDeviceSpan();

    linalg::ElementWiseKernel(
        ctx_, gpair, [=] XGBOOST_DEVICE(std::size_t i, std::size_t j) mutable {
          auto residual = predt(i, j) - labels(i, 0);
          auto residual_scale = scale_view[j];
          auto w = weight[i];
          if (!(residual_scale > 0.0f) || w == 0.0f) {
            gpair(i, j) = GradientPair{0.0f, 0.0f};
            return;
          }

          auto x = residual / (kSmoothingScale * residual_scale);
          auto tanh_x = tanhf(x);
          auto ratio = x == 0.0f ? 1.0f : tanh_x / x;
          ratio = fmaxf(ratio, kMinSurrogateRatio);
          auto grad = 0.5f * residual_scale * (tanh_x + 1.0f - 2.0f * alpha[j]);
          auto hess = 0.5f / kSmoothingScale * ratio;
          gpair(i, j) = GradientPair{w * grad, w * hess};
        });
  }

  void PredTransform(HostDeviceVector<float>* io_preds) const override {
    CHECK(!alpha_.Empty());
    CHECK_EQ(io_preds->Size() % alpha_.Size(), 0);
    // quantile_alpha is validated in ascending order. Sort each row's predictions directly to
    // prevent crossing. This insertion sort is simple and device portable.
    auto n_alphas = alpha_.Size();
    common::Transform<>::Init(
        [n_alphas] XGBOOST_DEVICE(std::size_t row, common::Span<float> predts) {
          auto offset = row * n_alphas;
          for (std::size_t i{1}; i < n_alphas; ++i) {
            auto value = predts[offset + i];
            auto pos = i;
            while (pos > 0 && predts[offset + pos - 1] > value) {
              predts[offset + pos] = predts[offset + pos - 1];
              --pos;
            }
            predts[offset + pos] = value;
          }
        },
        common::Range{0, static_cast<std::int64_t>(io_preds->Size() / n_alphas)}, ctx_->Threads(),
        io_preds->Device())
        .Eval(io_preds);
  }

  void InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) const override {
    CHECK(!alpha_.Empty());

    auto n_targets = this->Targets(info);
    base_score->SetDevice(ctx_->Device());
    base_score->Reshape(n_targets);

    if (ctx_->IsCUDA()) {
#if defined(XGBOOST_USE_CUDA)
      alpha_.SetDevice(ctx_->Device());
      auto d_alpha = alpha_.ConstDeviceSpan();
      auto d_labels = info.labels.View(ctx_->Device());
      auto seg_it = dh::MakeTransformIterator<std::size_t>(
          thrust::make_counting_iterator(0ul),
          [=] XGBOOST_DEVICE(std::size_t i) { return i * d_labels.Shape(0); });
      CHECK_EQ(d_labels.Shape(1), 1);
      auto val_it = dh::MakeTransformIterator<float>(thrust::make_counting_iterator(0ul),
                                                     [=] XGBOOST_DEVICE(std::size_t i) {
                                                       auto sample_idx = i % d_labels.Shape(0);
                                                       return d_labels(sample_idx, 0);
                                                     });
      auto n = d_labels.Size() * d_alpha.size();
      CHECK_EQ(base_score->Size(), d_alpha.size());
      if (info.weights_.Empty()) {
        common::SegmentedQuantile(ctx_, d_alpha.data(), seg_it, seg_it + d_alpha.size() + 1, val_it,
                                  val_it + n, base_score->Data());
      } else {
        info.weights_.SetDevice(ctx_->Device());
        auto d_weights = info.weights_.ConstDeviceSpan();
        auto weight_it = dh::MakeTransformIterator<float>(thrust::make_counting_iterator(0ul),
                                                          [=] XGBOOST_DEVICE(std::size_t i) {
                                                            auto sample_idx = i % d_labels.Shape(0);
                                                            return d_weights[sample_idx];
                                                          });
        common::SegmentedWeightedQuantile(ctx_, d_alpha.data(), seg_it, seg_it + d_alpha.size() + 1,
                                          val_it, val_it + n, weight_it, weight_it + n,
                                          base_score->Data());
      }
#else
      common::AssertGPUSupport();
#endif  // defined(XGBOOST_USE_CUDA)
    } else {
      auto quantiles = base_score->HostView();
      auto h_weights = info.weights_.ConstHostVector();
      for (bst_target_t t{0}; t < n_targets; ++t) {
        auto alpha = param_.quantile_alpha[t];
        auto h_labels = info.labels.HostView();
        if (h_weights.empty()) {
          quantiles(t) =
              common::Quantile(ctx_, alpha, linalg::cbegin(h_labels), linalg::cend(h_labels));
        } else {
          CHECK_EQ(h_weights.size(), h_labels.Size());
          quantiles(t) = common::WeightedQuantile(ctx_, alpha, linalg::cbegin(h_labels),
                                                  linalg::cend(h_labels), std::cbegin(h_weights));
        }
      }
    }

    // Global mean. There's no strong preference on whether weighted mean should be used
    // with weighted quantiles. The proper way to do this might be using an approximated
    // quantile algorithm with stream inputs, but it's also much more expensive.
    auto intercept = base_score->View(this->ctx_->Device());
    collective::SafeColl(collective::GlobalSum(ctx_, intercept));
    double n_workers = collective::GetWorldSize();
    linalg::VecScaDiv(ctx_, intercept, n_workers);
  }
  std::set<std::string> Configure(Args const& args) override {
    auto used = UpdateAndGetUsedParameters(&param_, args);
    param_.Validate();
    this->alpha_.HostVector() = param_.quantile_alpha.Get();
    return used;
  }
  [[nodiscard]] ObjInfo Task() const override { return {ObjInfo::kRegression, false}; }
  static char const* Name() { return "reg:quantileerror"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(Name());
    out["quantile_loss_param"] = ToJson(param_);
  }
  void LoadConfig(Json const& in) override {
    CHECK_EQ(get<String const>(in["name"]), Name());
    FromJson(in["quantile_loss_param"], &param_);
    param_.Validate();
    alpha_.HostVector() = param_.quantile_alpha.Get();
  }

  [[nodiscard]] const char* DefaultEvalMetric() const override { return "quantile"; }
  [[nodiscard]] Json DefaultMetricConfig() const override {
    CHECK(param_.GetInitialised());
    Json config{Object{}};
    config["name"] = String{this->DefaultEvalMetric()};
    config["quantile_loss_param"] = ToJson(param_);
    return config;
  }
};

XGBOOST_REGISTER_OBJECTIVE(QuantileRegression, QuantileRegression::Name())
    .describe("Regression with a smooth approximation to quantile loss.")
    .set_body([]() { return new QuantileRegression(); });

#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(quantile_obj_gpu);
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::obj
