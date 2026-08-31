/**
 * Copyright 2023-2026, XGBoost contributors
 * \file quantile_obj.cc
 * \brief CPU implementations and registration of the quantile objective.
 */
#include "quantile_obj.h"

#include <dmlc/registry.h>

#include <algorithm>  // for max
#include <cmath>      // for fabsf, fmaxf, sqrtf, tanhf
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t
#include <vector>     // for vector

#include "../collective/aggregator.h"       // for GlobalSum
#include "../common/kernel.h"               // for DispatchKernel, KernelRegistration
#include "../common/linalg_op.h"            // for ElementWiseKernel
#include "../common/math.h"                 // for CloseTo
#include "../common/numeric.h"              // for Reduce
#include "../common/optional_weight.h"      // for MakeOptionalWeights, SumOptionalWeights
#include "../common/quantile_loss_utils.h"  // for QuantileLossParam
#include "../common/threading_utils.h"      // for ParallelFor
#include "../common/transform.h"            // for Transform
#include "init_estimation.h"                // for CheckInitInputs, FitIntercept
#include "xgboost/json.h"                   // for FromJson, Json, Object, String, ToJson
#include "xgboost/logging.h"                // for CHECK
#include "xgboost/objective.h"              // for ObjFunction

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(quantile_obj);

namespace {
// Fixed internal constants selected by the quantile experiments associated with
// https://github.com/dmlc/xgboost/issues/12351. Sweeps over quantile level, sample size,
// learning rate, and regularization did not find an alpha- or sample-size-dependent bandwidth
// rule that consistently improved c = 0.04. The relative 3e-4 curvature floor was the
// conservative safeguard for unregularized tail updates. Neither constant is user-facing.
constexpr float kSmoothingScale{0.04f};
constexpr float kMinSurrogateRatio{3.0e-4f};

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
void QuantileGradientCpu(Context const* ctx, HostDeviceVector<float> const& preds,
                         MetaInfo const& info, bst_target_t n_targets,
                         HostDeviceVector<float> const& alpha,
                         linalg::Matrix<GradientPair>* out_gpair) {
  auto labels = info.labels.HostView();
  auto predt =
      linalg::MakeTensorView(DeviceOrd::CPU(), preds.ConstHostSpan(), info.num_row_, n_targets);
  auto weights = common::MakeOptionalWeights(DeviceOrd::CPU(), info.weights_);
  auto alpha_h = alpha.ConstHostSpan();

  HostDeviceVector<float> root_residual(info.num_row_, 0.0f, DeviceOrd::CPU());
  std::vector<double> scale_stats(n_targets + 1, 0.0);
  for (bst_target_t target{0}; target < n_targets; ++target) {
    if (info.num_row_ != 0) {
      common::Transform<>::Init(
          [=](std::size_t i, common::Span<float> residual) {
            residual[i] = weights[i] * sqrtf(fabsf(predt(i, target) - labels(i, 0)));
          },
          common::Range{0, static_cast<std::int64_t>(info.num_row_)}, ctx->Threads(),
          DeviceOrd::CPU())
          .Eval(&root_residual);
      scale_stats[target] = common::Reduce(ctx, root_residual);
    }
  }
  scale_stats.back() = common::SumOptionalWeights(ctx, weights, info.num_row_);
  auto cpu_ctx = ctx->MakeCPU();
  collective::SafeColl(
      collective::GlobalSum(&cpu_ctx, linalg::MakeVec(scale_stats.data(), scale_stats.size())));

  std::vector<float> scale(n_targets, 0.0f);
  for (bst_target_t target{0}; target < n_targets; ++target) {
    if (scale_stats.back() != 0.0) {
      auto root_mean = scale_stats[target] / scale_stats.back();
      scale[target] = static_cast<float>(root_mean * root_mean);
    }
  }

  out_gpair->SetDevice(DeviceOrd::CPU());
  out_gpair->Reshape(info.num_row_, n_targets);
  auto gpair = out_gpair->HostView();
  linalg::cpu_impl::ElementWiseKernel(
      gpair, ctx->Threads(), [=](std::size_t i, std::size_t j) mutable {
        auto residual = predt(i, j) - labels(i, 0);
        auto residual_scale = scale[j];
        auto weight = weights[i];
        if (!(residual_scale > 0.0f) || weight == 0.0f) {
          gpair(i, j) = GradientPair{0.0f, 0.0f};
          return;
        }

        auto x = residual / (kSmoothingScale * residual_scale);
        auto tanh_x = tanhf(x);
        auto ratio = x == 0.0f ? 1.0f : tanh_x / x;
        ratio = fmaxf(ratio, kMinSurrogateRatio);
        auto grad = 0.5f * residual_scale * (tanh_x + 1.0f - 2.0f * alpha_h[j]);
        auto hess = 0.5f / kSmoothingScale * ratio;
        gpair(i, j) = GradientPair{weight * grad, weight * hess};
      });
}

void QuantileTransformCpu(Context const* ctx, HostDeviceVector<float>* predictions,
                          std::size_t n_alphas) {
  auto values = predictions->HostSpan();
  auto n_rows = values.size() / n_alphas;
  common::ParallelFor(n_rows, ctx->Threads(), [&](std::size_t row) {
    auto predictions = values.subspan(row * n_alphas, n_alphas);
    std::sort(predictions.begin(), predictions.end());
  });
}

const auto kRegisterQuantileGradientCpu =
    common::KernelRegistration<QuantileGradientKernel>{DeviceOrd::kCPU, &QuantileGradientCpu};
const auto kRegisterQuantileTransformCpu =
    common::KernelRegistration<QuantileTransformKernel>{DeviceOrd::kCPU, &QuantileTransformCpu};
}  // namespace

class QuantileRegression : public FitIntercept {
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

    auto n_targets = this->Targets(info);
    auto n_alphas = alpha_.Size();
    CHECK_NE(n_alphas, 0);
    CHECK_GE(n_targets, n_alphas);
    CHECK_EQ(preds.Size(), info.num_row_ * n_targets);
    CHECK_EQ(info.labels.Shape(1), 1)
        << "Multi-target for quantile regression is not yet supported.";

    common::DispatchKernel<QuantileGradientKernel>(ctx_, preds, info, n_targets, alpha_, out_gpair);
  }

  void PredTransform(HostDeviceVector<float>* predictions) const override {
    CHECK(!alpha_.Empty());
    CHECK_EQ(predictions->Size() % alpha_.Size(), 0);
    common::DispatchKernel<QuantileTransformKernel>(ctx_, predictions, alpha_.Size());
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
}  // namespace xgboost::obj
