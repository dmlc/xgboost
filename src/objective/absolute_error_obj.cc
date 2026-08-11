/**
 * Copyright 2026, XGBoost Contributors
 * \file absolute_error_obj.cc
 * \brief CPU implementation and registration of the absolute-error objective.
 */
#include "absolute_error_obj.h"

#include <dmlc/registry.h>

#include <algorithm>  // for max
#include <cmath>      // for fabsf, hypotf, sqrtf
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t
#include <memory>     // for unique_ptr
#include <vector>     // for vector

#include "../collective/aggregator.h"   // for GlobalSum
#include "../common/common.h"           // for CloseTo
#include "../common/kernel.h"           // for DispatchKernel, KernelRegistration
#include "../common/linalg_op.h"        // for ElementWiseKernel
#include "../common/math.h"             // for CloseTo
#include "../common/numeric.h"          // for Reduce
#include "../common/optional_weight.h"  // for MakeOptionalWeights
#include "../common/stats.h"            // for SampleMean, WeightedSampleMean
#include "../common/transform.h"        // for Transform
#include "../tree/fit_stump.h"          // for FitStump
#include "init_estimation.h"            // for CheckInitInputs
#include "xgboost/json.h"               // for Json, Object, String
#include "xgboost/logging.h"            // for CHECK, LOG
#include "xgboost/objective.h"          // for ObjFunction
#include "xgboost/string_view.h"        // for StringView

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(absolute_error_obj);

namespace {
void AbsoluteErrorGradientCpu(Context const* ctx, HostDeviceVector<float> const& preds,
                              MetaInfo const& info, bst_target_t n_targets,
                              linalg::Matrix<GradientPair>* out_gpair) {
  auto labels = info.labels.HostView();
  auto predt =
      linalg::MakeTensorView(DeviceOrd::CPU(), preds.ConstHostSpan(), info.num_row_, n_targets);
  auto weights = common::MakeOptionalWeights(DeviceOrd::CPU(), info.weights_);

  HostDeviceVector<float> root_residual(info.num_row_, 0.0f, DeviceOrd::CPU());
  std::vector<double> scale_stats(n_targets + 1, 0.0);
  for (bst_target_t target{0}; target < n_targets; ++target) {
    common::Transform<>::Init(
        [=](std::size_t i, common::Span<float> residual) {
          residual[i] = weights[i] * sqrtf(fabsf(predt(i, target) - labels(i, target)));
        },
        common::Range{0, static_cast<std::int64_t>(info.num_row_)}, ctx->Threads(),
        DeviceOrd::CPU())
        .Eval(&root_residual);
    scale_stats[target] = common::Reduce(ctx, root_residual);
  }
  scale_stats.back() = common::SumOptionalWeights(ctx, weights, info.num_row_);
  auto cpu_ctx = ctx->MakeCPU();
  collective::SafeColl(
      collective::GlobalSum(&cpu_ctx, linalg::MakeVec(scale_stats.data(), scale_stats.size())));

  std::vector<float> scale(n_targets, 0.0f);
  for (bst_target_t target{0}; target < n_targets; ++target) {
    if (!common::CloseTo(scale_stats.back(), 0.0)) {
      auto root_mean = scale_stats[target] / scale_stats.back();
      scale[target] = static_cast<float>(root_mean * root_mean);
    }
  }

  out_gpair->SetDevice(DeviceOrd::CPU());
  out_gpair->Reshape(info.num_row_, n_targets);
  auto gpair = out_gpair->HostView();
  linalg::cpu_impl::ElementWiseKernel(
      gpair, ctx->Threads(), [=](std::size_t i, std::size_t j) mutable {
        auto residual = predt(i, j) - labels(i, j);
        auto norm = hypotf(scale[j], residual);
        auto curvature = norm > 0.0f ? scale[j] / norm : 1.0f;
        auto weight = weights[i];
        gpair(i, j) = {weight * residual * curvature, weight * curvature};
      });
}

void AbsoluteErrorInitEstimationCpu(Context const* ctx, MetaInfo const& info,
                                    bst_target_t n_targets, linalg::Vector<float>* base_score) {
  double sum_weight = info.weights_.Empty() ? static_cast<double>(info.num_row_)
                                            : common::Reduce(ctx, info.weights_);
  collective::SafeColl(collective::GlobalSum(ctx, linalg::MakeVec(&sum_weight, std::size_t{1})));
  if (common::CloseTo(sum_weight, 0.0)) {
    LOG(WARNING) << "Sum of weights is close to 0.0, skipping base score estimation.";
    *base_score = linalg::Zeros<float>(ctx, n_targets);
    return;
  }

  linalg::Vector<float> mean;
  if (info.weights_.Empty()) {
    common::SampleMean(ctx, info.labels, &mean);
  } else {
    common::WeightedSampleMean(ctx, info.labels, info.weights_, &mean);
  }
  CHECK_EQ(mean.Size(), n_targets);

  HostDeviceVector<float> predt(info.labels.Size(), 0.0f, DeviceOrd::CPU());
  auto predt_h =
      linalg::MakeTensorView(DeviceOrd::CPU(), predt.HostSpan(), info.num_row_, n_targets);
  auto mean_h = mean.HostView();
  linalg::cpu_impl::ElementWiseKernel(
      predt_h, ctx->Threads(),
      [=](std::size_t i, std::size_t target) mutable { predt_h(i, target) = mean_h(target); });

  linalg::Matrix<GradientPair> gpair;
  AbsoluteErrorGradientCpu(ctx, predt, info, n_targets, &gpair);
  tree::FitStump(ctx, gpair, n_targets, base_score);

  auto out = base_score->HostView();
  for (bst_target_t target{0}; target < n_targets; ++target) {
    out(target) += mean_h(target);
  }
}

auto const kRegisterAbsoluteErrorGradientCpu =
    common::KernelRegistration<AbsoluteErrorGradientKernel>{DeviceOrd::kCPU,
                                                            &AbsoluteErrorGradientCpu};
auto const kRegisterAbsoluteErrorInitEstimationCpu =
    common::KernelRegistration<AbsoluteErrorInitEstimationKernel>{DeviceOrd::kCPU,
                                                                  &AbsoluteErrorInitEstimationCpu};
}  // namespace

/**
 * @brief Smooth MM approximation to the mean absolute error.
 *
 * At each boosting iteration and for each target, choose the automatic scale
 *
 *   delta = E_w[sqrt(abs(prediction - label))]^2.
 *
 * For residual r, q = sqrt(1 + (r / delta)^2), the pseudo-Huber gradient is r / q.
 * We use 1 / q as the Hessian instead of the exact pseudo-Huber Hessian 1 / q^3. This is
 * the majorization curvature that produces a stable IRLS update while approaching the L1
 * gradient as the residual scale contracts.
 */
class MeanAbsoluteError : public ObjFunction {
 public:
  std::set<std::string> Configure(Args const&) override { return {}; }
  [[nodiscard]] ObjInfo Task() const override { return {ObjInfo::kRegression, false}; }
  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    return std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info, std::int32_t,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    CheckInitInputs(info);
    CHECK_EQ(info.labels.Size(), preds.Size()) << "Invalid shape of labels.";
    common::DispatchKernel<AbsoluteErrorGradientKernel>(ctx_, preds, info, this->Targets(info),
                                                        out_gpair);
  }

  void InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) const override {
    CheckInitInputs(info);
    common::DispatchKernel<AbsoluteErrorInitEstimationKernel>(ctx_, info, this->Targets(info),
                                                              base_score);
  }

  [[nodiscard]] const char* DefaultEvalMetric() const override { return "mae"; }
  void SaveConfig(Json* out) const override { (*out)["name"] = String("reg:absoluteerror"); }
  void LoadConfig(Json const& in) override {
    CHECK_EQ(StringView{get<String const>(in["name"])}, StringView{"reg:absoluteerror"});
  }
};

XGBOOST_REGISTER_OBJECTIVE(MeanAbsoluteError, "reg:absoluteerror")
    .describe("Mean absolute error with automatic smooth majorization.")
    .set_body([]() { return new MeanAbsoluteError(); });
}  // namespace xgboost::obj
