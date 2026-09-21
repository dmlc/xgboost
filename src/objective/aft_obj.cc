/**
 * Copyright 2019-2026, XGBoost Contributors
 * \file aft_obj.cc
 * \brief CPU implementation and registration of the AFT objective.
 */
#include "aft_obj.h"

#include <dmlc/registry.h>

#include <cmath>    // for exp, log
#include <cstddef>  // for size_t

#include "../common/kernel.h"           // for DispatchKernel, KernelRegistration
#include "../common/threading_utils.h"  // for ParallelFor
#include "xgboost/json.h"               // for Json
#include "xgboost/logging.h"            // for CHECK
#include "xgboost/objective.h"          // for ObjFunction

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(aft_obj);

namespace {
template <typename Distribution>
void AFTGradientCpuImpl(HostDeviceVector<float> const& preds, MetaInfo const& info, float scale,
                        std::int32_t n_threads, linalg::Matrix<GradientPair>* out_gpair) {
  auto predt = preds.ConstHostSpan();
  auto lower = info.labels_lower_bound_.ConstHostSpan();
  auto upper = info.labels_upper_bound_.ConstHostSpan();
  auto weights = info.weights_.ConstHostSpan();
  auto gpair = out_gpair->HostView();
  bool is_null_weight = weights.empty();
  common::ParallelFor(predt.size(), n_threads, [=](std::size_t i) mutable {
    auto grad = static_cast<float>(
        common::AFTLoss<Distribution>::Gradient(lower[i], upper[i], predt[i], scale));
    auto hess = static_cast<float>(
        common::AFTLoss<Distribution>::Hessian(lower[i], upper[i], predt[i], scale));
    auto weight = is_null_weight ? 1.0f : weights[i];
    gpair(i, 0) = {grad * weight, hess * weight};
  });
}

void AFTGradientCpu(Context const* ctx, HostDeviceVector<float> const& preds, MetaInfo const& info,
                    common::ProbabilityDistributionType distribution, float scale,
                    linalg::Matrix<GradientPair>* out_gpair) {
  out_gpair->SetDevice(DeviceOrd::CPU());
  out_gpair->Reshape(preds.Size(), 1);
  switch (distribution) {
    case common::ProbabilityDistributionType::kNormal:
      AFTGradientCpuImpl<common::NormalDistribution>(preds, info, scale, ctx->Threads(), out_gpair);
      break;
    case common::ProbabilityDistributionType::kLogistic:
      AFTGradientCpuImpl<common::LogisticDistribution>(preds, info, scale, ctx->Threads(),
                                                       out_gpair);
      break;
    case common::ProbabilityDistributionType::kExtreme:
      AFTGradientCpuImpl<common::ExtremeDistribution>(preds, info, scale, ctx->Threads(),
                                                      out_gpair);
      break;
    default:
      LOG(FATAL) << "Unrecognized distribution";
  }
}

void AFTPredTransformCpu(Context const* ctx, HostDeviceVector<float>* predictions) {
  auto values = predictions->HostSpan();
  common::ParallelFor(values.size(), ctx->Threads(),
                      [=](std::size_t i) { values[i] = std::exp(values[i]); });
}

void AFTProbToMarginCpu(Context const*, linalg::Vector<float>* base_score) {
  auto values = base_score->HostView();
  for (std::size_t i{0}; i < values.Size(); ++i) {
    values(i) = std::log(values(i));
  }
}

auto const kRegisterAFTGradientCpu =
    common::KernelRegistration<AFTGradientKernel>{DeviceOrd::kCPU, &AFTGradientCpu};
auto const kRegisterAFTPredTransformCpu =
    common::KernelRegistration<AFTPredTransformKernel>{DeviceOrd::kCPU, &AFTPredTransformCpu};
auto const kRegisterAFTProbToMarginCpu =
    common::KernelRegistration<AFTProbToMarginKernel>{DeviceOrd::kCPU, &AFTProbToMarginCpu};
}  // namespace

class AFTObj : public ObjFunction {
 public:
  std::set<std::string> Configure(Args const& args) override {
    return UpdateAndGetUsedParameters(&param_, args);
  }
  ObjInfo Task() const override { return ObjInfo::kSurvival; }

  void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info, std::int32_t,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    auto ndata = preds.Size();
    CHECK_EQ(info.labels_lower_bound_.Size(), ndata);
    CHECK_EQ(info.labels_upper_bound_.Size(), ndata);
    if (!info.weights_.Empty()) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }
    common::DispatchKernel<AFTGradientKernel>(ctx_, preds, info, param_.aft_loss_distribution,
                                              param_.aft_loss_distribution_scale, out_gpair);
  }

  void PredTransform(HostDeviceVector<float>* predictions) const override {
    // Trees give us a prediction in log scale, so exponentiate.
    common::DispatchKernel<AFTPredTransformKernel>(ctx_, predictions);
  }
  // Do nothing here, since the AFT metric expects untransformed prediction scores.
  void EvalTransform(HostDeviceVector<float>*) override {}
  void ProbToMargin(linalg::Vector<float>* base_score) const override {
    common::DispatchKernel<AFTProbToMarginKernel>(ctx_, base_score);
  }
  char const* DefaultEvalMetric() const override { return "aft-nloglik"; }

  void SaveConfig(Json* out) const override {
    (*out)["name"] = String("survival:aft");
    (*out)["aft_loss_param"] = ToJson(param_);
  }
  void LoadConfig(Json const& in) override { FromJson(in["aft_loss_param"], &param_); }
  Json DefaultMetricConfig() const override {
    Json config{Object{}};
    config["name"] = String{this->DefaultEvalMetric()};
    config["aft_loss_param"] = ToJson(param_);
    return config;
  }

 private:
  common::AFTParam param_;
};

XGBOOST_REGISTER_OBJECTIVE(AFTObj, "survival:aft").describe("AFT loss function").set_body([]() {
  return new AFTObj();
});
}  // namespace xgboost::obj
