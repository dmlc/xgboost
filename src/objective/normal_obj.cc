/**
 * Copyright 2026, XGBoost Contributors
 * \file normal_obj.cc
 * \brief CPU implementation and registration of normal distribution regression.
 */
#include "normal_obj.h"

#include <dmlc/registry.h>

#include <cmath>    // for log
#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <set>      // for set
#include <string>   // for string

#include "../common/kernel.h"           // for DispatchKernel, KernelRegistration
#include "../common/linalg_op.h"        // for ElementWiseKernel
#include "../common/optional_weight.h"  // for MakeOptionalWeights
#include "../common/stats.h"            // for SampleMean, WeightedSampleMean
#include "init_estimation.h"            // for CheckInitInputs
#include "xgboost/json.h"               // for Json, String
#include "xgboost/logging.h"            // for CHECK
#include "xgboost/objective.h"          // for ObjFunction

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(normal_obj);

namespace {
void NormalGradientCpu(Context const* ctx, HostDeviceVector<float> const& preds,
                       MetaInfo const& info, linalg::Matrix<GradientPair>* out_gpair) {
  auto predt = linalg::MakeTensorView(DeviceOrd::CPU(), preds.ConstHostSpan(), info.num_row_, 2);
  auto labels = info.labels.HostView();
  auto weights = common::MakeOptionalWeights(DeviceOrd::CPU(), info.weights_);

  out_gpair->SetDevice(DeviceOrd::CPU());
  out_gpair->Reshape(info.num_row_, 2);
  auto gpair = out_gpair->HostView();
  linalg::cpu_impl::ElementWiseKernel(labels, ctx->Threads(),
                                      [=](std::size_t i, std::size_t) mutable {
                                        NormalGradient{}(predt(i, 0), predt(i, 1), labels(i, 0),
                                                         weights[i], &gpair(i, 0), &gpair(i, 1));
                                      });
}

void NormalInitEstimationCpu(Context const* ctx, MetaInfo const& info,
                             linalg::Vector<float>* base_score) {
  linalg::Vector<float> mean;
  if (info.weights_.Empty()) {
    common::SampleMean(ctx, info.labels, &mean);
  } else {
    common::WeightedSampleMean(ctx, info.labels, info.weights_, &mean);
  }
  CHECK_EQ(mean.Size(), 1);

  linalg::Matrix<float> squared_residual;
  squared_residual.SetDevice(DeviceOrd::CPU());
  squared_residual.Reshape(info.num_row_, 1);
  auto residual = squared_residual.HostView();
  auto labels = info.labels.HostView();
  auto mean_value = mean.HostView()(0);
  linalg::cpu_impl::ElementWiseKernel(residual, ctx->Threads(),
                                      [=](std::size_t i, std::size_t) mutable {
                                        auto diff = labels(i, 0) - mean_value;
                                        residual(i, 0) = diff * diff;
                                      });

  linalg::Vector<float> variance;
  if (info.weights_.Empty()) {
    common::SampleMean(ctx, squared_residual, &variance);
  } else {
    common::WeightedSampleMean(ctx, squared_residual, info.weights_, &variance);
  }
  CHECK_EQ(variance.Size(), 1);

  base_score->SetDevice(DeviceOrd::CPU());
  base_score->Reshape(2);
  auto out = base_score->HostView();
  out(0) = mean_value;
  out(1) = std::log(variance.HostView()(0) + kNormalMinVariance);
}

auto const kRegisterNormalGradientCpu =
    common::KernelRegistration<NormalGradientKernel>{DeviceOrd::kCPU, &NormalGradientCpu};
auto const kRegisterNormalInitCpu = common::KernelRegistration<NormalInitEstimationKernel>{
    DeviceOrd::kCPU, &NormalInitEstimationCpu};
}  // namespace

class NormalRegression : public ObjFunction {
 public:
  std::set<std::string> Configure(Args const&) override { return {}; }
  [[nodiscard]] ObjInfo Task() const override { return ObjInfo::kRegression; }
  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    CHECK_LE(info.labels.Shape(1), 1) << "Normal regression requires a single response column.";
    return 2;
  }

  void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info, std::int32_t,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    CheckInitInputs(info);
    CHECK_EQ(info.labels.Shape(1), 1) << "Normal regression requires a single response column.";
    CHECK_EQ(preds.Size(), info.num_row_ * 2)
        << "Normal regression requires two predictions per row: mean and log variance.";
    if (!info.weights_.Empty()) {
      CHECK_EQ(info.weights_.Size(), info.num_row_)
          << "Number of weights should be equal to the number of data points.";
    }
    common::DispatchKernel<NormalGradientKernel>(ctx_, preds, info, out_gpair);
  }

  void InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) const override {
    CheckInitInputs(info);
    CHECK_EQ(info.labels.Shape(1), 1) << "Normal regression requires a single response column.";
    if (!info.weights_.Empty()) {
      CHECK_EQ(info.weights_.Size(), info.num_row_)
          << "Number of weights should be equal to the number of data points.";
    }
    common::DispatchKernel<NormalInitEstimationKernel>(ctx_, info, base_score);
  }

  [[nodiscard]] const char* DefaultEvalMetric() const override { return "normal-nloglik"; }

  void SaveConfig(Json* out) const override { (*out)["name"] = String{"reg:normal"}; }
  void LoadConfig(Json const& in) override {
    CHECK_EQ(get<String const>(in["name"]), "reg:normal");
  }
};

XGBOOST_REGISTER_OBJECTIVE(NormalRegression, "reg:normal")
    .describe("Normal distribution regression with mean and log-variance outputs.")
    .set_body([]() { return new NormalRegression(); });
}  // namespace xgboost::obj
