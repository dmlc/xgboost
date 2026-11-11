/**
 * Copyright 2026, XGBoost Contributors
 */
#include "../common/cuda_compat.cuh"
#include "../common/cuda_rt_utils.h"  // for SetDevice
#include "../common/kernel.h"
#include "../tree/gpu_hist/sampler.cuh"
#include "row_sampler.h"

namespace xgboost::gbm {
namespace {
void CopyGradient(Context const* ctx, linalg::Matrix<GradientPair> const& in,
                  linalg::Matrix<GradientPair>* out) {
  in.SetDevice(ctx->Device());
  out->SetDevice(ctx->Device());
  out->Reshape(in.Shape());
  out->Data()->Copy(*in.Data());
}

class CUDARowSampler : public RowSampler {
  GradientContainer const& in_;
  tree::cuda_impl::Sampler sampler_;

 public:
  CUDARowSampler(Context const* ctx, tree::TrainParam const& param, GradientContainer const& in)
      : in_{in}, sampler_{in.gpair.Shape(0), param.subsample, param.sampling_method} {
    in.gpair.SetDevice(ctx->Device());
    sampler_.Prepare(ctx, in.gpair.View(ctx->Device()));
  }

  void Sample(Context const* ctx, GradientContainer* out) override {
    out->gpair.SetDevice(ctx->Device());
    out->gpair.Reshape(in_.gpair.Shape());
    // Fuse copying and sampling, without changing the original gradients.
    sampler_.Sample(ctx, in_.gpair.View(ctx->Device()), out->gpair.View(ctx->Device()));
    if (in_.HasValueGrad()) {
      CopyGradient(ctx, in_.value_gpair, &out->value_gpair);
      sampler_.ApplySampling(ctx, &out->value_gpair);
    } else {
      out->value_gpair = {};
    }
  }
};

std::unique_ptr<RowSampler> MakeRowSamplerCUDA(Context const* ctx, tree::TrainParam const& param,
                                               GradientContainer const& in) {
  // Scratch allocation must use the training device, before accessing the input.
  curt::SetDevice(ctx->Ordinal());
  return std::make_unique<CUDARowSampler>(ctx, param, in);
}

auto const kSampleRowsCUDA =
    common::KernelRegistration<MakeRowSamplerKernel>{DeviceOrd::kCUDA, MakeRowSamplerCUDA};
}  // namespace
}  // namespace xgboost::gbm
