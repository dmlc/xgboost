/**
 * Copyright 2026, XGBoost Contributors
 */
#include "row_sampler.h"

#include <algorithm>  // for copy
#include <vector>     // for vector

#include "../common/kernel.h"
#include "../common/linalg_op.h"
#include "../common/threading_utils.h"
#include "../tree/hist/sampler.h"

namespace xgboost::gbm {
namespace {
class CPURowSampler : public RowSampler {
  GradientContainer const& in_;
  tree::cpu_impl::Sampler sampler_;
  std::vector<float> weights_;

 public:
  CPURowSampler(Context const* ctx, tree::TrainParam const& param, GradientContainer const& in)
      : in_{in}, sampler_{param}, weights_(in.HasValueGrad() ? in.gpair.Shape(0) : 0) {
    sampler_.Prepare(ctx, in.gpair.HostView());
  }

  void Sample(Context const* ctx, GradientContainer* out) override {
    auto split = in_.gpair.HostView();
    if (out->gpair.Shape() != split.Shape() || !out->gpair.HostView().FContiguous()) {
      out->gpair = linalg::Matrix<GradientPair>{split.Shape(), ctx->Device(), linalg::Order::kF};
    }
    auto sampled = out->gpair.HostView();
    std::copy(linalg::cbegin(split), linalg::cend(split), linalg::begin(sampled));

    sampler_.SamplePrepared(ctx, sampled, common::Span{weights_});
    if (in_.HasValueGrad()) {
      auto value = in_.value_gpair.HostView();
      out->value_gpair.SetDevice(ctx->Device());
      out->value_gpair.Reshape(value.Shape());
      auto sampled_value = out->value_gpair.HostView();
      common::ParallelFor(value.Shape(0), ctx->Threads(), [&](auto i) {
        for (bst_target_t t = 0; t < value.Shape(1); ++t) {
          sampled_value(i, t) = value(i, t) * weights_[i];
        }
      });
    } else {
      out->value_gpair = {};
    }
  }
};

std::unique_ptr<RowSampler> MakeRowSamplerCPU(Context const* ctx, tree::TrainParam const& param,
                                              GradientContainer const& in) {
  return std::make_unique<CPURowSampler>(ctx, param, in);
}

auto const kSampleRowsCPU =
    common::KernelRegistration<MakeRowSamplerKernel>{DeviceOrd::kCPU, MakeRowSamplerCPU};
}  // namespace
}  // namespace xgboost::gbm
