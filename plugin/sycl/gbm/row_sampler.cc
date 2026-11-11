/**
 * Copyright 2026, XGBoost Contributors
 */
#include "../../../src/gbm/row_sampler.h"

#include <cstddef>  // for size_t
#include <cstdint>  // for int64_t
#include <limits>   // for numeric_limits
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/random>
#include <sycl/sycl.hpp>

#include "../../../src/common/kernel.h"
#include "../../../src/tree/hist/sampler.h"
#include "../data.h"
#include "../device_manager.h"

namespace xgboost::sycl {
namespace {
class SYCLRowSampler : public gbm::RowSampler {
  GradientContainer const& in_;
  float subsample_;
  bool gradient_based_;
  std::size_t sample_rows_;
  USMVector<float, MemoryType::on_device> regularized_;
  USMVector<float, MemoryType::on_device> threshold_;

 public:
  SYCLRowSampler(Context const* ctx, xgboost::tree::TrainParam const& param,
                 GradientContainer const& in)
      : in_{in},
        subsample_{param.subsample},
        gradient_based_{param.sampling_method == xgboost::tree::TrainParam::kGradientBased},
        sample_rows_{static_cast<std::size_t>(in.gpair.Shape(0) * param.subsample)} {
    DeviceManager manager;
    auto* queue = manager.GetQueue(ctx->Device());
    auto split = in.gpair.View(ctx->Device());
    auto n_rows = split.Shape(0);
    auto n_targets = split.Shape(1);
    if (n_rows == 0) {
      return;
    }
    auto sample_rows = sample_rows_;
    bool gradient_based = param.sampling_method == xgboost::tree::TrainParam::kGradientBased;
    auto needs_threshold = gradient_based && sample_rows != 0 && sample_rows < n_rows;
    auto workspace_rows = needs_threshold ? n_rows : 0;
    regularized_.ResizeNoCopy(queue, workspace_rows);
    USMVector<float, MemoryType::on_device> sorted{queue, workspace_rows};
    USMVector<float, MemoryType::on_device> sums{queue, workspace_rows};
    threshold_.ResizeNoCopy(queue, needs_threshold ? 1ul : 0ul);
    auto* rag = regularized_.Data();
    auto* sorted_ptr = sorted.Data();
    auto* prefix = sums.Data();
    auto* cutoff = threshold_.Data();
    if (needs_threshold) {
      auto grad_op = xgboost::tree::MvsGradOp{xgboost::tree::kDefaultMvsLambda};
      queue
          ->parallel_for(::sycl::range<1>{n_rows},
                         [=](::sycl::id<1> id) {
                           auto i = id[0];
                           float sum = 0.0f;
                           for (std::size_t t = 0; t < n_targets; ++t) {
                             sum += grad_op(split(i, t));
                           }
                           sorted_ptr[i] = rag[i] = ::sycl::sqrt(sum);
                         })
          .wait_and_throw();
      auto policy = oneapi::dpl::execution::make_device_policy(*queue);
      oneapi::dpl::sort(policy, sorted_ptr, sorted_ptr + n_rows);
      oneapi::dpl::inclusive_scan(policy, sorted_ptr, sorted_ptr + n_rows, prefix);
      queue->wait_and_throw();
      queue
          ->single_task([=] {
            std::int64_t lo = 0, hi = n_rows - 1;
            float u = prefix[n_rows - 1] / sample_rows;
            while (lo <= hi) {
              auto i = lo + (hi - lo) / 2;
              auto above = n_rows - i - 1;
              auto denom = static_cast<float>(sample_rows) - static_cast<float>(above);
              if (denom <= 0.0f) {
                lo = i + 1;
                continue;
              }
              auto candidate = prefix[i] / denom;
              auto upper =
                  i + 1 < n_rows ? sorted_ptr[i + 1] : std::numeric_limits<float>::infinity();
              if (candidate > sorted_ptr[i] && candidate <= upper) {
                u = candidate;
                break;
              }
              if (candidate <= sorted_ptr[i]) {
                hi = i - 1;
              } else {
                lo = i + 1;
              }
            }
            cutoff[0] = u;
          })
          .wait_and_throw();
    }
  }

  void Sample(Context const* ctx, GradientContainer* out) override {
    DeviceManager manager;
    auto* queue = manager.GetQueue(ctx->Device());
    auto split = in_.gpair.View(ctx->Device());
    out->gpair.SetDevice(ctx->Device());
    out->gpair.Reshape(split.Shape());
    auto sampled = out->gpair.View(ctx->Device());
    auto n_rows = split.Shape(0);
    auto n_targets = split.Shape(1);
    bool has_value = in_.HasValueGrad();
    if (has_value) {
      out->value_gpair.SetDevice(ctx->Device());
      out->value_gpair.Reshape(in_.value_gpair.Shape());
    } else {
      out->value_gpair = {};
    }
    auto value = in_.value_gpair.View(ctx->Device());
    auto sampled_value = out->value_gpair.View(ctx->Device());
    if (n_rows == 0) {
      return;
    }

    auto sample_rows = sample_rows_;
    auto gradient_based = gradient_based_;
    auto* rag = regularized_.Data();
    auto* cutoff = threshold_.Data();
    auto seed = ctx->Rng()();
    auto subsample = subsample_;
    queue
        ->parallel_for(::sycl::range<1>{n_rows},
                       [=](::sycl::id<1> id) {
                         auto i = id[0];
                         auto split_out = sampled;
                         auto value_out = sampled_value;
                         float p = subsample;
                         if (gradient_based) {
                           p = sample_rows == 0        ? 0.0f
                               : sample_rows >= n_rows ? 1.0f
                                                       : rag[i] / ::sycl::fmax(cutoff[0], kRtEps);
                         }
                         oneapi::dpl::minstd_rand engine(seed, i);
                         oneapi::dpl::uniform_real_distribution<float> dist;
                         bool selected = p >= 1.0f || (p > 0.0f && dist(engine) <= p);
                         float weight =
                             gradient_based && selected ? 1.0f / ::sycl::fmin(p, 1.0f) : 1.0f;
                         for (std::size_t t = 0; t < n_targets; ++t) {
                           split_out(i, t) = selected ? split(i, t) * weight : GradientPair{};
                         }
                         if (has_value) {
                           for (std::size_t t = 0; t < value.Shape(1); ++t) {
                             value_out(i, t) = selected ? value(i, t) * weight : GradientPair{};
                           }
                         }
                       })
        .wait_and_throw();
  }
};

std::unique_ptr<gbm::RowSampler> MakeRowSamplerSYCL(Context const* ctx,
                                                    xgboost::tree::TrainParam const& param,
                                                    GradientContainer const& in) {
  return std::make_unique<SYCLRowSampler>(ctx, param, in);
}

auto const kSampleRows = xgboost::common::KernelRegistration<gbm::MakeRowSamplerKernel>{
    {DeviceOrd::kSyclDefault, DeviceOrd::kSyclCPU, DeviceOrd::kSyclGPU}, MakeRowSamplerSYCL};
}  // namespace
}  // namespace xgboost::sycl
