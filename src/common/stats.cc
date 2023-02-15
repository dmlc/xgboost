/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#include "stats.h"

#include <cstddef>                       // std::size_t
#include <numeric>                       // std::accumulate

#include "common.h"                      // OptionalWeights
#include "linalg_op.h"
#include "threading_utils.h"             // ParallelFor, MemStackAllocator
#include "transform_iterator.h"          // MakeIndexTransformIter
#include "xgboost/context.h"             // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/linalg.h"              // Tensor, UnravelIndex, Apply
#include "xgboost/logging.h"             // CHECK_EQ

namespace xgboost {
namespace common {
void Median(Context const* ctx, linalg::Tensor<float, 2> const& t,
            HostDeviceVector<float> const& weights, linalg::Tensor<float, 1>* out) {
  if (!ctx->IsCPU()) {
    weights.SetDevice(ctx->gpu_id);
    auto opt_weights = OptionalWeights(weights.ConstDeviceSpan());
    auto t_v = t.View(ctx->gpu_id);
    cuda_impl::Median(ctx, t_v, opt_weights, out);
  }

  auto opt_weights = OptionalWeights(weights.ConstHostSpan());
  auto t_v = t.HostView();
  out->Reshape(t.Shape(1));
  auto h_out = out->HostView();
  for (std::size_t i{0}; i < t.Shape(1); ++i) {
    auto ti_v = t_v.Slice(linalg::All(), i);
    auto iter = linalg::cbegin(ti_v);
    float q{0};
    if (opt_weights.Empty()) {
      q = common::Quantile(ctx, 0.5, iter, iter + ti_v.Size());
    } else {
      CHECK_NE(t_v.Shape(1), 0);
      auto w_it = common::MakeIndexTransformIter([&](std::size_t i) { return opt_weights[i]; });
      q = common::WeightedQuantile(ctx, 0.5, iter, iter + ti_v.Size(), w_it);
    }
    h_out(i) = q;
  }
}

void Mean(Context const* ctx, linalg::Vector<float> const& v, linalg::Vector<float>* out) {
  v.SetDevice(ctx->gpu_id);
  out->SetDevice(ctx->gpu_id);
  out->Reshape(1);

  if (ctx->IsCPU()) {
    auto h_v = v.HostView();
    float n = v.Size();
    MemStackAllocator<float, DefaultMaxThreads()> tloc(ctx->Threads(), 0.0f);
    ParallelFor(v.Size(), ctx->Threads(),
                [&](auto i) { tloc[omp_get_thread_num()] += h_v(i) / n; });
    auto ret = std::accumulate(tloc.cbegin(), tloc.cend(), .0f);
    out->HostView()(0) = ret;
  } else {
    cuda_impl::Mean(ctx, v.View(ctx->gpu_id), out->View(ctx->gpu_id));
  }
}
}  // namespace common
}  // namespace xgboost
