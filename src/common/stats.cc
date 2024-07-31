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

namespace xgboost::common {
void Median(Context const* ctx, linalg::Tensor<float, 2> const& t,
            HostDeviceVector<float> const& weights, linalg::Tensor<float, 1>* out) {
  if (ctx->IsCUDA()) {
    weights.SetDevice(ctx->Device());
    auto opt_weights = OptionalWeights(weights.ConstDeviceSpan());
    auto t_v = t.View(ctx->Device());
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
  v.SetDevice(ctx->Device());
  out->SetDevice(ctx->Device());
  out->Reshape(1);

  if (ctx->IsCUDA()) {
    cuda_impl::Mean(ctx, v.View(ctx->Device()), out->View(ctx->Device()));
  } else {
    auto h_v = v.HostView();
    float n = v.Size();
    MemStackAllocator<float, DefaultMaxThreads()> tloc(ctx->Threads(), 0.0f);
    ParallelFor(v.Size(), ctx->Threads(),
                [&](auto i) { tloc[omp_get_thread_num()] += h_v(i) / n; });
    auto ret = std::accumulate(tloc.cbegin(), tloc.cend(), .0f);
    out->HostView()(0) = ret;
  }
}

void SampleMean(Context const* ctx, linalg::Matrix<float> const& v, linalg::Vector<float>* out) {
  *out = linalg::Zeros<float>(ctx, v.Shape(1));
  if (ctx->IsCPU()) {
    auto h_v = v.HostView();
    CHECK(h_v.CContiguous());
    auto n_rows_f32 = static_cast<float>(v.Shape(0));
    auto n_columns = v.Shape(1);
    auto h_out = out->HostView();
    for (std::size_t j = 0; j < n_columns; ++j) {
      MemStackAllocator<float, DefaultMaxThreads()> mean_tloc(ctx->Threads(), 0.0f);
      ParallelFor(v.Shape(0), ctx->Threads(),
                  [&](auto i) { mean_tloc[omp_get_thread_num()] += (h_v(i, j) / n_rows_f32); });
      auto mean = std::accumulate(mean_tloc.cbegin(), mean_tloc.cend(), 0.0f);
      h_out(j) = mean;
    }
  } else {
    auto d_v = v.View(ctx->Device());
    auto d_out = out->View(ctx->Device());
    cuda_impl::SampleMean(ctx, d_v, d_out);
  }
}

void WeightedMean(Context const* ctx,
                  const std::vector<float> &v,
                  const std::vector<float> &w,
                  linalg::Vector<float>* out) {
  out->SetDevice(ctx->Device());
  out->Reshape(1);

  MemStackAllocator<float, DefaultMaxThreads()> tloc_w(ctx->Threads(), 0.0f);
  ParallelFor(w.size(), ctx->Threads(),
              [&](auto i) { tloc_w[omp_get_thread_num()] += w[i]; });
  auto sumw = std::accumulate(tloc_w.cbegin(), tloc_w.cend(), .0f);

  MemStackAllocator<float, DefaultMaxThreads()> tloc_v(ctx->Threads(), 0.0f);
  ParallelFor(v.size(), ctx->Threads(),
              [&](auto i) { tloc_v[omp_get_thread_num()] += v[i] * w[i] / sumw; });
  auto ret = std::accumulate(tloc_v.cbegin(), tloc_v.cend(), .0f);
  out->HostView()(0) = ret;
}
}  // namespace xgboost::common
