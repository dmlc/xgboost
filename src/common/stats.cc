/**
 * Copyright 2022-2026, XGBoost Contributors
 */
#include "stats.h"

#include <cstddef>  // std::size_t
#include <numeric>  // std::accumulate

#include "../collective/aggregator.h"    // for GlobalSum
#include "linalg_op.h"                   // for Matrix
#include "numeric.h"                     // for TransformReduce
#include "optional_weight.h"             // OptionalWeights
#include "transform_iterator.h"          // MakeIndexTransformIter
#include "xgboost/context.h"             // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/linalg.h"              // Tensor, UnravelIndex, Apply
#include "xgboost/logging.h"             // CHECK_EQ

namespace xgboost::common {
void Median(Context const* ctx, linalg::Matrix<float> const& t,
            HostDeviceVector<float> const& weights, linalg::Tensor<float, 1>* out) {
  if (ctx->IsCUDA()) {
    weights.SetDevice(ctx->Device());
    auto opt_weights = OptionalWeights(weights.ConstDeviceSpan());
    auto t_v = t.View(ctx->Device());
    cuda_impl::Median(ctx, t_v, opt_weights, out);
    return;
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

void Mean(Context const* ctx, linalg::VectorView<float const> v, linalg::Vector<float>* out) {
  out->SetDevice(ctx->Device());
  out->Reshape(1);

  if (ctx->IsCUDA()) {
    cuda_impl::Mean(ctx, v, out->View(ctx->Device()));
  } else if (ctx->IsSycl()) {
    sycl_impl::Mean(ctx, v, out->View(ctx->Device()));
  } else {
    auto h_v = v;
    float n = v.Size();
    out->HostView()(0) =
        TransformReduce(v.Size(), ctx->Threads(), 0.0f, [&](std::size_t i) { return h_v(i) / n; });
  }
}

void SampleMean(Context const* ctx, linalg::Matrix<float> const& v, linalg::Vector<float>* out) {
  *out = linalg::Zeros<float>(ctx, std::max(v.Shape(1), decltype(v.Shape(1)){1}));
  if (!ctx->IsCUDA()) {
    auto h_v = v.HostView();
    CHECK(h_v.CContiguous());
    std::int64_t n_samples = v.Shape(0);
    SafeColl(collective::GlobalSum(ctx, linalg::MakeVec(&n_samples, 1)));
    auto n_columns = v.Shape(1);
    auto h_out = out->HostView();

    auto n_rows_f64 = static_cast<double>(n_samples);
    for (std::size_t j = 0; j < n_columns; ++j) {
      h_out(j) = TransformReduce(v.Shape(0), ctx->Threads(), 0.0,
                                 [&](std::size_t i) { return h_v(i, j) / n_rows_f64; });
    }
    SafeColl(collective::GlobalSum(ctx, h_out));
  } else {
    auto d_v = v.View(ctx->Device());
    auto d_out = out->View(ctx->Device());
    cuda_impl::SampleMean(ctx, d_v, d_out);
  }
}

void WeightedSampleMean(Context const* ctx, linalg::Matrix<float> const& v,
                        HostDeviceVector<float> const& w, linalg::Vector<float>* out) {
  *out = linalg::Zeros<float>(ctx, std::max(v.Shape(1), decltype(v.Shape(1)){1}));
  CHECK_EQ(v.Shape(0), w.Size());
  if (!ctx->IsCUDA()) {
    auto h_v = v.HostView();
    auto h_w = w.ConstHostSpan();
    auto sum_w = std::accumulate(h_w.data(), h_w.data() + h_w.size(), 0.0);
    SafeColl(collective::GlobalSum(ctx, linalg::MakeVec(&sum_w, 1)));
    CHECK_GT(sum_w, 0.0) << "weights must contain at least one non-zero value.";
    auto h_out = out->HostView();
    for (std::size_t j = 0; j < v.Shape(1); ++j) {
      h_out(j) = TransformReduce(v.Shape(0), ctx->Threads(), 0.0,
                                 [&](std::size_t i) { return h_v(i, j) / sum_w * h_w(i); });
    }
    SafeColl(collective::GlobalSum(ctx, h_out));
  } else {
    auto d_v = v.View(ctx->Device());
    w.SetDevice(ctx->Device());
    auto d_w = w.ConstDeviceSpan();
    auto d_out = out->View(ctx->Device());
    cuda_impl::WeightedSampleMean(ctx, d_v, d_w, d_out);
  }
}
}  // namespace xgboost::common
