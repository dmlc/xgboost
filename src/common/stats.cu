/**
 * Copyright 2022-2023 by XGBoost Contributors
 */

#include <thrust/iterator/counting_iterator.h>  // thrust::make_counting_iterator

#include <cstddef>                              // size_t

#include "cuda_context.cuh"                     // CUDAContext
#include "device_helpers.cuh"                   // dh::MakeTransformIterator, tcbegin, tcend
#include "optional_weight.h"                    // common::OptionalWeights
#include "stats.cuh"          // common::SegmentedQuantile, common::SegmentedWeightedQuantile
#include "xgboost/base.h"     // XGBOOST_DEVICE
#include "xgboost/context.h"  // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/linalg.h"              // linalg::TensorView, UnravelIndex, Apply

namespace xgboost {
namespace common {
namespace cuda_impl {
void Median(Context const* ctx, linalg::TensorView<float const, 2> t,
            common::OptionalWeights weights, linalg::Tensor<float, 1>* out) {
  CHECK_GE(t.Shape(1), 1);
  HostDeviceVector<std::size_t> segments(t.Shape(1) + 1, 0);
  segments.SetDevice(ctx->gpu_id);
  auto d_segments = segments.DeviceSpan();
  dh::LaunchN(d_segments.size(), ctx->CUDACtx()->Stream(),
              [=] XGBOOST_DEVICE(std::size_t i) { d_segments[i] = t.Shape(0) * i; });
  auto val_it = dh::MakeTransformIterator<float>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) {
        return linalg::detail::Apply(t, linalg::UnravelIndex(i, t.Shape()));
      });

  out->SetDevice(ctx->gpu_id);
  out->Reshape(t.Shape(1));
  if (weights.Empty()) {
    common::SegmentedQuantile(ctx, 0.5, dh::tcbegin(d_segments), dh::tcend(d_segments), val_it,
                              val_it + t.Size(), out->Data());
  } else {
    CHECK_NE(t.Shape(1), 0);
    auto w_it = dh::MakeTransformIterator<float>(thrust::make_counting_iterator(0ul),
                                                 [=] XGBOOST_DEVICE(std::size_t i) {
                                                   auto sample_idx = i / t.Shape(1);
                                                   return weights[sample_idx];
                                                 });
    common::SegmentedWeightedQuantile(ctx, 0.5, dh::tcbegin(d_segments), dh::tcend(d_segments),
                                      val_it, val_it + t.Size(), w_it, w_it + t.Size(),
                                      out->Data());
  }
}

void Mean(Context const* ctx, linalg::VectorView<float const> v, linalg::VectorView<float> out) {
  float n = v.Size();
  auto it = dh::MakeTransformIterator<float>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t i) { return v(i) / n; });
  std::size_t bytes;
  CHECK_EQ(out.Size(), 1);
  auto s = ctx->CUDACtx()->Stream();
  cub::DeviceReduce::Sum(nullptr, bytes, it, out.Values().data(), v.Size(), s);
  dh::TemporaryArray<char> temp{bytes};
  cub::DeviceReduce::Sum(temp.data().get(), bytes, it, out.Values().data(), v.Size(), s);
}
}  // namespace cuda_impl
}  // namespace common
}  // namespace xgboost
