/*!
 * Copyright 2022 by XGBoost Contributors
 */

#include <thrust/iterator/counting_iterator.h>  // thrust::make_counting_iterator

#include "common.h"            // common::OptionalWeights
#include "device_helpers.cuh"  // dh::MakeTransformIterator, tcbegin, tcend
#include "stats.cuh"           // common::SegmentedQuantile, common::SegmentedWeightedQuantile
#include "xgboost/context.h"   // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/linalg.h"              // linalg::TensorView, UnravelIndex, Apply

namespace xgboost {
namespace common {
namespace cuda_impl {
float Median(Context const* ctx, linalg::TensorView<float const, 2> t,
             common::OptionalWeights weights) {
  HostDeviceVector<size_t> segments{0, t.Size()};
  segments.SetDevice(ctx->gpu_id);
  auto d_segments = segments.ConstDeviceSpan();
  auto val_it = dh::MakeTransformIterator<float>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) {
        return linalg::detail::Apply(t, linalg::UnravelIndex(i, t.Shape()));
      });

  HostDeviceVector<float> quantile{0};
  quantile.SetDevice(ctx->gpu_id);
  if (weights.Empty()) {
    common::SegmentedQuantile(ctx, 0.5, dh::tcbegin(d_segments), dh::tcend(d_segments), val_it,
                              val_it + t.Size(), &quantile);
  } else {
    CHECK_NE(t.Shape(1), 0);
    auto w_it = dh::MakeTransformIterator<float>(thrust::make_counting_iterator(0ul),
                                                 [=] XGBOOST_DEVICE(size_t i) {
                                                   auto sample_idx = i / t.Shape(1);
                                                   return weights[sample_idx];
                                                 });
    common::SegmentedWeightedQuantile(ctx, 0.5, dh::tcbegin(d_segments), dh::tcend(d_segments),
                                      val_it, val_it + t.Size(), w_it, w_it + t.Size(), &quantile);
  }
  CHECK_EQ(quantile.Size(), 1);
  return quantile.HostVector().front();
}

void Mean(Context const* ctx, linalg::VectorView<float const> v, linalg::VectorView<float> out) {
  float n = v.Size();
  auto it = dh::MakeTransformIterator<float>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t i) { return v(i) / n; });
  std::size_t bytes;
  CHECK_EQ(out.Size(), 1);
  cub::DeviceReduce::Sum(nullptr, bytes, it, out.Values().data(), v.Size());
  dh::TemporaryArray<char> temp{bytes};
  cub::DeviceReduce::Sum(temp.data().get(), bytes, it, out.Values().data(), v.Size());
}
}  // namespace cuda_impl
}  // namespace common
}  // namespace xgboost
