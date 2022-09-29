/*!
 * Copyright 2022 by XGBoost Contributors
 */

#include <thrust/iterator/counting_iterator.h>  // thrust::make_counting_iterator

#include "common.h"            // common::OptionalWeights
#include "device_helpers.cuh"  // dh::MakeTransformIterator, tcbegin, tcend
#include "stats.cuh"           // common::SegmentedQuantile, common::SegmentedWeightedQuantile
#include "xgboost/generic_parameters.h"  // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/linalg.h"              // linalg::TensorView, UnravelIndex, Apply

namespace xgboost {
namespace common {
namespace cuda {
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
}  // namespace cuda
}  // namespace common
}  // namespace xgboost
