#include "stats.h"

#include <numeric>  // std::accumulate

#include "common.h"                      // OptionalWeights
#include "threading_utils.h"             // ParallelFor, MemStackAllocator
#include "transform_iterator.h"          // MakeIndexTransformIter
#include "xgboost/generic_parameters.h"  // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/linalg.h"              // Tensor, UnravelIndex, Apply
#include "xgboost/logging.h"             // CHECK_EQ

namespace xgboost {
namespace common {
float Median(Context const* ctx, linalg::Tensor<float, 2> const& t,
             HostDeviceVector<float> const& weights) {
  CHECK_LE(t.Shape(1), 1) << "Matrix is not yet supported.";
  if (!ctx->IsCPU()) {
    weights.SetDevice(ctx->gpu_id);
    auto opt_weights = OptionalWeights(weights.ConstDeviceSpan());
    auto t_v = t.View(ctx->gpu_id);
    return cuda_impl::Median(ctx, t_v, opt_weights);
  }

  auto opt_weights = OptionalWeights(weights.ConstHostSpan());
  auto t_v = t.HostView();
  auto iter = common::MakeIndexTransformIter(
      [&](size_t i) { return linalg::detail::Apply(t_v, linalg::UnravelIndex(i, t_v.Shape())); });
  float q{0};
  if (opt_weights.Empty()) {
    q = common::Quantile(0.5, iter, iter + t_v.Size());
  } else {
    CHECK_NE(t_v.Shape(1), 0);
    auto w_it = common::MakeIndexTransformIter([&](size_t i) {
      auto sample_idx = i / t_v.Shape(1);
      return opt_weights[sample_idx];
    });
    q = common::WeightedQuantile(0.5, iter, iter + t_v.Size(), w_it);
  }
  return q;
}
}  // namespace common
}  // namespace xgboost
