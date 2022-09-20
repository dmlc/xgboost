#include "stats.h"

#include <numeric>  // std::accumulate

#include "common.h"                      // OptionalWeights, MakeIndexTransformIter
#include "threading_utils.h"             // ParallelFor, MemStackAllocator
#include "xgboost/generic_parameters.h"  // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/linalg.h"              // Tensor, UnravelIndex, Apply
#include "xgboost/logging.h"             // CHECK_EQ

namespace xgboost {
namespace common {

float Median(Context const* ctx, linalg::Tensor<float, 2> const& t,
             HostDeviceVector<float> const& weights) {
  CHECK_EQ(t.Shape(1), 0) << "Matrix is not yet supported.";
  if (!ctx->IsCPU()) {
    weights.SetDevice(ctx->gpu_id);
    auto opt_weights = OptionalWeights(weights.ConstDeviceSpan());
    auto t_v = t.View(ctx->gpu_id);
    return cuda::Median(ctx, t_v, opt_weights);
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

float Mean(Context const* ctx, linalg::Tensor<float, 2> const& t,
           HostDeviceVector<float> const& weights) {
  if (!weights.Empty()) {
    CHECK_EQ(weights.Size(), t.Shape(0)) << "Weight is assigned for each row.";
  }
  if (!ctx->IsCPU()) {
    weights.SetDevice(ctx->gpu_id);
    auto opt_weights = OptionalWeights(weights.ConstDeviceSpan());
    auto t_v = t.View(ctx->gpu_id);
    cuda::Mean(ctx, t_v, opt_weights);
  }

  auto opt_weights = OptionalWeights(weights.ConstHostSpan());
  auto t_v = t.HostView();

  MemStackAllocator<float, 128> mean_tloc(ctx->Threads(), 0.0f);
  auto iter = common::MakeIndexTransformIter(
      [&](size_t i) { return linalg::detail::Apply(t_v, linalg::UnravelIndex(i, t_v.Shape())); });

  double size = t_v.Shape(0);
  CHECK_NE(size, 0);
  ParallelFor(t_v.Size(), ctx->Threads(), [&](auto i) {
    auto tidx = omp_get_thread_num();
    auto ridx = std::get<0>(linalg::UnravelIndex(i, t_v.Shape()));
    mean_tloc[tidx] += iter[i] * opt_weights[ridx] / size;
  });
  auto mean = std::accumulate(mean_tloc.cbegin(), mean_tloc.cend(), 0.0f);
  return mean;
}
}  // namespace common
}  // namespace xgboost
