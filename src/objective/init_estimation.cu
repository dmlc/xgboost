/**
 * Copyright 2022 by XGBoost Contributors
 *
 * \brief Utilities for estimating initial score.
 */
#if !defined(NOMINMAX) && defined(_WIN32)
#define NOMINMAX
#endif                                          // !defined(NOMINMAX)
#include <thrust/iterator/counting_iterator.h>  // thrust::make_counting_iterator

#include <algorithm>  // std::max
#include <cinttypes>  // std::uint64_t
#include <cstddef>    // std::size_t

#include "../common/device_helpers.cuh"  // dh::MakeTransformIterator
#include "../common/numeric.cuh"         // Reduce
#include "init_estimation.h"
#include "rabit/rabit.h"
#include "xgboost/generic_parameters.h"  // Context

namespace xgboost {
namespace obj {
namespace cuda_impl {
double FitStump(Context const* ctx, HostDeviceVector<GradientPair> const& gpair) {
  gpair.SetDevice(ctx->gpu_id);
  auto const& d_gpair = gpair.ConstDeviceSpan();
  auto it = dh::MakeTransformIterator<GradientPairPrecise>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(std::size_t i) -> GradientPairPrecise {
        return GradientPairPrecise{d_gpair[i]};
      });
  auto sum = common::cuda_impl::Reduce(ctx, it, it + d_gpair.size(), GradientPairPrecise{});
  return -sum.GetGrad() / std::max(sum.GetHess(), 1e-6);
}
}  // namespace cuda_impl
}  // namespace obj
}  // namespace xgboost
