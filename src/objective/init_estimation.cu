/**
 * Copyright 2022 by XGBoost Contributors
 *
 * \brief Utilities for estimating initial score.
 */
#if !defined(NOMINMAX) && defined(_WIN32)
#define NOMINMAX
#endif                                          // !defined(NOMINMAX)
#include <thrust/execution_policy.h>            // cuda::par
#include <thrust/iterator/counting_iterator.h>  // thrust::make_counting_iterator

#include <algorithm>  // std::max
#include <cstddef>    // std::size_t

#include "../collective/communicator-inl.h"  // Allreduce
#include "../common/device_helpers.cuh"      // dh::MakeTransformIterator, dh::Reduce
#include "init_estimation.h"
#include "xgboost/base.h"                // GradientPairPrecise, GradientPair, XGBOOST_DEVICE
#include "xgboost/generic_parameters.h"  // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector

namespace xgboost {
namespace obj {
namespace cuda_impl {
double FitStump(Context const* ctx, HostDeviceVector<GradientPair> const& gpair) {
  gpair.SetDevice(ctx->gpu_id);
  auto d_gpair = gpair.ConstDeviceSpan();
  auto it = dh::MakeTransformIterator<GradientPairPrecise>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(std::size_t i) -> GradientPairPrecise {
        return GradientPairPrecise{d_gpair[i]};
      });
  dh::XGBCachingDeviceAllocator<char> alloc;
  auto sum = dh::Reduce(thrust::cuda::par(alloc), it, it + d_gpair.size(), GradientPairPrecise{},
                        thrust::plus<GradientPairPrecise>{});
  static_assert(sizeof(sum) == sizeof(double) * 2, "");
  collective::Allreduce<collective::Operation::kSum>(reinterpret_cast<double*>(&sum), 2);
  return -sum.GetGrad() / std::max(sum.GetHess(), 1e-6);
}
}  // namespace cuda_impl
}  // namespace obj
}  // namespace xgboost
