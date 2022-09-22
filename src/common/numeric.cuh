/*!
 * Copyright 2022 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_NUMERIC_CUH_
#define XGBOOST_COMMON_NUMERIC_CUH_
#include <xgboost/generic_parameters.h>  // Context

#include "device_helpers.cuh"  // Reduce
#include "numeric.h"

namespace xgboost {
namespace common {
namespace cuda_impl {
template <typename It, typename V = typename It::value_type>
V Reduce(Context const* /*ctx unused*/, It first, It second, V const& init) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  auto res = dh::Reduce(thrust::cuda::par(alloc), first, second, init, thrust::plus<V>{});
  return res;
}
}  // namespace cuda_impl
}  // namespace common
}  // namespace xgboost
#endif
