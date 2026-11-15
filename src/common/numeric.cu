/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include <thrust/execution_policy.h>

#include <cuda/std/functional>  // for plus

#include "device_helpers.cuh"  // dh::Reduce, dh::XGBCachingDeviceAllocator
#include "numeric.h"
#include "xgboost/context.h"             // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector

namespace xgboost::common::cuda_impl {
double Reduce(Context const* ctx, HostDeviceVector<float> const& values) {
  values.SetDevice(ctx->Device());
  auto const d_values = values.ConstDeviceSpan();
  dh::XGBCachingDeviceAllocator<char> alloc;
  return dh::Reduce(thrust::cuda::par(alloc), dh::tcbegin(d_values), dh::tcend(d_values), 0.0,
                    cuda::std::plus<float>{});
}
}  // namespace xgboost::common::cuda_impl
