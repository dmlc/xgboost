/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include <thrust/execution_policy.h>
#include <thrust/functional.h>  // thrust:plus

#include "device_helpers.cuh"  // dh::Reduce, safe_cuda, dh::XGBCachingDeviceAllocator
#include "numeric.cuh"
#include "numeric.h"
#include "xgboost/generic_parameters.h"  // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector

namespace xgboost {
namespace common {
namespace cuda_impl {
double Reduce(Context const* ctx, HostDeviceVector<float> const& values) {
  values.SetDevice(ctx->gpu_id);
  auto const d_values = values.ConstDeviceSpan();
  return Reduce(ctx, d_values.cbegin(), d_values.cend(), 0.0);
}
}  // namespace cuda_impl
}  // namespace common
}  // namespace xgboost
