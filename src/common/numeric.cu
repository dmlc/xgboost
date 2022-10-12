/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include <thrust/execution_policy.h>

#include "numeric.cuh"         // Reduce
#include "numeric.h"
#include "device_helpers.cuh"
#include "xgboost/generic_parameters.h"  // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector

namespace xgboost {
namespace common {
namespace cuda_impl {
double Reduce(Context const* ctx, HostDeviceVector<float> const& values) {
  values.SetDevice(ctx->gpu_id);
  auto const d_values = values.ConstDeviceSpan();
  return Reduce(ctx, dh::tcbegin(d_values), dh::tcend(d_values), 0.0);
}
}  // namespace cuda_impl
}  // namespace common
}  // namespace xgboost
