/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include <thrust/execution_policy.h>
#include <thrust/functional.h>  // thrust:plus

#include "device_helpers.cuh"  // dh::Reduce
#include "numeric.h"
#include "xgboost/generic_parameters.h"  // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/logging.h"             // CHECK_GE

namespace xgboost {
namespace common {
namespace cuda {
double Reduce(Context const* ctx, HostDeviceVector<float> const& values) {
  CHECK_GE(ctx->gpu_id, 0);
  dh::safe_cuda(cudaSetDevice(ctx->gpu_id));
  auto const d_values = values.ConstDeviceSpan();
  dh::XGBCachingDeviceAllocator<char> alloc;
  auto res = dh::Reduce(thrust::cuda::par(alloc), d_values.data(),
                        d_values.data() + d_values.size(), 0.0, thrust::plus<double>{});
  return res;
}
}  // namespace cuda
}  // namespace common
}  // namespace xgboost
