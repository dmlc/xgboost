/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include <thrust/execution_policy.h>
#include <thrust/functional.h>  // thrust:plus

#include "algorithm.h"
#include "device_helpers.cuh"            // dh::Reduce
#include "xgboost/generic_parameters.h"  // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector

namespace xgboost {
namespace common {
namespace cuda {
double Reduce(Context const* ctx, HostDeviceVector<float> const& values) {
  auto const d_values = values.ConstDeviceSpan();
  dh::XGBCachingDeviceAllocator<char> alloc;
  auto res = dh::Reduce(thrust::cuda::par(alloc), d_values.data(),
                        d_values.data() + d_values.size(), 0.0, thrust::plus<double>{});
  return res;
}
}  // namespace cuda
}  // namespace common
}  // namespace xgboost
