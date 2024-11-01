/**
 * Copyright 2024, XGBoost Contributors
 */
#include <thrust/equal.h>  // for equal

#include "../common/device_helpers.cuh"  // for tcbegin
#include "../common/error_msg.h"         // for InconsistentFeatureTypes
#include "validation.h"

namespace xgboost::data::cuda_impl {
void CheckFeatureTypes(HostDeviceVector<FeatureType> const& lhs,
                       HostDeviceVector<FeatureType> const& rhs) {
  auto device = lhs.DeviceCanRead() ? lhs.Device() : rhs.Device();
  CHECK(device.IsCUDA());
  lhs.SetDevice(device), rhs.SetDevice(device);
  auto const& d_lhs = lhs.ConstDeviceSpan();
  auto const& d_rhs = rhs.ConstDeviceSpan();
  auto ft_is_same = thrust::equal(dh::tcbegin(d_lhs), dh::tcend(d_lhs), dh::tcbegin(d_rhs));
  CHECK(ft_is_same) << error::InconsistentFeatureTypes();
}
}  // namespace xgboost::data::cuda_impl
