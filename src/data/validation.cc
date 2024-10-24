/**
 * Copyright 2024, XGBoost Contributors
 */
#include "validation.h"

#include "../common/error_msg.h"  // for InconsistentFeatureTypes

#if !defined(XGBOOST_USE_CUDA)

#include "../common/common.h"  // for AssertGPUSupport

#endif  // !defined(XGBOOST_USE_CUDA)

namespace xgboost::data {
void CheckFeatureTypes(HostDeviceVector<FeatureType> const& lhs,
                       HostDeviceVector<FeatureType> const& rhs) {
  CHECK_EQ(lhs.Size(), rhs.Size()) << error::InconsistentFeatureTypes();
  if (lhs.DeviceCanRead() || rhs.DeviceCanRead()) {
    return cuda_impl::CheckFeatureTypes(lhs, rhs);
  }
  auto const& h_lhs = lhs.ConstHostVector();
  auto const& h_rhs = rhs.ConstHostVector();
  auto ft_is_same = std::equal(h_lhs.cbegin(), h_lhs.cend(), h_rhs.cbegin());
  CHECK(ft_is_same) << error::InconsistentFeatureTypes();
}

#if !defined(XGBOOST_USE_CUDA)
namespace cuda_impl {
void CheckFeatureTypes(HostDeviceVector<FeatureType> const&, HostDeviceVector<FeatureType> const&) {
  common::AssertGPUSupport();
}
}  // namespace cuda_impl
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::data
