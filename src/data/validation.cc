/**
 * Copyright 2024, XGBoost Contributors
 */
#include "validation.h"

#include "../common/error_msg.h"  // for InconsistentFeatureTypes

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
}  // namespace xgboost::data
