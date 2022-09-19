/*!
 * Copyright 2022 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_ALGORITHM_H_
#define XGBOOST_COMMON_ALGORITHM_H_
#include <algorithm>  // std::upper_bound
#include <cinttypes>  // std::size_t

#include "xgboost/generic_parameters.h"  // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector

namespace xgboost {
namespace common {
template <typename It, typename Idx>
auto SegmentId(It first, It last, Idx idx) {
  std::size_t segment_id = std::upper_bound(first, last, idx) - 1 - first;
  return segment_id;
}

namespace cuda {
double Reduce(Context const* ctx, HostDeviceVector<float> const& values);
}

/**
 * \brief Reduction with summation.
 */
double Reduce(Context const* ctx, HostDeviceVector<float> const& values);
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_ALGORITHM_H_
