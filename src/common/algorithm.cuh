/*!
 * Copyright 2022 by XGBoost Contributors
 */
#pragma once

#include <thrust/binary_search.h>     // thrust::upper_bound
#include <thrust/execution_policy.h>  // thrust::seq

#include "xgboost/base.h"
#include "xgboost/span.h"

namespace xgboost {
namespace common {
namespace cuda {
template <typename It>
size_t XGBOOST_DEVICE SegmentId(It first, It last, size_t idx) {
  size_t segment_id = thrust::upper_bound(thrust::seq, first, last, idx) - 1 - first;
  return segment_id;
}

template <typename T>
size_t XGBOOST_DEVICE SegmentId(Span<T> segments_ptr, size_t idx) {
  return SegmentId(segments_ptr.cbegin(), segments_ptr.cend(), idx);
}
}  // namespace cuda
}  // namespace common
}  // namespace xgboost
