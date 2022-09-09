/*!
 * Copyright 2022 by XGBoost Contributors
 */
#pragma once
#include <algorithm>  // std::upper_bound
#include <cinttypes>  // std::size_t

namespace xgboost {
namespace common {
template <typename It, typename Idx>
auto SegmentId(It first, It last, Idx idx) {
  std::size_t segment_id = std::upper_bound(first, last, idx) - 1 - first;
  return segment_id;
}
}  // namespace common
}  // namespace xgboost
