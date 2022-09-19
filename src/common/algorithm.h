/*!
 * Copyright 2022 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_ALGORITHM_H_
#define XGBOOST_COMMON_ALGORITHM_H_
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
#endif  // XGBOOST_COMMON_ALGORITHM_H_
