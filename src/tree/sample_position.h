/**
 * Copyright 2024, XGBoost Contributors
 */
#pragma once
#include "xgboost/base.h"  // for bst_node_t

namespace xgboost::tree {
// Utility for maniputing the node index. This is used by the tree methods and the
// adaptive objectives to share the node index. A row is invalid if it's not used in the
// last iteration (due to sampling). For these rows, the corresponding tree node index is
// negated.
struct SamplePosition {
  [[nodiscard]] bst_node_t static XGBOOST_HOST_DEV_INLINE Encode(bst_node_t nidx, bool is_valid) {
    return is_valid ? nidx : ~nidx;
  }
  [[nodiscard]] bst_node_t static XGBOOST_HOST_DEV_INLINE Decode(bst_node_t nidx) {
    return IsValid(nidx) ? nidx : ~nidx;
  }
  [[nodiscard]] bool static XGBOOST_HOST_DEV_INLINE IsValid(bst_node_t nidx) { return nidx >= 0; }
};
}  // namespace xgboost::tree
