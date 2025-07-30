/**
 *  Copyright 2019-2025, XGBoost Contributors
 */
#pragma once

#include "../common/math.h"  // for CheckNAN
#include "xgboost/base.h"    // for bst_idx_t
#include "xgboost/data.h"    // for Entry

namespace xgboost::data {
struct COOTuple {
  COOTuple() = default;
  XGBOOST_DEVICE COOTuple(bst_idx_t row_idx, bst_idx_t column_idx, float value)
      : row_idx(row_idx), column_idx(column_idx), value(value) {}

  bst_idx_t row_idx{0};
  bst_idx_t column_idx{0};
  float value{0};
};

struct IsValidFunctor {
  float missing;

  XGBOOST_DEVICE explicit IsValidFunctor(float missing) : missing(missing) {}

  XGBOOST_DEVICE bool operator()(float value) const {
    return !(common::CheckNAN(value) || value == missing);
  }

  XGBOOST_DEVICE bool operator()(const data::COOTuple& e) const {
    return !(common::CheckNAN(e.value) || e.value == missing);
  }

  XGBOOST_DEVICE bool operator()(const Entry& e) const {
    return !(common::CheckNAN(e.fvalue) || e.fvalue == missing);
  }
};
}  // namespace xgboost::data
