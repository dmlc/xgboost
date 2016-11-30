/*!
 * Copyright 2016 Rory mitchell
*/
#pragma once
#include "types.cuh"
#include "../../../src/tree/param.h"

// When we split on a value which has no left neighbour, define its left
// neighbour as having left_fvalue = current_fvalue - FVALUE_EPS
// This produces a split value slightly lower than the current instance
#define FVALUE_EPS 0.0001

namespace xgboost {
namespace tree {


__device__ __forceinline__ float
device_calc_loss_chg(const GPUTrainingParam &param, const gpu_gpair &scan,
                     const gpu_gpair &missing, const gpu_gpair &parent_sum,
                     const float &parent_gain, bool missing_left) {
  gpu_gpair left = scan;

  if (missing_left) {
    left += missing;
  }

  gpu_gpair right = parent_sum - left;

  float left_gain = CalcGain(param, left.grad(), left.hess());
  float right_gain = CalcGain(param, right.grad(), right.hess());
  return left_gain + right_gain - parent_gain;
}

__device__ __forceinline__ float
loss_chg_missing(const gpu_gpair &scan, const gpu_gpair &missing,
                 const gpu_gpair &parent_sum, const float &parent_gain,
                 const GPUTrainingParam &param, bool &missing_left_out) { // NOLINT
  float missing_left_loss =
      device_calc_loss_chg(param, scan, missing, parent_sum, parent_gain, true);
  float missing_right_loss = device_calc_loss_chg(
      param, scan, missing, parent_sum, parent_gain, false);

  if (missing_left_loss >= missing_right_loss) {
    missing_left_out = true;
    return missing_left_loss;
  } else {
    missing_left_out = false;
    return missing_right_loss;
  }
}
}  // namespace tree
}  // namespace xgboost
