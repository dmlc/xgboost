/*
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "../common.cuh"
#include "gradients.cuh"


namespace xgboost {
namespace tree {
namespace exact {

HOST_DEV_INLINE float device_calc_loss_chg(const TrainParam &param,
                                           const gpu_gpair &scan,
                                           const gpu_gpair &missing,
                                           const gpu_gpair &parent_sum,
                                           const float &parent_gain,
                                           bool missing_left) {
  gpu_gpair left = scan;
  if (missing_left) {
    left += missing;
  }
  gpu_gpair right = parent_sum - left;
  float left_gain = CalcGain(param, left.g, left.h);
  float right_gain = CalcGain(param, right.g, right.h);
  return left_gain + right_gain - parent_gain;
}

HOST_DEV_INLINE float loss_chg_missing(const gpu_gpair &scan,
                                       const gpu_gpair &missing,
                                       const gpu_gpair &parent_sum,
                                       const float &parent_gain,
                                       const TrainParam &param,
                                       bool &missing_left_out) {
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

}  // namespace exact
}  // namespace tree
}  // namespace xgboost
