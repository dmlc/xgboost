/*!
 * Copyright 2016 Rory Mitchell
 */

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


namespace xgboost {
namespace tree {
namespace exact {

/**
 * @struct gpu_gpair gradients.cuh
 * @brief The first/second order gradients for iteratively building the tree
 */
struct gpu_gpair {
  /** the 'g_i' as it appears in the xgboost paper */
  float g;
  /** the 'h_i' as it appears in the xgboost paper */
  float h;

  HOST_DEV_INLINE gpu_gpair(): g(0.f), h(0.f) {}
  HOST_DEV_INLINE gpu_gpair(const float& _g, const float& _h): g(_g), h(_h) {}
  HOST_DEV_INLINE gpu_gpair(const gpu_gpair& a): g(a.g), h(a.h) {}

  /**
   * @brief Checks whether the hessian is more than the defined weight
   * @param minWeight minimum weight to be compared against
   * @return true if the hessian is greater than the minWeight
   * @note this is useful in deciding whether to further split to child node
   */
  HOST_DEV_INLINE bool isSplittable(float minWeight) const {
    return (h > minWeight);
  }

  HOST_DEV_INLINE gpu_gpair& operator+=(const gpu_gpair& a) {
    g += a.g;
    h += a.h;
    return *this;
  }

  HOST_DEV_INLINE gpu_gpair& operator-=(const gpu_gpair& a) {
    g -= a.g;
    h -= a.h;
    return *this;
  }

  HOST_DEV_INLINE friend gpu_gpair operator+(const gpu_gpair& a,
                                             const gpu_gpair& b) {
    return gpu_gpair(a.g+b.g, a.h+b.h);
  }

  HOST_DEV_INLINE friend gpu_gpair operator-(const gpu_gpair& a,
                                             const gpu_gpair& b) {
    return gpu_gpair(a.g-b.g, a.h-b.h);
  }

  HOST_DEV_INLINE gpu_gpair(int value) {
    *this = gpu_gpair((float)value, (float)value);
  }
};


/**
 * @brief Gradient value getter function
 * @param id the index into the vals or instIds array to which to fetch
 * @param vals the gradient value buffer
 * @param instIds instance index buffer
 * @return the expected gradient value
 */
HOST_DEV_INLINE gpu_gpair get(int id, const gpu_gpair* vals, const int* instIds) {
  id = instIds[id];
  return vals[id];
}

}  // namespace exact
}  // namespace tree
}  // namespace xgboost
