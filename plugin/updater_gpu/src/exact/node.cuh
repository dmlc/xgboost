/*
 * Copyright (c) 2017, NVIDIA CORPORATION, Xgboost contributors.  All rights
 * reserved.
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

/**
 * @struct Split node.cuh
 * @brief Abstraction of a possible split in the decision tree
 */
struct Split {
  /** the optimal gain score for this node */
  float score;
  /** index where to split in the DMatrix */
  int index;

  HOST_DEV_INLINE Split() : score(-FLT_MAX), index(INT_MAX) {}

  /**
   * @brief Whether the split info is valid to be used to create a new child
   * @param minSplitLoss minimum score above which decision to split is made
   * @return true if splittable, else false
   */
  HOST_DEV_INLINE bool isSplittable(float minSplitLoss) const {
    return ((score >= minSplitLoss) && (index != INT_MAX));
  }
};

/**
 * @struct Segment node.cuh
 * @brief Space inefficient, but super easy to implement structure to define
 *   the start and end of a segment in the input array
 */
struct Segment {
  /** start index of the segment */
  int start;
  /** end index of the segment */
  int end;

  HOST_DEV_INLINE Segment() : start(-1), end(-1) {}

  /** Checks whether the current structure defines a valid segment */
  HOST_DEV_INLINE bool isValid() const {
    return !((start == -1) || (end == -1));
  }
};

/**
 * @enum NodeType node.cuh
 * @brief Useful to decribe the node type in a dense BFS-order tree array
 */
enum NodeType {
  /** a non-leaf node */
  NODE = 0,
  /** leaf node */
  LEAF,
  /** unused node */
  UNUSED
};

/**
 * @brief Absolute BFS order IDs to col-wise unique IDs based on user input
 * @param tid the index of the element that this thread should access
 * @param abs the array of absolute IDs
 * @param colIds the array of column IDs for each element
 * @param nodeStart the start of the node ID at this level
 * @param nKeys number of nodes at this level.
 * @return the uniq key
 */

HOST_DEV_INLINE int abs2uniqKey(int tid, const node_id_t* abs,
                                const int* colIds, node_id_t nodeStart,
                                int nKeys) {
  int a = abs[tid];
  if (a == UNUSED_NODE) return a;
  return ((a - nodeStart) + (colIds[tid] * nKeys));
}

}  // namespace tree
}  // namespace xgboost
