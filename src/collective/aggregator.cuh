/**
 * Copyright 2023 by XGBoost contributors
 *
 * Higher level functions built on top the Communicator API, taking care of behavioral differences
 * between row-split vs column-split distributed training, and horizontal vs vertical federated
 * learning.
 */
#pragma once
#include <xgboost/data.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "communicator-inl.cuh"

namespace xgboost {
namespace collective {

/**
 * @brief Find the global sum of the given values across all workers.
 *
 * This only applies when the data is split row-wise (horizontally). When data is split
 * column-wise (vertically), the original values are returned.
 *
 * @tparam T The type of the values.
 * @param info MetaInfo about the DMatrix.
 * @param device The device id.
 * @param values Pointer to the inputs to sum.
 * @param size Number of values to sum.
 */
template <typename T>
void GlobalSum(MetaInfo const& info, int device, T* values, size_t size) {
  if (info.IsRowSplit()) {
    collective::AllReduce<collective::Operation::kSum>(device, values, size);
  }
}
}  // namespace collective
}  // namespace xgboost
