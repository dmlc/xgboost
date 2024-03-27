/**
 * Copyright 2023-2024, XGBoost contributors
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

#include "allreduce.h"
#include "xgboost/collective/result.h"  // for Result

namespace xgboost::collective {

/**
 * @brief Find the global sum of the given values across all workers.
 *
 * This only applies when the data is split row-wise (horizontally). When data is split
 * column-wise (vertically), the original values are returned.
 *
 * @tparam T The type of the values.
 *
 * @param info MetaInfo about the DMatrix.
 * @param values Pointer to the inputs to sum.
 * @param size Number of values to sum.
 */
template <typename T, std::int32_t kDim>
[[nodiscard]] Result GlobalSum(Context const* ctx, MetaInfo const& info,
                               linalg::TensorView<T, kDim> values) {
  if (info.IsRowSplit()) {
    return collective::Allreduce(ctx, values, collective::Op::kSum);
  }
  return Success();
}
}  // namespace xgboost::collective
