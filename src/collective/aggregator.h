/**
 * Copyright 2023 by XGBoost contributors
 *
 * Higher level functions built on top the Communicator API, taking care of behavioral differences
 * between row-split vs column-split distributed training, and horizontal vs vertical federated
 * learning.
 */
#pragma once
#include <xgboost/data.h>

#include <string>
#include <utility>
#include <vector>

#include "communicator-inl.h"

namespace xgboost {
namespace collective {

/**
 * @brief Apply the given function where the labels are.
 *
 * Normally all the workers have access to the labels, so the function is just applied locally. In
 * vertical federated learning, we assume labels are only available on worker 0, so the function is
 * applied there, with the results broadcast to other workers.
 *
 * @tparam Function The function used to calculate the results.
 * @tparam Args Arguments to the function.
 * @param info MetaInfo about the DMatrix.
 * @param buffer The buffer storing the results.
 * @param size The size of the buffer.
 * @param function The function used to calculate the results.
 * @param args Arguments to the function.
 */
template <typename Function, typename T, typename... Args>
void ApplyWithLabels(MetaInfo const& info, T* buffer, size_t size, Function&& function,
                     Args&&... args) {
  if (info.IsVerticalFederated()) {
    // We assume labels are only available on worker 0, so the calculation is done there and result
    // broadcast to other workers.
    std::string message;
    if (collective::GetRank() == 0) {
      try {
        std::forward<Function>(function)(std::forward<Args>(args)...);
      } catch (dmlc::Error& e) {
        message = e.what();
      }
    }

    collective::Broadcast(&message, 0);
    if (message.empty()) {
      collective::Broadcast(buffer, size, 0);
    } else {
      LOG(FATAL) << &message[0];
    }
  } else {
    std::forward<Function>(function)(std::forward<Args>(args)...);
  }
}

/**
 * @brief Find the global max of the given value across all workers.
 *
 * This only applies when the data is split row-wise (horizontally). When data is split
 * column-wise (vertically), the local value is returned.
 *
 * @tparam T The type of the value.
 * @param info MetaInfo about the DMatrix.
 * @param value The input for finding the global max.
 * @return The global max of the input.
 */
template <typename T>
T GlobalMax(MetaInfo const& info, T value) {
  if (info.IsRowSplit()) {
    collective::Allreduce<collective::Operation::kMax>(&value, 1);
  }
  return value;
}

/**
 * @brief Find the global ratio of the given two values across all workers.
 *
 * This only applies when the data is split row-wise (horizontally). When data is split
 * column-wise (vertically), the local ratio is returned.
 *
 * @tparam T The type of the values.
 * @param info MetaInfo about the DMatrix.
 * @param dividend The dividend of the ratio.
 * @param divisor The divisor of the ratio.
 * @return The global ratio of the two inputs.
 */
template <typename T>
T GlobalRatio(MetaInfo const& info, T dividend, T divisor) {
  if (info.IsRowSplit()) {
    std::array<T, 2> results{dividend, divisor};
    collective::Allreduce<collective::Operation::kSum>(results.data(), results.size());
    dividend = results[0];
    divisor = results[1];
  }
  if (divisor <= 0) {
    return std::numeric_limits<T>::quiet_NaN();
  } else {
    return dividend / divisor;
  }
}

/**
 * @brief Find the global sum of the given values across all workers.
 *
 * This only applies when the data is split row-wise (horizontally). When data is split
 * column-wise (vertically), the original values are returned.
 *
 * @tparam T The type of the values.
 * @param info MetaInfo about the DMatrix.
 * @param values The inputs to sum.
 */
template <typename T>
void GlobalSum(MetaInfo const& info, common::Span<T> const& values) {
  if (info.IsRowSplit()) {
    collective::Allreduce<collective::Operation::kSum>(values.data(), values.size());
  }
}

template <typename T>
void GlobalSum(MetaInfo const& info, std::vector<T>& values) {
  if (info.IsRowSplit()) {
    collective::Allreduce<collective::Operation::kSum>(values.data(), values.size());
  }
}

template <typename T, size_t N>
void GlobalSum(MetaInfo const& info, T (&values)[N]) {
  if (info.IsRowSplit()) {
    collective::Allreduce<collective::Operation::kSum>(values, N);
  }
}

template <typename T>
void GlobalSum(MetaInfo const& info, T* values, size_t size) {
  if (info.IsRowSplit()) {
    collective::Allreduce<collective::Operation::kSum>(values, size);
  }
}
}  // namespace collective
}  // namespace xgboost
