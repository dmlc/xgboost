/**
 * Copyright 2023-2024, XGBoost contributors
 *
 * Higher level functions built on top the Communicator API, taking care of behavioral differences
 * between row-split vs column-split distributed training, and horizontal vs vertical federated
 * learning.
 */
#pragma once
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "communicator-inl.h"
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/data.h"               // for MetaINfo

namespace xgboost::collective {

/**
 * @brief Apply the given function where the labels are.
 *
 * Normally all the workers have access to the labels, so the function is just applied locally. In
 * vertical federated learning, we assume labels are only available on worker 0, so the function is
 * applied there, with the results broadcast to other workers.
 *
 * @tparam Function The function used to calculate the results.
 * @param info MetaInfo about the DMatrix.
 * @param buffer The buffer storing the results.
 * @param size The size of the buffer.
 * @param function The function used to calculate the results.
 */
template <typename FN>
void ApplyWithLabels(Context const*, MetaInfo const& info, void* buffer, std::size_t size,
                     FN&& function) {
  if (info.IsVerticalFederated()) {
    // We assume labels are only available on worker 0, so the calculation is done there and result
    // broadcast to other workers.
    std::string message;
    if (collective::GetRank() == 0) {
      try {
        std::forward<FN>(function)();
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
    std::forward<FN>(function)();
  }
}

/**
 * @brief Apply the given function where the labels are.
 *
 * Normally all the workers have access to the labels, so the function is just applied locally. In
 * vertical federated learning, we assume labels are only available on worker 0, so the function is
 * applied there, with the results broadcast to other workers.
 *
 * @tparam T Type of the HostDeviceVector storing the results.
 * @tparam Function The function used to calculate the results.
 * @param info MetaInfo about the DMatrix.
 * @param result The HostDeviceVector storing the results.
 * @param function The function used to calculate the results.
 */
template <typename T, typename Function>
void ApplyWithLabels(Context const*, MetaInfo const& info, HostDeviceVector<T>* result,
                     Function&& function) {
  if (info.IsVerticalFederated()) {
    // We assume labels are only available on worker 0, so the calculation is done there and result
    // broadcast to other workers.
    std::string message;
    if (collective::GetRank() == 0) {
      try {
        std::forward<Function>(function)();
      } catch (dmlc::Error& e) {
        message = e.what();
      }
    }

    collective::Broadcast(&message, 0);
    if (!message.empty()) {
      LOG(FATAL) << &message[0];
      return;
    }

    std::size_t size{};
    if (collective::GetRank() == 0) {
      size = result->Size();
    }
    collective::Broadcast(&size, sizeof(std::size_t), 0);

    result->Resize(size);
    collective::Broadcast(result->HostPointer(), size * sizeof(T), 0);
  } else {
    std::forward<Function>(function)();
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
std::enable_if_t<std::is_trivially_copy_assignable_v<T>, T> GlobalMax(Context const*,
                                                                      MetaInfo const& info,
                                                                      T value) {
  if (info.IsRowSplit()) {
    collective::Allreduce<collective::Operation::kMax>(&value, 1);
  }
  return value;
}

/**
 * @brief Find the global sum of the given values across all workers.
 *
 * This only applies when the data is split row-wise (horizontally). When data is split
 * column-wise (vertically), the original values are returned.
 *
 * @tparam T The type of the values.
 * @param info MetaInfo about the DMatrix.
 * @param values Pointer to the inputs to sum.
 * @param size Number of values to sum.
 */
template <typename T, std::int32_t kDim>
[[nodiscard]] Result GlobalSum(Context const*, MetaInfo const& info,
                               linalg::TensorView<T, kDim> values) {
  if (info.IsRowSplit()) {
    collective::Allreduce<collective::Operation::kSum>(values.Values().data(), values.Size());
  }
  return Success();
}

template <typename Container>
[[nodiscard]] Result GlobalSum(Context const* ctx, MetaInfo const& info, Container* values) {
  return GlobalSum(ctx, info, values->data(), values->size());
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
T GlobalRatio(Context const* ctx, MetaInfo const& info, T dividend, T divisor) {
  std::array<T, 2> results{dividend, divisor};
  auto rc = GlobalSum(ctx, info, linalg::MakeVec(results.data(), results.size()));
  SafeColl(rc);
  std::tie(dividend, divisor) = std::tuple_cat(results);
  if (divisor <= 0) {
    return std::numeric_limits<T>::quiet_NaN();
  } else {
    return dividend / divisor;
  }
}
}  // namespace xgboost::collective
