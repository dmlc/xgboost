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

#include "allreduce.h"
#include "broadcast.h"
#include "comm.h"
#include "communicator-inl.h"
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/data.h"               // for MetaINfo

namespace xgboost::collective {
namespace detail {
template <typename Fn>
[[nodiscard]] Result TryApplyWithLabels(Context const* ctx, Fn&& fn) {
  std::string msg;
  if (collective::GetRank() == 0) {
    try {
      fn();
    } catch (dmlc::Error const& e) {
      msg = e.what();
    }
  }
  std::size_t msg_size{msg.size()};
  auto rc = Success() << [&] {
    auto rc = collective::Broadcast(ctx, linalg::MakeVec(&msg_size, 1), 0);
    return rc;
  } << [&] {
    if (msg_size > 0) {
      msg.resize(msg_size);
      return collective::Broadcast(ctx, linalg::MakeVec(msg.data(), msg.size()), 0);
    }
    return Success();
  } << [&] {
    if (msg_size > 0) {
      LOG(FATAL) << msg;
    }
    return Success();
  };
  return rc;
}
}  // namespace detail

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
template <typename Fn>
void ApplyWithLabels(Context const* ctx, MetaInfo const& info, void* buffer, std::size_t size,
                     Fn&& fn) {
  if (info.IsVerticalFederated()) {
    auto rc = detail::TryApplyWithLabels(ctx, fn) << [&] {
      // We assume labels are only available on worker 0, so the calculation is done there and
      // result broadcast to other workers.
      return collective::Broadcast(
          ctx, linalg::MakeVec(reinterpret_cast<std::int8_t*>(buffer), size), 0);
    };
    SafeColl(rc);
  } else {
    std::forward<Fn>(fn)();
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
template <typename T, typename Fn>
void ApplyWithLabels(Context const* ctx, MetaInfo const& info, HostDeviceVector<T>* result,
                     Fn&& fn) {
  if (info.IsVerticalFederated()) {
    // We assume labels are only available on worker 0, so the calculation is done there and result
    // broadcast to other workers.
    auto rc = detail::TryApplyWithLabels(ctx, fn);

    std::size_t size{result->Size()};
    rc = std::move(rc) << [&] {
      return collective::Broadcast(ctx, linalg::MakeVec(&size, 1), 0);
    } << [&] {
      result->Resize(size);
      return collective::Broadcast(ctx, linalg::MakeVec(result->HostPointer(), size), 0);
    };
    SafeColl(rc);
  } else {
    std::forward<Fn>(fn)();
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
std::enable_if_t<std::is_trivially_copy_assignable_v<T>, T> GlobalMax(Context const* ctx,
                                                                      MetaInfo const& info,
                                                                      T value) {
  if (info.IsRowSplit()) {
    auto rc = collective::Allreduce(ctx, linalg::MakeVec(&value, 1), collective::Op::kMax);
    SafeColl(rc);
  }
  return value;
}

template <typename T, std::int32_t kDim>
[[nodiscard]] Result GlobalSum(Context const* ctx, bool is_column_split,
                               linalg::TensorView<T, kDim> values) {
  if (!is_column_split) {
    return collective::Allreduce(ctx, values, collective::Op::kSum);
  }
  return Success();
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
[[nodiscard]] Result GlobalSum(Context const* ctx, MetaInfo const& info,
                               linalg::TensorView<T, kDim> values) {
  return GlobalSum(ctx, info.IsColumnSplit(), values);
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
