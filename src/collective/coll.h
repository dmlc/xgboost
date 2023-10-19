/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once
#include <cstddef>  // for size_t
#include <cstdint>  // for int8_t, int64_t
#include <memory>   // for enable_shared_from_this

#include "../data/array_interface.h"    // for ArrayInterfaceHandler
#include "comm.h"                       // for Comm
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/context.h"            // for Context
#include "xgboost/span.h"               // for Span

namespace xgboost::collective {
/**
 * @brief Interface and base implementation for collective.
 */
class Coll : public std::enable_shared_from_this<Coll> {
 public:
  Coll() = default;
  virtual ~Coll() noexcept(false) {}  // NOLINT

  /**
   * @brief Allreduce
   *
   * @param [in,out] data Data buffer for input and output.
   * @param [in] type data type.
   * @param [in] op Reduce operation. For custom operation, user needs to reach down to
   *             the CPU implementation.
   */
  [[nodiscard]] virtual Result Allreduce(Context const* ctx, Comm const& comm,
                                         common::Span<std::int8_t> data,
                                         ArrayInterfaceHandler::Type type, Op op);
  /**
   * @brief Broadcast
   *
   * @param [in,out] data Data buffer for input and output.
   * @param [in] root Root rank for broadcast.
   */
  [[nodiscard]] virtual Result Broadcast(Context const* ctx, Comm const& comm,
                                         common::Span<std::int8_t> data, std::int32_t root);
  /**
   * @brief Allgather
   *
   * @param [in,out] data Data buffer for input and output.
   * @param [in] size Size of data for each worker.
   */
  [[nodiscard]] virtual Result Allgather(Context const* ctx, Comm const& comm,
                                         common::Span<std::int8_t> data, std::size_t size);
  /**
   * @brief Allgather with variable length.
   *
   * @param [in] data Input data for the current worker.
   * @param [in] sizes Size of the input from each worker.
   * @param [out] recv_segments pre-allocated offset for each worker in the output, size
   *        should be equal to (world + 1).
   * @param [out] recv pre-allocated buffer for output.
   */
  [[nodiscard]] virtual Result AllgatherV(Context const* ctx, Comm const& comm,
                                          common::Span<std::int8_t const> data,
                                          common::Span<std::int64_t const> sizes,
                                          common::Span<std::int64_t> recv_segments,
                                          common::Span<std::int8_t> recv);
};
}  // namespace xgboost::collective
