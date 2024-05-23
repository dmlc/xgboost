/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once
#include <cstdint>  // for int8_t, int64_t
#include <memory>   // for enable_shared_from_this

#include "../data/array_interface.h"    // for ArrayInterfaceHandler
#include "comm.h"                       // for Comm
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/span.h"               // for Span

namespace xgboost::collective {
enum class AllgatherVAlgo {
  kRing = 0,   // use ring-based allgather-v
  kBcast = 1,  // use broadcast-based allgather-v
};

/**
 * @brief Interface and base implementation for collective.
 */
class Coll : public std::enable_shared_from_this<Coll> {
 public:
  Coll() = default;
  virtual ~Coll() noexcept(false) {}  // NOLINT

  virtual Coll* MakeCUDAVar();

  /**
   * @brief Allreduce
   *
   * @param [in,out] data Data buffer for input and output.
   * @param [in] type data type.
   * @param [in] op Reduce operation. For custom operation, user needs to reach down to
   *             the CPU implementation.
   */
  [[nodiscard]] virtual Result Allreduce(Comm const& comm, common::Span<std::int8_t> data,
                                         ArrayInterfaceHandler::Type type, Op op);
  /**
   * @brief Broadcast
   *
   * @param [in,out] data Data buffer for input and output.
   * @param [in] root Root rank for broadcast.
   */
  [[nodiscard]] virtual Result Broadcast(Comm const& comm, common::Span<std::int8_t> data,
                                         std::int32_t root);
  /**
   * @brief Allgather
   *
   * @param [in,out] data Data buffer for input and output.
   */
  [[nodiscard]] virtual Result Allgather(Comm const& comm, common::Span<std::int8_t> data);
  /**
   * @brief Allgather with variable length.
   *
   * @param [in] data Input data for the current worker.
   * @param [in] sizes Size of the input from each worker.
   * @param [out] recv_segments pre-allocated offset buffer for each worker in the output,
   *              size should be equal to (world + 1). GPU ring-based implementation
   *              doesn't use the buffer.
   * @param [out] recv pre-allocated buffer for output.
   */
  [[nodiscard]] virtual Result AllgatherV(Comm const& comm, common::Span<std::int8_t const> data,
                                          common::Span<std::int64_t const> sizes,
                                          common::Span<std::int64_t> recv_segments,
                                          common::Span<std::int8_t> recv, AllgatherVAlgo algo);
};
}  // namespace xgboost::collective
