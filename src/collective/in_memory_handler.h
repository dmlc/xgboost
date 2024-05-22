/**
 * Copyright 2022-2023, XGBoost contributors
 */
#pragma once
#include <condition_variable>
#include <map>
#include <string>

#include "../data/array_interface.h"
#include "comm.h"

namespace xgboost::collective {
/**
 * @brief Handles collective communication primitives in memory.
 *
 * This class is thread safe.
 */
class InMemoryHandler {
 public:
  /**
   * @brief Default constructor.
   *
   * This is used when multiple objects/threads are accessing the same handler and need to
   * initialize it collectively.
   */
  InMemoryHandler() = default;

  /**
   * @brief Construct a handler with the given world size.
   * @param world Number of workers.
   *
   * This is used when the handler only needs to be initialized once with a known world size.
   */
  explicit InMemoryHandler(std::int32_t world) : world_size_{world} {}

  /**
   * @brief Initialize the handler with the world size and rank.
   * @param world_size Number of workers.
   * @param rank Index of the worker.
   *
   * This is used when multiple objects/threads are accessing the same handler and need to
   * initialize it collectively.
   */
  void Init(std::int32_t world_size, std::int32_t rank);

  /**
   * @brief Shut down the handler.
   * @param sequence_number Call sequence number.
   * @param rank Index of the worker.
   *
   * This is used when multiple objects/threads are accessing the same handler and need to
   * shut it down collectively.
   */
  void Shutdown(uint64_t sequence_number, std::int32_t rank);

  /**
   * @brief Perform allgather.
   * @param input The input buffer.
   * @param bytes Number of bytes in the input buffer.
   * @param output The output buffer.
   * @param sequence_number Call sequence number.
   * @param rank Index of the worker.
   */
  void Allgather(char const* input, std::size_t bytes, std::string* output,
                 std::size_t sequence_number, std::int32_t rank);

  /**
   * @brief Perform variable-length allgather.
   * @param input The input buffer.
   * @param bytes Number of bytes in the input buffer.
   * @param output The output buffer.
   * @param sequence_number Call sequence number.
   * @param rank Index of the worker.
   */
  void AllgatherV(char const* input, std::size_t bytes, std::string* output,
                  std::size_t sequence_number, std::int32_t rank);

  /**
   * @brief Perform allreduce.
   * @param input The input buffer.
   * @param bytes Number of bytes in the input buffer.
   * @param output The output buffer.
   * @param sequence_number Call sequence number.
   * @param rank Index of the worker.
   * @param data_type Type of the data.
   * @param op The reduce operation.
   */
  void Allreduce(char const* input, std::size_t bytes, std::string* output,
                 std::size_t sequence_number, std::int32_t rank,
                 ArrayInterfaceHandler::Type data_type, Op op);

  /**
   * @brief Perform broadcast.
   * @param input The input buffer.
   * @param bytes Number of bytes in the input buffer.
   * @param output The output buffer.
   * @param sequence_number Call sequence number.
   * @param rank Index of the worker.
   * @param root Index of the worker to broadcast from.
   */
  void Broadcast(char const* input, std::size_t bytes, std::string* output,
                 std::size_t sequence_number, std::int32_t rank, std::int32_t root);

 private:
  /**
   * @brief Handle a collective communication primitive.
   * @tparam HandlerFunctor The functor used to perform the specific primitive.
   * @param input The input buffer.
   * @param size Size of the input in terms of the data type.
   * @param output The output buffer.
   * @param sequence_number Call sequence number.
   * @param rank Index of the worker.
   * @param functor The functor instance used to perform the specific primitive.
   */
  template <class HandlerFunctor>
  void Handle(char const* input, std::size_t size, std::string* output, std::size_t sequence_number,
              std::int32_t rank, HandlerFunctor const& functor);

  std::int32_t world_size_{};  /// Number of workers.
  std::int64_t received_{};     /// Number of calls received with the current sequence.
  std::int64_t sent_{};        /// Number of calls completed with the current sequence.
  std::string buffer_{};      /// A shared common buffer.
  std::map<std::size_t, std::string_view> aux_{};  /// A shared auxiliary map.
  uint64_t sequence_number_{};                     /// Call sequence number.
  mutable std::mutex mutex_;                       /// Lock.
  mutable std::condition_variable cv_;             /// Conditional variable to wait on.
};
}  // namespace xgboost::collective
