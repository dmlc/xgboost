/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <condition_variable>
#include <string>

#include "communicator.h"

namespace xgboost {
namespace collective {

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
   * @param world_size Number of workers.
   *
   * This is used when the handler only needs to be initialized once with a known world size.
   */
  explicit InMemoryHandler(int worldSize) : world_size_{worldSize} {}

  /**
   * @brief Initialize the handler with the world size and rank.
   * @param world_size Number of workers.
   * @param rank Index of the worker.
   *
   * This is used when multiple objects/threads are accessing the same handler and need to
   * initialize it collectively.
   */
  void Init(int world_size, int rank);

  /**
   * @brief Shut down the handler.
   * @param sequence_number Call sequence number.
   * @param rank Index of the worker.
   *
   * This is used when multiple objects/threads are accessing the same handler and need to
   * shut it down collectively.
   */
  void Shutdown(uint64_t sequence_number, int rank);

  /**
   * @brief Perform allgather.
   * @param input The input buffer.
   * @param bytes Number of bytes in the input buffer.
   * @param output The output buffer.
   * @param sequence_number Call sequence number.
   * @param rank Index of the worker.
   */
  void Allgather(char const* input, std::size_t bytes, std::string* output,
                 std::size_t sequence_number, int rank);

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
                 std::size_t sequence_number, int rank, DataType data_type, Operation op);

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
                 std::size_t sequence_number, int rank, int root);

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
              int rank, HandlerFunctor const& functor);

  int world_size_{};                    /// Number of workers.
  int received_{};                      /// Number of calls received with the current sequence.
  int sent_{};                          /// Number of calls completed with the current sequence.
  std::string buffer_{};                /// A shared common buffer.
  uint64_t sequence_number_{};          /// Call sequence number.
  mutable std::mutex mutex_;            /// Lock.
  mutable std::condition_variable cv_;  /// Conditional variable to wait on.
};

}  // namespace collective
}  // namespace xgboost
