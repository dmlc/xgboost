/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <xgboost/logging.h>

namespace xgboost {
namespace collective {

/** @brief Defines the integral and floating data types. */
enum class DataType { kInt, kFloat, kDouble };

inline std::size_t GetTypeSize(DataType data_type) {
  std::size_t size{0};
  switch (data_type) {
    case DataType::kInt:
      size = sizeof(int);
      break;
    case DataType::kFloat:
      size = sizeof(float);
      break;
    case DataType::kDouble:
      size = sizeof(double);
      break;
  }
  return size;
}

/** @brief Defines the reduction operation. */
enum class Operation { kMax, kSum };

/**
 * @brief A communicator class that handles collective communication.
 */
class Communicator {
 public:
  /**
   * @brief Construct a new communicator.
   *
   * @param world_size Total number of processes.
   * @param rank       Rank of the current process.
   */
  Communicator(int world_size, int rank) : world_size_(world_size), rank_(rank) {
    if (world_size < 1) {
      LOG(FATAL) << "World size " << world_size << " is less than 1.";
    }
    if (rank < 0) {
      LOG(FATAL) << "Rank " << rank << " is less than 0.";
    }
    if (rank >= world_size) {
      LOG(FATAL) << "Rank " << rank << " is greater than world_size - 1: " << world_size - 1 << ".";
    }
  }

  virtual ~Communicator() = default;

  /** @brief Get the total number of processes. */
  int GetWorldSize() const { return world_size_; }

  /** @brief Get the rank of the current processes. */
  int GetRank() const { return rank_; }

  /** @brief Whether the communicator is running in distributed mode. */
  bool IsDistributed() const { return world_size_ > 1; };

  /**
   * @brief Combines values from all processes and distributes the result back to all processes.
   *
   * @param send_receive_buffer Buffer storing the data.
   * @param count               Number of elements in the buffer.
   * @param data_type           Data type stored in the buffer.
   * @param op                  The operation to perform.
   */
  virtual void AllReduce(void *send_receive_buffer, std::size_t count, DataType data_type,
                         Operation op) = 0;

  /**
   * @brief Broadcasts a message from the process with rank `root` to all other processes of the
   * group.
   *
   * @param send_receive_buffer Buffer storing the data.
   * @param size                Size of the data in bytes.
   * @param root                Rank of broadcast root.
   */
  virtual void Broadcast(void *send_receive_buffer, std::size_t size, int root) = 0;

 private:
  int const world_size_;
  int const rank_;
};

}  // namespace collective
}  // namespace xgboost
