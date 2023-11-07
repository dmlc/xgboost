/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <vector>

#include "../common/device_helpers.cuh"

namespace xgboost {
namespace collective {

/**
 * @brief Collective communicator for device buffers.
 */
class DeviceCommunicator {
 public:
  virtual ~DeviceCommunicator() = default;

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
   * @brief Gather values from all all processes.
   *
   * This assumes all ranks have the same size.
   *
   * @param send_buffer    Buffer storing the data to be sent.
   * @param receive_buffer Buffer storing the gathered data.
   * @param send_size      Size of the sent data in bytes.
   */
  virtual void AllGather(void const *send_buffer, void *receive_buffer, std::size_t send_size) = 0;

  /**
   * @brief Gather variable-length values from all processes.
   * @param send_buffer Buffer storing the input data.
   * @param length_bytes Length in bytes of the input data.
   * @param segments Size of each segment.
   * @param receive_buffer Buffer storing the output data.
   */
  virtual void AllGatherV(void const *send_buffer, size_t length_bytes,
                          std::vector<size_t> *segments,
                          dh::caching_device_vector<char> *receive_buffer) = 0;

  /** @brief Synchronize device operations. */
  virtual void Synchronize() = 0;
};

}  // namespace collective
}  // namespace xgboost
