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
   * @brief Sum values from all processes and distribute the result back to all processes.
   * @param send_receive_buffer Buffer storing the data.
   * @param count               Number of elements in the buffer.
   */
  virtual void AllReduceSum(float *send_receive_buffer, size_t count) = 0;

  /**
   * @brief Sum values from all processes and distribute the result back to all processes.
   * @param send_receive_buffer Buffer storing the data.
   * @param count               Number of elements in the buffer.
   */
  virtual void AllReduceSum(double *send_receive_buffer, size_t count) = 0;

  /**
   * @brief Sum values from all processes and distribute the result back to all processes.
   * @param send_receive_buffer Buffer storing the data.
   * @param count               Number of elements in the buffer.
   */
  virtual void AllReduceSum(int64_t *send_receive_buffer, size_t count) = 0;

  /**
   * @brief Sum values from all processes and distribute the result back to all processes.
   * @param send_receive_buffer Buffer storing the data.
   * @param count               Number of elements in the buffer.
   */
  virtual void AllReduceSum(uint64_t *send_receive_buffer, size_t count) = 0;

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
