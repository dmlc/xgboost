/**
 * Copyright 2023 by XGBoost contributors
 */
#pragma once
#include <string>
#include <vector>

#include "communicator.h"
#include "device_communicator.cuh"

namespace xgboost {
namespace collective {

/**
 * @brief Reduce values from all processes and distribute the result back to all processes.
 * @param device              ID of the device.
 * @param send_receive_buffer Buffer storing the data.
 * @param count               Number of elements in the buffer.
 */
template <Operation op>
inline void AllReduce(int device, std::int8_t *send_receive_buffer, size_t count) {
  Communicator::GetDevice(device)->AllReduce(send_receive_buffer, count, DataType::kInt8, op);
}

template <Operation op>
inline void AllReduce(int device, std::uint8_t *send_receive_buffer, size_t count) {
  Communicator::GetDevice(device)->AllReduce(send_receive_buffer, count, DataType::kUInt8, op);
}

template <Operation op>
inline void AllReduce(int device, std::int32_t *send_receive_buffer, size_t count) {
  Communicator::GetDevice(device)->AllReduce(send_receive_buffer, count, DataType::kInt32, op);
}

template <Operation op>
inline void AllReduce(int device, std::uint32_t *send_receive_buffer, size_t count) {
  Communicator::GetDevice(device)->AllReduce(send_receive_buffer, count, DataType::kUInt32, op);
}

template <Operation op>
inline void AllReduce(int device, std::int64_t *send_receive_buffer, size_t count) {
  Communicator::GetDevice(device)->AllReduce(send_receive_buffer, count, DataType::kInt64, op);
}

template <Operation op>
inline void AllReduce(int device, std::uint64_t *send_receive_buffer, size_t count) {
  Communicator::GetDevice(device)->AllReduce(send_receive_buffer, count, DataType::kUInt64, op);
}

template <Operation op>
inline void AllReduce(int device, float *send_receive_buffer, size_t count) {
  Communicator::GetDevice(device)->AllReduce(send_receive_buffer, count, DataType::kFloat, op);
}

template <Operation op>
inline void AllReduce(int device, double *send_receive_buffer, size_t count) {
  Communicator::GetDevice(device)->AllReduce(send_receive_buffer, count, DataType::kDouble, op);
}

/**
 * @brief Gather variable-length values from all processes.
 * @param device         ID of the device.
 * @param send_buffer    Buffer storing the input data.
 * @param length_bytes   Length in bytes of the input data.
 * @param segments       Size of each segment.
 * @param receive_buffer Buffer storing the output data.
 */
inline void AllGatherV(int device, void const *send_buffer, size_t length_bytes,
                       std::vector<size_t> *segments,
                       dh::caching_device_vector<char> *receive_buffer) {
  Communicator::GetDevice(device)->AllGatherV(send_buffer, length_bytes, segments, receive_buffer);
}

/**
 * @brief Synchronize device operations.
 * @param device ID of the device.
 */
inline void Synchronize(int device) { Communicator::GetDevice(device)->Synchronize(); }

}  // namespace collective
}  // namespace xgboost
