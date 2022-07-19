/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <vector>

#include "../common/device_helpers.cuh"

namespace xgboost {
namespace collective {

class DeviceCommunicator {
 public:
  virtual ~DeviceCommunicator() = default;

  virtual void AllReduceSum(double *send_receive_buffer, int count) = 0;

  virtual void AllGatherV(void const *send_buffer, size_t length_bytes,
                          std::vector<size_t> *segments,
                          dh::caching_device_vector<char> *receive_buffer) = 0;

  virtual void Synchronize() = 0;
};

}  // namespace collective
}  // namespace xgboost
