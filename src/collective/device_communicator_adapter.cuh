/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once

#include <numeric>  // for accumulate

#include "communicator.h"
#include "device_communicator.cuh"

namespace xgboost {
namespace collective {

class DeviceCommunicatorAdapter : public DeviceCommunicator {
 public:
  explicit DeviceCommunicatorAdapter(int device_ordinal)
      : device_ordinal_{device_ordinal}, world_size_{GetWorldSize()}, rank_{GetRank()} {
    if (device_ordinal_ < 0) {
      LOG(FATAL) << "Invalid device ordinal: " << device_ordinal_;
    }
  }

  ~DeviceCommunicatorAdapter() override = default;

  void AllReduce(void *send_receive_buffer, std::size_t count, DataType data_type,
                 Operation op) override {
    if (world_size_ == 1) {
      return;
    }

    dh::safe_cuda(cudaSetDevice(device_ordinal_));
    auto size = count * GetTypeSize(data_type);
    host_buffer_.resize(size);
    dh::safe_cuda(cudaMemcpy(host_buffer_.data(), send_receive_buffer, size, cudaMemcpyDefault));
    Allreduce(host_buffer_.data(), count, data_type, op);
    dh::safe_cuda(cudaMemcpy(send_receive_buffer, host_buffer_.data(), size, cudaMemcpyDefault));
  }

  void AllGather(void const *send_buffer, void *receive_buffer, std::size_t send_size) override {
    if (world_size_ == 1) {
      return;
    }

    dh::safe_cuda(cudaSetDevice(device_ordinal_));
    host_buffer_.resize(send_size);
    dh::safe_cuda(cudaMemcpy(host_buffer_.data(), send_buffer, send_size, cudaMemcpyDefault));
    auto const output = Allgather(host_buffer_);
    dh::safe_cuda(cudaMemcpy(receive_buffer, output.data(), output.size(), cudaMemcpyDefault));
  }

  void AllGatherV(void const *send_buffer, size_t length_bytes, std::vector<std::size_t> *segments,
                  dh::caching_device_vector<char> *receive_buffer) override {
    if (world_size_ == 1) {
      return;
    }

    dh::safe_cuda(cudaSetDevice(device_ordinal_));

    segments->clear();
    segments->resize(world_size_, 0);
    segments->at(rank_) = length_bytes;
    Allreduce(segments->data(), segments->size(), DataType::kUInt64, Operation::kMax);
    auto total_bytes = std::accumulate(segments->cbegin(), segments->cend(), 0UL);
    receive_buffer->resize(total_bytes);

    host_buffer_.resize(total_bytes);
    size_t offset = 0;
    for (int32_t i = 0; i < world_size_; ++i) {
      size_t as_bytes = segments->at(i);
      if (i == rank_) {
        dh::safe_cuda(cudaMemcpy(host_buffer_.data() + offset, send_buffer, segments->at(rank_),
                                 cudaMemcpyDefault));
      }
      Broadcast(host_buffer_.data() + offset, as_bytes, i);
      offset += as_bytes;
    }
    dh::safe_cuda(cudaMemcpy(receive_buffer->data().get(), host_buffer_.data(), total_bytes,
                             cudaMemcpyDefault));
  }

  void Synchronize() override {
    // Noop.
  }

 private:
  int const device_ordinal_;
  int const world_size_;
  int const rank_;
  /// Host buffer used to call communicator functions.
  std::vector<char> host_buffer_{};
};

}  // namespace collective
}  // namespace xgboost
