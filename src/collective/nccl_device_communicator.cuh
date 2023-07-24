/*!
 * Copyright 2022-2023 XGBoost contributors
 */
#pragma once

#include "../common/device_helpers.cuh"
#include "communicator.h"
#include "device_communicator.cuh"

namespace xgboost {
namespace collective {

class NcclDeviceCommunicator : public DeviceCommunicator {
 public:
  /**
   * @brief Construct a new NCCL communicator.
   * @param device_ordinal The GPU device id.
   * @param needs_sync Whether extra CUDA stream synchronization is needed.
   *
   * In multi-GPU tests when multiple NCCL communicators are created in the same process, sometimes
   * a deadlock happens because NCCL kernels are blocking. The extra CUDA stream synchronization
   * makes sure that the NCCL kernels are caught up, thus avoiding the deadlock.
   *
   * The Rabit communicator runs with one process per GPU, so the additional synchronization is not
   * needed. The in-memory communicator is used in tests with multiple threads, each thread
   * representing a rank/worker, so the additional synchronization is needed to avoid deadlocks.
   */
  explicit NcclDeviceCommunicator(int device_ordinal, bool needs_sync);
  ~NcclDeviceCommunicator() override;
  void AllReduce(void *send_receive_buffer, std::size_t count, DataType data_type,
                 Operation op) override;
  void AllGatherV(void const *send_buffer, size_t length_bytes, std::vector<std::size_t> *segments,
                  dh::caching_device_vector<char> *receive_buffer) override;
  void Synchronize() override;

 private:
  static constexpr std::size_t kUuidLength =
      sizeof(std::declval<cudaDeviceProp>().uuid) / sizeof(uint64_t);

  void GetCudaUUID(xgboost::common::Span<uint64_t, kUuidLength> const &uuid) const {
    cudaDeviceProp prob{};
    dh::safe_cuda(cudaGetDeviceProperties(&prob, device_ordinal_));
    std::memcpy(uuid.data(), static_cast<void *>(&(prob.uuid)), sizeof(prob.uuid));
  }

  static std::string PrintUUID(xgboost::common::Span<uint64_t, kUuidLength> const &uuid) {
    std::stringstream ss;
    for (auto v : uuid) {
      ss << std::hex << v;
    }
    return ss.str();
  }

  /**
   * \fn  ncclUniqueId GetUniqueId()
   *
   * \brief Gets the Unique ID from NCCL to be used in setting up interprocess
   * communication
   *
   * \return the Unique ID
   */
  ncclUniqueId GetUniqueId() {
    static const int kRootRank = 0;
    ncclUniqueId id;
    if (rank_ == kRootRank) {
      dh::safe_nccl(ncclGetUniqueId(&id));
    }
    Broadcast(static_cast<void *>(&id), sizeof(ncclUniqueId), static_cast<int>(kRootRank));
    return id;
  }

  void BitwiseAllReduce(void *send_receive_buffer, std::size_t count, DataType data_type,
                        Operation op);

  int const device_ordinal_;
  bool const needs_sync_;
  int const world_size_;
  int const rank_;
  ncclComm_t nccl_comm_{};
  cudaStream_t cuda_stream_{};
  ncclUniqueId nccl_unique_id_{};
  size_t allreduce_bytes_{0};  // Keep statistics of the number of bytes communicated.
  size_t allreduce_calls_{0};  // Keep statistics of the number of reduce calls.
};

}  // namespace collective
}  // namespace xgboost
