/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once

#include "../common/device_helpers.cuh"
#include "communicator.h"
#include "device_communicator.cuh"

namespace xgboost {
namespace collective {

class NcclDeviceCommunicator : public DeviceCommunicator {
 public:
  NcclDeviceCommunicator(int device_ordinal, Communicator *communicator)
      : device_ordinal_{device_ordinal}, communicator_{communicator} {
    if (device_ordinal_ < 0) {
      LOG(FATAL) << "Invalid device ordinal: " << device_ordinal_;
    }
    if (communicator_ == nullptr) {
      LOG(FATAL) << "Communicator cannot be null.";
    }

    int32_t const rank = communicator_->GetRank();
    int32_t const world = communicator_->GetWorldSize();

    std::vector<uint64_t> uuids(world * kUuidLength, 0);
    auto s_uuid = xgboost::common::Span<uint64_t>{uuids.data(), uuids.size()};
    auto s_this_uuid = s_uuid.subspan(rank * kUuidLength, kUuidLength);
    GetCudaUUID(s_this_uuid);

    // TODO(rongou): replace this with allgather.
    communicator_->AllReduce(uuids.data(), uuids.size(), DataType::kUInt64, Operation::kSum);

    std::vector<xgboost::common::Span<uint64_t, kUuidLength>> converted(world);
    size_t j = 0;
    for (size_t i = 0; i < uuids.size(); i += kUuidLength) {
      converted[j] = xgboost::common::Span<uint64_t, kUuidLength>{uuids.data() + i, kUuidLength};
      j++;
    }

    auto iter = std::unique(converted.begin(), converted.end());
    auto n_uniques = std::distance(converted.begin(), iter);

    CHECK_EQ(n_uniques, world)
        << "Multiple processes within communication group running on same CUDA "
        << "device is not supported. " << PrintUUID(s_this_uuid) << "\n";

    nccl_unique_id_ = GetUniqueId();
    dh::safe_nccl(ncclCommInitRank(&nccl_comm_, world, nccl_unique_id_, rank));
    dh::safe_cuda(cudaStreamCreate(&cuda_stream_));
  }

  ~NcclDeviceCommunicator() override {
    dh::safe_cuda(cudaStreamDestroy(cuda_stream_));
    ncclCommDestroy(nccl_comm_);
    if (xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug)) {
      LOG(CONSOLE) << "======== NCCL Statistics========";
      LOG(CONSOLE) << "AllReduce calls: " << allreduce_calls_;
      LOG(CONSOLE) << "AllReduce total MiB communicated: " << allreduce_bytes_ / 1048576;
    }
  }

  void AllReduceSum(double *send_receive_buffer, int count) override {
    dh::safe_cuda(cudaSetDevice(device_ordinal_));
    dh::safe_nccl(ncclAllReduce(send_receive_buffer, send_receive_buffer, count, ncclDouble,
                                ncclSum, nccl_comm_, cuda_stream_));
    allreduce_bytes_ += count * sizeof(double);
    allreduce_calls_ += 1;
  }

  void AllGatherV(void const *send_buffer, size_t length_bytes, std::vector<std::size_t> *segments,
                  dh::caching_device_vector<char> *receive_buffer) override {
    dh::safe_cuda(cudaSetDevice(device_ordinal_));
    int const world_size = communicator_->GetWorldSize();
    int const rank = communicator_->GetRank();

    segments->clear();
    segments->resize(world_size, 0);
    segments->at(rank) = length_bytes;
    communicator_->AllReduce(segments->data(), segments->size(), DataType::kUInt64,
                             Operation::kMax);
    auto total_bytes = std::accumulate(segments->cbegin(), segments->cend(), 0UL);
    receive_buffer->resize(total_bytes);

    size_t offset = 0;
    dh::safe_nccl(ncclGroupStart());
    for (int32_t i = 0; i < world_size; ++i) {
      size_t as_bytes = segments->at(i);
      dh::safe_nccl(ncclBroadcast(send_buffer, receive_buffer->data().get() + offset, as_bytes,
                                  ncclChar, i, nccl_comm_, cuda_stream_));
      offset += as_bytes;
    }
    dh::safe_nccl(ncclGroupEnd());
  }

  void Synchronize() override {
    dh::safe_cuda(cudaSetDevice(device_ordinal_));
    dh::safe_cuda(cudaStreamSynchronize(cuda_stream_));
  }

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
    if (communicator_->GetRank() == kRootRank) {
      dh::safe_nccl(ncclGetUniqueId(&id));
    }
    communicator_->Broadcast(static_cast<void *>(&id), sizeof(ncclUniqueId),
                             static_cast<int>(kRootRank));
    return id;
  }

  int const device_ordinal_;
  Communicator *communicator_;
  ncclComm_t nccl_comm_{};
  cudaStream_t cuda_stream_{};
  ncclUniqueId nccl_unique_id_{};
  size_t allreduce_bytes_{0};  // Keep statistics of the number of bytes communicated.
  size_t allreduce_calls_{0};  // Keep statistics of the number of reduce calls.
};

}  // namespace collective
}  // namespace xgboost
