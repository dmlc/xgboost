/*!
 * Copyright 2017-2019 XGBoost contributors
 *
 * \brief Utilities for CUDA.
 */
#ifdef XGBOOST_USE_NCCL
#include <nccl.h>
#endif  // #ifdef XGBOOST_USE_NCCL
#include <sstream>

#include "device_helpers.cuh"

namespace dh {

#if __CUDACC_VER_MAJOR__ > 9
constexpr std::size_t kUuidLength =
    sizeof(std::declval<cudaDeviceProp>().uuid) / sizeof(uint64_t);

void GetCudaUUID(int world_size, int rank, int device_ord,
                 xgboost::common::Span<uint64_t, kUuidLength> uuid) {
  cudaDeviceProp prob;
  safe_cuda(cudaGetDeviceProperties(&prob, device_ord));
  std::memcpy(uuid.data(), static_cast<void*>(&(prob.uuid)), sizeof(prob.uuid));
}

std::string PrintUUID(xgboost::common::Span<uint64_t, kUuidLength> uuid) {
  std::stringstream ss;
  for (auto v : uuid) {
    ss << std::hex << v;
  }
  return ss.str();
}

#endif  // __CUDACC_VER_MAJOR__ > 9

void AllReducer::Init(int _device_ordinal) {
#ifdef XGBOOST_USE_NCCL
  LOG(DEBUG) << "Running nccl init on: " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__;

  device_ordinal_ = _device_ordinal;
  int32_t const rank = rabit::GetRank();

#if __CUDACC_VER_MAJOR__ > 9
  int32_t const world = rabit::GetWorldSize();

  std::vector<uint64_t> uuids(world * kUuidLength, 0);
  auto s_uuid = xgboost::common::Span<uint64_t>{uuids.data(), uuids.size()};
  auto s_this_uuid = s_uuid.subspan(rank * kUuidLength, kUuidLength);
  GetCudaUUID(world, rank, device_ordinal_, s_this_uuid);

  // No allgather yet.
  rabit::Allreduce<rabit::op::Sum, uint64_t>(uuids.data(), uuids.size());

  std::vector<xgboost::common::Span<uint64_t, kUuidLength>> converted(world);;
  size_t j = 0;
  for (size_t i = 0; i < uuids.size(); i += kUuidLength) {
    converted[j] =
        xgboost::common::Span<uint64_t, kUuidLength>{uuids.data() + i, kUuidLength};
    j++;
  }

  auto iter = std::unique(converted.begin(), converted.end());
  auto n_uniques = std::distance(converted.begin(), iter);
  CHECK_EQ(n_uniques, world)
      << "Multiple processes within communication group running on same CUDA "
      << "device is not supported";
#endif  // __CUDACC_VER_MAJOR__ > 9

  id_ = GetUniqueId();
  dh::safe_cuda(cudaSetDevice(device_ordinal_));
  dh::safe_nccl(ncclCommInitRank(&comm_, rabit::GetWorldSize(), id_, rank));
  safe_cuda(cudaStreamCreate(&stream_));
  initialised_ = true;
#else
  if (rabit::IsDistributed()) {
    LOG(FATAL) << "XGBoost is not compiled with NCCL.";
  }
#endif  // XGBOOST_USE_NCCL
}

void AllReducer::AllGather(void const *data, size_t length_bytes,
                           std::vector<size_t> *segments,
                           dh::caching_device_vector<char> *recvbuf) {
#ifdef XGBOOST_USE_NCCL
  CHECK(initialised_);
  dh::safe_cuda(cudaSetDevice(device_ordinal_));
  size_t world = rabit::GetWorldSize();
  segments->clear();
  segments->resize(world, 0);
  segments->at(rabit::GetRank()) = length_bytes;
  rabit::Allreduce<rabit::op::Max>(segments->data(), segments->size());
  auto total_bytes = std::accumulate(segments->cbegin(), segments->cend(), 0);
  recvbuf->resize(total_bytes);

  size_t offset = 0;
  safe_nccl(ncclGroupStart());
  for (int32_t i = 0; i < world; ++i) {
    size_t as_bytes = segments->at(i);
    safe_nccl(
        ncclBroadcast(data, recvbuf->data().get() + offset,
                      as_bytes, ncclChar, i, comm_, stream_));
    offset += as_bytes;
  }
  safe_nccl(ncclGroupEnd());
#endif  // XGBOOST_USE_NCCL
}

AllReducer::~AllReducer() {
#ifdef XGBOOST_USE_NCCL
  if (initialised_) {
    dh::safe_cuda(cudaStreamDestroy(stream_));
    ncclCommDestroy(comm_);
  }
  if (xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug)) {
    LOG(CONSOLE) << "======== NCCL Statistics========";
    LOG(CONSOLE) << "AllReduce calls: " << allreduce_calls_;
    LOG(CONSOLE) << "AllReduce total MiB communicated: " << allreduce_bytes_/1048576;
  }
#endif  // XGBOOST_USE_NCCL
}

}  // namespace dh
