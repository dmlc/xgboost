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

  device_ordinal = _device_ordinal;
  int32_t const rank = rabit::GetRank();

#if __CUDACC_VER_MAJOR__ > 9
  int32_t const world = rabit::GetWorldSize();

  std::vector<uint64_t> uuids(world * kUuidLength, 0);
  auto s_uuid = xgboost::common::Span<uint64_t>{uuids.data(), uuids.size()};
  auto s_this_uuid = s_uuid.subspan(rank * kUuidLength, kUuidLength);
  GetCudaUUID(world, rank, device_ordinal, s_this_uuid);

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

  id = GetUniqueId();
  dh::safe_cuda(cudaSetDevice(device_ordinal));
  dh::safe_nccl(ncclCommInitRank(&comm, rabit::GetWorldSize(), id, rank));
  safe_cuda(cudaStreamCreate(&stream));
  initialised_ = true;
#endif  // XGBOOST_USE_NCCL
}

AllReducer::~AllReducer() {
#ifdef XGBOOST_USE_NCCL
  if (initialised_) {
    dh::safe_cuda(cudaStreamDestroy(stream));
    ncclCommDestroy(comm);
  }
  if (xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug)) {
    LOG(CONSOLE) << "======== NCCL Statistics========";
    LOG(CONSOLE) << "AllReduce calls: " << allreduce_calls_;
    LOG(CONSOLE) << "AllReduce total MiB communicated: " << allreduce_bytes_/1048576;
  }
#endif  // XGBOOST_USE_NCCL
}

}  // namespace dh
