/**
 * Copyright 2023, XGBoost Contributors
 */
#if defined(XGBOOST_USE_NCCL)
#include <algorithm>  // for sort
#include <cstddef>    // for size_t
#include <cstdint>    // for uint64_t, int8_t
#include <cstring>    // for memcpy
#include <memory>     // for shared_ptr
#include <sstream>    // for stringstream
#include <vector>     // for vector

#include "../common/device_helpers.cuh"  // for DefaultStream
#include "../common/type.h"              // for EraseType
#include "broadcast.h"                   // for Broadcast
#include "comm.cuh"                      // for NCCLComm
#include "comm.h"                        // for Comm
#include "xgboost/collective/result.h"   // for Result
#include "xgboost/span.h"                // for Span

namespace xgboost::collective {
namespace {
Result GetUniqueId(Comm const& comm, ncclUniqueId* pid) {
  static const int kRootRank = 0;
  ncclUniqueId id;
  if (comm.Rank() == kRootRank) {
    dh::safe_nccl(ncclGetUniqueId(&id));
  }
  auto rc = Broadcast(comm, common::Span{reinterpret_cast<std::int8_t*>(&id), sizeof(ncclUniqueId)},
                      kRootRank);
  if (!rc.OK()) {
    return rc;
  }
  *pid = id;
  return Success();
}

inline constexpr std::size_t kUuidLength =
    sizeof(std::declval<cudaDeviceProp>().uuid) / sizeof(std::uint64_t);

void GetCudaUUID(xgboost::common::Span<std::uint64_t, kUuidLength> const& uuid, DeviceOrd device) {
  cudaDeviceProp prob{};
  dh::safe_cuda(cudaGetDeviceProperties(&prob, device.ordinal));
  std::memcpy(uuid.data(), static_cast<void*>(&(prob.uuid)), sizeof(prob.uuid));
}

static std::string PrintUUID(xgboost::common::Span<std::uint64_t, kUuidLength> const& uuid) {
  std::stringstream ss;
  for (auto v : uuid) {
    ss << std::hex << v;
  }
  return ss.str();
}
}  // namespace

Comm* Comm::MakeCUDAVar(Context const* ctx, std::shared_ptr<Coll> pimpl) {
  return new NCCLComm{ctx, *this, pimpl};
}

NCCLComm::NCCLComm(Context const* ctx, Comm const& root, std::shared_ptr<Coll> pimpl)
    : Comm{root.TrackerInfo().host, root.TrackerInfo().port, root.Timeout(), root.Retry(),
           root.TaskID()},
      stream_{dh::DefaultStream()} {
  this->world_ = root.World();
  this->rank_ = root.Rank();
  this->domain_ = root.Domain();
  if (!root.IsDistributed()) {
    return;
  }

  dh::safe_cuda(cudaSetDevice(ctx->Ordinal()));

  std::vector<std::uint64_t> uuids(root.World() * kUuidLength, 0);
  auto s_uuid = xgboost::common::Span<std::uint64_t>{uuids.data(), uuids.size()};
  auto s_this_uuid = s_uuid.subspan(root.Rank() * kUuidLength, kUuidLength);
  GetCudaUUID(s_this_uuid, ctx->Device());

  auto rc = pimpl->Allgather(root, common::EraseType(s_uuid), s_this_uuid.size_bytes());
  CHECK(rc.OK()) << rc.Report();

  std::vector<xgboost::common::Span<std::uint64_t, kUuidLength>> converted(root.World());
  std::size_t j = 0;
  for (size_t i = 0; i < uuids.size(); i += kUuidLength) {
    converted[j] = s_uuid.subspan(i, kUuidLength);
    j++;
  }

  std::sort(converted.begin(), converted.end());
  auto iter = std::unique(converted.begin(), converted.end());
  auto n_uniques = std::distance(converted.begin(), iter);

  CHECK_EQ(n_uniques, root.World())
      << "Multiple processes within communication group running on same CUDA "
      << "device is not supported. " << PrintUUID(s_this_uuid) << "\n";

  rc = GetUniqueId(root, &nccl_unique_id_);
  CHECK(rc.OK()) << rc.Report();
  dh::safe_nccl(ncclCommInitRank(&nccl_comm_, root.World(), nccl_unique_id_, root.Rank()));

  for (std::int32_t r = 0; r < root.World(); ++r) {
    this->channels_.emplace_back(
        std::make_shared<NCCLChannel>(root, r, nccl_comm_, dh::DefaultStream()));
  }
}

NCCLComm::~NCCLComm() {
  if (nccl_comm_) {
    dh::safe_nccl(ncclCommDestroy(nccl_comm_));
  }
}
}  // namespace xgboost::collective
#endif  // defined(XGBOOST_USE_NCCL)
