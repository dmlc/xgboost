/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#if defined(XGBOOST_USE_NCCL)
#include <algorithm>  // for sort
#include <cstddef>    // for size_t
#include <cstdint>    // for uint64_t, int8_t
#include <cstring>    // for memcpy
#include <memory>     // for shared_ptr
#include <sstream>    // for stringstream
#include <vector>     // for vector

#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/cuda_rt_utils.h"     // for SetDevice
#include "../common/device_helpers.cuh"  // for DefaultStream
#include "../common/type.h"              // for EraseType
#include "comm.cuh"                      // for NCCLComm
#include "comm.h"                        // for Comm
#include "nccl_stub.h"                   // for NcclStub
#include "xgboost/collective/result.h"   // for Result
#include "xgboost/span.h"                // for Span

namespace xgboost::collective {
namespace {
Result GetUniqueId(Comm const& comm, std::shared_ptr<NcclStub> stub, std::shared_ptr<Coll> coll,
                   ncclUniqueId* pid) {
  static const int kRootRank = 0;
  ncclUniqueId id;
  if (comm.Rank() == kRootRank) {
    auto rc = stub->GetUniqueId(&id);
    SafeColl(rc);
  }
  auto rc = coll->Broadcast(
      comm, common::Span{reinterpret_cast<std::int8_t*>(&id), sizeof(ncclUniqueId)}, kRootRank);
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

std::string PrintUUID(xgboost::common::Span<std::uint64_t, kUuidLength> const& uuid) {
  std::stringstream ss;
  for (auto v : uuid) {
    ss << std::hex << v;
  }
  return ss.str();
}
}  // namespace

Comm* RabitComm::MakeCUDAVar(Context const* ctx, std::shared_ptr<Coll> pimpl) const {
  return new NCCLComm{ctx, *this, pimpl, StringView{this->nccl_path_}};
}

NCCLComm::NCCLComm(Context const* ctx, Comm const& root, std::shared_ptr<Coll> pimpl,
                   StringView nccl_path)
    : Comm{root.TrackerInfo().host, root.TrackerInfo().port, root.Timeout(), root.Retry(),
           root.TaskID()},
      stream_{ctx->CUDACtx()->Stream()} {
  this->world_ = root.World();
  this->rank_ = root.Rank();
  this->domain_ = root.Domain();
  if (!root.IsDistributed()) {
    return;
  }

  curt::SetDevice(ctx->Ordinal());
  stub_ = std::make_shared<NcclStub>(nccl_path);

  std::vector<std::uint64_t> uuids(root.World() * kUuidLength, 0);
  auto s_uuid = xgboost::common::Span<std::uint64_t>{uuids.data(), uuids.size()};
  auto s_this_uuid = s_uuid.subspan(root.Rank() * kUuidLength, kUuidLength);
  GetCudaUUID(s_this_uuid, ctx->Device());

  auto rc = pimpl->Allgather(root, common::EraseType(s_uuid));
  SafeColl(rc);

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

  rc = std::move(rc) << [&] {
    return GetUniqueId(root, this->stub_, pimpl, &nccl_unique_id_);
  } << [&] {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    return this->stub_->CommInitRankConfig(&nccl_comm_, root.World(), nccl_unique_id_, root.Rank(),
                                           &config);
  } << [&] {
    return BusyWait(this->stub_, this->nccl_comm_, this->Timeout());
  };
  SafeColl(rc);

  for (std::int32_t r = 0; r < root.World(); ++r) {
    this->channels_.emplace_back(
        std::make_shared<NCCLChannel>(root, r, nccl_comm_, stub_, dh::DefaultStream()));
  }
}

NCCLComm::~NCCLComm() {
  if (nccl_comm_) {
    auto rc = Success() << [this] {
      return this->stub_->CommFinalize(this->nccl_comm_);
    } << [this] {
      auto rc = BusyWait(this->stub_, this->nccl_comm_, this->Timeout());
      if (!rc.OK()) {
        return std::move(rc) + this->stub_->CommAbort(this->nccl_comm_);
      }
      return rc;
    } << [this] {
      return this->stub_->CommDestroy(this->nccl_comm_);
    };
    if (!rc.OK()) {
      LOG(WARNING) << rc.Report();
    }
  }
  nccl_comm_ = nullptr;
}
}  // namespace xgboost::collective
#endif  // defined(XGBOOST_USE_NCCL)
