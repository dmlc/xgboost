/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once

#ifdef XGBOOST_USE_NCCL
#include "nccl.h"
#endif  // XGBOOST_USE_NCCL
#include "../common/device_helpers.cuh"
#include "coll.h"
#include "comm.h"
#include "xgboost/context.h"

namespace xgboost::collective {

inline Result GetCUDAResult(cudaError rc) {
  if (rc == cudaSuccess) {
    return Success();
  }
  std::string msg = thrust::system_error(rc, thrust::cuda_category()).what();
  return Fail(msg);
}

class NCCLComm : public Comm {
  ncclComm_t nccl_comm_{nullptr};
  ncclUniqueId nccl_unique_id_{};
  dh::CUDAStreamView stream_;

 public:
  [[nodiscard]] ncclComm_t Handle() const { return nccl_comm_; }

  explicit NCCLComm(Context const* ctx, Comm const& root, std::shared_ptr<Coll> pimpl);
  [[nodiscard]] Result LogTracker(std::string) const override {
    LOG(FATAL) << "Device comm is used for logging.";
    return Fail("Undefined.");
  }
  ~NCCLComm() override;
  [[nodiscard]] bool IsFederated() const override { return false; }
  [[nodiscard]] dh::CUDAStreamView Stream() const { return stream_; }
  [[nodiscard]] Result Block() const override {
    auto rc = this->Stream().Sync(false);
    return GetCUDAResult(rc);
  }
};

class NCCLChannel : public Channel {
  std::int32_t rank_{-1};
  ncclComm_t nccl_comm_{};
  dh::CUDAStreamView stream_;

 public:
  explicit NCCLChannel(Comm const& comm, std::int32_t rank, ncclComm_t nccl_comm,
                       dh::CUDAStreamView stream)
      : rank_{rank}, nccl_comm_{nccl_comm}, Channel{comm, nullptr}, stream_{stream} {}

  void SendAll(std::int8_t const* ptr, std::size_t n) override {
    dh::safe_nccl(ncclSend(ptr, n, ncclInt8, rank_, nccl_comm_, stream_));
  }
  void RecvAll(std::int8_t* ptr, std::size_t n) override {
    dh::safe_nccl(ncclRecv(ptr, n, ncclInt8, rank_, nccl_comm_, stream_));
  }
  [[nodiscard]] Result Block() override {
    auto rc = stream_.Sync(false);
    return GetCUDAResult(rc);
  }
};
}  // namespace xgboost::collective
