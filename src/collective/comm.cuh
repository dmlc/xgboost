/**
 * Copyright 2023-2025, XGBoost Contributors
 */
#pragma once

#ifdef XGBOOST_USE_NCCL
#include "nccl.h"
#endif  // XGBOOST_USE_NCCL

#include <cstdint>  // for int32_t
#include <memory>   // for shared_ptr
#include <utility>  // for move
#include <vector>   // for vector

#include "../common/cuda_stream.h"  // for StreamRef
#include "coll.h"
#include "comm.h"
#include "nccl_stub.h"  // for NcclStub
#include "xgboost/context.h"

namespace xgboost::collective {

inline Result GetCUDAResult(cudaError rc) {
  if (rc == cudaSuccess) {
    return Success();
  }
  std::string msg = thrust::system_error(rc, thrust::cuda_category()).what();
  return Fail(msg);
}

#if defined(XGBOOST_USE_NCCL)
class NCCLComm : public Comm {
 public:
  struct Subgroup {
    std::vector<std::int32_t> active_ranks;
    ncclComm_t comm{nullptr};
    std::shared_ptr<curt::Stream> stream;
  };

 private:
  ncclComm_t nccl_comm_{nullptr};
  std::shared_ptr<NcclStub> stub_;
  ncclUniqueId nccl_unique_id_{};
  curt::StreamRef stream_;
  std::string nccl_path_;
  mutable std::vector<Subgroup> subgroups_;

 public:
  [[nodiscard]] ncclComm_t Handle() const { return nccl_comm_; }
  auto Stub() const { return stub_; }

  explicit NCCLComm(Context const* ctx, Comm const& root, std::shared_ptr<Coll> pimpl,
                    StringView nccl_path);
  [[nodiscard]] Result LogTracker(std::string) const override {
    LOG(FATAL) << "Device comm is used for logging.";
    return Fail("Undefined.");
  }
  ~NCCLComm() override;
  [[nodiscard]] bool IsFederated() const override { return false; }
  [[nodiscard]] curt::StreamRef Stream() const { return stream_; }
  [[nodiscard]] Result Block() const override {
    auto rc = this->Stream().Sync(false);
    return GetCUDAResult(rc);
  }
  [[nodiscard]] Result Shutdown() final {
    this->ResetState();
    return Success();
  }
  [[nodiscard]] std::size_t GetSubgroup(std::vector<std::int32_t> const& active_ranks) const;
  [[nodiscard]] Subgroup const& SubgroupAt(std::size_t idx) const { return subgroups_.at(idx); }
};

class NCCLChannel : public Channel {
  std::int32_t rank_{-1};
  ncclComm_t nccl_comm_{};
  std::shared_ptr<NcclStub> stub_;
  curt::StreamRef stream_;

 public:
  explicit NCCLChannel(Comm const& comm, std::int32_t rank, ncclComm_t nccl_comm,
                       std::shared_ptr<NcclStub> stub, curt::StreamRef stream)
      : rank_{rank},
        nccl_comm_{nccl_comm},
        stub_{std::move(stub)},
        Channel{comm, nullptr},
        stream_{std::move(stream)} {}

  [[nodiscard]] Result SendAll(std::int8_t const* ptr, std::size_t n) override {
    return stub_->Send(ptr, n, ncclInt8, rank_, nccl_comm_, stream_);
  }
  [[nodiscard]] Result RecvAll(std::int8_t* ptr, std::size_t n) override {
    return stub_->Recv(ptr, n, ncclInt8, rank_, nccl_comm_, stream_);
  }
  [[nodiscard]] Result Block() override {
    auto rc = stream_.Sync(false);
    return GetCUDAResult(rc);
  }
};

#endif  //  defined(XGBOOST_USE_NCCL)
}  // namespace xgboost::collective
