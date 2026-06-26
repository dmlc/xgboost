/**
 * Copyright 2023-2026, XGBoost Contributors
 */
#pragma once

#ifdef XGBOOST_USE_NCCL
#include "nccl.h"
#endif  // XGBOOST_USE_NCCL

#include <cstdint>      // for int32_t
#include <memory>       // for shared_ptr
#include <type_traits>  // for enable_if_t, invoke_result_t, is_same_v
#include <utility>      // for move, forward

#include "../common/cuda_stream.h"  // for StreamRef, Stream, Event
#include "../common/utils.h"        // for MakeCleanup
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
template <typename Fn>
[[nodiscard]] std::enable_if_t<std::is_same_v<std::invoke_result_t<Fn>, Result>, Result>
BracketNccl(curt::StreamRef user_stream, curt::StreamRef nccl_stream, Fn&& fn) {
  curt::Event before;
  before.Record(user_stream);
  nccl_stream.Wait(before);

  auto after = common::MakeCleanup([&] {
    curt::Event ev;
    ev.Record(nccl_stream);
    user_stream.Wait(ev);
  });

  return std::forward<Fn>(fn)();
}
#endif  // defined(XGBOOST_USE_NCCL)

#if defined(XGBOOST_USE_NCCL)
class NCCLComm : public Comm {
 private:
  // Declared first so among this class's own members it is destroyed last
  curt::Stream stream_;
  std::shared_ptr<NcclStub> stub_;
  ncclComm_t nccl_comm_{nullptr};
  ncclUniqueId nccl_unique_id_{};
  std::string nccl_path_;

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
  [[nodiscard]] curt::StreamRef Stream() const { return stream_.View(); }
  [[nodiscard]] Result Block() const override {
    auto rc = this->Stream().Sync(false);
    return GetCUDAResult(rc);
  }
  [[nodiscard]] Result Shutdown() final {
    this->ResetState();
    return Success();
  }
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
