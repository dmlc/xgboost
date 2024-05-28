/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#pragma once
#include <chrono>   // for seconds
#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t, int64_t
#include <memory>   // for shared_ptr
#include <string>   // for string
#include <thread>   // for thread
#include <utility>  // for move
#include <vector>   // for vector

#include "loop.h"                       // for Loop
#include "protocol.h"                   // for PeerInfo
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/collective/socket.h"  // for TCPSocket, GetHostName
#include "xgboost/context.h"            // for Context
#include "xgboost/span.h"               // for Span

namespace xgboost::collective {

inline constexpr std::int64_t DefaultTimeoutSec() { return 60 * 30; }  // 30min
inline constexpr std::int32_t DefaultRetry() { return 3; }

// indexing into the ring
inline std::int32_t BootstrapNext(std::int32_t r, std::int32_t world) {
  auto nrank = (r + world + 1) % world;
  return nrank;
}

inline std::int32_t BootstrapPrev(std::int32_t r, std::int32_t world) {
  auto nrank = (r + world - 1) % world;
  return nrank;
}

inline StringView DefaultNcclName() { return "libnccl.so.2"; }

class Channel;
class Coll;

/**
 * @brief Base communicator storing info about the tracker and other communicators.
 */
class Comm : public std::enable_shared_from_this<Comm> {
 protected:
  std::int32_t world_{-1};
  std::int32_t rank_{0};
  std::chrono::seconds timeout_{DefaultTimeoutSec()};
  std::int32_t retry_{DefaultRetry()};

  proto::PeerInfo tracker_;
  SockDomain domain_{SockDomain::kV4};

  std::thread error_worker_;
  std::int32_t error_port_;

  std::string task_id_;
  std::vector<std::shared_ptr<Channel>> channels_;
  std::shared_ptr<Loop> loop_{nullptr};  // fixme: require federated comm to have a timeout

  void ResetState() {
    this->world_ = -1;
    this->rank_ = 0;
    this->timeout_ = std::chrono::seconds{DefaultTimeoutSec()};

    tracker_ = proto::PeerInfo{};
    this->task_id_.clear();
    channels_.clear();

    loop_.reset();
  }

 public:
  Comm() = default;
  Comm(std::string const& host, std::int32_t port, std::chrono::seconds timeout, std::int32_t retry,
       std::string task_id);
  virtual ~Comm() noexcept(false) {}  // NOLINT

  Comm(Comm const& that) = delete;
  Comm& operator=(Comm const& that) = delete;
  Comm(Comm&& that) = delete;
  Comm& operator=(Comm&& that) = delete;

  [[nodiscard]] auto TrackerInfo() const { return tracker_; }
  [[nodiscard]] Result ConnectTracker(TCPSocket* out) const;
  [[nodiscard]] auto Domain() const { return domain_; }
  [[nodiscard]] auto Timeout() const { return timeout_; }
  [[nodiscard]] auto Retry() const { return retry_; }
  [[nodiscard]] auto TaskID() const { return task_id_; }

  [[nodiscard]] auto Rank() const noexcept { return rank_; }
  [[nodiscard]] auto World() const noexcept { return IsDistributed() ? world_ : 1; }
  [[nodiscard]] bool IsDistributed() const noexcept { return world_ != -1; }
  void Submit(Loop::Op op) const {
    CHECK(loop_);
    loop_->Submit(std::move(op));
  }
  [[nodiscard]] virtual Result Block() const { return loop_->Block(); }

  [[nodiscard]] virtual std::shared_ptr<Channel> Chan(std::int32_t rank) const {
    return channels_.at(rank);
  }
  [[nodiscard]] virtual bool IsFederated() const = 0;
  [[nodiscard]] virtual Result LogTracker(std::string msg) const = 0;

  [[nodiscard]] virtual Result SignalError(Result const&) { return Success(); }
  /**
   * @brief Get a string ID for the current process.
   */
  [[nodiscard]] virtual Result ProcessorName(std::string* out) const {
    auto rc = GetHostName(out);
    return rc;
  }
  [[nodiscard]] virtual Result Shutdown() = 0;
};

/**
 * @brief Base class for CPU-based communicator.
 */
class HostComm : public Comm {
 public:
  using Comm::Comm;
  [[nodiscard]] virtual Comm* MakeCUDAVar(Context const* ctx,
                                          std::shared_ptr<Coll> pimpl) const = 0;
};

class RabitComm : public HostComm {
  std::string nccl_path_ = std::string{DefaultNcclName()};

  [[nodiscard]] Result Bootstrap(std::chrono::seconds timeout, std::int32_t retry,
                                 std::string task_id);

 public:
  // bootstrapping construction.
  RabitComm() = default;
  RabitComm(std::string const& tracker_host, std::int32_t tracker_port,
            std::chrono::seconds timeout, std::int32_t retry, std::string task_id,
            StringView nccl_path);
  ~RabitComm() noexcept(false) override;

  [[nodiscard]] bool IsFederated() const override { return false; }
  [[nodiscard]] Result LogTracker(std::string msg) const override;

  [[nodiscard]] Result SignalError(Result const&) override;
  [[nodiscard]] Result Shutdown() final;

  [[nodiscard]] Comm* MakeCUDAVar(Context const* ctx, std::shared_ptr<Coll> pimpl) const override;
};

/**
 * @brief Communication channel between workers.
 */
class Channel {
  std::shared_ptr<TCPSocket> sock_{nullptr};
  Result rc_;
  Comm const& comm_;

 public:
  explicit Channel(Comm const& comm, std::shared_ptr<TCPSocket> sock)
      : sock_{std::move(sock)}, comm_{comm} {}

  [[nodiscard]] virtual Result SendAll(std::int8_t const* ptr, std::size_t n) {
    Loop::Op op{Loop::Op::kWrite, comm_.Rank(), const_cast<std::int8_t*>(ptr), n, sock_.get(), 0};
    CHECK(sock_.get());
    comm_.Submit(std::move(op));
    return Success();
  }
  [[nodiscard]] Result SendAll(common::Span<std::int8_t const> data) {
    return this->SendAll(data.data(), data.size_bytes());
  }

  [[nodiscard]] virtual Result RecvAll(std::int8_t* ptr, std::size_t n) {
    Loop::Op op{Loop::Op::kRead, comm_.Rank(), ptr, n, sock_.get(), 0};
    CHECK(sock_.get());
    comm_.Submit(std::move(op));
    return Success();
  }
  [[nodiscard]] Result RecvAll(common::Span<std::int8_t> data) {
    return this->RecvAll(data.data(), data.size_bytes());
  }

  [[nodiscard]] auto Socket() const { return sock_; }
  [[nodiscard]] virtual Result Block() { return comm_.Block(); }
};

enum class Op { kMax = 0, kMin = 1, kSum = 2, kBitwiseAND = 3, kBitwiseOR = 4, kBitwiseXOR = 5 };
}  // namespace xgboost::collective
