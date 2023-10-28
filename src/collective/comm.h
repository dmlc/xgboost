/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once
#include <chrono>   // for seconds
#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <memory>   // for shared_ptr
#include <string>   // for string
#include <thread>   // for thread
#include <utility>  // for move
#include <vector>   // for vector

#include "loop.h"                       // for Loop
#include "protocol.h"                   // for PeerInfo
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/collective/socket.h"  // for TCPSocket
#include "xgboost/context.h"            // for Context
#include "xgboost/span.h"               // for Span

namespace xgboost::collective {

inline constexpr std::int32_t DefaultTimeoutSec() { return 300; }  // 5min
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

class Channel;
class Coll;

/**
 * @brief Base communicator storing info about the tracker and other communicators.
 */
class Comm {
 protected:
  std::int32_t world_{-1};
  std::int32_t rank_{0};
  std::chrono::seconds timeout_{DefaultTimeoutSec()};
  std::int32_t retry_{DefaultRetry()};

  proto::PeerInfo tracker_;
  SockDomain domain_{SockDomain::kV4};
  std::thread error_worker_;
  std::string task_id_;
  std::vector<std::shared_ptr<Channel>> channels_;
  std::shared_ptr<Loop> loop_{new Loop{std::chrono::seconds{
      DefaultTimeoutSec()}}};  // fixme: require federated comm to have a timeout

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

  [[nodiscard]] auto Rank() const { return rank_; }
  [[nodiscard]] auto World() const { return IsDistributed() ? world_ : 1; }
  [[nodiscard]] bool IsDistributed() const { return world_ != -1; }
  void Submit(Loop::Op op) const { loop_->Submit(op); }
  [[nodiscard]] virtual Result Block() const { return loop_->Block(); }

  [[nodiscard]] virtual std::shared_ptr<Channel> Chan(std::int32_t rank) const {
    return channels_.at(rank);
  }
  [[nodiscard]] virtual bool IsFederated() const = 0;
  [[nodiscard]] virtual Result LogTracker(std::string msg) const = 0;

  [[nodiscard]] virtual Result SignalError(Result const&) { return Success(); }

  Comm* MakeCUDAVar(Context const* ctx, std::shared_ptr<Coll> pimpl);
};

class RabitComm : public Comm {
  [[nodiscard]] Result Bootstrap(std::chrono::seconds timeout, std::int32_t retry,
                                 std::string task_id);
  [[nodiscard]] Result Shutdown();

 public:
  // bootstrapping construction.
  RabitComm() = default;
  // ctor for testing where environment is known.
  RabitComm(std::string const& host, std::int32_t port, std::chrono::seconds timeout,
            std::int32_t retry, std::string task_id);
  ~RabitComm() noexcept(false) override;

  [[nodiscard]] bool IsFederated() const override { return false; }
  [[nodiscard]] Result LogTracker(std::string msg) const override;

  [[nodiscard]] Result SignalError(Result const&) override;
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

  virtual void SendAll(std::int8_t const* ptr, std::size_t n) {
    Loop::Op op{Loop::Op::kWrite, comm_.Rank(), const_cast<std::int8_t*>(ptr), n, sock_.get(), 0};
    CHECK(sock_.get());
    comm_.Submit(std::move(op));
  }
  void SendAll(common::Span<std::int8_t const> data) {
    this->SendAll(data.data(), data.size_bytes());
  }

  virtual void RecvAll(std::int8_t* ptr, std::size_t n) {
    Loop::Op op{Loop::Op::kRead, comm_.Rank(), ptr, n, sock_.get(), 0};
    CHECK(sock_.get());
    comm_.Submit(std::move(op));
  }
  void RecvAll(common::Span<std::int8_t> data) { this->RecvAll(data.data(), data.size_bytes()); }

  [[nodiscard]] auto Socket() const { return sock_; }
  [[nodiscard]] virtual Result Block() { return comm_.Block(); }
};

enum class Op { kMax = 0, kMin = 1, kSum = 2, kBitwiseAND = 3, kBitwiseOR = 4, kBitwiseXOR = 5 };
}  // namespace xgboost::collective
