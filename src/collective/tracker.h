/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once
#include <chrono>   // for seconds
#include <cstdint>  // for int32_t
#include <future>   // for future
#include <string>   // for string
#include <utility>  // for pair
#include <vector>   // for vector

#include "protocol.h"
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/collective/socket.h"  // for TCPSocket
#include "xgboost/json.h"               // for Json

namespace xgboost::collective {
/**
 *
 * @brief Implementation of RABIT tracker.
 *
 * * What is a tracker
 *
 *   The implementation of collective follows what RABIT did in the past. It requires a
 *   tracker to coordinate initialization and error recovery of workers. While the
 *   original implementation attempted to attain error resislient inside the collective
 *   module, which turned out be too challenging due to large amount of external
 *   states. The new implementation here differs from RABIT in the way that neither state
 *   recovery nor resislient is handled inside the collective, it merely provides the
 *   mechanism to signal error to other workers through the use of a centralized tracker.
 *
 *   There are three major functionalities provided the a tracker, namely:
 *   - Initialization. Share the node addresses among all workers.
 *   - Logging.
 *   - Signal error. If an exception is thrown in one (or many) of the workers, it can
 *     signal an error to the tracker and the tracker will notify other workers.
 */
class Tracker {
 protected:
  std::int32_t n_workers_{0};
  std::int32_t port_{-1};
  std::chrono::seconds timeout_{0};

 public:
  explicit Tracker(Json const& config);
  Tracker(std::int32_t n_worders, std::int32_t port, std::chrono::seconds timeout)
      : n_workers_{n_worders}, port_{port}, timeout_{timeout} {}

  virtual ~Tracker() noexcept(false){};  // NOLINT
  [[nodiscard]] virtual std::future<Result> Run() = 0;
  [[nodiscard]] virtual Json WorkerArgs() const = 0;
  [[nodiscard]] std::chrono::seconds Timeout() const { return timeout_; }
};

class RabitTracker : public Tracker {
  // a wrapper for connected worker socket.
  class WorkerProxy {
    TCPSocket sock_;
    proto::PeerInfo info_;
    std::int32_t eport_{0};
    std::int32_t world_{-1};
    std::string task_id_;

    proto::CMD cmd_{proto::CMD::kInvalid};
    std::string msg_;
    std::int32_t code_{0};
    Result rc_;

   public:
    explicit WorkerProxy(std::int32_t world, TCPSocket sock, SockAddrV4 addr);
    WorkerProxy(WorkerProxy const& that) = delete;
    WorkerProxy(WorkerProxy&& that) = default;
    WorkerProxy& operator=(WorkerProxy const&) = delete;
    WorkerProxy& operator=(WorkerProxy&&) = default;

    [[nodiscard]] auto Host() const { return info_.host; }
    [[nodiscard]] auto TaskID() const { return task_id_; }
    [[nodiscard]] auto Port() const { return info_.port; }
    [[nodiscard]] auto Rank() const { return info_.rank; }
    [[nodiscard]] auto ErrorPort() const { return eport_; }
    [[nodiscard]] auto Command() const { return cmd_; }
    [[nodiscard]] auto Msg() const { return msg_; }
    [[nodiscard]] auto Code() const { return code_; }

    [[nodiscard]] Result const& Status() const { return rc_; }
    [[nodiscard]] Result& Status() { return rc_; }

    void Send(StringView value) { this->sock_.Send(value); }
  };
  // provide an ordering for workers, this helps us get deterministic topology.
  struct WorkerCmp {
    [[nodiscard]] bool operator()(WorkerProxy const& lhs, WorkerProxy const& rhs) {
      auto const& lh = lhs.Host();
      auto const& rh = rhs.Host();

      if (lh != rh) {
        return lh < rh;
      }
      return lhs.TaskID() < rhs.TaskID();
    }
  };

 private:
  std::string host_;
  // record for how to reach out to workers if error happens.
  std::vector<std::pair<std::string, std::int32_t>> worker_error_handles_;
  // listening socket for incoming workers.
  TCPSocket listener_;

  Result Bootstrap(std::vector<WorkerProxy>* p_workers);

 public:
  explicit RabitTracker(StringView host, std::int32_t n_worders, std::int32_t port,
                        std::chrono::seconds timeout)
      : Tracker{n_worders, port, timeout}, host_{host.c_str(), host.size()} {
    listener_ = TCPSocket::Create(SockDomain::kV4);
    auto rc = listener_.Bind(host, &this->port_);
    CHECK(rc.OK()) << rc.Report();
    listener_.Listen();
  }

  explicit RabitTracker(Json const& config);
  ~RabitTracker() noexcept(false) override = default;

  std::future<Result> Run() override;

  [[nodiscard]] std::int32_t Port() const { return port_; }
  [[nodiscard]] Json WorkerArgs() const override {
    Json args{Object{}};
    args["DMLC_TRACKER_URI"] = String{host_};
    args["DMLC_TRACKER_PORT"] = this->Port();
    return args;
  }
};

// Prob the public IP address of the host, need a better method.
//
// This is directly translated from the previous Python implementation, we should find a
// more riguous approach, can use some expertise in network programming.
[[nodiscard]] Result GetHostAddress(std::string* out);
}  // namespace xgboost::collective
