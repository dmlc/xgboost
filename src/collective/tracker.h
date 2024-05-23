/**
 * Copyright 2023-2024, XGBoost Contributors
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
inline bool HasTimeout(std::chrono::seconds timeout) { return timeout.count() > 0; }
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
 public:
  enum class SortBy : std::int8_t {
    kHost = 0,
    kTask = 1,
  };

 protected:
  // How to sort the workers, either by host name or by task ID. When using a multi-GPU
  // setting, multiple workers can occupy the same host, in which case one should sort
  // workers by task. Due to compatibility reason, the task ID is not always available, so
  // we use host as the default.
  SortBy sortby_;

 protected:
  std::int32_t n_workers_{0};
  std::int32_t port_{-1};
  std::chrono::seconds timeout_{-1};
  std::atomic<bool> ready_{false};

 public:
  explicit Tracker(Json const& config);
  virtual ~Tracker() = default;

  [[nodiscard]] Result WaitUntilReady() const;

  [[nodiscard]] virtual std::future<Result> Run() = 0;
  [[nodiscard]] virtual Json WorkerArgs() const = 0;
  [[nodiscard]] std::chrono::seconds Timeout() const { return timeout_; }
  [[nodiscard]] virtual std::int32_t Port() const { return port_; }
  /**
   * @brief Flag to indicate whether the server is running.
   */
  [[nodiscard]] bool Ready() const { return ready_; }
  /**
   * @brief Shutdown the tracker, cannot be restarted again. Useful when the tracker hangs while
   *        calling accept.
   */
  virtual Result Stop() { return Success(); }
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
    explicit WorkerProxy(std::int32_t world, TCPSocket sock, SockAddress addr);
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
  // Provide an ordering for workers, this helps us get deterministic topology.
  struct WorkerCmp {
    SortBy sortby;
    explicit WorkerCmp(SortBy sortby) : sortby{sortby} {}

    [[nodiscard]] bool operator()(WorkerProxy const& lhs, WorkerProxy const& rhs) {
      auto const& lh = sortby == Tracker::SortBy::kHost ? lhs.Host() : lhs.TaskID();
      auto const& rh = sortby == Tracker::SortBy::kHost ? rhs.Host() : rhs.TaskID();

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
  // mutex for protecting the listener, used to prevent race when it's listening while
  // another thread tries to shut it down.
  std::mutex listener_mu_;

  Result Bootstrap(std::vector<WorkerProxy>* p_workers);

 public:
  explicit RabitTracker(Json const& config);
  ~RabitTracker() override = default;

  std::future<Result> Run() override;
  [[nodiscard]] Json WorkerArgs() const override;
  // Stop the tracker without waiting. This is to prevent the tracker from hanging when
  // one of the workers failes to start.
  [[nodiscard]] Result Stop() override;
};

// Prob the public IP address of the host, need a better method.
//
// This is directly translated from the previous Python implementation, we should find a
// more riguous approach, can use some expertise in network programming.
[[nodiscard]] Result GetHostAddress(std::string* out);
}  // namespace xgboost::collective
