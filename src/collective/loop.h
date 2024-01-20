/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#pragma once
#include <chrono>              // for seconds
#include <condition_variable>  // for condition_variable
#include <cstddef>             // for size_t
#include <cstdint>             // for int8_t, int32_t
#include <exception>           // for exception_ptr
#include <mutex>               // for unique_lock, mutex
#include <queue>               // for queue
#include <thread>              // for thread

#include "../common/timer.h"            // for Monitor
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/collective/socket.h"  // for TCPSocket

namespace xgboost::collective {
class Loop {
 public:
  struct Op {
    enum Code : std::int8_t { kRead = 0, kWrite = 1, kBlock = 2 } code;
    std::int32_t rank{-1};
    std::int8_t* ptr{nullptr};
    std::size_t n{0};
    TCPSocket* sock{nullptr};
    std::size_t off{0};

    explicit Op(Code c) : code{c} { CHECK(c == kBlock); }
    Op(Code c, std::int32_t rank, std::int8_t* ptr, std::size_t n, TCPSocket* sock, std::size_t off)
        : code{c}, rank{rank}, ptr{ptr}, n{n}, sock{sock}, off{off} {}
    Op(Op const&) = default;
    Op& operator=(Op const&) = default;
    Op(Op&&) = default;
    Op& operator=(Op&&) = default;
  };

 private:
  std::thread worker_;  // thread worker to execute the tasks

  std::condition_variable cv_;        // CV used to notify a new submit call
  std::condition_variable block_cv_;  // CV used to notify the blocking call
  bool block_done_{false};            // Flag to indicate whether the blocking call has finished.

  std::queue<Op> queue_;  // event queue
  std::mutex mu_;         // mutex to protect the queue, cv, and block_done

  std::chrono::seconds timeout_;

  Result rc_;
  std::mutex rc_lock_;  // lock for transferring error info.

  bool stop_{false};
  std::exception_ptr curr_exce_{nullptr};
  common::Monitor mutable timer_;

  Result EmptyQueue(std::queue<Op>* p_queue) const;
  // The cunsumer function that runs inside a worker thread.
  void Process();

 public:
  /**
   * @brief Stop the worker thread.
   */
  Result Stop();

  void Submit(Op op) {
    std::unique_lock lock{mu_};
    queue_.push(op);
    lock.unlock();
    cv_.notify_one();
  }

  /**
   * @brief Block the event loop until all ops are finished. In the case of failure, this
   *        loop should be not be used for new operations.
   */
  [[nodiscard]] Result Block();

  explicit Loop(std::chrono::seconds timeout);

  ~Loop() noexcept(false) {
    // The worker will be joined in the stop function.
    this->Stop();
  }
};
}  // namespace xgboost::collective
