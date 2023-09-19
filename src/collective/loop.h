/**
 * Copyright 2023, XGBoost Contributors
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
#include <utility>             // for move

#include "../common/timer.h"            // for Monitor
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/collective/socket.h"  // for TCPSocket

namespace xgboost::collective {
class Loop {
 public:
  struct Op {
    enum Code : std::int8_t { kRead = 0, kWrite = 1 } code;
    std::int32_t rank{-1};
    std::int8_t* ptr{nullptr};
    std::size_t n{0};
    TCPSocket* sock{nullptr};
    std::size_t off{0};

    Op(Code c, std::int32_t rank, std::int8_t* ptr, std::size_t n, TCPSocket* sock, std::size_t off)
        : code{c}, rank{rank}, ptr{ptr}, n{n}, sock{sock}, off{off} {}
    Op(Op const&) = default;
    Op& operator=(Op const&) = default;
    Op(Op&&) = default;
    Op& operator=(Op&&) = default;
  };

 private:
  std::thread worker_;
  std::condition_variable cv_;
  std::mutex mu_;
  std::queue<Op> queue_;
  std::chrono::seconds timeout_;
  Result rc_;
  bool stop_{false};
  std::exception_ptr curr_exce_{nullptr};
  common::Monitor timer_;

  Result EmptyQueue();
  void Process();

 public:
  Result Stop();

  void Submit(Op op) {
    // producer
    std::unique_lock lock{mu_};
    queue_.push(op);
    lock.unlock();
    cv_.notify_one();
  }

  [[nodiscard]] Result Block() {
    {
      std::unique_lock lock{mu_};
      cv_.notify_all();
    }
    std::unique_lock lock{mu_};
    cv_.wait(lock, [this] { return this->queue_.empty() || stop_; });
    return std::move(rc_);
  }

  explicit Loop(std::chrono::seconds timeout);

  ~Loop() noexcept(false) {
    this->Stop();

    if (worker_.joinable()) {
      worker_.join();
    }
  }
};
}  // namespace xgboost::collective
