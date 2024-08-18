/**
 * Copyright 2024, XGBoost Contributors
 */
#pragma once
#include <condition_variable>  // for condition_variable
#include <cstdint>             // for int32_t
#include <functional>          // for function
#include <future>              // for promise
#include <memory>              // for make_shared
#include <mutex>               // for mutex, unique_lock
#include <queue>               // for queue
#include <thread>              // for thread
#include <type_traits>         // for invoke_result_t
#include <utility>             // for move
#include <vector>              // for vector

namespace xgboost::common {
/**
 * @brief Simple implementation of a thread pool.
 */
class ThreadPool {
  std::mutex mu_;
  std::queue<std::function<void()>> tasks_;
  std::condition_variable cv_;
  std::vector<std::thread> pool_;
  bool stop_{false};

 public:
  /**
   * @param n_threads The number of threads this pool should hold.
   * @param init_fn   Function called once during thread creation.
   */
  template <typename InitFn>
  explicit ThreadPool(std::int32_t n_threads, InitFn&& init_fn) {
    for (std::int32_t i = 0; i < n_threads; ++i) {
      pool_.emplace_back([&, init_fn = std::forward<InitFn>(init_fn)] {
        init_fn();

        while (true) {
          std::unique_lock lock{mu_};
          cv_.wait(lock, [this] { return !this->tasks_.empty() || stop_; });

          if (this->stop_) {
            while (!tasks_.empty()) {
              auto fn = tasks_.front();
              tasks_.pop();
              fn();
            }
            return;
          }

          auto fn = tasks_.front();
          tasks_.pop();
          lock.unlock();
          fn();
        }
      });
    }
  }

  ~ThreadPool() {
    std::unique_lock lock{mu_};
    stop_ = true;
    lock.unlock();

    for (auto& t : pool_) {
      if (t.joinable()) {
        std::unique_lock lock{mu_};
        this->cv_.notify_one();
        lock.unlock();
      }
    }

    for (auto& t : pool_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  /**
   * @brief Submit a function that doesn't take any argument.
   */
  template <typename Fn, typename R = std::invoke_result_t<Fn>>
  auto Submit(Fn&& fn) {
    // Use shared ptr to make the task copy constructible.
    auto p{std::make_shared<std::promise<R>>()};
    auto fut = p->get_future();
    auto ffn = std::function{[task = std::move(p), fn = std::forward<Fn>(fn)]() mutable {
      if constexpr (std::is_void_v<R>) {
        fn();
        task->set_value();
      } else {
        task->set_value(fn());
      }
    }};

    std::unique_lock lock{mu_};
    this->tasks_.push(std::move(ffn));
    lock.unlock();

    cv_.notify_one();
    return fut;
  }
};
}  // namespace xgboost::common
