/**
 * Copyright 2024-2026, XGBoost Contributors
 */
#pragma once
#include <condition_variable>  // for condition_variable
#include <cstdint>             // for int32_t
#include <exception>           // for current_exception
#include <functional>          // for function
#include <future>              // for promise
#include <memory>              // for make_shared
#include <mutex>               // for mutex, unique_lock
#include <queue>               // for queue
#include <string>              // for string
#include <thread>              // for thread
#include <type_traits>         // for invoke_result_t
#include <utility>             // for move
#include <vector>              // for vector

#include "utils.h"                // for MakeCleanup
#include "xgboost/string_view.h"  // for StringView

namespace xgboost::common {
/**
 * @brief Simple implementation of a thread pool.
 */
class ThreadPool {
  std::string name_;
  std::function<void()> worker_fn_;
  mutable std::mutex grow_mu_;
  std::mutex mu_;
  std::queue<std::function<void()>> tasks_;
  std::condition_variable cv_;
  std::vector<std::thread> pool_;
  bool stop_{false};

 public:
  /**
   * @param name      Name prefix for threads.
   * @param n_threads The number of threads this pool should hold.
   * @param init_fn   Copyable function copied to and called once on each worker.
   */
  template <typename InitFn>
  explicit ThreadPool(StringView name, std::int32_t n_threads, InitFn&& init_fn)
      : name_{static_cast<std::string>(name)}, worker_fn_{[this, init_fn] {
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
        }} {
    this->EnsureCapacity(n_threads);
  }

  void EnsureCapacity(std::int32_t n_threads);

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
      try {
        if constexpr (std::is_void_v<R>) {
          fn();
          task->set_value();
        } else {
          task->set_value(fn());
        }
      } catch (...) {
        task->set_exception(std::current_exception());
      }
    }};

    std::unique_lock lock{mu_};
    this->tasks_.push(std::move(ffn));
    lock.unlock();

    cv_.notify_one();
    return fut;
  }

  [[nodiscard]] auto NumWorkers() const {
    std::unique_lock lock{grow_mu_};
    // Use the same type as the typical n_threads parameter in XGBoost.
    return static_cast<std::int32_t>(pool_.size());
  }
};

ThreadPool* GetGlobalThreadPool(std::int32_t n_threads);

template <typename Index, typename Func>
void GlobalThreadPoolExecutor(Index size, std::int32_t n_threads, Func&& fn) {
  auto& pool = *GetGlobalThreadPool(n_threads);
  std::vector<std::future<void>> futures;
  futures.reserve(n_threads);

  // The drain is used to handle submit failure.
  auto drain = common::MakeCleanup([&] {
    for (auto& future : futures) {
      if (future.valid()) {
        future.wait();
      }
    }
  });

  auto const block_size = DivRoundUp(size, static_cast<std::size_t>(n_threads));
  auto worker_fn = [&](Index block_id) {
    auto const begin = block_id * block_size;
    auto const end = std::min(begin + block_size, size);
    for (auto i = begin; i < end; ++i) {
      fn(i);
    }
  };
  for (std::int32_t t = 0; t < n_threads; ++t) {
    futures.emplace_back(pool.Submit(worker_fn));
  }
  // After submitting all tasks:
  for (auto& fut : futures) {
    fut.wait();  // Waits without rethrowing the task's exception.
  }
  for (auto& fut : futures) {
    fut.get();
  }
}
}  // namespace xgboost::common
