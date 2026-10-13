/**
 * Copyright 2024-2026, XGBoost Contributors
 */
#pragma once
#include <algorithm>           // for max, min
#include <condition_variable>  // for condition_variable
#include <cstddef>             // for size_t
#include <cstdint>             // for int32_t
#include <exception>           // for current_exception
#include <functional>          // for function
#include <future>              // for future, promise
#include <memory>              // for make_shared
#include <mutex>               // for mutex, unique_lock
#include <queue>               // for queue
#include <string>              // for string
#include <thread>              // for thread
#include <type_traits>         // for invoke_result_t, is_void_v
#include <utility>             // for forward, move
#include <vector>              // for vector

#include "executor.h"             // for Sched
#include "utils.h"                // for MakeCleanup
#include "xgboost/logging.h"      // for CHECK_GE
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
              // Drain
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

struct GlobalThreadPoolExecutor {
 private:
  inline static thread_local bool in_parallel_{false};

 public:
  [[nodiscard]] static std::int32_t MaxNumWorkers() noexcept;

  // Scheduling hints currently use contiguous blocks in the pool executor.
  template <typename Index, typename Func>
  static void Execute(Index size, std::int32_t n_threads, Sched, Func&& fn) {
    CHECK_GE(n_threads, 1);
    if (size <= 0) {
      return;
    }
    n_threads = in_parallel_ ? 1 : std::min<std::size_t>(n_threads, size);
    auto const length = static_cast<std::size_t>(size);
    auto const block_size = length / n_threads + (length % n_threads != 0);
    auto run = [&](std::int32_t t) {
      auto&& callback = detail::BindWorker(fn, [t] { return Worker{t}; });
      auto const begin = t * block_size;
      auto const end = std::min(begin + block_size, length);
      for (auto i = begin; i < end; ++i) {
        callback(static_cast<Index>(i));
      }
    };
    if (n_threads == 1) {
      run(0);
      return;
    }
    auto& pool = *GetGlobalThreadPool(n_threads);
    std::vector<std::future<void>> futures;
    futures.reserve(n_threads);

    // Keep borrowed callback state alive on either submission or task failure.
    auto drain = common::MakeCleanup([&] {
      for (auto& future : futures) {
        if (future.valid()) {
          future.wait();
        }
      }
    });

    for (std::int32_t t = 0; t < n_threads; ++t) {
      futures.emplace_back(pool.Submit([&, t] {
        in_parallel_ = true;
        auto reset = common::MakeCleanup([] { in_parallel_ = false; });
        run(t);
      }));
    }
    for (auto& fut : futures) {
      fut.get();
    }
  }

  template <typename Space, typename Func>
  static void Execute2d(Space const& space, std::int32_t n_threads, Func&& fn) {
    Execute(space.Size(), n_threads, Sched::Static(),
            detail::TransformCallback(fn, [&](auto& callback, auto i) {
              callback(space.GetFirstDimension(i), space.GetRange(i));
            }));
  }
};
}  // namespace xgboost::common
