#include <gtest/gtest.h>

#include <future>
#include <queue>

namespace xgboost::common {
class ThreadPool {
  std::mutex mu_;
  std::queue<std::function<void()>> tasks_;
  std::condition_variable cv_;
  std::vector<std::thread> pool_;

 public:
  explicit ThreadPool(std::int32_t n_threads) {
    for (std::int32_t i = 0; i < n_threads; ++i) {
      pool_.emplace_back([&] {
        std::unique_lock lock{mu_};
        cv_.wait(lock, [this] { return !this->pool_.empty(); });
      });
    }
  }
  template <typename Fn>
  auto Submit(Fn&& fn) {
    std::shared_ptr<std::promise<int>> task{std::make_shared<std::promise<int>>()};
    auto fut = task->get_future();
    auto ffn = std::function{[task = std::move(task), fn = std::move(fn)]() mutable {
      auto v = fn();
      task->set_value(v);
    }};
    tasks_.push(std::move(ffn));

    cv_.notify_one();

    return fut;
  }
};

TEST(ThreadPool, Basic) {
  ThreadPool pool{3};
  pool.Submit([] {
    std::cout << "hello" << std::endl;
    return 3;
  });
}
}  // namespace xgboost::common
