/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <chrono>   // for seconds
#include <cstdint>  // for int32_t
#include <string>   // for string
#include <thread>   // for thread
#include <vector>   // for vector

#include "../../../src/collective/comm.h"
#include "test_worker.h"

namespace xgboost::collective {
namespace {
class PrintWorker : public WorkerForTest {
 public:
  using WorkerForTest::WorkerForTest;

  void Print() {
    auto rc = comm_.LogTracker("ack:" + std::to_string(this->comm_.Rank()));
    ASSERT_TRUE(rc.OK()) << rc.Report();
  }
};
}  // namespace

TEST_F(TrackerTest, Bootstrap) {
  RabitTracker tracker{host, n_workers, 0, timeout};
  auto fut = tracker.Run();

  std::vector<std::thread> workers;
  std::int32_t port = tracker.Port();

  for (std::int32_t i = 0; i < n_workers; ++i) {
    workers.emplace_back([=] { WorkerForTest worker{host, port, timeout, n_workers, i}; });
  }
  for (auto &w : workers) {
    w.join();
  }

  ASSERT_TRUE(fut.get().OK());
}

TEST_F(TrackerTest, Print) {
  RabitTracker tracker{host, n_workers, 0, timeout};
  auto fut = tracker.Run();

  std::vector<std::thread> workers;
  std::int32_t port = tracker.Port();

  for (std::int32_t i = 0; i < n_workers; ++i) {
    workers.emplace_back([=] {
      PrintWorker worker{host, port, timeout, n_workers, i};
      worker.Print();
    });
  }

  for (auto &w : workers) {
    w.join();
  }

  ASSERT_TRUE(fut.get().OK());
}

TEST_F(TrackerTest, GetHostAddress) { ASSERT_TRUE(host.find("127.") == std::string::npos); }
}  // namespace xgboost::collective
