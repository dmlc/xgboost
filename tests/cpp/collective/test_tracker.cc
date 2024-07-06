/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>   // for seconds
#include <cstdint>  // for int32_t
#include <string>   // for string
#include <thread>   // for thread
#include <vector>   // for vector

#include "../../../src/collective/comm.h"
#include "../helpers.h"  // for GMockThrow
#include "test_worker.h"

namespace xgboost::collective {
namespace {
class PrintWorker : public WorkerForTest {
 public:
  using WorkerForTest::WorkerForTest;

  void Print() {
    auto rc = comm_.LogTracker("ack:" + std::to_string(this->comm_.Rank()));
    SafeColl(rc);
  }
};
}  // namespace

TEST_F(TrackerTest, Bootstrap) {
  RabitTracker tracker{MakeTrackerConfig(host, n_workers, timeout)};
  ASSERT_TRUE(HasTimeout(tracker.Timeout()));
  ASSERT_FALSE(tracker.Ready());
  auto fut = tracker.Run();

  std::vector<std::thread> workers;

  auto args = tracker.WorkerArgs();
  ASSERT_TRUE(tracker.Ready());
  ASSERT_EQ(get<String const>(args["dmlc_tracker_uri"]), host);

  std::int32_t port = tracker.Port();

  for (std::int32_t i = 0; i < n_workers; ++i) {
    workers.emplace_back([=] { WorkerForTest worker{host, port, timeout, n_workers, i}; });
  }
  for (auto &w : workers) {
    w.join();
  }
  SafeColl(fut.get());

  ASSERT_FALSE(HasTimeout(std::chrono::seconds{-1}));
  ASSERT_FALSE(HasTimeout(std::chrono::seconds{0}));
}

TEST_F(TrackerTest, Print) {
  RabitTracker tracker{MakeTrackerConfig(host, n_workers, timeout)};
  auto fut = tracker.Run();

  std::vector<std::thread> workers;
  auto rc = tracker.WaitUntilReady();
  SafeColl(rc);

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

  SafeColl(fut.get());
}

TEST_F(TrackerTest, GetHostAddress) { ASSERT_TRUE(host.find("127.") == std::string::npos); }

/**
 * Test connecting the tracker after it has finished. This should not hang the workers.
 */
TEST_F(TrackerTest, AfterShutdown) {
  RabitTracker tracker{MakeTrackerConfig(host, n_workers, timeout)};
  auto fut = tracker.Run();

  std::vector<std::thread> workers;
  auto rc = tracker.WaitUntilReady();
  SafeColl(rc);

  std::int32_t port = tracker.Port();

  // Launch no-op workers to cause the tracker to shutdown.
  for (std::int32_t i = 0; i < n_workers; ++i) {
    workers.emplace_back([=] { WorkerForTest worker{host, port, timeout, n_workers, i}; });
  }

  for (auto &w : workers) {
    w.join();
  }

  SafeColl(fut.get());

  // Launch workers again, they should fail.
  workers.clear();
  for (std::int32_t i = 0; i < n_workers; ++i) {
    auto assert_that = [=] {
      WorkerForTest worker{host, port, timeout, n_workers, i};
    };
    // On a Linux platform, the connection will be refused, on Apple platform, this gets
    // an operation now in progress poll failure, on Windows, it's a timeout error.
#if defined(__linux__)
    workers.emplace_back([=] { ASSERT_THAT(assert_that, GMockThrow("Connection refused")); });
#else
    workers.emplace_back([=] { ASSERT_THAT(assert_that, GMockThrow("Failed to connect to")); });
#endif
  }
  for (auto &w : workers) {
    w.join();
  }
}
}  // namespace xgboost::collective
