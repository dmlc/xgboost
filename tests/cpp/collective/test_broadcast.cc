/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/collective/socket.h>

#include <cstdint>  // for int32_t
#include <string>   // for string
#include <thread>   // for thread
#include <vector>   // for vector

#include "../../../src/collective/broadcast.h"  // for Broadcast
#include "../../../src/collective/tracker.h"    // for GetHostAddress, Tracker
#include "test_worker.h"                        // for WorkerForTest

namespace xgboost::collective {
namespace {
class Worker : public WorkerForTest {
 public:
  using WorkerForTest::WorkerForTest;

  void Run() {
    for (std::int32_t r = 0; r < comm_.World(); ++r) {
      // basic test
      std::vector<std::int32_t> data(1, comm_.Rank());
      auto rc = Broadcast(this->comm_, common::Span{data.data(), data.size()}, r);
      ASSERT_TRUE(rc.OK()) << rc.Report();
      ASSERT_EQ(data[0], r);
    }

    for (std::int32_t r = 0; r < comm_.World(); ++r) {
      std::vector<std::int32_t> data(1 << 16, comm_.Rank());
      auto rc = Broadcast(this->comm_, common::Span{data.data(), data.size()}, r);
      ASSERT_TRUE(rc.OK()) << rc.Report();
      ASSERT_EQ(data[0], r);
    }
  }
};

class BroadcastTest : public SocketTest {};
}  // namespace

TEST_F(BroadcastTest, Basic) {
  std::int32_t n_workers = std::min(24u, std::thread::hardware_concurrency());
  std::chrono::seconds timeout{3};

  std::string host;
  ASSERT_TRUE(GetHostAddress(&host).OK());
  RabitTracker tracker{StringView{host}, n_workers, 0, timeout};
  auto fut = tracker.Run();

  std::vector<std::thread> workers;
  std::int32_t port = tracker.Port();

  for (std::int32_t i = 0; i < n_workers; ++i) {
    workers.emplace_back([=] {
      Worker worker{host, port, timeout, n_workers, i};
      worker.Run();
    });
  }

  for (auto& t : workers) {
    t.join();
  }

  ASSERT_TRUE(fut.get().OK());
}
}  // namespace xgboost::collective
