/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/collective/socket.h>

#include <cstdint>  // for int32_t
#include <string>   // for string
#include <thread>   // for thread
#include <vector>   // for vector

#include "../../../src/collective/broadcast.h"  // for Broadcast
#include "test_worker.h"                        // for WorkerForTest, TestDistributed

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
      SafeColl(rc);
      ASSERT_EQ(data[0], r);
    }

    for (std::int32_t r = 0; r < comm_.World(); ++r) {
      std::vector<std::int32_t> data(1 << 16, comm_.Rank());
      auto rc = Broadcast(this->comm_, common::Span{data.data(), data.size()}, r);
      SafeColl(rc);
      ASSERT_EQ(data[0], r);
    }
  }
};

class BroadcastTest : public SocketTest {};
}  // namespace

TEST_F(BroadcastTest, Basic) {
  std::int32_t n_workers = std::min(2u, std::thread::hardware_concurrency());
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    Worker worker{host, port, timeout, n_workers, r};
    worker.Run();
  });
}
}  // namespace xgboost::collective
