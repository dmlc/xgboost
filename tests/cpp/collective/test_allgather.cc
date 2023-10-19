/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>   // for ASSERT_EQ
#include <xgboost/span.h>  // for Span, oper...

#include <algorithm>  // for min
#include <chrono>     // for seconds
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t
#include <numeric>    // for iota
#include <string>     // for string
#include <thread>     // for thread
#include <vector>     // for vector

#include "../../../src/collective/allgather.h"  // for RingAllgather
#include "../../../src/collective/comm.h"       // for RabitComm
#include "gtest/gtest.h"                        // for AssertionR...
#include "test_worker.h"                        // for TestDistri...
#include "xgboost/collective/result.h"          // for Result

namespace xgboost::collective {
namespace {
class AllgatherTest : public TrackerTest {};

class Worker : public WorkerForTest {
 public:
  using WorkerForTest::WorkerForTest;

  void Run() {
    {
      // basic test
      std::vector<std::int32_t> data(comm_.World(), 0);
      data[comm_.Rank()] = comm_.Rank();

      auto rc = RingAllgather(this->comm_, common::Span{data.data(), data.size()}, 1);
      ASSERT_TRUE(rc.OK()) << rc.Report();

      for (std::int32_t r = 0; r < comm_.World(); ++r) {
        ASSERT_EQ(data[r], r);
      }
    }
    {
      // test for limited socket buffer
      this->LimitSockBuf(4096);

      std::size_t n = 8192;  // n_bytes = 8192 * sizeof(int)
      std::vector<std::int32_t> data(comm_.World() * n, 0);
      auto s_data = common::Span{data.data(), data.size()};
      auto seg = s_data.subspan(comm_.Rank() * n, n);
      std::iota(seg.begin(), seg.end(), comm_.Rank());

      auto rc = RingAllgather(comm_, common::Span{data.data(), data.size()}, n);
      ASSERT_TRUE(rc.OK()) << rc.Report();

      for (std::int32_t r = 0; r < comm_.World(); ++r) {
        auto seg = s_data.subspan(r * n, n);
        for (std::int32_t i = 0; i < static_cast<std::int32_t>(seg.size()); ++i) {
          auto v = seg[i];
          ASSERT_EQ(v, r + i);
        }
      }
    }
  }

  void TestV() {
    {
      // basic test
      std::int32_t n{comm_.Rank()};
      std::vector<std::int32_t> result;
      auto rc = RingAllgatherV(comm_, common::Span{&n, 1}, &result);
      ASSERT_TRUE(rc.OK()) << rc.Report();
      for (std::int32_t i = 0; i < comm_.World(); ++i) {
        ASSERT_EQ(result[i], i);
      }
    }

    {
      // V test
      std::vector<std::int32_t> data(comm_.Rank() + 1, comm_.Rank());
      std::vector<std::int32_t> result;
      auto rc = RingAllgatherV(comm_, common::Span{data.data(), data.size()}, &result);
      ASSERT_TRUE(rc.OK()) << rc.Report();
      ASSERT_EQ(result.size(), (1 + comm_.World()) * comm_.World() / 2);
      std::int32_t k{0};
      for (std::int32_t r = 0; r < comm_.World(); ++r) {
        auto seg = common::Span{result.data(), result.size()}.subspan(k, (r + 1));
        if (comm_.Rank() == 0) {
          for (auto v : seg) {
            ASSERT_EQ(v, r);
          }
          k += seg.size();
        }
      }
    }
  }
};
}  // namespace

TEST_F(AllgatherTest, Basic) {
  std::int32_t n_workers = std::min(7u, std::thread::hardware_concurrency());
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    Worker worker{host, port, timeout, n_workers, r};
    worker.Run();
  });
}

TEST_F(AllgatherTest, V) {
  std::int32_t n_workers = std::min(7u, std::thread::hardware_concurrency());
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    Worker worker{host, port, timeout, n_workers, r};
    worker.TestV();
  });
}
}  // namespace xgboost::collective
