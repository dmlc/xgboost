/**
 * Copyright 2023-2024, XGBoost Contributors
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
#include "../../../src/collective/coll.h"       // for Coll
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

      auto rc = RingAllgather(this->comm_, common::Span{data.data(), data.size()});
      SafeColl(rc);

      for (std::int32_t r = 0; r < comm_.World(); ++r) {
        ASSERT_EQ(data[r], r);
      }
    }
    {
      // test for limited socket buffer
      this->LimitSockBuf(4096);

      std::size_t n = 8192;  // n_bytes = 8192 * sizeof(int)
      std::vector<std::int32_t> data(comm_.World() * n, 0);
      auto s_data = common::Span<std::int32_t>{data};
      auto seg = s_data.subspan(comm_.Rank() * n, n);
      std::iota(seg.begin(), seg.end(), comm_.Rank());

      auto rc = RingAllgather(comm_, common::Span{data.data(), data.size()});
      SafeColl(rc);

      for (std::int32_t r = 0; r < comm_.World(); ++r) {
        auto seg = s_data.subspan(r * n, n);
        for (std::int32_t i = 0; i < static_cast<std::int32_t>(seg.size()); ++i) {
          auto v = seg[i];
          ASSERT_EQ(v, r + i);
        }
      }
    }
  }

  void CheckV(common::Span<std::int32_t> result) {
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
  void TestVRing() {
    // V test
    std::vector<std::int32_t> data(comm_.Rank() + 1, comm_.Rank());
    std::vector<std::int32_t> result;
    auto rc = RingAllgatherV(comm_, common::Span{data.data(), data.size()}, &result);
    SafeColl(rc);
    ASSERT_EQ(result.size(), (1 + comm_.World()) * comm_.World() / 2);
    CheckV(result);
  }

  void TestVBasic() {
    // basic test
    std::int32_t n{comm_.Rank()};
    std::vector<std::int32_t> result;
    auto rc = RingAllgatherV(comm_, common::Span{&n, 1}, &result);
    SafeColl(rc);
    for (std::int32_t i = 0; i < comm_.World(); ++i) {
      ASSERT_EQ(result[i], i);
    }
  }

  void TestVAlgo() {
    // V test, broadcast
    std::vector<std::int32_t> data(comm_.Rank() + 1, comm_.Rank());
    auto s_data = common::Span{data.data(), data.size()};

    std::vector<std::int64_t> sizes(comm_.World(), 0);
    sizes[comm_.Rank()] = s_data.size_bytes();
    auto rc = RingAllgather(comm_, common::Span{sizes.data(), sizes.size()});
    SafeColl(rc);
    std::shared_ptr<Coll> pcoll{new Coll{}};

    std::vector<std::int64_t> recv_segments(comm_.World() + 1, 0);
    std::vector<std::int32_t> recv(std::accumulate(sizes.cbegin(), sizes.cend(), 0));

    auto s_recv = common::Span{recv.data(), recv.size()};

    rc = pcoll->AllgatherV(comm_, common::EraseType(s_data),
                           common::Span{sizes.data(), sizes.size()},
                           common::Span{recv_segments.data(), recv_segments.size()},
                           common::EraseType(s_recv), AllgatherVAlgo::kBcast);
    SafeColl(rc);
    CheckV(s_recv);

    // Test inplace
    auto test_inplace = [&] (AllgatherVAlgo algo) {
      std::fill_n(s_recv.data(), s_recv.size(), 0);
      auto current = s_recv.subspan(recv_segments[comm_.Rank()],
                                    recv_segments[comm_.Rank() + 1] - recv_segments[comm_.Rank()]);
      std::copy_n(data.data(), data.size(), current.data());
      rc = pcoll->AllgatherV(comm_, common::EraseType(current),
                             common::Span{sizes.data(), sizes.size()},
                             common::Span{recv_segments.data(), recv_segments.size()},
                             common::EraseType(s_recv), algo);
      SafeColl(rc);
      CheckV(s_recv);
    };

    test_inplace(AllgatherVAlgo::kBcast);
    test_inplace(AllgatherVAlgo::kRing);
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

TEST_F(AllgatherTest, VBasic) {
  std::int32_t n_workers = std::min(7u, std::thread::hardware_concurrency());
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    Worker worker{host, port, timeout, n_workers, r};
    worker.TestVBasic();
  });
}

TEST_F(AllgatherTest, VRing) {
  std::int32_t n_workers = std::min(7u, std::thread::hardware_concurrency());
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    Worker worker{host, port, timeout, n_workers, r};
    worker.TestVRing();
  });
}

TEST_F(AllgatherTest, VAlgo) {
  std::int32_t n_workers = std::min(7u, std::thread::hardware_concurrency());
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    Worker worker{host, port, timeout, n_workers, r};
    worker.TestVAlgo();
  });
}

TEST(VectorAllgatherV, Basic) {
  std::int32_t n_workers{3};
  TestDistributedGlobal(n_workers, []() {
    auto n_workers = collective::GetWorldSize();
    ASSERT_EQ(n_workers, 3);
    auto rank = collective::GetRank();
    // Construct input that has different length for each worker.
    std::vector<std::vector<char>> inputs;
    for (std::int32_t i = 0; i < rank + 1; ++i) {
      std::vector<char> in;
      for (std::int32_t j = 0; j < rank + 1; ++j) {
        in.push_back(static_cast<char>(j));
      }
      inputs.emplace_back(std::move(in));
    }

    Context ctx;
    auto outputs = VectorAllgatherV(&ctx, inputs);

    ASSERT_EQ(outputs.size(), (1 + n_workers) * n_workers / 2);
    auto const& res = outputs;

    for (std::int32_t i = 0; i < n_workers; ++i) {
      std::int32_t k = 0;
      for (auto v : res[i]) {
        ASSERT_EQ(v, k++);
      }
    }
  });
}
}  // namespace xgboost::collective
