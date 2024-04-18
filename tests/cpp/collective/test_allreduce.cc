/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <numeric>  // for iota

#include "../../../src/collective/allreduce.h"
#include "../../../src/collective/coll.h"  // for Coll
#include "../../../src/common/type.h"  // for EraseType
#include "test_worker.h"               // for WorkerForTest, TestDistributed

namespace xgboost::collective {
namespace {
class AllreduceWorker : public WorkerForTest {
 public:
  using WorkerForTest::WorkerForTest;

  void Basic() {
    {
      std::vector<double> data(13, 0.0);
      auto rc = Allreduce(comm_, common::Span{data.data(), data.size()}, [](auto lhs, auto rhs) {
        for (std::size_t i = 0; i < rhs.size(); ++i) {
          rhs[i] += lhs[i];
        }
      });
      ASSERT_TRUE(rc.OK());
      ASSERT_EQ(std::accumulate(data.cbegin(), data.cend(), 0.0), 0.0);
    }
    {
      std::vector<double> data(1, 1.0);
      auto rc = Allreduce(comm_, common::Span{data.data(), data.size()}, [](auto lhs, auto rhs) {
        for (std::size_t i = 0; i < rhs.size(); ++i) {
          rhs[i] += lhs[i];
        }
      });
      ASSERT_TRUE(rc.OK());
      ASSERT_EQ(data[0], static_cast<double>(comm_.World()));
    }
  }

  void Acc() {
    std::vector<double> data(314, 1.5);
    auto rc = Allreduce(comm_, common::Span{data.data(), data.size()}, [](auto lhs, auto rhs) {
      for (std::size_t i = 0; i < rhs.size(); ++i) {
        rhs[i] += lhs[i];
      }
    });
    ASSERT_TRUE(rc.OK());
    for (std::size_t i = 0; i < data.size(); ++i) {
      auto v = data[i];
      ASSERT_EQ(v, 1.5 * static_cast<double>(comm_.World())) << i;
    }
  }

  void BitOr() {
    std::vector<std::uint32_t> data(comm_.World(), 0);
    data[comm_.Rank()] = ~std::uint32_t{0};
    auto pcoll = std::shared_ptr<Coll>{new Coll{}};
    auto rc = pcoll->Allreduce(comm_, common::EraseType(common::Span{data.data(), data.size()}),
                               ArrayInterfaceHandler::kU4, Op::kBitwiseOR);
    SafeColl(rc);
    for (auto v : data) {
      ASSERT_EQ(v, ~std::uint32_t{0});
    }
  }
};

class AllreduceTest : public SocketTest {};
}  // namespace

TEST_F(AllreduceTest, Basic) {
  std::int32_t n_workers = std::min(7u, std::thread::hardware_concurrency());
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    AllreduceWorker worker{host, port, timeout, n_workers, r};
    worker.Basic();
  });
}

TEST_F(AllreduceTest, Sum) {
  std::int32_t n_workers = std::min(7u, std::thread::hardware_concurrency());
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    AllreduceWorker worker{host, port, timeout, n_workers, r};
    worker.Acc();
  });
}

TEST_F(AllreduceTest, BitOr) {
  std::int32_t n_workers = std::min(7u, std::thread::hardware_concurrency());
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    AllreduceWorker worker{host, port, timeout, n_workers, r};
    worker.BitOr();
  });
}
}  // namespace xgboost::collective
