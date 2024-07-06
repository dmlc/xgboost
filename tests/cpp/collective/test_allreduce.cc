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
      SafeColl(rc);
      ASSERT_EQ(std::accumulate(data.cbegin(), data.cend(), 0.0), 0.0);
    }
    {
      std::vector<double> data(1, 1.0);
      auto rc = Allreduce(comm_, common::Span{data.data(), data.size()}, [](auto lhs, auto rhs) {
        for (std::size_t i = 0; i < rhs.size(); ++i) {
          rhs[i] += lhs[i];
        }
      });
      SafeColl(rc);
      ASSERT_EQ(data[0], static_cast<double>(comm_.World()));
    }
  }

  void Restricted() {
    this->LimitSockBuf(4096);

    std::size_t n = 4096 * 4;
    std::vector<std::int32_t> data(comm_.World() * n, 1);
    auto rc = Allreduce(comm_, common::Span{data.data(), data.size()}, [](auto lhs, auto rhs) {
      for (std::size_t i = 0; i < rhs.size(); ++i) {
        rhs[i] += lhs[i];
      }
    });
    SafeColl(rc);
    for (auto v : data) {
      ASSERT_EQ(v, comm_.World());
    }
  }

  void Acc() {
    std::vector<double> data(314, 1.5);
    auto rc = Allreduce(comm_, common::Span{data.data(), data.size()}, [](auto lhs, auto rhs) {
      for (std::size_t i = 0; i < rhs.size(); ++i) {
        rhs[i] += lhs[i];
      }
    });
    SafeColl(rc);
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

TEST_F(AllreduceTest, Restricted) {
  std::int32_t n_workers = std::min(3u, std::thread::hardware_concurrency());
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    AllreduceWorker worker{host, port, timeout, n_workers, r};
    worker.Restricted();
  });
}

TEST(AllreduceGlobal, Basic) {
  auto n_workers = 3;
  TestDistributedGlobal(n_workers, [&]() {
    std::vector<float> values(n_workers * 2, 0);
    auto rank = GetRank();
    auto s_values = common::Span{values.data(), values.size()};
    auto self = s_values.subspan(rank * 2, 2);
    for (auto& v : self) {
      v = 1.0f;
    }
    Context ctx;
    auto rc =
        Allreduce(&ctx, linalg::MakeVec(s_values.data(), s_values.size()), collective::Op::kSum);
    SafeColl(rc);
    for (auto v : s_values) {
      ASSERT_EQ(v, 1);
    }
  });
}

TEST(AllreduceGlobal, Small) {
  // Test when the data is not large enougth to be divided by the number of workers
  auto n_workers = 8;
  TestDistributedGlobal(n_workers, [&]() {
    std::uint64_t value{1};
    Context ctx;
    auto rc = Allreduce(&ctx, linalg::MakeVec(&value, 1), collective::Op::kSum);
    SafeColl(rc);
    ASSERT_EQ(value, n_workers);
  });
}
}  // namespace xgboost::collective
