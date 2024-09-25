/**
 * Copyright 2022-2023, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/span.h>  // for Span

#include <array>  // for array

#include "../../../../src/common/type.h"   // for EraseType
#include "../../collective/test_worker.h"  // for SocketTest
#include "federated_coll.h"                // for FederatedColl
#include "federated_comm.h"                // for FederatedComm
#include "test_worker.h"                   // for TestFederated

namespace xgboost::collective {
namespace {
class FederatedCollTest : public SocketTest {};
}  // namespace

TEST_F(FederatedCollTest, Allreduce) {
  std::int32_t n_workers = std::min(std::thread::hardware_concurrency(), 3u);
  TestFederated(n_workers, [=](std::shared_ptr<FederatedComm> comm, std::int32_t) {
    std::array<std::int32_t, 5> buffer = {1, 2, 3, 4, 5};
    std::array<std::int32_t, 5> expected;
    std::transform(buffer.cbegin(), buffer.cend(), expected.begin(),
                   [=](auto i) { return i * n_workers; });

    auto coll = std::make_shared<FederatedColl>();
    auto rc = coll->Allreduce(*comm, common::EraseType(common::Span{buffer.data(), buffer.size()}),
                              ArrayInterfaceHandler::kI4, Op::kSum);
    SafeColl(rc);
    for (auto i = 0; i < 5; i++) {
      ASSERT_EQ(buffer[i], expected[i]);
    }
  });
}

TEST_F(FederatedCollTest, Broadcast) {
  std::int32_t n_workers = std::min(std::thread::hardware_concurrency(), 3u);
  TestFederated(n_workers, [=](std::shared_ptr<FederatedComm> comm, std::int32_t) {
    FederatedColl coll{};
    auto rc = Success();
    if (comm->Rank() == 0) {
      std::string buffer{"hello"};
      rc = coll.Broadcast(*comm, common::EraseType(common::Span{buffer.data(), buffer.size()}), 0);
      ASSERT_EQ(buffer, "hello");
    } else {
      std::string buffer{"     "};
      rc = coll.Broadcast(*comm, common::EraseType(common::Span{buffer.data(), buffer.size()}), 0);
      ASSERT_EQ(buffer, "hello");
    }
    SafeColl(rc);
  });
}

TEST_F(FederatedCollTest, Allgather) {
  std::int32_t n_workers = std::min(std::thread::hardware_concurrency(), 3u);
  TestFederated(n_workers, [=](std::shared_ptr<FederatedComm> comm, std::int32_t) {
    FederatedColl coll{};

    std::vector<std::int32_t> buffer(n_workers, 0);
    buffer[comm->Rank()] = comm->Rank();
    auto rc = coll.Allgather(*comm, common::EraseType(common::Span{buffer.data(), buffer.size()}));
    SafeColl(rc);
    for (auto i = 0; i < n_workers; i++) {
      ASSERT_EQ(buffer[i], i);
    }
  });
}

TEST_F(FederatedCollTest, AllgatherV) {
  std::int32_t n_workers = 2;
  TestFederated(n_workers, [=](std::shared_ptr<FederatedComm> comm, std::int32_t) {
    FederatedColl coll{};

    std::vector<std::string_view> inputs{"Federated", " Learning!!!"};
    std::vector<std::int64_t> recv_segments(inputs.size() + 1, 0);
    std::string r;
    std::vector<std::int64_t> sizes{static_cast<std::int64_t>(inputs[0].size()),
                                    static_cast<std::int64_t>(inputs[1].size())};
    r.resize(sizes[0] + sizes[1]);

    auto rc = coll.AllgatherV(
        *comm,
        common::EraseType(common::Span{inputs[comm->Rank()].data(), inputs[comm->Rank()].size()}),
        common::Span{sizes.data(), sizes.size()}, recv_segments,
        common::EraseType(common::Span{r.data(), r.size()}), AllgatherVAlgo::kRing);

    EXPECT_EQ(r, "Federated Learning!!!");
    SafeColl(rc);
  });
}
}  // namespace xgboost::collective
