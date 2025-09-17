/**
 * Copyright 2022-2023, XGBoost contributors
 */

#include <gtest/gtest.h>
#include <xgboost/collective/result.h>  // for Result

#include "../../../../src/collective/allreduce.h"
#include "../../../../src/common/cuda_rt_utils.h"     // for AllVisibleGPUs
#include "../../../../src/common/device_helpers.cuh"  // for device_vector
#include "../../../../src/common/type.h"              // for EraseType
#include "../../collective/test_worker.h"             // for SocketTest
#include "../../helpers.h"                            // for MakeCUDACtx
#include "federated_coll.cuh"
#include "test_worker.h"  // for TestFederated

namespace xgboost::collective {
namespace {
class FederatedCollTestGPU : public SocketTest {};

struct Worker {
  std::shared_ptr<FederatedColl> impl;
  std::shared_ptr<Comm> nccl_comm;
  std::shared_ptr<CUDAFederatedColl> coll;

  Worker(std::shared_ptr<FederatedComm> comm, std::int32_t rank) {
    auto ctx = MakeCUDACtx(rank);
    impl = std::make_shared<FederatedColl>();
    nccl_comm.reset(comm->MakeCUDAVar(&ctx, impl));
    coll = std::make_shared<CUDAFederatedColl>(impl);
  }
};

void TestAllreduce(std::shared_ptr<FederatedComm> comm, std::int32_t rank, std::int32_t n_workers) {
  Worker w{comm, rank};

  dh::device_vector<std::int32_t> buffer{std::vector<std::int32_t>{1, 2, 3, 4, 5}};
  dh::device_vector<std::int32_t> expected(buffer.size());
  thrust::transform(buffer.cbegin(), buffer.cend(), expected.begin(),
                    [=] XGBOOST_DEVICE(std::int32_t i) { return i * n_workers; });

  auto rc = w.coll->Allreduce(*w.nccl_comm, common::EraseType(dh::ToSpan(buffer)),
                              ArrayInterfaceHandler::kI4, Op::kSum);
  SafeColl(rc);
  for (auto i = 0; i < 5; i++) {
    ASSERT_EQ(buffer[i], expected[i]);
  }
}

void TestBroadcast(std::shared_ptr<FederatedComm> comm, std::int32_t rank) {
  Worker w{comm, rank};

  auto rc = Success();
  std::vector<std::int32_t> expect{0, 1, 2, 3};

  if (comm->Rank() == 0) {
    dh::device_vector<std::int32_t> buffer{expect};
    rc = w.coll->Broadcast(*w.nccl_comm, common::EraseType(dh::ToSpan(buffer)), 0);
    std::vector<std::int32_t> expect{0, 1, 2, 3};
    ASSERT_EQ(buffer, expect);
  } else {
    dh::device_vector<std::int32_t> buffer(std::vector<std::int32_t>{4, 5, 6, 7});
    rc = w.coll->Broadcast(*w.nccl_comm, common::EraseType(dh::ToSpan(buffer)), 0);
    ASSERT_EQ(buffer, expect);
  }
  SafeColl(rc);
}

void TestAllgather(std::shared_ptr<FederatedComm> comm, std::int32_t rank, std::int32_t n_workers) {
  Worker w{comm, rank};

  dh::device_vector<std::int32_t> buffer(n_workers, 0);
  buffer[comm->Rank()] = comm->Rank();
  auto rc = w.coll->Allgather(*w.nccl_comm, common::EraseType(dh::ToSpan(buffer)));
  SafeColl(rc);
  for (auto i = 0; i < n_workers; i++) {
    ASSERT_EQ(buffer[i], i);
  }
}

void TestAllgatherV(std::shared_ptr<FederatedComm> comm, std::int32_t rank) {
  Worker w{comm, rank};

  std::vector<dh::device_vector<std::int32_t>> inputs{std::vector<std::int32_t>{1, 2, 3},
                                                      std::vector<std::int32_t>{4, 5}};
  std::vector<std::int64_t> recv_segments(inputs.size() + 1, 0);
  dh::device_vector<std::int32_t> r;
  std::vector<std::int64_t> sizes{static_cast<std::int64_t>(inputs[0].size()),
                                  static_cast<std::int64_t>(inputs[1].size())};
  r.resize(sizes[0] + sizes[1]);

  auto rc = w.coll->AllgatherV(*w.nccl_comm, common::EraseType(dh::ToSpan(inputs[comm->Rank()])),
                               common::Span{sizes.data(), sizes.size()}, recv_segments,
                               common::EraseType(dh::ToSpan(r)), AllgatherVAlgo::kRing);
  SafeColl(rc);

  ASSERT_EQ(r[0], 1);
  for (std::size_t i = 1; i < r.size(); ++i) {
    ASSERT_EQ(r[i], r[i - 1] + 1);
  }
}
}  // namespace

TEST_F(FederatedCollTestGPU, Allreduce) {
  std::int32_t n_workers = curt::AllVisibleGPUs();
  TestFederated(n_workers, [=](std::shared_ptr<FederatedComm> comm, std::int32_t rank) {
    TestAllreduce(comm, rank, n_workers);
  });
}

TEST(FederatedCollGPUGlobal, Allreduce) {
  std::int32_t n_workers = curt::AllVisibleGPUs();
  TestFederatedGlobal(n_workers, [&] {
    auto r = collective::GetRank();
    auto world = collective::GetWorldSize();
    CHECK_EQ(n_workers, world);

    dh::device_vector<std::uint32_t> values(3, r);
    auto ctx = MakeCUDACtx(r);
    auto rc = collective::Allreduce(
        &ctx, linalg::MakeVec(values.data().get(), values.size(), DeviceOrd::CUDA(r)),
        Op::kBitwiseOR);
    SafeColl(rc);

    std::vector<std::uint32_t> expected(values.size(), 0);
    for (std::int32_t rank = 0; rank < world; ++rank) {
      for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] |= rank;
      }
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
      CHECK_EQ(expected[i], values[i]);
    }
  });
}

TEST_F(FederatedCollTestGPU, Broadcast) {
  std::int32_t n_workers = curt::AllVisibleGPUs();
  TestFederated(n_workers, [=](std::shared_ptr<FederatedComm> comm, std::int32_t rank) {
    TestBroadcast(comm, rank);
  });
}

TEST_F(FederatedCollTestGPU, Allgather) {
  std::int32_t n_workers = curt::AllVisibleGPUs();
  TestFederated(n_workers, [=](std::shared_ptr<FederatedComm> comm, std::int32_t rank) {
    TestAllgather(comm, rank, n_workers);
  });
}

TEST_F(FederatedCollTestGPU, AllgatherV) {
  std::int32_t n_workers = 2;
  if (curt::AllVisibleGPUs() < n_workers) {
    GTEST_SKIP_("At least 2 GPUs are required for the test.");
  }
  TestFederated(n_workers, [=](std::shared_ptr<FederatedComm> comm, std::int32_t rank) {
    TestAllgatherV(comm, rank);
  });
}
}  // namespace xgboost::collective
