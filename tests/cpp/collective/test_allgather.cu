/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#if defined(XGBOOST_USE_NCCL)
#include <gtest/gtest.h>
#include <thrust/device_vector.h>  // for device_vector
#include <thrust/equal.h>          // for equal
#include <xgboost/span.h>          // for Span

#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t, int64_t
#include <vector>   // for vector

#include "../../../src/collective/allgather.h"     // for RingAllgather
#include "../../../src/common/device_helpers.cuh"  // for ToSpan,  device_vector
#include "../../../src/common/type.h"              // for EraseType
#include "test_worker.cuh"                         // for NCCLWorkerForTest
#include "test_worker.h"                           // for TestDistributed, WorkerForTest

namespace xgboost::collective {
namespace {
class Worker : public NCCLWorkerForTest {
 public:
  using NCCLWorkerForTest::NCCLWorkerForTest;

  void TestV(AllgatherVAlgo algo) {
    {
      // basic test
      std::size_t n = 1;
      // create data
      dh::device_vector<std::int32_t> data(n, comm_.Rank());
      auto s_data = common::EraseType(common::Span{data.data().get(), data.size()});
      // get size
      std::vector<std::int64_t> sizes(comm_.World(), -1);
      sizes[comm_.Rank()] = s_data.size_bytes();
      auto rc = RingAllgather(comm_, common::Span{sizes.data(), sizes.size()});
      SafeColl(rc);
      // create result
      dh::device_vector<std::int32_t> result(comm_.World(), -1);
      auto s_result = common::EraseType(dh::ToSpan(result));

      std::vector<std::int64_t> recv_seg(nccl_comm_->World() + 1, 0);
      rc = nccl_coll_->AllgatherV(*nccl_comm_, s_data, common::Span{sizes.data(), sizes.size()},
                                  common::Span{recv_seg.data(), recv_seg.size()}, s_result, algo);
      SafeColl(rc);

      for (std::int32_t i = 0; i < comm_.World(); ++i) {
        ASSERT_EQ(result[i], i);
      }
    }
    {
      // V test
      std::size_t n = 256 * 256;
      // create data
      dh::device_vector<std::int32_t> data(n * nccl_comm_->Rank(), nccl_comm_->Rank());
      auto s_data = common::EraseType(common::Span{data.data().get(), data.size()});
      // get size
      std::vector<std::int64_t> sizes(nccl_comm_->World(), 0);
      sizes[comm_.Rank()] = dh::ToSpan(data).size_bytes();
      auto rc = RingAllgather(comm_, common::Span{sizes.data(), sizes.size()});
      SafeColl(rc);
      auto n_bytes = std::accumulate(sizes.cbegin(), sizes.cend(), 0);
      // create result
      dh::device_vector<std::int32_t> result(n_bytes / sizeof(std::int32_t), -1);
      auto s_result = common::EraseType(dh::ToSpan(result));

      std::vector<std::int64_t> recv_seg(nccl_comm_->World() + 1, 0);
      rc = nccl_coll_->AllgatherV(*nccl_comm_, s_data, common::Span{sizes.data(), sizes.size()},
                                  common::Span{recv_seg.data(), recv_seg.size()}, s_result, algo);
      SafeColl(rc);
      // check segment size
      if (algo != AllgatherVAlgo::kBcast) {
        auto size = recv_seg[nccl_comm_->Rank() + 1] - recv_seg[nccl_comm_->Rank()];
        ASSERT_EQ(size, n * nccl_comm_->Rank() * sizeof(std::int32_t));
        ASSERT_EQ(size, sizes[nccl_comm_->Rank()]);
      }
      // check data
      std::size_t k{0};
      for (std::int32_t r = 0; r < nccl_comm_->World(); ++r) {
        std::size_t s = n * r;
        auto current = dh::ToSpan(result).subspan(k, s);
        std::vector<std::int32_t> h_data(current.size());
        dh::CopyDeviceSpanToVector(&h_data, current);
        for (auto v : h_data) {
          ASSERT_EQ(v, r);
        }
        k += s;
      }
    }
  }
};

class MGPUAllgatherTest : public SocketTest {};
}  // namespace

TEST_F(MGPUAllgatherTest, MGPUTestVRing) {
  auto n_workers = curt::AllVisibleGPUs();
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    Worker w{host, port, timeout, n_workers, r};
    w.Setup();
    w.TestV(AllgatherVAlgo::kRing);
    w.TestV(AllgatherVAlgo::kBcast);
  });
}

TEST_F(MGPUAllgatherTest, MGPUTestVBcast) {
  auto n_workers = curt::AllVisibleGPUs();
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    Worker w{host, port, timeout, n_workers, r};
    w.Setup();
    w.TestV(AllgatherVAlgo::kBcast);
  });
}
}  // namespace xgboost::collective
#endif  // defined(XGBOOST_USE_NCCL)
