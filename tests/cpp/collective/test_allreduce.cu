/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#if defined(XGBOOST_USE_NCCL)
#include <gtest/gtest.h>
#include <thrust/host_vector.h>  // for host_vector

#include "../../../src/collective/comm.cuh"        // for NCCLComm
#include "../../../src/common/cuda_rt_utils.h"     // for AllVisibleGPUs
#include "../../../src/common/device_helpers.cuh"  // for ToSpan,  device_vector
#include "../../../src/common/type.h"              // for EraseType
#include "test_worker.cuh"                         // for NCCLWorkerForTest
#include "test_worker.h"                           // for WorkerForTest, TestDistributed

namespace xgboost::collective {
namespace {
class MGPUAllreduceTest : public SocketTest {};

class Worker : public NCCLWorkerForTest {
 public:
  using NCCLWorkerForTest::NCCLWorkerForTest;

  bool SkipIfOld() {
    auto nccl = dynamic_cast<NCCLComm const*>(nccl_comm_.get());
    std::int32_t major = 0, minor = 0, patch = 0;
    SafeColl(nccl->Stub()->GetVersion(&major, &minor, &patch));
    CHECK_GE(major, 2);
    bool too_old = minor < 23;
    if (too_old) {
      LOG(INFO) << "NCCL compile version:" << NCCL_VERSION_CODE << " runtime version:" << major
                << "." << minor << "." << patch;
    }
    return too_old;
  }

  void BitOr() {
    dh::device_vector<std::uint32_t> data(comm_.World(), 0);
    data[comm_.Rank()] = ~std::uint32_t{0};
    auto rc = nccl_coll_->Allreduce(*nccl_comm_, common::EraseType(dh::ToSpan(data)),
                                    ArrayInterfaceHandler::kU4, Op::kBitwiseOR);
    SafeColl(rc);
    thrust::host_vector<std::uint32_t> h_data(data.size());
    thrust::copy(data.cbegin(), data.cend(), h_data.begin());
    for (auto v : h_data) {
      ASSERT_EQ(v, ~std::uint32_t{0});
    }
  }

  void Acc() {
    dh::device_vector<double> data(314, 1.5);
    auto rc = nccl_coll_->Allreduce(*nccl_comm_, common::EraseType(dh::ToSpan(data)),
                                    ArrayInterfaceHandler::kF8, Op::kSum);
    SafeColl(rc);
    for (std::size_t i = 0; i < data.size(); ++i) {
      auto v = data[i];
      ASSERT_EQ(v, 1.5 * static_cast<double>(comm_.World())) << i;
    }
  }

  Result NoCheck() {
    dh::device_vector<double> data(314, 1.5);
    auto rc = nccl_coll_->Allreduce(*nccl_comm_, common::EraseType(dh::ToSpan(data)),
                                    ArrayInterfaceHandler::kF8, Op::kSum);
    return rc;
  }

  ~Worker() noexcept(false) override = default;
};
}  // namespace

TEST_F(MGPUAllreduceTest, BitOr) {
  auto n_workers = curt::AllVisibleGPUs();
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    Worker w{host, port, timeout, n_workers, r};
    w.Setup();
    w.BitOr();
  });
}

TEST_F(MGPUAllreduceTest, Sum) {
  auto n_workers = curt::AllVisibleGPUs();
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    Worker w{host, port, timeout, n_workers, r};
    w.Setup();
    w.Acc();
  });
}

TEST_F(MGPUAllreduceTest, Timeout) {
  auto n_workers = curt::AllVisibleGPUs();
  if (n_workers <= 1) {
    GTEST_SKIP_("Requires more than one GPU to run.");
  }
  using std::chrono_literals::operator""s;

  TestDistributed(
      n_workers,
      [=](std::string host, std::int32_t port, std::chrono::seconds, std::int32_t r) {
        auto w = std::make_unique<Worker>(host, port, 1s, n_workers, r);
        w->Setup();
        if (w->SkipIfOld()) {
          GTEST_SKIP_("nccl is too old.");
          return;
        }
        // 1s for worker timeout, sleeping for 2s should trigger a timeout error.
        if (r == 0) {
          std::this_thread::sleep_for(2s);
        }
        auto rc = w->NoCheck();
        if (r == 1) {
          auto rep = rc.Report();
          ASSERT_NE(rep.find("NCCL timeout:"), std::string::npos) << rep;
        }

        w.reset();
      },
      // We use 8s for the tracker to make sure shutdown is successful.
      8s);

  TestDistributed(
      n_workers,
      [=](std::string host, std::int32_t port, std::chrono::seconds, std::int32_t r) {
        auto w = std::make_unique<Worker>(host, port, 1s, n_workers, r);
        w->Setup();
        if (w->SkipIfOld()) {
          GTEST_SKIP_("nccl is too old.");
          return;
        }
        // Only one of the workers is doing allreduce.
        if (r == 0) {
          auto rc = w->NoCheck();
          ASSERT_NE(rc.Report().find("NCCL timeout:"), std::string::npos) << rc.Report();
        }

        w.reset();
      },
      // We use 8s for the tracker to make sure shutdown is successful.
      8s);
}
}  // namespace xgboost::collective
#endif  // defined(XGBOOST_USE_NCCL)
