/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/json.h>  // for Json

#include <chrono>   // for seconds
#include <cstdint>  // for int32_t
#include <string>   // for string
#include <thread>   // for thread

#include "../../../src/collective/comm.h"
#include "../../../src/collective/comm_group.h"
#include "../../../src/common/common.h"  // for AllVisibleGPUs
#include "../helpers.h"                  // for MakeCUDACtx
#include "test_worker.h"                 // for TestDistributed

namespace xgboost::collective {
namespace {
class CommGroupTest : public SocketTest {};
}  // namespace

TEST_F(CommGroupTest, Basic) {
  std::int32_t n_workers = std::min(std::thread::hardware_concurrency(), 5u);
  TestDistributed(n_workers, [&](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    Context ctx;
    auto config = MakeDistributedTestConfig(host, port, timeout, r);
    std::unique_ptr<CommGroup> ptr{CommGroup::Create(config)};
    ASSERT_TRUE(ptr->IsDistributed());
    ASSERT_EQ(ptr->World(), n_workers);
    auto const& comm = ptr->Ctx(&ctx, DeviceOrd::CPU());
    ASSERT_EQ(comm.TaskID(), std::to_string(r));
    ASSERT_EQ(comm.Retry(), 2);
  });
}

#if defined(XGBOOST_USE_NCCL)
TEST_F(CommGroupTest, BasicGPU) {
  std::int32_t n_workers = curt::AllVisibleGPUs();
  TestDistributed(n_workers, [&](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    auto ctx = MakeCUDACtx(r);
    auto config = MakeDistributedTestConfig(host, port, timeout, r);
    std::unique_ptr<CommGroup> ptr{CommGroup::Create(config)};
    auto const& comm = ptr->Ctx(&ctx, DeviceOrd::CUDA(0));
    ASSERT_EQ(comm.TaskID(), std::to_string(r));
    ASSERT_EQ(comm.Retry(), 2);
  });
}
#endif  // for defined(XGBOOST_USE_NCCL)
}  // namespace xgboost::collective
