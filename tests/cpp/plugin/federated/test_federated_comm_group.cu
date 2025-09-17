/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/json.h>  // for Json

#include "../../../../src/collective/comm_group.h"
#include "../../../../src/common/cuda_rt_utils.h"  // for AllVisibleGPUs
#include "../../helpers.h"
#include "test_worker.h"

namespace xgboost::collective {
TEST(CommGroup, FederatedGPU) {
  std::int32_t n_workers = curt::AllVisibleGPUs();
  TestFederatedGroup(n_workers, [&](std::shared_ptr<CommGroup> comm_group, std::int32_t r) {
    Context ctx = MakeCUDACtx(0);
    auto const& comm = comm_group->Ctx(&ctx, DeviceOrd::CUDA(0));
    ASSERT_EQ(comm_group->Rank(), r);
    ASSERT_EQ(comm.TaskID(), std::to_string(r));
    ASSERT_EQ(comm.Retry(), 2);
  });
}
}  // namespace xgboost::collective
