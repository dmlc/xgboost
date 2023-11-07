/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/json.h>  // for Json

#include "../../../../src/collective/comm_group.h"
#include "../../helpers.h"
#include "test_worker.h"

namespace xgboost::collective {
TEST(CommGroup, Federated) {
  std::int32_t n_workers = common::AllVisibleGPUs();
  TestFederatedGroup(n_workers, [&](std::shared_ptr<CommGroup> comm_group, std::int32_t r) {
    Context ctx;
    ASSERT_EQ(comm_group->Rank(), r);
    auto const& comm = comm_group->Ctx(&ctx, DeviceOrd::CPU());
    ASSERT_EQ(comm.TaskID(), std::to_string(r));
    ASSERT_EQ(comm.Retry(), 2);
  });
}
}  // namespace xgboost::collective
