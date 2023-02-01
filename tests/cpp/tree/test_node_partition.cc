/**
 * Copyright 2023 by XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/task.h>
#include <xgboost/tree_updater.h>

namespace xgboost {
TEST(Updater, HasNodePosition) {
  Context ctx;
  ObjInfo task{ObjInfo::kRegression, true, true};
  std::unique_ptr<TreeUpdater> up{TreeUpdater::Create("grow_histmaker", &ctx, task)};
  ASSERT_TRUE(up->HasNodePosition());

  up.reset(TreeUpdater::Create("grow_quantile_histmaker", &ctx, task));
  ASSERT_TRUE(up->HasNodePosition());

#if defined(XGBOOST_USE_CUDA)
  ctx.gpu_id = 0;
  up.reset(TreeUpdater::Create("grow_gpu_hist", &ctx, task));
  ASSERT_TRUE(up->HasNodePosition());
#endif  // defined(XGBOOST_USE_CUDA)
}
}  // namespace xgboost
