/**
 * Copyright 2020-2024 by XGBoost contributors
 */
#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/json.h>
#include <xgboost/task.h>
#include "../../../plugin/sycl/tree/updater_quantile_hist.h"       // for QuantileHistMaker
#pragma GCC diagnostic pop

namespace xgboost::sycl::tree {
TEST(SyclQuantileHistMaker, Basic) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  ObjInfo task{ObjInfo::kRegression};
  std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_quantile_histmaker_sycl", &ctx, &task)};

  ASSERT_EQ(updater->Name(), "grow_quantile_histmaker_sycl");
}

TEST(SyclQuantileHistMaker, JsonIO) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  ObjInfo task{ObjInfo::kRegression};
  Json config {Object()};
  {
    std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_quantile_histmaker_sycl", &ctx, &task)};
    updater->Configure({{"max_depth", std::to_string(42)}});
    updater->Configure({{"single_precision_histogram", std::to_string(true)}});
    updater->SaveConfig(&config);
  }

  {
    std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_quantile_histmaker_sycl", &ctx, &task)};
    updater->LoadConfig(config);

    Json new_config {Object()};
    updater->SaveConfig(&new_config);

    ASSERT_EQ(config, new_config);

    auto max_depth = atoi(get<String const>(new_config["train_param"]["max_depth"]).c_str());
    ASSERT_EQ(max_depth, 42);

    auto single_precision_histogram = atoi(get<String const>(new_config["sycl_hist_train_param"]["single_precision_histogram"]).c_str());
    ASSERT_EQ(single_precision_histogram, 1);
  }
  
}
}  // namespace xgboost::sycl::tree
