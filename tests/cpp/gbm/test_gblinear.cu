/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/global_config.h>  // for GlobalConfigThreadLocalStore
#include <xgboost/json.h>           // for Json, Object
#include <xgboost/learner.h>        // for Learner

#include <algorithm>  // for transform
#include <string>     // for string
#include <utility>    // for swap

#include "../helpers.h"  // for RandomDataGenerator

namespace xgboost {
TEST(GBlinear, DispatchUpdater) {
  auto verbosity = 3;
  std::swap(GlobalConfigThreadLocalStore::Get()->verbosity, verbosity);

  auto test = [](std::string device) {
    auto p_fmat = RandomDataGenerator{10, 10, 0.0f}.GenerateDMatrix(true);
    std::unique_ptr<Learner> learner{Learner::Create({p_fmat})};
    learner->SetParams(
        Args{{"booster", "gblinear"}, {"updater", "coord_descent"}, {"device", device}});
    learner->Configure();
    for (std::int32_t iter = 0; iter < 3; ++iter) {
      learner->UpdateOneIter(iter, p_fmat);
    }
    Json config{Object{}};
    ::testing::internal::CaptureStderr();
    learner->SaveConfig(&config);
    auto str = ::testing::internal::GetCapturedStderr();
    std::transform(device.cbegin(), device.cend(), device.begin(),
                   [](char c) { return std::toupper(c); });
    ASSERT_NE(str.find(device), std::string::npos);
  };
  test("cpu");
  test("gpu");

  std::swap(GlobalConfigThreadLocalStore::Get()->verbosity, verbosity);
}
}  // namespace xgboost
