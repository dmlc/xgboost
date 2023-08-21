/**
 * Copyright 2019-2023, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <memory>
#include <utility>  // for swap

#include "../helpers.h"
#include "xgboost/context.h"
#include "xgboost/gbm.h"
#include "xgboost/json.h"
#include "xgboost/learner.h"
#include "xgboost/logging.h"

namespace xgboost::gbm {
TEST(GBLinear, JsonIO) {
  size_t constexpr kRows = 16, kCols = 16;

  Context ctx;
  LearnerModelParam mparam{MakeMP(kCols, .5, 1)};

  std::unique_ptr<GradientBooster> gbm{
      CreateTrainedGBM("gblinear", Args{}, kRows, kCols, &mparam, &ctx)};
  Json model { Object() };
  gbm->SaveModel(&model);
  ASSERT_TRUE(IsA<Object>(model));

  std::string model_str;
  Json::Dump(model, &model_str);

  model = Json::Load(StringView{model_str.c_str(), model_str.size()});
  ASSERT_TRUE(IsA<Object>(model));

  {
    model = model["model"];
    auto weights = get<Array>(model["weights"]);
    ASSERT_EQ(weights.size(), 17);
  }
}

TEST(GBlinear, DispatchUpdater) {
  auto verbosity = 2;
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
}  // namespace xgboost::gbm
