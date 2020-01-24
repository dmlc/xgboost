/*!
 * Copyright 2019 by Contributors
 */
#include <gtest/gtest.h>

#include <memory>
#include <sstream>

#include "../helpers.h"
#include "xgboost/json.h"
#include "xgboost/logging.h"
#include "xgboost/gbm.h"
#include "xgboost/generic_parameters.h"
#include "xgboost/learner.h"

namespace xgboost {
namespace gbm {

TEST(GBLinear, Json_IO) {
  size_t constexpr kRows = 16, kCols = 16;

  LearnerModelParam param;
  param.num_feature = kCols;
  param.num_output_group = 1;

  GenericParameter gparam;
  gparam.Init(Args{});

  std::unique_ptr<GradientBooster> gbm {
    CreateTrainedGBM("gblinear", Args{}, kRows, kCols, &param, &gparam) };
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

}  // namespace gbm
}  // namespace xgboost
