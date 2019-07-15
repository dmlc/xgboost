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
#include "test_gbm.h"

namespace xgboost {
namespace gbm {

TEST(Linear, Json_IO) {
  size_t constexpr kRows = 16, kCols = 16;
  auto gbm = ConstructGBM("gblinear", {{"num_feature", "16"}}, kRows, kCols);

  Json model {Object()};
  gbm->Save(&model);

  std::stringstream ss;
  Json::Dump(model, &ss);

  // delete pp_dmat;

  auto model_str = ss.str();
  model = Json::Load({model_str.c_str(), model_str.size()}, true);
  model = model["model"];

  {
    auto weights = get<Array>(model["linear/weights"]);
    ASSERT_EQ(weights.size(), 17);
    auto model_param = get<Object>(model["model_param"]);
    ASSERT_EQ(get<String>(model_param["num_feature"]), "16");
    ASSERT_EQ(get<String>(model_param["num_output_group"]), "1");
  }

  {
    model = Json::Load({model_str.c_str(), model_str.size()});
    model = model["model"];
    auto weights = get<Raw>(model["linear/weights"]);
    ASSERT_EQ(weights.front(), '[');
    ASSERT_EQ(weights.back(), ']');
  }

}

}  // namespace gbm
}  // namespace xgboost
