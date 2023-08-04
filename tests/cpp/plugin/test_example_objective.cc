#include <gtest/gtest.h>
#include <xgboost/objective.h>
#include <string>
#include "../helpers.h"

namespace xgboost {
TEST(Plugin, ExampleObjective) {
  xgboost::Context ctx = MakeCUDACtx(GPUIDX);
  auto* obj = xgboost::ObjFunction::Create("mylogistic", &ctx);
  ASSERT_EQ(obj->DefaultEvalMetric(), std::string{"logloss"});
  delete obj;
}
}  // namespace xgboost
