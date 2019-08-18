#include <gtest/gtest.h>
#include <xgboost/objective.h>
#include <string>
#include "../helpers.h"

namespace xgboost {

TEST(Plugin, ExampleObjective) {
  xgboost::GenericParameter tparam = CreateEmptyGenericParam(0, 0);
  auto * obj = xgboost::ObjFunction::Create("mylogistic", &tparam);
  ASSERT_EQ(obj->DefaultEvalMetric(), std::string{"error"});
  delete obj;
}

}  // namespace xgboost
