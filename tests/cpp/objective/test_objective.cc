// Copyright by Contributors
#include <gtest/gtest.h>
#include <xgboost/objective.h>
#include <xgboost/generic_parameters.h>

#include "../helpers.h"

TEST(Objective, UnknownFunction) {
  xgboost::ObjFunction* obj = nullptr;
  xgboost::GenericParameter tparam;
  std::vector<std::pair<std::string, std::string>> args;
  tparam.UpdateAllowUnknown(args);

  EXPECT_ANY_THROW(obj = xgboost::ObjFunction::Create("unknown_name", &tparam));
  EXPECT_NO_THROW(obj = xgboost::ObjFunction::Create("reg:squarederror", &tparam));
  if (obj) {
    delete obj;
  }
}

namespace xgboost {
TEST(Objective, PredTransform) {
  // Test that show PredTransform uses the same device with predictor.
  xgboost::GenericParameter tparam;
  Args args{{"gpu_id", "0"}};  // set to deviec
  tparam.UpdateAllowUnknown(args);
  size_t n = 100;

  for (const auto &entry :
       ::dmlc::Registry<::xgboost::ObjFunctionReg>::List()) {
    std::unique_ptr<xgboost::ObjFunction> obj{
        xgboost::ObjFunction::Create(entry->name, &tparam)};
    HostDeviceVector<float> predts;
    predts.Resize(n); // prediction is performed on host.
    ASSERT_FALSE(predts.DeviceCanRead()) << entry->name;
    obj->PredTransform(&predts);
    ASSERT_FALSE(predts.DeviceCanRead()) << entry->name;
    ASSERT_TRUE(predts.HostCanWrite());
  }
}
} // namespace xgboost
