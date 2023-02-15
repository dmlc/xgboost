/**
 * Copyright 2016-2023 by XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>
#include <xgboost/objective.h>

#include "../helpers.h"

TEST(Objective, UnknownFunction) {
  xgboost::ObjFunction* obj = nullptr;
  xgboost::Context tparam;
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
  xgboost::Context tparam;
  tparam.UpdateAllowUnknown(Args{{"gpu_id", "0"}});
  size_t n = 100;

  for (const auto& entry : ::dmlc::Registry<::xgboost::ObjFunctionReg>::List()) {
    std::unique_ptr<xgboost::ObjFunction> obj{xgboost::ObjFunction::Create(entry->name, &tparam)};
    if (entry->name.find("multi") != std::string::npos) {
      obj->Configure(Args{{"num_class", "2"}});
    }
    if (entry->name.find("quantile") != std::string::npos) {
      obj->Configure(Args{{"quantile_alpha", "0.5"}});
    }
    HostDeviceVector<float> predts;
    predts.Resize(n, 3.14f);  // prediction is performed on host.
    ASSERT_FALSE(predts.DeviceCanRead());
    obj->PredTransform(&predts);
    ASSERT_FALSE(predts.DeviceCanRead());
    ASSERT_TRUE(predts.HostCanWrite());
  }
}
} // namespace xgboost
