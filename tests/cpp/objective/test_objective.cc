/**
 * Copyright 2016-2023 by XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>
#include <xgboost/objective.h>

#include "../helpers.h"
#include "../objective_helpers.h"

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

class TestDefaultObjConfig : public ::testing::TestWithParam<std::string> {
  Context ctx_;

 public:
  void Run(std::string objective) {
    auto Xy = MakeFmatForObjTest(objective, 10, 10);
    std::unique_ptr<Learner> learner{Learner::Create({Xy})};
    std::unique_ptr<ObjFunction> objfn{ObjFunction::Create(objective, &ctx_)};

    learner->SetParam("objective", objective);
    if (objective.find("multi") != std::string::npos) {
      learner->SetParam("num_class", "3");
      objfn->Configure(Args{{"num_class", "3"}});
    } else if (objective.find("quantile") != std::string::npos) {
      learner->SetParam("quantile_alpha", "0.5");
      objfn->Configure(Args{{"quantile_alpha", "0.5"}});
    } else {
      objfn->Configure(Args{});
    }
    learner->Configure();
    learner->UpdateOneIter(0, Xy);
    learner->EvalOneIter(0, {Xy}, {"train"});
    Json config{Object{}};
    learner->SaveConfig(&config);
    auto jobj = get<Object const>(config["learner"]["objective"]);

    ASSERT_TRUE(jobj.find("name") != jobj.cend());
    // FIXME(jiamingy): We should have the following check, but some legacy parameter like
    // "pos_weight", "delta_step" in objectives are not in metrics.

    // if (jobj.size() > 1) {
    //   ASSERT_FALSE(IsA<Null>(objfn->DefaultMetricConfig()));
    // }
    auto mconfig = objfn->DefaultMetricConfig();
    if (!IsA<Null>(mconfig)) {
      // make sure metric can handle it
      std::unique_ptr<Metric> metricfn{Metric::Create(get<String const>(mconfig["name"]), &ctx_)};
      metricfn->LoadConfig(mconfig);
      Json loaded(Object{});
      metricfn->SaveConfig(&loaded);
      metricfn->Configure(Args{});
      ASSERT_EQ(mconfig, loaded);
    }
  }
};

TEST_P(TestDefaultObjConfig, Objective) {
  std::string objective = GetParam();
  this->Run(objective);
}

INSTANTIATE_TEST_SUITE_P(Objective, TestDefaultObjConfig,
                         ::testing::ValuesIn(MakeObjNamesForTest()),
                         [](const ::testing::TestParamInfo<TestDefaultObjConfig::ParamType>& info) {
                           return ObjTestNameGenerator(info);
                         });
} // namespace xgboost
