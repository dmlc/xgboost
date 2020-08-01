#include <gtest/gtest.h>
#include <xgboost/learner.h>

#include <xgboost/gbm.h>
#include "../helpers.h"


namespace xgboost {
TEST(Hist, SplitOneHot) {
  std::vector<float> x{0, 1};
  auto encoded = OneHotEncodeFeature(x, 2);
  ASSERT_EQ(encoded.size(), x.size() * 2);
  auto m = GetDMatrixFromData(encoded, 2, 2);

  HostDeviceVector<GradientPair> gradient{{1.0f, 2.0f}, {1.0f, 2.0f}};
  GenericParameter runtime;
  runtime.UpdateAllowUnknown(Args{});
  LearnerModelParam mparam;
  mparam.num_feature = 2;
  mparam.num_output_group = 1;

  std::unique_ptr<GradientBooster> gbm {
      GradientBooster::Create("gbtree", &runtime, &mparam)};
  gbm->Configure(Args{{"tree_method", "hist"}});

  PredictionCacheEntry prediction;
  prediction.predictions.Resize(2);
  gbm->DoBoost(m.get(), &gradient, &prediction);

  Json out { Object() };
  gbm->SaveModel(&out);
  float split_cond = get<Number>(out["model"]["trees"][0]["split_conditions"][0]);
  // ASSERT_EQ(split_cond, 1.0f);
  std::cout << out << std::endl;
}
}  // namespace xgboost
