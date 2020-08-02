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

  HostDeviceVector<GradientPair> gradient{{1.0f, 1.0f}, {2.0f, 1.0f}};
  GenericParameter runtime;
  runtime.UpdateAllowUnknown(Args{});
  LearnerModelParam mparam;
  mparam.num_feature = 2;
  mparam.num_output_group = 1;

  std::unique_ptr<GradientBooster> gbm {
      GradientBooster::Create("gbtree", &runtime, &mparam)};
  gbm->Configure(Args{{"tree_method", "hist"}, {"reg_lambda", "0"}});

  PredictionCacheEntry prediction;
  prediction.predictions.Resize(2);
  gbm->DoBoost(m.get(), &gradient, &prediction);

  Json out { Object() };
  gbm->SaveModel(&out);
  float split_cond = get<Number>(out["model"]["trees"][0]["split_conditions"][0]);
  ASSERT_EQ(split_cond, 1.0f);
}

TEST(Hist, TrivialSplit) {
  std::vector<float> encoded{0, 0, 1, 2};
  size_t constexpr kCols = 1;

  encoded[1] = std::numeric_limits<float>::quiet_NaN();
  auto m = GetDMatrixFromData(encoded, encoded.size() / kCols, kCols);
  ASSERT_EQ(m->Info().num_row_, encoded.size() / kCols);

  GenericParameter runtime;
  runtime.UpdateAllowUnknown(Args{});
  LearnerModelParam mparam;
  mparam.num_output_group = 1;
  mparam.num_feature = 1;

  std::unique_ptr<GradientBooster> gbm {
      GradientBooster::Create("gbtree", &runtime, &mparam)};
  gbm->Configure(Args{{"tree_method", "hist"}, {"reg_lambda", "0"}});

  omp_set_num_threads(1);
  PredictionCacheEntry prediction;
  prediction.predictions.Resize(encoded.size(), kCols);

  HostDeviceVector<GradientPair> gradient{{1.0f, 4.0f}, {1.0f, 4.0f}, {2.0f, 1.0f}, {2.0f, 1.0f}};
  ASSERT_EQ(m->Info().num_row_, gradient.Size());
  gbm->DoBoost(m.get(), &gradient, &prediction);

  Json out { Object() };
  gbm->SaveModel(&out);
  std::cout << out << std::endl;
}
}  // namespace xgboost
