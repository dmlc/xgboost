/**
 * Copyright 2024-2026, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>
#include <xgboost/learner.h>

#include <cmath>   // for tanh
#include <memory>  // for unique_ptr

#include "../helpers.h"
#include "test_quantile_obj.h"

namespace xgboost {
TEST(Objective, DeclareUnifiedTest(Quantile)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestQuantile(&ctx);
}

TEST(Objective, DeclareUnifiedTest(QuantileIntercept)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestQuantileIntercept(&ctx);
}

TEST(Objective, DeclareUnifiedTest(QuantileRegularization)) {
  auto Xy = GetDMatrixFromData({0.0f}, 1, 1);
  Xy->Info().labels.Reshape(1, 1);
  Xy->Info().labels.HostView()(0, 0) = 1.0f;

  auto train = [&](float reg_lambda) {
    std::unique_ptr<Learner> learner{Learner::Create({Xy})};
    learner->Configure(Args{{"tree_method", "exact"},
                            {"objective", "reg:quantileerror"},
                            {"quantile_alpha", "0.5"},
                            {"base_score", "0.5"},
                            {"eta", "1"},
                            {"max_depth", "1"},
                            {"min_child_weight", "0"},
                            {"reg_alpha", "0"},
                            {"reg_lambda", std::to_string(reg_lambda)}});
    learner->Configure();
    learner->UpdateOneIter(0, Xy);
    HostDeviceVector<float> predt;
    learner->Predict(Xy, false, &predt, 0, 0);
    return predt.HostVector().front();
  };

  float residual{-0.5f};
  float residual_scale = std::abs(residual);
  float x = residual / (0.04f * residual_scale);
  float grad = 0.5f * residual_scale * std::tanh(x);
  float curvature = 0.5f / 0.04f * std::tanh(x) / x;
  ASSERT_NEAR(train(0.0f), 0.5f - grad / curvature, 1.0e-5f);
  ASSERT_NEAR(train(1.0f), 0.5f - grad / (curvature + 1.0f), 1.0e-5f);
}

TEST(Objective, DeclareUnifiedTest(QuantileMonotoneConstraint)) {
  auto Xy = GetDMatrixFromData({0.0f, 1.0f, 2.0f, 3.0f}, 4, 1);
  Xy->Info().labels.Reshape(4, 1);
  Xy->Info().labels.Data()->HostVector() = {3.0f, 2.0f, 1.0f, 0.0f};

  std::unique_ptr<Learner> learner{Learner::Create({Xy})};
  learner->Configure(Args{{"tree_method", "hist"},
                          {"objective", "reg:quantileerror"},
                          {"quantile_alpha", "0.5"},
                          {"monotone_constraints", "(1)"},
                          {"min_child_weight", "0"}});
  learner->Configure();
  for (std::int32_t iter{0}; iter < 8; ++iter) {
    learner->UpdateOneIter(iter, Xy);
  }
  HostDeviceVector<float> predt;
  learner->Predict(Xy, false, &predt, 0, 0);
  auto const& h_predt = predt.HostVector();
  for (std::size_t i{1}; i < h_predt.size(); ++i) {
    ASSERT_LE(h_predt[i - 1], h_predt[i]);
  }
}
}  // namespace xgboost
