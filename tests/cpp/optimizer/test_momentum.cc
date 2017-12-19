// Copyright by Contributors
#include <gtest/gtest.h>
#include <xgboost/optimizer.h>
#include "../helpers.h"

namespace xgboost {
TEST(momentum_optimizer, Test) {
  auto optimizer =
      std::unique_ptr<Optimizer>(Optimizer::Create("momentum_optimizer"));
  optimizer->Init({std::pair<std::string, std::string>("momentum", "0.5")});

  std::vector<bst_gpair> gpair(10, bst_gpair(0.5f, 1.0f));

  optimizer->OptimizeGradients(&gpair);

  for (auto &g : gpair) {
    ASSERT_EQ(g.GetGrad(), 0.5f);
    ASSERT_EQ(g.GetHess(), 1.0f);
  }

  optimizer->OptimizeGradients(&gpair);

  for (auto &g : gpair) {
    ASSERT_EQ(g.GetGrad(), 0.5f + 0.5f * 0.5f);
    ASSERT_EQ(g.GetHess(), 1.0f);
  }
}
TEST(nesterov_optimizer, Test) {
  auto optimizer =
      std::unique_ptr<Optimizer>(Optimizer::Create("nesterov_optimizer"));
  optimizer->Init({std::pair<std::string, std::string>("momentum", "0.5")});

  std::vector<bst_gpair> gpair(10, bst_gpair(0.5f, 1.0f));

  optimizer->OptimizeGradients(&gpair);

  for (auto &g : gpair) {
    ASSERT_EQ(g.GetGrad(), 0.5f);
    ASSERT_EQ(g.GetHess(), 1.0f);
  }

  optimizer->OptimizeGradients(&gpair);

  for (auto &g : gpair) {
    ASSERT_EQ(g.GetGrad(), 0.5f + 0.5f * 0.5f);
    ASSERT_EQ(g.GetHess(), 1.0f);
  }

  auto dmat = CreateDMatrix(100, 5, 0);
  auto gbm = std::unique_ptr<GradientBooster>(GradientBooster::Create("gbtree",{dmat},0.5));
  gbm->Configure({ std::pair<std::string, std::string>("num_feature","5") });

  std::vector<float > predictions(100, 0.0f);
  optimizer->OptimizePredictions(&predictions, gbm.get(), dmat.get());

  for (auto &p : predictions) {
    ASSERT_EQ(p, 0.0f);
  }
}
}  // namespace xgboost
