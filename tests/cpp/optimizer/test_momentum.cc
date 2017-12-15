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
}  // namespace xgboost
