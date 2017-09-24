// Copyright by Contributors
#include <gtest/gtest.h>
#include "helpers.h"
#include "xgboost/learner.h"

namespace xgboost {
TEST(learner, Test) {
  typedef std::pair<std::string, std::string> arg;
  auto args = {arg("tree_method", "exact")};
  auto mat = {CreateDMatrix(10, 10, 0)};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->Configure(args);
}
}  // namespace xgboost