// Copyright by Contributors
#include <gtest/gtest.h>
#include <vector>
#include "helpers.h"
#include "xgboost/learner.h"

namespace xgboost {
TEST(learner, Test) {
  typedef std::pair<std::string, std::string> arg;
  auto args = {arg("tree_method", "exact")};
  auto mat_ptr = CreateDMatrix(10, 10, 0);
  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {*mat_ptr};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->Configure(args);

  delete mat_ptr;
}
}  // namespace xgboost
