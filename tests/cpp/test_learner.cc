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

TEST(learner, SelectTreeMethod) {
  using arg = std::pair<std::string, std::string>;
  auto mat_ptr = CreateDMatrix(10, 10, 0);
  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {*mat_ptr};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));

  // Test if `tree_method` can be set
  learner->Configure({arg("tree_method", "approx")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_histmaker,prune");
  learner->Configure({arg("tree_method", "exact")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_colmaker,prune");
  learner->Configure({arg("tree_method", "hist")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_fast_histmaker");
#ifdef XGBOOST_USE_CUDA
  learner->Configure({arg("tree_method", "gpu_exact")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_gpu,prune");
  learner->Configure({arg("tree_method", "gpu_hist")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_gpu_hist");
#endif

  delete mat_ptr;
}

}  // namespace xgboost
