#include <gtest/gtest.h>
#include <xgboost/logging.h>
#include <memory>
#include "../../../src/tree/split_evaluator.h"

namespace xgboost {
namespace tree {

TEST(SplitEvaluator, Interaction) {
  std::string constraints_str = R"interaction([[0, 1], [1, 2, 3]])interaction";
  std::vector<std::pair<std::string, std::string>> args{
    {"interaction_constraints", constraints_str},
    {"num_feature", "8"}};
  {
    std::unique_ptr<SplitEvaluator> eval{
        SplitEvaluator::Create("elastic_net,interaction")};
    eval->Init(args);

    eval->AddSplit(0, 1, 2, /*feature_id=*/4, 0, 0);
    eval->AddSplit(2, 3, 4, /*feature_id=*/5, 0, 0);
    ASSERT_FALSE(eval->CheckFeatureConstraint(2, /*feature_id=*/0));
    ASSERT_FALSE(eval->CheckFeatureConstraint(2, /*feature_id=*/1));

    ASSERT_TRUE(eval->CheckFeatureConstraint(2, /*feature_id=*/4));
    ASSERT_FALSE(eval->CheckFeatureConstraint(2, /*feature_id=*/5));

    std::vector<int32_t> accepted_features; // for node 3
    for (int32_t f = 0; f < 8; ++f) {
      if (eval->CheckFeatureConstraint(3, f)) {
        accepted_features.emplace_back(f);
      }
    }
    std::vector<int32_t> solutions{4, 5};
    ASSERT_EQ(accepted_features.size(), solutions.size());
    for (size_t f = 0; f < accepted_features.size(); ++f) {
      ASSERT_EQ(accepted_features[f], solutions[f]);
    }
  }

  {
    std::unique_ptr<SplitEvaluator> eval{
        SplitEvaluator::Create("elastic_net,interaction")};
    eval->Init(args);
    eval->AddSplit(/*node_id=*/0, /*left_id=*/1, /*right_id=*/2, /*feature_id=*/4, 0, 0);
    std::vector<int32_t> accepted_features; // for node 1
    for (int32_t f = 0; f < 8; ++f) {
      if (eval->CheckFeatureConstraint(1, f)) {
        accepted_features.emplace_back(f);
      }
    }
    ASSERT_EQ(accepted_features.size(), 1);
    ASSERT_EQ(accepted_features[0], 4);
  }
}

}  // namespace tree
}  // namespace xgboost
