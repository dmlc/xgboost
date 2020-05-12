#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/logging.h>

#include <memory>
#include <string>

#include "../../../src/tree/constraints.h"

namespace xgboost {
namespace tree {

TEST(CPUFeatureInteractionConstraint, Empty) {
  TrainParam param;
  param.UpdateAllowUnknown(Args{});
  bst_feature_t constexpr kFeatures = 6;

  FeatureInteractionConstraintHost constraints;
  constraints.Configure(param, kFeatures);

  // no-op
  constraints.Split(/*node_id=*/0, /*feature_id=*/2, /*left_id=*/1, /*right_id=*/2);

  std::vector<bst_feature_t> h_input_feature_list {0, 1, 2, 3, 4, 5};
  common::Span<bst_feature_t> s_input_feature_list = common::Span<bst_feature_t>{h_input_feature_list};

  for (auto f : h_input_feature_list) {
    constraints.Query(f, 1);
  }

  // no-op
  ASSERT_TRUE(constraints.Query(94389, 12309));
}

TEST(CPUFeatureInteractionConstraint, Basic) {
  std::string const constraints_str = R"constraint([[1, 2], [2, 3, 4]])constraint";

  std::vector<std::pair<std::string, std::string>> args{
    {"interaction_constraints", constraints_str}};
  TrainParam param;
  param.interaction_constraints = constraints_str;
  bst_feature_t constexpr kFeatures = 6;

  FeatureInteractionConstraintHost constraints;
  constraints.Configure(param, kFeatures);
  constraints.Split(/*node_id=*/0, /*feature_id=*/2, /*left_id=*/1, /*right_id=*/2);

  std::vector<bst_feature_t> h_input_feature_list{0, 1, 2, 3, 4, 5};

  ASSERT_TRUE(constraints.Query(1, 1));
  ASSERT_TRUE(constraints.Query(1, 2));
  ASSERT_TRUE(constraints.Query(1, 3));
  ASSERT_TRUE(constraints.Query(1, 4));

  ASSERT_FALSE(constraints.Query(1, 0));
  ASSERT_FALSE(constraints.Query(1, 5));
}

}  // namespace tree
}  // namespace xgboost
