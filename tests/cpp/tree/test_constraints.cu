/*!
 * Copyright 2019 XGBoost contributors
 */
#include <gtest/gtest.h>
#include "../../../src/tree/constraints.cuh"
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/tree/param.h"

namespace xgboost {
namespace tree {

TEST(GPUInteractionConstraint, Init) {
  std::string const constraints_str = R"constraints(
[[0, 2], [1, 3, 4], [5, 6], [3, 5]]
)constraints";
  TrainParam param;
  using Arg = std::vector<std::pair<std::string, std::string>>;
  Arg args {{"interaction_constraints", constraints_str}};
  param.Init(args);

  size_t constexpr kCols = 10;
  InteractionConstraints constraints{param.interaction_constraints, kCols};
  std::vector< std::set<int32_t> > feature_interactions_solution {
    {0, 2},
    {1, 3, 4},
    {0, 2},
    {1, 3, 4, 5},
    {1, 3, 4},
    {3, 5, 6},
    {5, 6}};
  for (size_t i = feature_interactions_solution.size(); i < kCols; ++i) {
    feature_interactions_solution.emplace_back(std::set<int32_t>{});
  }
  ASSERT_EQ(feature_interactions_solution.size(),
            constraints.feature_interactions_.size());
  for (size_t i = 0;
       i < feature_interactions_solution.size();
       ++i) {
    ASSERT_EQ(constraints.feature_interactions_[i], feature_interactions_solution[i]);
  }

  // Correct device vector
  std::vector<int32_t> flatten_feature_interactions_solution;
  for (auto interactions : feature_interactions_solution) {
    for (auto feat : interactions) {
      flatten_feature_interactions_solution.emplace_back(feat);
    }
  }
  for (auto i = flatten_feature_interactions_solution.size();
       i < kCols;
       ++i) {
    flatten_feature_interactions_solution.emplace_back(19);
  }
  ASSERT_EQ(flatten_feature_interactions_solution.size(),
            constraints.d_feature_interactions_.size());
  for (size_t i = 0; i < constraints.d_feature_interactions_.size();
       ++i) {
    ASSERT_EQ(constraints.d_feature_interactions_[i],
              flatten_feature_interactions_solution.at(i));
  }

  // Correct ptr
  std::vector<int32_t> interactions_ptr_solution {
    0, 2, 5, 7, 11, 14, 17, 19, 19, 19, 19};
  ASSERT_EQ(constraints.d_feature_interactions_ptr_.size(),
            interactions_ptr_solution.size());
  for (size_t i = 0; i < constraints.d_feature_interactions_ptr_.size(); ++i) {
    ASSERT_EQ(interactions_ptr_solution.at(i),
              constraints.d_feature_interactions_ptr_[i]);
  }
}

TEST(GPUInteractionConstraint, ApplySplit) {
  std::string const constraints_str = R"constraints(
[[0, 2], [1, 3, 4], [5, 6], [3, 5]]
)constraints";
  TrainParam param;
  using Arg = std::vector<std::pair<std::string, std::string>>;
  Arg args {{"interaction_constraints", constraints_str}};
  param.Init(args);

  size_t constexpr kCols = 10;

  InteractionConstraints constraints(param.interaction_constraints, kCols);

  constraints.ApplySplit(0, 1, 2, /* feaature_id = */ 3);
  ASSERT_EQ(constraints.node_interactions_.size(), 3);
  ASSERT_EQ(constraints.d_node_interactions_.size(), 2);
  ASSERT_EQ(constraints.d_node_interactions_ptr_.size(), 4);

  std::vector<int32_t> flatten_node_interactions_solution(2);
  thrust::copy(constraints.d_node_interactions_.cbegin(),
               constraints.d_node_interactions_.cend(),
               flatten_node_interactions_solution.begin());
  for (size_t i = 0; i < flatten_node_interactions_solution.size(); ++i) {
    ASSERT_EQ(flatten_node_interactions_solution.at(i),
              constraints.d_node_interactions_[i]);
  }
  std::vector<int32_t> flatten_node_interactions_ptr_solution {
    0, 0, 1, 2
  };
  for (size_t i = 0; i < flatten_node_interactions_ptr_solution.size();
       ++i) {
    ASSERT_EQ(flatten_node_interactions_ptr_solution.at(i),
              constraints.d_node_interactions_ptr_[i]);
  }
}

// Avoid using private method in gtest.
void TestEvaluateSplit() {
  std::string const constraints_str = R"constraints(
[[0, 2], [1, 3, 4], [5, 6], [3, 5]]
)constraints";
  TrainParam param;
  using Arg = std::vector<std::pair<std::string, std::string>>;
  Arg args {{"interaction_constraints", constraints_str}};
  param.Init(args);

  size_t constexpr kCols = 10;

  InteractionConstraints constraints(param.interaction_constraints, kCols);
  constraints.ApplySplit(0, 1, 2, 3);

  auto device_constraints = constraints.split_evaluator_;

  thrust::device_vector<float> d_gain(1);
  auto s_gain = dh::ToSpan(d_gain);

  // Please refer to Init test for how `feather_interactions_` looke like.
  for (auto nid : {1, 2}) {
    dh::LaunchN(0, 1, [=]__device__(size_t idx) {
        s_gain[0] = device_constraints.EvaluateSplit(nid, /* fid = */ 0, 1.1f);
      });
    std::vector<float> h_gain(1);
    thrust::copy(d_gain.cbegin(), d_gain.cend(), h_gain.begin());
    ASSERT_EQ(h_gain[0], -std::numeric_limits<bst_float>::infinity());

    dh::LaunchN(0, 1, [=]__device__(size_t idx) {
        s_gain[0] = device_constraints.EvaluateSplit(nid, /* fid = */ 3, 1.1f);
      });
    thrust::copy(d_gain.cbegin(), d_gain.cend(), h_gain.begin());
    ASSERT_NEAR(h_gain[0], 1.1f, kRtEps);
  }
}

TEST(GPUInteractionConstraint, EvaluateSplit) {
  TestEvaluateSplit();
}

}  // namespace tree
}  // namespace xgboost