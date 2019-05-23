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

  int32_t constexpr kCols = 10;
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

  auto feature_set = std::make_shared<HostDeviceVector<int32_t>>();
  auto& h_feature_set = feature_set->HostVector();

  h_feature_set.emplace_back(0);
  h_feature_set.emplace_back(1);
  h_feature_set.emplace_back(6);

  {
    // only {1, 3, 4, 5} are allowed
    auto result = constraints.GetAllowedFeatures(feature_set, 1);
    auto& h_result = result->HostVector();
    std::vector<int32_t> solution {1};
    ASSERT_EQ(h_result.size(), solution.size());
    ASSERT_EQ(h_result[0], solution[0]);
  }
  {
    // no constraint for root
    auto result = constraints.GetAllowedFeatures(feature_set, /* nid = */ 0);
    auto& h_result = result->HostVector();
    std::vector<int32_t> solution {0, 1, 6};
    ASSERT_EQ(h_result.size(), solution.size());
  }

  constraints.ApplySplit(1, 3, 4, /* feaature_id = */ 0);
  h_feature_set.emplace_back(5);
  h_feature_set.emplace_back(6);

  {
    // node 3 merges constraints from node 1
    auto result = constraints.GetAllowedFeatures(feature_set, 3);
    auto& h_result = result->HostVector();
    std::vector<int32_t> solution {0, 1, 5};
    ASSERT_EQ(h_result.size(), solution.size());
    for (size_t i = 0; i < solution.size(); ++i) {
      ASSERT_EQ(h_result[i], solution[i]);
    }
  }
}
}  // namespace tree
}  // namespace xgboost