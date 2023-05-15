#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/logging.h>

#include <memory>
#include <string>

#include "../../../src/tree/constraints.h"
#include "../../../src/tree/hist/evaluate_splits.h"
#include "../helpers.h"

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

TEST(CPUMonoConstraint, Basic) {
  std::size_t kRows{64}, kCols{16};
  Context ctx;

  TrainParam param;
  std::vector<std::int32_t> mono(kCols, 1);
  I32Array arr;
  for (std::size_t i = 0; i < kCols; ++i) {
    arr.GetArray().push_back(mono[i]);
  }
  Json jarr{std::move(arr)};
  std::string str_mono;
  Json::Dump(jarr, &str_mono);
  str_mono.front() = '(';
  str_mono.back() = ')';

  param.UpdateAllowUnknown(Args{{"monotone_constraints", str_mono}});

  auto Xy = RandomDataGenerator{kRows, kCols, 0.0}.GenerateDMatrix(true);
  auto sampler = std::make_shared<common::ColumnSampler>();

  HistEvaluator evalutor{&ctx, &param, Xy->Info(), sampler};
  evalutor.InitRoot(GradStats{2.0, 2.0});

  SplitEntry split;
  split.Update(1.0f, 0, 3.0, false, false, GradStats{1.0, 1.0}, GradStats{1.0, 1.0});
  CPUExpandEntry entry{0, 0, split};
  RegTree tree{1, static_cast<bst_feature_t>(kCols)};
  evalutor.ApplyTreeSplit(entry, &tree);

  ASSERT_TRUE(evalutor.Evaluator().has_constraint);
}
}  // namespace tree
}  // namespace xgboost
