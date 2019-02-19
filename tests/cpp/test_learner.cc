// Copyright by Contributors
#include <gtest/gtest.h>
#include <vector>
#include "helpers.h"
#include "xgboost/learner.h"

namespace xgboost {

TEST(Learner, Basic) {
  typedef std::pair<std::string, std::string> Arg;
  auto args = {Arg("tree_method", "exact")};
  auto mat_ptr = CreateDMatrix(10, 10, 0);
  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {*mat_ptr};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->Configure(args);

  delete mat_ptr;
}

TEST(Learner, SelectTreeMethod) {
  using Arg = std::pair<std::string, std::string>;
  auto mat_ptr = CreateDMatrix(10, 10, 0);
  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {*mat_ptr};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));

  // Test if `tree_method` can be set
  learner->Configure({Arg("tree_method", "approx")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_histmaker,prune");
  learner->Configure({Arg("tree_method", "exact")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_colmaker,prune");
  learner->Configure({Arg("tree_method", "hist")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_quantile_histmaker");
  learner->Configure({Arg{"booster", "dart"}, Arg{"tree_method", "hist"}});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_quantile_histmaker");
#ifdef XGBOOST_USE_CUDA
  learner->Configure({Arg("tree_method", "gpu_exact")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_gpu,prune");
  learner->Configure({Arg("tree_method", "gpu_hist")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_gpu_hist");
  learner->Configure({Arg{"booster", "dart"}, Arg{"tree_method", "gpu_hist"}});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_gpu_hist");
#endif

  delete mat_ptr;
}

TEST(Learner, CheckGroup) {
  using Arg = std::pair<std::string, std::string>;
  size_t constexpr kNumGroups = 4;
  size_t constexpr kNumRows = 17;
  size_t constexpr kNumCols = 15;

  auto pp_mat = CreateDMatrix(kNumRows, kNumCols, 0);
  auto& p_mat = *pp_mat;
  std::vector<bst_float> weight(kNumGroups);
  std::vector<bst_int> group(kNumGroups);
  group[0] = 2;
  group[1] = 3;
  group[2] = 7;
  group[3] = 5;
  std::vector<bst_float> labels (kNumRows);
  for (size_t i = 0; i < kNumRows; ++i) {
    labels[i] = i % 2;
  }

  p_mat->Info().SetInfo(
      "weight", static_cast<void*>(weight.data()), DataType::kFloat32, kNumGroups);
  p_mat->Info().SetInfo(
      "group", group.data(), DataType::kUInt32, kNumGroups);
  p_mat->Info().SetInfo("label", labels.data(), DataType::kFloat32, kNumRows);

  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {p_mat};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->Configure({Arg{"objective", "rank:pairwise"}});
  learner->InitModel();

  EXPECT_NO_THROW(learner->UpdateOneIter(0, p_mat.get()));

  group.resize(kNumGroups+1);
  group[3] = 4;
  group[4] = 1;
  p_mat->Info().SetInfo("group", group.data(), DataType::kUInt32, kNumGroups+1);
  EXPECT_ANY_THROW(learner->UpdateOneIter(0, p_mat.get()));

  delete pp_mat;
}

}  // namespace xgboost
