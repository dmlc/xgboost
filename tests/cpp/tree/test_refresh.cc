/**
 * Copyright 2018-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/task.h>  // for ObjInfo
#include <xgboost/tree_updater.h>

#include <memory>
#include <string>
#include <vector>

#include "../../../src/tree/param.h"  // for TrainParam
#include "../helpers.h"

namespace xgboost::tree {
TEST(Updater, Refresh) {
  bst_row_t constexpr kRows = 8;
  bst_feature_t constexpr kCols = 16;

  HostDeviceVector<GradientPair> gpair =
      { {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f},
        {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f} };
  std::shared_ptr<DMatrix> p_dmat{
    RandomDataGenerator{kRows, kCols, 0.4f}.Seed(3).GenerateDMatrix()};
  std::vector<std::pair<std::string, std::string>> cfg{
      {"reg_alpha", "0.0"},
      {"num_feature", std::to_string(kCols)},
      {"reg_lambda", "1"}};

  RegTree tree = RegTree{1u, kCols};
  auto ctx = CreateEmptyGenericParam(GPUIDX);
  std::vector<RegTree*> trees{&tree};

  ObjInfo task{ObjInfo::kRegression};
  std::unique_ptr<TreeUpdater> refresher(TreeUpdater::Create("refresh", &ctx, &task));

  tree.ExpandNode(0, 2, 0.2f, false, 0.0, 0.2f, 0.8f, 0.0f, 0.0f,
                  /*left_sum=*/0.0f, /*right_sum=*/0.0f);
  int cleft = tree[0].LeftChild();
  int cright = tree[0].RightChild();

  tree.Stat(cleft).base_weight = 1.2;
  tree.Stat(cright).base_weight = 1.3;

  std::vector<HostDeviceVector<bst_node_t>> position;
  tree::TrainParam param;
  param.UpdateAllowUnknown(cfg);

  refresher->Update(&param, &gpair, p_dmat.get(), position, trees);

  bst_float constexpr kEps = 1e-6;
  ASSERT_NEAR(-0.183392, tree[cright].LeafValue(), kEps);
  ASSERT_NEAR(-0.224489, tree.Stat(0).loss_chg, kEps);
  ASSERT_NEAR(0, tree.Stat(cleft).loss_chg, kEps);
  ASSERT_NEAR(0, tree.Stat(1).loss_chg, kEps);
  ASSERT_NEAR(0, tree.Stat(2).loss_chg, kEps);
}
}  // namespace xgboost::tree
