/**
 * Copyright 2018-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/learner.h>
#include <xgboost/tree_updater.h>

#include <memory>
#include <string>
#include <vector>

#include "../../../src/tree/param.h"  // for TrainParam
#include "../helpers.h"

namespace xgboost::tree {
TEST(Updater, Prune) {
  int constexpr kCols = 16;

  std::vector<std::pair<std::string, std::string>> cfg;
  cfg.emplace_back("num_feature", std::to_string(kCols));
  cfg.emplace_back("min_split_loss", "10");

  // These data are just place holders.
  HostDeviceVector<GradientPair> gpair =
      { {0.50f, 0.25f}, {0.50f, 0.25f}, {0.50f, 0.25f}, {0.50f, 0.25f},
        {0.25f, 0.24f}, {0.25f, 0.24f}, {0.25f, 0.24f}, {0.25f, 0.24f} };
  std::shared_ptr<DMatrix> p_dmat {
    RandomDataGenerator{32, 10, 0}.GenerateDMatrix() };

  auto ctx = CreateEmptyGenericParam(GPUIDX);

  // prepare tree
  RegTree tree = RegTree{1u, kCols};
  std::vector<RegTree*> trees {&tree};
  // prepare pruner
  TrainParam param;
  param.UpdateAllowUnknown(cfg);

  ObjInfo task{ObjInfo::kRegression};
  std::unique_ptr<TreeUpdater> pruner(TreeUpdater::Create("prune", &ctx, &task));

  // loss_chg < min_split_loss;
  std::vector<HostDeviceVector<bst_node_t>> position(trees.size());
  tree.ExpandNode(0, 0, 0, true, 0.0f, 0.3f, 0.4f, 0.0f, 0.0f,
                  /*left_sum=*/0.0f, /*right_sum=*/0.0f);
  pruner->Update(&param, &gpair, p_dmat.get(), position, trees);

  ASSERT_EQ(tree.NumExtraNodes(), 0);

  // loss_chg > min_split_loss;
  tree.ExpandNode(0, 0, 0, true, 0.0f, 0.3f, 0.4f, 11.0f, 0.0f,
                  /*left_sum=*/0.0f, /*right_sum=*/0.0f);
  pruner->Update(&param, &gpair, p_dmat.get(), position, trees);

  ASSERT_EQ(tree.NumExtraNodes(), 2);

  // loss_chg == min_split_loss;
  tree.Stat(0).loss_chg = 10;
  pruner->Update(&param, &gpair, p_dmat.get(), position, trees);

  ASSERT_EQ(tree.NumExtraNodes(), 2);

  // Test depth
  // loss_chg > min_split_loss
  tree.ExpandNode(tree[0].LeftChild(),
                  0, 0.5f, true, 0.3, 0.4, 0.5,
                  /*loss_chg=*/18.0f, 0.0f,
                  /*left_sum=*/0.0f, /*right_sum=*/0.0f);
  tree.ExpandNode(tree[0].RightChild(),
                  0, 0.5f, true, 0.3, 0.4, 0.5,
                  /*loss_chg=*/19.0f, 0.0f,
                  /*left_sum=*/0.0f, /*right_sum=*/0.0f);

  cfg.emplace_back("max_depth", "1");
  param.UpdateAllowUnknown(cfg);
  pruner->Update(&param, &gpair, p_dmat.get(), position, trees);
  ASSERT_EQ(tree.NumExtraNodes(), 2);

  tree.ExpandNode(tree[0].LeftChild(),
                  0, 0.5f, true, 0.3, 0.4, 0.5,
                  /*loss_chg=*/18.0f, 0.0f,
                  /*left_sum=*/0.0f, /*right_sum=*/0.0f);
  cfg.emplace_back("min_split_loss", "0");
  param.UpdateAllowUnknown(cfg);

  pruner->Update(&param, &gpair, p_dmat.get(), position, trees);
  ASSERT_EQ(tree.NumExtraNodes(), 2);
}
}  // namespace xgboost::tree
