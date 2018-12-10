/*!
 * Copyright 2018 by Contributors
 */
#include "../helpers.h"
#include "../../../src/common/host_device_vector.h"
#include <xgboost/tree_updater.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <memory>

namespace xgboost {
namespace tree {

TEST(Updater, Prune) {
  int constexpr n_rows = 32, n_cols = 16;

  std::vector<std::pair<std::string, std::string>> cfg;
  cfg.push_back(std::pair<std::string, std::string>(
      "num_feature", std::to_string(n_cols)));
  cfg.push_back(std::pair<std::string, std::string>(
      "min_split_loss", "10"));
  cfg.push_back(std::pair<std::string, std::string>(
      "silent", "1"));

  // These data are just place holders.
  HostDeviceVector<GradientPair> gpair =
      { {0.50f, 0.25f}, {0.50f, 0.25f}, {0.50f, 0.25f}, {0.50f, 0.25f},
        {0.25f, 0.24f}, {0.25f, 0.24f}, {0.25f, 0.24f}, {0.25f, 0.24f} };
  auto dmat = CreateDMatrix(32, 16, 0.4, 3);

  // prepare tree
  RegTree tree = RegTree();
  tree.param.InitAllowUnknown(cfg);
  std::vector<RegTree*> trees {&tree};
  // prepare pruner
  std::unique_ptr<TreeUpdater> pruner(TreeUpdater::Create("prune"));
  pruner->Init(cfg);

  // loss_chg < min_split_loss;
  tree.AddChilds(0);
  int cleft = tree.GetNode(0).LeftChild();
  int cright = tree.GetNode(0).RightChild();
  tree.GetNode(cleft).SetLeaf(0.3f, 0);
  tree.GetNode(cright).SetLeaf(0.4f, 0);
  pruner->Update(&gpair, dmat->get(), trees);

  ASSERT_EQ(tree.NumExtraNodes(), 0);

  // loss_chg > min_split_loss;
  tree.AddChilds(0);
  cleft = tree.GetNode(0).LeftChild();
  cright = tree.GetNode(0).RightChild();
  tree.GetNode(cleft).SetLeaf(0.3f, 0);
  tree.GetNode(cright).SetLeaf(0.4f, 0);
  tree.Stat(0).loss_chg = 11;
  pruner->Update(&gpair, dmat->get(), trees);

  ASSERT_EQ(tree.NumExtraNodes(), 2);

  // loss_chg == min_split_loss;
  tree.Stat(0).loss_chg = 10;
  pruner->Update(&gpair, dmat->get(), trees);

  ASSERT_EQ(tree.NumExtraNodes(), 2);

  delete dmat;
}

}  // namespace tree
}  // namespace xgboost
