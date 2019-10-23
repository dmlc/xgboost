/*!
 * Copyright 2018-2019 by Contributors
 */
#include "../helpers.h"
#include <xgboost/host_device_vector.h>
#include <xgboost/tree_updater.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <memory>

namespace xgboost {
namespace tree {

TEST(Updater, Prune) {
  int constexpr kNCols = 16;

  std::vector<std::pair<std::string, std::string>> cfg;
  cfg.emplace_back(std::pair<std::string, std::string>(
      "num_feature", std::to_string(kNCols)));
  cfg.emplace_back(std::pair<std::string, std::string>(
      "min_split_loss", "10"));
  cfg.emplace_back(std::pair<std::string, std::string>(
      "silent", "1"));

  // These data are just place holders.
  HostDeviceVector<GradientPair> gpair =
      { {0.50f, 0.25f}, {0.50f, 0.25f}, {0.50f, 0.25f}, {0.50f, 0.25f},
        {0.25f, 0.24f}, {0.25f, 0.24f}, {0.25f, 0.24f}, {0.25f, 0.24f} };
  auto dmat = CreateDMatrix(32, 16, 0.4, 3);

  auto lparam = CreateEmptyGenericParam(GPUIDX);

  // prepare tree
  RegTree tree = RegTree();
  tree.param.UpdateAllowUnknown(cfg);
  std::vector<RegTree*> trees {&tree};
  // prepare pruner
  std::unique_ptr<TreeUpdater> pruner(TreeUpdater::Create("prune", &lparam));
  pruner->Configure(cfg);

  // loss_chg < min_split_loss;
  tree.ExpandNode(0, 0, 0, true, 0.0f, 0.3f, 0.4f, 0.0f, 0.0f);
  pruner->Update(&gpair, dmat->get(), trees);

  ASSERT_EQ(tree.NumExtraNodes(), 0);

  // loss_chg > min_split_loss;
  tree.ExpandNode(0, 0, 0, true, 0.0f, 0.3f, 0.4f, 11.0f, 0.0f);
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
