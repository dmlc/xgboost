/*!
 * Copyright 2018-2019 by Contributors
 */
#include <xgboost/host_device_vector.h>
#include <xgboost/tree_updater.h>
#include <gtest/gtest.h>

#include <vector>
#include <string>
#include <memory>

#include "../helpers.h"

namespace xgboost {
namespace tree {

TEST(Updater, Refresh) {
  int constexpr kNRows = 8, kNCols = 16;

  HostDeviceVector<GradientPair> gpair =
      { {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f},
        {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f} };
  auto dmat = CreateDMatrix(kNRows, kNCols, 0.4, 3);
  std::vector<std::pair<std::string, std::string>> cfg {
    {"reg_alpha", "0.0"},
    {"num_feature", std::to_string(kNCols)},
    {"reg_lambda", "1"}};

  RegTree tree = RegTree();
  auto lparam = CreateEmptyGenericParam(GPUIDX);
  tree.param.UpdateAllowUnknown(cfg);
  std::vector<RegTree*> trees {&tree};
  std::unique_ptr<TreeUpdater> refresher(TreeUpdater::Create("refresh", &lparam));

  tree.ExpandNode(0, 2, 0.2f, false, 0.0, 0.2f, 0.8f, 0.0f, 0.0f);
  int cleft = tree[0].LeftChild();
  int cright = tree[0].RightChild();

  tree.Stat(cleft).base_weight = 1.2;
  tree.Stat(cright).base_weight = 1.3;

  refresher->Configure(cfg);
  refresher->Update(&gpair, dmat->get(), trees);

  bst_float constexpr kEps = 1e-6;
  ASSERT_NEAR(-0.183392, tree[cright].LeafValue(), kEps);
  ASSERT_NEAR(-0.224489, tree.Stat(0).loss_chg, kEps);
  ASSERT_NEAR(0, tree.Stat(cleft).loss_chg, kEps);
  ASSERT_NEAR(0, tree.Stat(1).loss_chg, kEps);
  ASSERT_NEAR(0, tree.Stat(2).loss_chg, kEps);

  delete dmat;
}

}  // namespace tree
}  // namespace xgboost
