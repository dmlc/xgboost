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

TEST(Updater, Refresh) {
  int constexpr n_rows = 8, n_cols = 16;

  HostDeviceVector<GradientPair> gpair =
      { {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f}, {0.23f, 0.24f},
        {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f}, {0.27f, 0.29f} };
  auto dmat = CreateDMatrix(n_rows, n_cols, 0.4, 3);
  std::vector<std::pair<std::string, std::string>> cfg {
    {"reg_alpha", "0.0"},
    {"num_feature", std::to_string(n_cols)},
    {"reg_lambda", "1"}};

  RegTree tree = RegTree();
  tree.InitModel();
  tree.param.InitAllowUnknown(cfg);
  std::vector<RegTree*> trees {&tree};
  std::unique_ptr<TreeUpdater> refresher(TreeUpdater::Create("refresh"));

  tree.AddChilds(0);
  int cleft = tree[0].LeftChild();
  int cright = tree[0].RightChild();
  tree[cleft].SetLeaf(0.2f, 0);
  tree[cright].SetLeaf(0.8f, 0);
  tree[0].SetSplit(2, 0.2f);

  tree.Stat(cleft).base_weight = 1.2;
  tree.Stat(cright).base_weight = 1.3;

  refresher->Init(cfg);
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
