// Copyright by Contributors
#include <gtest/gtest.h>
#include <xgboost/tree_model.h>
#include "../helpers.h"
#include "dmlc/filesystem.h"

namespace xgboost {
// Manually construct tree in binary format
// Do not use structs in case they change
// We want to preserve backwards compatibility
TEST(Tree, Load) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/tree.model";
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(tmp_file.c_str(), "w"));

  // Write params
  EXPECT_EQ(sizeof(TreeParam), (31 + 6) * sizeof(int));
  int num_roots = 1;
  int num_nodes = 2;
  int num_deleted = 0;
  int max_depth = 1;
  int num_feature = 0;
  int size_leaf_vector = 0;
  int reserved[31];
  fo->Write(&num_roots, sizeof(int));
  fo->Write(&num_nodes, sizeof(int));
  fo->Write(&num_deleted, sizeof(int));
  fo->Write(&max_depth, sizeof(int));
  fo->Write(&num_feature, sizeof(int));
  fo->Write(&size_leaf_vector, sizeof(int));
  fo->Write(reserved, sizeof(int) * 31);

  // Write 2 nodes
  EXPECT_EQ(sizeof(RegTree::Node),
            3 * sizeof(int) + 1 * sizeof(unsigned) + sizeof(float));
  int parent = -1;
  int cleft = 1;
  int cright = -1;
  unsigned sindex = 5;
  float split_or_weight = 0.5;
  fo->Write(&parent, sizeof(int));
  fo->Write(&cleft, sizeof(int));
  fo->Write(&cright, sizeof(int));
  fo->Write(&sindex, sizeof(unsigned));
  fo->Write(&split_or_weight, sizeof(float));
  parent = 0;
  cleft = -1;
  cright = -1;
  sindex = 2;
  split_or_weight = 0.1;
  fo->Write(&parent, sizeof(int));
  fo->Write(&cleft, sizeof(int));
  fo->Write(&cright, sizeof(int));
  fo->Write(&sindex, sizeof(unsigned));
  fo->Write(&split_or_weight, sizeof(float));

  // Write 2x node stats
  EXPECT_EQ(sizeof(RTreeNodeStat), 3 * sizeof(float) + sizeof(int));
  bst_float loss_chg = 5.0;
  bst_float sum_hess = 1.0;
  bst_float base_weight = 3.0;
  int leaf_child_cnt = 0;
  fo->Write(&loss_chg, sizeof(float));
  fo->Write(&sum_hess, sizeof(float));
  fo->Write(&base_weight, sizeof(float));
  fo->Write(&leaf_child_cnt, sizeof(int));

  loss_chg = 50.0;
  sum_hess = 10.0;
  base_weight = 30.0;
  leaf_child_cnt = 0;
  fo->Write(&loss_chg, sizeof(float));
  fo->Write(&sum_hess, sizeof(float));
  fo->Write(&base_weight, sizeof(float));
  fo->Write(&leaf_child_cnt, sizeof(int));
  fo.reset();
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(tmp_file.c_str(), "r"));

  xgboost::RegTree tree;
  tree.Load(fi.get());
  EXPECT_EQ(tree.GetDepth(1), 1);
  EXPECT_EQ(tree[0].SplitCond(), 0.5f);
  EXPECT_EQ(tree[0].SplitIndex(), 5);
  EXPECT_EQ(tree[1].LeafValue(), 0.1f);
  EXPECT_TRUE(tree[1].IsLeaf());
}

TEST(Tree, AllocateNode) {
  RegTree tree;
  tree.ExpandNode(
      0, 0, 0.0f, false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  tree.CollapseToLeaf(0, 0);
  ASSERT_EQ(tree.NumExtraNodes(), 0);

  tree.ExpandNode(
      0, 0, 0.0f, false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  ASSERT_EQ(tree.NumExtraNodes(), 2);

  auto& nodes = tree.GetNodes();
  ASSERT_FALSE(nodes.at(1).IsDeleted());
  ASSERT_TRUE(nodes.at(1).IsLeaf());
  ASSERT_TRUE(nodes.at(2).IsLeaf());
}
}  // namespace xgboost
