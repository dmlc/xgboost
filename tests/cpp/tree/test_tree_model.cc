// Copyright by Contributors
#include <gtest/gtest.h>
#include <xgboost/tree_model.h>
#include "../helpers.h"
#include "dmlc/filesystem.h"
#include "xgboost/json_io.h"

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

RegTree ConstructTree() {
  RegTree tree;
  tree.ExpandNode(
      /*nid=*/0, /*split_index=*/0, /*split_value=*/0.0f,
      /*default_left=*/true,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  auto left = tree[0].LeftChild();
  auto right = tree[0].RightChild();
  tree.ExpandNode(
      /*nid=*/left, /*split_index=*/1, /*split_value=*/1.0f,
      /*default_left=*/false,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  tree.ExpandNode(
      /*nid=*/right, /*split_index=*/2, /*split_value=*/2.0f,
      /*default_left=*/false,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  return tree;
}

TEST(Tree, DumpJson) {
  auto tree = ConstructTree();
  FeatureMap fmap;
  auto str = tree.DumpModel(fmap, true, "json");
  size_t n_leaves = 0;
  size_t iter = 0;
  while ((iter = str.find("leaf", iter + 1)) != std::string::npos) {
    n_leaves++;
  }
  ASSERT_EQ(n_leaves, 4);

  size_t n_conditions = 0;
  iter = 0;
  while ((iter = str.find("split_condition", iter + 1)) != std::string::npos) {
    n_conditions++;
  }
  ASSERT_EQ(n_conditions, 3);

  fmap.PushBack(0, "feat_0", "i");
  fmap.PushBack(1, "feat_1", "q");
  fmap.PushBack(2, "feat_2", "int");

  str = tree.DumpModel(fmap, true, "json");
  ASSERT_NE(str.find(R"("split": "feat_0")"), std::string::npos);
  ASSERT_NE(str.find(R"("split": "feat_1")"), std::string::npos);
  ASSERT_NE(str.find(R"("split": "feat_2")"), std::string::npos);

  str = tree.DumpModel(fmap, false, "json");
  ASSERT_EQ(str.find("cover"), std::string::npos);
}

TEST(Tree, DumpText) {
  auto tree = ConstructTree();
  FeatureMap fmap;
  auto str = tree.DumpModel(fmap, true, "text");
  size_t n_leaves = 0;
  size_t iter = 0;
  while ((iter = str.find("leaf", iter + 1)) != std::string::npos) {
    n_leaves++;
  }
  ASSERT_EQ(n_leaves, 4);

  iter = 0;
  size_t n_conditions = 0;
  while ((iter = str.find("gain", iter + 1)) != std::string::npos) {
    n_conditions++;
  }
  ASSERT_EQ(n_conditions, 3);

  ASSERT_NE(str.find("[f0<0]"), std::string::npos);
  ASSERT_NE(str.find("[f1<1]"), std::string::npos);
  ASSERT_NE(str.find("[f2<2]"), std::string::npos);

  fmap.PushBack(0, "feat_0", "i");
  fmap.PushBack(1, "feat_1", "q");
  fmap.PushBack(2, "feat_2", "int");

  str = tree.DumpModel(fmap, true, "text");
  ASSERT_NE(str.find("[feat_0]"), std::string::npos);
  ASSERT_NE(str.find("[feat_1<1]"), std::string::npos);
  ASSERT_NE(str.find("[feat_2<2]"), std::string::npos);

  str = tree.DumpModel(fmap, false, "text");
  ASSERT_EQ(str.find("cover"), std::string::npos);
}

TEST(Tree, DumpDot) {
  auto tree = ConstructTree();
  FeatureMap fmap;
  auto str = tree.DumpModel(fmap, true, "dot");

  size_t n_leaves = 0;
  size_t iter = 0;
  while ((iter = str.find("leaf", iter + 1)) != std::string::npos) {
    n_leaves++;
  }
  ASSERT_EQ(n_leaves, 4);

  size_t n_edges = 0;
  iter = 0;
  while ((iter = str.find("->", iter + 1)) != std::string::npos) {
    n_edges++;
  }
  ASSERT_EQ(n_edges, 6);

  fmap.PushBack(0, "feat_0", "i");
  fmap.PushBack(1, "feat_1", "q");
  fmap.PushBack(2, "feat_2", "int");

  str = tree.DumpModel(fmap, true, "dot");
  ASSERT_NE(str.find(R"("feat_0")"), std::string::npos);
  ASSERT_NE(str.find(R"(feat_1<1)"), std::string::npos);
  ASSERT_NE(str.find(R"(feat_2<2)"), std::string::npos);

  str = tree.DumpModel(fmap, true, R"(dot:{"graph_attrs": {"bgcolor": "#FFFF00"}})");
  ASSERT_NE(str.find(R"(graph [ bgcolor="#FFFF00" ])"), std::string::npos);
}

TEST(Tree, Json_IO) {
  RegTree tree;
  tree.ExpandNode(0, 0, 0.0f, false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  Json j_tree{Object()};
  tree.SaveModel(&j_tree);
  std::stringstream ss;
  Json::Dump(j_tree, &ss);

  auto tparam = j_tree["tree_param"];
  ASSERT_EQ(get<String>(tparam["num_feature"]), "0");
  ASSERT_EQ(get<String>(tparam["num_nodes"]), "3");
  ASSERT_EQ(get<String>(tparam["size_leaf_vector"]), "0");

  ASSERT_EQ(get<Array const>(j_tree["left_children"]).size(), 3);
  ASSERT_EQ(get<Array const>(j_tree["right_children"]).size(), 3);
  ASSERT_EQ(get<Array const>(j_tree["parents"]).size(), 3);
  ASSERT_EQ(get<Array const>(j_tree["split_indices"]).size(), 3);
  ASSERT_EQ(get<Array const>(j_tree["split_conditions"]).size(), 3);
  ASSERT_EQ(get<Array const>(j_tree["default_left"]).size(), 3);

  RegTree loaded_tree;
  loaded_tree.LoadModel(j_tree);
  ASSERT_EQ(loaded_tree.param.num_nodes, 3);
}

}  // namespace xgboost
