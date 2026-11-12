/**
 * Copyright 2024-2026, XGBoost contributors
 *
 * @brief Unit tests for Tree-G updater
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/context.h>
#include <xgboost/data.h>
#include <xgboost/gbm.h>
#include <xgboost/gradient.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "../../../plugin/treeg/booster.h"
#include "../../../plugin/treeg/model.h"
#include "../../../plugin/treeg/updater.h"
#include "../../../src/tree/param.h"  // for tree::CalcGain, tree::CalcWeight
#include "../helpers.h"

namespace xgboost {
namespace treeg {

// ========== Test Helpers ==========

/**
 * @brief Create a simple in-memory DMatrix for testing.
 *
 * Creates a DMatrix with format [X | A | A^2] where:
 * - X: n x k feature matrix
 * - A: n x n adjacency matrix
 * - A^2: n x n second-order adjacency matrix
 */
class TestDMatrix {
 public:
  TestDMatrix(bst_node_t n_vertices, bst_feature_t n_features, bst_node_t max_walk_len)
      : n_vertices_{n_vertices}, n_features_{n_features}, max_walk_len_{max_walk_len} {
    // DMatrix column layout: [X | A^1 | A^2 | ... | A^max_walk_len]
    //   - X occupies columns [0, n_features)
    //   - A^d occupies columns [n_features + (d-1)*n, n_features + d*n)
    // A^0 = I is implicit and handled by the model at d=0 (raw feature lookup).
    total_features_ = n_features + n_vertices * max_walk_len;

    // Initialize data storage
    data_.resize(n_vertices_);
  }

  void SetFeature(bst_node_t vertex, bst_feature_t feature, bst_float value) {
    CHECK_LT(vertex, n_vertices_);
    CHECK_LT(feature, n_features_);
    data_[vertex][feature] = value;
  }

  /**
   * @brief Set an entry in the adjacency power matrix A^walk_len.
   * @param walk_len  Must be >= 1.  A^0 = I is implicit.
   */
  void SetAdjacency(bst_node_t from_vertex, bst_node_t to_vertex, bst_node_t walk_len,
                    bst_float value) {
    CHECK_LT(from_vertex, n_vertices_);
    CHECK_LT(to_vertex, n_vertices_);
    CHECK_GE(walk_len, 1) << "A^0 = I is implicit; use walk_len >= 1";
    CHECK_LE(walk_len, max_walk_len_);

    bst_feature_t col = n_features_ + (walk_len - 1) * n_vertices_ + to_vertex;
    data_[from_vertex][col] = value;
  }

  std::shared_ptr<DMatrix> CreateDMatrix() {
    // Convert sparse data to dense format (fine for small test graphs)
    std::vector<float> dense_data(n_vertices_ * total_features_, 0.0f);

    for (bst_node_t i = 0; i < n_vertices_; ++i) {
      for (const auto& [fid, fvalue] : data_[i]) {
        dense_data[i * total_features_ + fid] = fvalue;
      }
    }

    // Use XGBoost's helper function for creating DMatrix from dense data
    return GetDMatrixFromData(dense_data, n_vertices_, total_features_);
  }

 private:
  bst_node_t n_vertices_;
  bst_feature_t n_features_;
  bst_node_t max_walk_len_;
  bst_feature_t total_features_;
  std::vector<std::map<bst_feature_t, bst_float>> data_;
};

// ========== GradStats Tests ==========

TEST(GradStats, BasicOperations) {
  GradStats stats1(10.0, 20.0);
  ASSERT_EQ(stats1.sum_grad, 10.0);
  ASSERT_EQ(stats1.sum_hess, 20.0);

  GradStats stats2(5.0, 10.0);

  // Test addition
  GradStats sum = stats1 + stats2;
  ASSERT_EQ(sum.sum_grad, 15.0);
  ASSERT_EQ(sum.sum_hess, 30.0);

  // Test += operator
  stats1 += stats2;
  ASSERT_EQ(stats1.sum_grad, 15.0);
  ASSERT_EQ(stats1.sum_hess, 30.0);
}

TEST(GradStats, SufficientWeight) {
  GradStats heavy(10.0, 20.0);
  GradStats light(5.0, 2.0);

  ASSERT_TRUE(heavy.HasSufficientWeight(10.0f));
  ASSERT_TRUE(heavy.HasSufficientWeight(20.0f));
  ASSERT_FALSE(heavy.HasSufficientWeight(25.0f));

  ASSERT_TRUE(light.HasSufficientWeight(1.0f));
  ASSERT_FALSE(light.HasSufficientWeight(5.0f));
}

TEST(GradStats, IsEmpty) {
  GradStats empty;
  ASSERT_TRUE(empty.IsEmpty());

  GradStats non_empty(1.0, 0.5);
  ASSERT_FALSE(non_empty.IsEmpty());
}

// ========== SplitEntry Tests ==========

TEST(SplitEntry, IsValid) {
  SplitEntry invalid;
  ASSERT_FALSE(invalid.IsValid());

  SplitEntry valid;
  valid.loss_chg = 1.5f;
  ASSERT_TRUE(valid.IsValid());
}

TEST(SplitEntry, Update) {
  SplitEntry current;
  current.loss_chg = 1.0f;

  SplitEntry better;
  better.loss_chg = 2.0f;
  better.split_feature = 5;

  SplitEntry worse;
  worse.loss_chg = 0.5f;

  // Update with better split
  ASSERT_TRUE(current.Update(better));
  ASSERT_EQ(current.loss_chg, 2.0f);
  ASSERT_EQ(current.split_feature, 5);

  // Update with worse split
  ASSERT_FALSE(current.Update(worse));
  ASSERT_EQ(current.loss_chg, 2.0f);
}

// ========== TreeGTrainParam Tests ==========

TEST(TreeGTrainParam, DefaultValues) {
  TreeGTrainParam param;

  ASSERT_EQ(param.learning_rate, 0.3f);
  ASSERT_EQ(param.min_split_loss, 0.0f);
  ASSERT_EQ(param.max_leaves, 32);
  ASSERT_EQ(param.min_child_weight, 1.0f);
  ASSERT_EQ(param.reg_lambda, 1.0f);
  ASSERT_EQ(param.reg_alpha, 0.0f);
  ASSERT_EQ(param.max_walk_len, 2);
  ASSERT_EQ(param.max_ancestor_distance, 2);
}

TEST(TreeGTrainParam, Configuration) {
  TreeGTrainParam param;

  Args args{{"learning_rate", "0.1"}, {"max_leaves", "64"},  {"min_child_weight", "5.0"},
            {"reg_lambda", "2.0"},    {"max_walk_len", "3"}, {"num_features", "10"},
            {"num_vertices", "100"}};

  param.UpdateAllowUnknown(args);

  ASSERT_EQ(param.learning_rate, 0.1f);
  ASSERT_EQ(param.max_leaves, 64);
  ASSERT_EQ(param.min_child_weight, 5.0f);
  ASSERT_EQ(param.reg_lambda, 2.0f);
  ASSERT_EQ(param.max_walk_len, 3);
  ASSERT_EQ(param.num_features, 10);
  ASSERT_EQ(param.num_vertices, 100);
}

// ========== TreeGUpdater Tests ==========

class TreeGUpdaterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ctx_ = std::make_unique<Context>();
    updater_ = std::make_unique<TreeGUpdater>(ctx_.get());
  }

  std::unique_ptr<Context> ctx_;
  std::unique_ptr<TreeGUpdater> updater_;
};

TEST_F(TreeGUpdaterTest, Configure) {
  Args args{{"learning_rate", "0.1"}, {"max_leaves", "16"},  {"min_child_weight", "2.0"},
            {"reg_lambda", "0.5"},    {"max_walk_len", "1"}, {"max_ancestor_distance", "1"},
            {"num_features", "5"},    {"num_vertices", "10"}};

  ASSERT_NO_THROW(updater_->Configure(args));
}

TEST_F(TreeGUpdaterTest, ConfigureInvalidParameters) {
  Args args{{"num_vertices", "0"},  // Invalid
            {"num_features", "5"}};

  ASSERT_THROW(updater_->Configure(args), dmlc::Error);
}

TEST_F(TreeGUpdaterTest, AttentionTypeSampleProbabilityOneMatchesDefaultBehavior) {
  constexpr bst_node_t kNumVertices = 4;
  constexpr bst_feature_t kNumFeatures = 2;
  constexpr bst_node_t kMaxWalkLen = 1;

  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);

  test_data.SetFeature(0, 0, 1.0f);
  test_data.SetFeature(0, 1, 0.5f);
  test_data.SetFeature(1, 0, 2.0f);
  test_data.SetFeature(1, 1, 1.5f);
  test_data.SetFeature(2, 0, 0.5f);
  test_data.SetFeature(2, 1, 0.2f);
  test_data.SetFeature(3, 0, 1.5f);
  test_data.SetFeature(3, 1, 1.0f);

  test_data.SetAdjacency(0, 1, 1, 1.0f);
  test_data.SetAdjacency(0, 2, 1, 1.0f);
  test_data.SetAdjacency(1, 0, 1, 1.0f);
  test_data.SetAdjacency(1, 3, 1, 1.0f);
  test_data.SetAdjacency(2, 0, 1, 1.0f);
  test_data.SetAdjacency(2, 3, 1, 1.0f);
  test_data.SetAdjacency(3, 1, 1, 1.0f);
  test_data.SetAdjacency(3, 2, 1, 1.0f);

  auto dmat = test_data.CreateDMatrix();

  std::vector<GradientPair> gpairs = {GradientPair(-0.5f, 0.25f), GradientPair(-0.3f, 0.21f),
                                      GradientPair(0.4f, 0.24f), GradientPair(0.5f, 0.25f)};

  Args base_args{{"learning_rate", "1.0"},
                 {"max_leaves", "3"},
                 {"min_child_weight", "0.2"},
                 {"reg_lambda", "0.0"},
                 {"reg_alpha", "0.0"},
                 {"min_split_loss", "0.0"},
                 {"max_walk_len", std::to_string(kMaxWalkLen)},
                 {"max_ancestor_distance", "1"},
                 {"num_features", std::to_string(kNumFeatures)},
                 {"num_vertices", std::to_string(kNumVertices)}};

  Context default_ctx;
  default_ctx.seed = 17;
  TreeGUpdater default_updater(&default_ctx);
  default_updater.Configure(base_args);

  RegTreeG default_tree;
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  default_updater.Update(gpairs, graph_data, &default_tree);

  Context prob_one_ctx;
  prob_one_ctx.seed = 17;
  TreeGUpdater prob_one_updater(&prob_one_ctx);
  Args prob_one_args = base_args;
  prob_one_args.emplace_back("attention_type_sample_probability", "1.0");
  prob_one_updater.Configure(prob_one_args);

  RegTreeG prob_one_tree;
  prob_one_updater.Update(gpairs, graph_data, &prob_one_tree);

  ASSERT_TRUE(default_tree == prob_one_tree)
      << "attention_type_sample_probability=1.0 must preserve the legacy split search";
}

TEST_F(TreeGUpdaterTest, AttentionTypeSampleProbabilityZeroForcesFallbackConfiguration) {
  ctx_->seed = 23;

  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 0;

  Args args{{"learning_rate", "1.0"},
            {"max_leaves", "4"},
            {"min_child_weight", "0.1"},
            {"reg_lambda", "0.0"},
            {"reg_alpha", "0.0"},
            {"min_split_loss", "0.0"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "2"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)},
            {"attention_type_sample_probability", "0.0"}};

  updater_->Configure(args);

  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    test_data.SetFeature(i, 0, static_cast<bst_float>(i));
  }
  auto dmat = test_data.CreateDMatrix();

  std::vector<GradientPair> gpairs = {GradientPair(-10.0f, 1.0f), GradientPair(-9.0f, 1.0f),
                                      GradientPair(-8.0f, 1.0f),  GradientPair(-7.0f, 1.0f),
                                      GradientPair(-6.0f, 1.0f),  GradientPair(6.0f, 1.0f),
                                      GradientPair(7.0f, 1.0f),   GradientPair(8.0f, 1.0f),
                                      GradientPair(9.0f, 1.0f),   GradientPair(10.0f, 1.0f)};

  RegTreeG tree;
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  updater_->Update(gpairs, graph_data, &tree);

  bst_node_t split_nodes = 0;
  for (bst_node_t i = 0; i < tree.GetParam().num_nodes; ++i) {
    if (tree[i].IsLeaf()) {
      continue;
    }
    ++split_nodes;
    ASSERT_EQ(tree[i].RestrictionTypeValue(), RestrictionType::kSource)
        << "Fallback must force r=1 (source restriction type) on every split node";
    ASSERT_EQ(tree[i].AncestorIndex(), RegTreeG::kInvalidNodeId)
        << "Fallback must force all-vertices attention (effective max attention depth = 0)";
  }

  ASSERT_GE(split_nodes, 1)
      << "Expected at least one split so the fallback configuration is exercised";
}

TEST_F(TreeGUpdaterTest, SimpleTreeGrowth) {
  // Setup parameters
  constexpr bst_node_t kNumVertices = 4;
  constexpr bst_feature_t kNumFeatures = 2;
  constexpr bst_node_t kMaxWalkLen = 1;

  Args args{{"learning_rate", "1.0"},
            {"max_leaves", "3"},
            {"min_child_weight", "0.2"},  // Lowered to allow splits with test data
            {"reg_lambda", "0.0"},
            {"reg_alpha", "0.0"},
            {"min_split_loss", "0.0"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "1"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};

  updater_->Configure(args);

  // Create simple test data
  // Graph: 0 -- 1
  //        |    |
  //        2 -- 3
  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);

  // Set features
  test_data.SetFeature(0, 0, 1.0);
  test_data.SetFeature(0, 1, 0.5);
  test_data.SetFeature(1, 0, 2.0);
  test_data.SetFeature(1, 1, 1.5);
  test_data.SetFeature(2, 0, 0.5);
  test_data.SetFeature(2, 1, 0.2);
  test_data.SetFeature(3, 0, 1.5);
  test_data.SetFeature(3, 1, 1.0);

  // Set adjacency matrix A (walk_len = 1)
  // Symmetric graph
  test_data.SetAdjacency(0, 1, 1, 1.0);
  test_data.SetAdjacency(0, 2, 1, 1.0);
  test_data.SetAdjacency(1, 0, 1, 1.0);
  test_data.SetAdjacency(1, 3, 1, 1.0);
  test_data.SetAdjacency(2, 0, 1, 1.0);
  test_data.SetAdjacency(2, 3, 1, 1.0);
  test_data.SetAdjacency(3, 1, 1, 1.0);
  test_data.SetAdjacency(3, 2, 1, 1.0);

  auto dmat = test_data.CreateDMatrix();

  // Create gradient pairs (simulate classification problem)
  std::vector<GradientPair> gpairs = {
      GradientPair(-0.5f, 0.25f),  // vertex 0
      GradientPair(-0.3f, 0.21f),  // vertex 1
      GradientPair(0.4f, 0.24f),   // vertex 2
      GradientPair(0.5f, 0.25f)    // vertex 3
  };

  // All vertices have equal weight

  // Create tree
  RegTreeG tree;

  // Update tree
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  ASSERT_NO_THROW(updater_->Update(gpairs, graph_data, &tree));

  // Verify tree structure
  ASSERT_GT(tree.GetParam().num_nodes, 1) << "Tree should have been expanded";

  // Root should be a split node (not a leaf)
  ASSERT_FALSE(tree[0].IsLeaf()) << "Root should be split";

  // Check that children exist
  bst_node_t left_child = tree[0].LeftChild();
  bst_node_t right_child = tree[0].RightChild();

  ASSERT_NE(left_child, RegTreeG::kInvalidNodeId);
  ASSERT_NE(right_child, RegTreeG::kInvalidNodeId);
}

TEST_F(TreeGUpdaterTest, WeightMasking) {
  // Test that vertices with zero weight are excluded from splits
  constexpr bst_node_t kNumVertices = 6;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 0;

  Args args{{"learning_rate", "1.0"},
            {"max_leaves", "2"},
            {"min_child_weight", "0.1"},
            {"reg_lambda", "0.0"},
            {"min_split_loss", "0.0"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "0"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};

  updater_->Configure(args);

  // Create simple data
  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);

  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    test_data.SetFeature(i, 0, static_cast<bst_float>(i));
  }

  auto dmat = test_data.CreateDMatrix();

  // Gradients for all vertices.  In production, the XGBoost objective
  // pre-applies sample weights to gradients.  Simulate that here:
  // training vertices (first 3) keep their gradients, validation
  // vertices (last 3, weight = 0) get zeroed-out gradients.
  std::vector<GradientPair> gpairs = {
      GradientPair(1.0f, 1.0f),  // weight=1: grad*1=1
      GradientPair(1.0f, 1.0f),  // weight=1: grad*1=1
      GradientPair(1.0f, 1.0f),  // weight=1: grad*1=1
      GradientPair(0.0f, 0.0f),  // weight=0: grad*0=0
      GradientPair(0.0f, 0.0f),  // weight=0: grad*0=0
      GradientPair(0.0f, 0.0f)   // weight=0: grad*0=0
  };

  RegTreeG tree;
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  updater_->Update(gpairs, graph_data, &tree);
  double expected_sum_hess = 3.0;  // 3 vertices with weight 1.0 and hess 1.0

  ASSERT_NEAR(tree.Stat(0).sum_hess, expected_sum_hess, 1e-5);
}

TEST_F(TreeGUpdaterTest, MinChildWeightConstraint) {
  // Test that splits respect min_child_weight constraint
  constexpr bst_node_t kNumVertices = 4;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 0;

  Args args{{"learning_rate", "1.0"},
            {"max_leaves", "10"},
            {"min_child_weight", "2.0"},  // Require at least 2 hessian sum per child
            {"reg_lambda", "0.0"},
            {"min_split_loss", "0.0"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "0"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};

  updater_->Configure(args);

  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);

  // Features: 1, 2, 3, 4
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    test_data.SetFeature(i, 0, static_cast<bst_float>(i + 1));
  }

  auto dmat = test_data.CreateDMatrix();

  // Each vertex has hessian = 1.0
  std::vector<GradientPair> gpairs(kNumVertices, GradientPair(1.0f, 1.0f));

  RegTreeG tree;
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  updater_->Update(gpairs, graph_data, &tree);
  if (!tree[0].IsLeaf()) {
    bst_node_t left_child = tree[0].LeftChild();
    bst_node_t right_child = tree[0].RightChild();

    ASSERT_GE(tree.Stat(left_child).sum_hess, 2.0f);
    ASSERT_GE(tree.Stat(right_child).sum_hess, 2.0f);
  }
}

TEST_F(TreeGUpdaterTest, MaxLeavesConstraint) {
  // Test that tree respects max_leaves constraint
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 0;
  constexpr bst_node_t kMaxLeaves = 4;

  Args args{{"learning_rate", "1.0"},
            {"max_leaves", std::to_string(kMaxLeaves)},
            {"min_child_weight", "0.1"},
            {"reg_lambda", "0.0"},
            {"min_split_loss", "0.0"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "0"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};

  updater_->Configure(args);

  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);

  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    test_data.SetFeature(i, 0, static_cast<bst_float>(i));
  }

  auto dmat = test_data.CreateDMatrix();

  // Create gradients with clear patterns for splitting
  std::vector<GradientPair> gpairs;
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    gpairs.emplace_back(static_cast<float>(i - 5), 1.0f);
  }

  RegTreeG tree;
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  updater_->Update(gpairs, graph_data, &tree);
  bst_node_t leaf_count = 0;
  for (bst_node_t i = 0; i < tree.GetParam().num_nodes; ++i) {
    if (tree[i].IsLeaf()) {
      leaf_count++;
    }
  }

  ASSERT_LE(leaf_count, kMaxLeaves) << "Tree has more leaves than max_leaves";
}

TEST_F(TreeGUpdaterTest, RegularizationEffects) {
  // Test that L2 regularization affects gain calculation
  constexpr bst_node_t kNumVertices = 6;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 0;

  // Test with no regularization
  {
    Args args{{"learning_rate", "1.0"},
              {"max_leaves", "5"},
              {"min_child_weight", "0.1"},
              {"reg_lambda", "0.0"},
              {"min_split_loss", "0.0"},
              {"max_walk_len", std::to_string(kMaxWalkLen)},
              {"max_ancestor_distance", "0"},
              {"num_features", std::to_string(kNumFeatures)},
              {"num_vertices", std::to_string(kNumVertices)}};

    updater_->Configure(args);

    TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);
    for (bst_node_t i = 0; i < kNumVertices; ++i) {
      test_data.SetFeature(i, 0, static_cast<bst_float>(i));
    }
    auto dmat = test_data.CreateDMatrix();
    std::vector<GradientPair> gpairs(kNumVertices, GradientPair(1.0f, 1.0f));
    auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);

    RegTreeG tree_no_reg;
    updater_->Update(gpairs, graph_data, &tree_no_reg);

    bst_node_t nodes_no_reg = tree_no_reg.GetParam().num_nodes;

    // Test with high regularization
    Args args_high_reg{{"learning_rate", "1.0"},
                       {"max_leaves", "5"},
                       {"min_child_weight", "0.1"},
                       {"reg_lambda", "10.0"},
                       {"min_split_loss", "0.0"},
                       {"max_walk_len", std::to_string(kMaxWalkLen)},
                       {"max_ancestor_distance", "0"},
                       {"num_features", std::to_string(kNumFeatures)},
                       {"num_vertices", std::to_string(kNumVertices)}};
    updater_->Configure(args_high_reg);

    RegTreeG tree_high_reg;
    updater_->Update(gpairs, graph_data, &tree_high_reg);

    bst_node_t nodes_high_reg = tree_high_reg.GetParam().num_nodes;

    // High regularization should result in fewer splits (more conservative)
    ASSERT_LE(nodes_high_reg, nodes_no_reg) << "High regularization should create simpler trees";
  }
}

// ========== Intercept Feature Tests ==========

TEST_F(TreeGUpdaterTest, InterceptConfigure) {
  // The updater should accept intercept=1 without error and propagate the flag
  // to the RegTreeG it constructs via InitRoot.
  constexpr bst_node_t kNumVertices = 4;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 1;

  Args args{{"learning_rate", "1.0"},
            {"max_leaves", "2"},
            {"min_child_weight", "0.1"},
            {"reg_lambda", "0.0"},
            {"min_split_loss", "0.0"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "0"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)},
            {"intercept", "1"}};

  ASSERT_NO_THROW(updater_->Configure(args));

  // Build a trivial DMatrix and run Update so we can inspect the tree.
  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    test_data.SetFeature(i, 0, static_cast<bst_float>(i));
  }
  auto dmat = test_data.CreateDMatrix();

  std::vector<GradientPair> gpairs(kNumVertices, GradientPair(1.0f, 1.0f));

  RegTreeG tree;
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  updater_->Update(gpairs, graph_data, &tree);
  ASSERT_TRUE(tree.Intercept());
}

TEST_F(TreeGUpdaterTest, InterceptTreeGrowthBasic) {
  // Smoke test: with intercept=true on a 4-vertex grid graph the tree should
  // still grow normally (no crashes, root gets split, children exist).
  constexpr bst_node_t kNumVertices = 4;
  constexpr bst_feature_t kNumFeatures = 2;
  constexpr bst_node_t kMaxWalkLen = 1;

  Args args{{"learning_rate", "1.0"},
            {"max_leaves", "3"},
            {"min_child_weight", "0.2"},
            {"reg_lambda", "0.0"},
            {"reg_alpha", "0.0"},
            {"min_split_loss", "0.0"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "1"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)},
            {"intercept", "1"}};

  updater_->Configure(args);

  // Graph: 0 -- 1
  //        |    |
  //        2 -- 3
  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);

  test_data.SetFeature(0, 0, 1.0);
  test_data.SetFeature(0, 1, 0.5);
  test_data.SetFeature(1, 0, 2.0);
  test_data.SetFeature(1, 1, 1.5);
  test_data.SetFeature(2, 0, 0.5);
  test_data.SetFeature(2, 1, 0.2);
  test_data.SetFeature(3, 0, 1.5);
  test_data.SetFeature(3, 1, 1.0);

  // Adjacency A^1
  test_data.SetAdjacency(0, 1, 1, 1.0);
  test_data.SetAdjacency(0, 2, 1, 1.0);
  test_data.SetAdjacency(1, 0, 1, 1.0);
  test_data.SetAdjacency(1, 3, 1, 1.0);
  test_data.SetAdjacency(2, 0, 1, 1.0);
  test_data.SetAdjacency(2, 3, 1, 1.0);
  test_data.SetAdjacency(3, 1, 1, 1.0);
  test_data.SetAdjacency(3, 2, 1, 1.0);

  auto dmat = test_data.CreateDMatrix();

  std::vector<GradientPair> gpairs = {GradientPair(-0.5f, 0.25f), GradientPair(-0.3f, 0.21f),
                                      GradientPair(0.4f, 0.24f), GradientPair(0.5f, 0.25f)};

  RegTreeG tree;
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  ASSERT_NO_THROW(updater_->Update(gpairs, graph_data, &tree));

  ASSERT_TRUE(tree.Intercept());
  ASSERT_GT(tree.GetParam().num_nodes, 1) << "Tree should have been expanded";
  ASSERT_FALSE(tree[0].IsLeaf()) << "Root should be split";

  bst_node_t left = tree[0].LeftChild();
  bst_node_t right = tree[0].RightChild();
  ASSERT_NE(left, RegTreeG::kInvalidNodeId);
  ASSERT_NE(right, RegTreeG::kInvalidNodeId);
}

TEST_F(TreeGUpdaterTest, InterceptFeatureSelectedAsSplit) {
  // When all real features are identical across vertices, no threshold from X
  // alone can produce a split.  But a star graph has different vertex degrees
  // (hub=5, leaves=1), so the intercept feature at d=1 Source computes the
  // row-sum of A (= degree).  With gradients designed so the hub has opposite
  // sign from leaves, the only viable split is on the intercept feature.
  //
  // Star graph (6 vertices):
  //   hub = vertex 0, connected to 1..5
  //   A^1 row sums: hub->5, leaves->1
  constexpr bst_node_t kNumVertices = 6;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 1;

  Args args{{"learning_rate", "1.0"},
            {"max_leaves", "2"},
            {"min_child_weight", "0.1"},
            {"reg_lambda", "0.0"},
            {"reg_alpha", "0.0"},
            {"min_split_loss", "0.0"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "0"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)},
            {"intercept", "1"}};

  updater_->Configure(args);

  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);

  // All real features set to 0.0 so that at every walk length the weighted
  // sum A^d @ mask @ X[·, 0] equals 0, producing no threshold variation.
  // Only the virtual intercept (which substitutes 1.0 for x_val) can create
  // non-zero, degree-dependent split values.
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    test_data.SetFeature(i, 0, 0.0f);
  }

  // Star adjacency A^1: hub (0) <-> leaves (1..5)
  for (bst_node_t leaf = 1; leaf < kNumVertices; ++leaf) {
    test_data.SetAdjacency(0, leaf, 1, 1.0);
    test_data.SetAdjacency(leaf, 0, 1, 1.0);
  }

  auto dmat = test_data.CreateDMatrix();

  // Hub has strong negative gradient; leaves have positive gradient.
  // This ensures the degree-based intercept split hub-vs-leaves has high gain.
  std::vector<GradientPair> gpairs;
  gpairs.emplace_back(-5.0f, 1.0f);  // hub
  for (bst_node_t i = 1; i < kNumVertices; ++i) {
    gpairs.emplace_back(1.0f, 1.0f);  // leaves
  }

  RegTreeG tree;
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  updater_->Update(gpairs, graph_data, &tree);
  ASSERT_FALSE(tree[0].IsLeaf()) << "Root should be split";

  // The winning split feature should be the virtual intercept (index == num_features).
  ASSERT_EQ(tree[0].SplitIndex(), kNumFeatures)
      << "Expected split on the intercept feature (index " << kNumFeatures << "), "
      << "got feature " << tree[0].SplitIndex();

  // Walk length must be >= 1 (intercept is skipped at d=0).
  ASSERT_GE(tree[0].WalkLen(), 1);
}

TEST_F(TreeGUpdaterTest, InterceptDoesNotAffectTreeWithoutFlag) {
  // Same star-graph + uniform-feature setup as Intercept_FeatureSelectedAsSplit,
  // but with intercept=false.  Since the single real feature is constant across
  // all vertices, no threshold produces gain -> tree stays as a single leaf.
  constexpr bst_node_t kNumVertices = 6;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 1;

  Args args{{"learning_rate", "1.0"},
            {"max_leaves", "2"},
            {"min_child_weight", "0.1"},
            {"reg_lambda", "0.0"},
            {"reg_alpha", "0.0"},
            {"min_split_loss", "0.0"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "0"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)},
            {"intercept", "0"}};

  updater_->Configure(args);

  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);

  // Real features all 0.0 — see Intercept_FeatureSelectedAsSplit for rationale.
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    test_data.SetFeature(i, 0, 0.0f);
  }

  // Star adjacency A^1: hub (0) <-> leaves (1..5)
  for (bst_node_t leaf = 1; leaf < kNumVertices; ++leaf) {
    test_data.SetAdjacency(0, leaf, 1, 1.0);
    test_data.SetAdjacency(leaf, 0, 1, 1.0);
  }

  auto dmat = test_data.CreateDMatrix();

  std::vector<GradientPair> gpairs;
  gpairs.emplace_back(-5.0f, 1.0f);
  for (bst_node_t i = 1; i < kNumVertices; ++i) {
    gpairs.emplace_back(1.0f, 1.0f);
  }

  RegTreeG tree;
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  updater_->Update(gpairs, graph_data, &tree);

  // With all-zero features and no intercept, no feature at any walk length
  // produces threshold variation — the tree must remain a single leaf.
  ASSERT_TRUE(tree[0].IsLeaf()) << "Without intercept the tree should remain a single leaf "
                                << "(zero features produce no threshold variation)";
}

// ========== colsample_bytree Tests ==========

TEST_F(TreeGUpdaterTest, ColsampleBytreeDefaultOneMatchesBaseline) {
  // With colsample_bytree=1.0 (default), the tree should be identical to one
  // built without specifying colsample_bytree at all.
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 4;
  constexpr bst_node_t kMaxWalkLen = 1;

  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    for (bst_feature_t k = 0; k < kNumFeatures; ++k) {
      test_data.SetFeature(i, k, static_cast<float>(i * kNumFeatures + k));
    }
    // Ring adjacency
    bst_node_t next = (i + 1) % kNumVertices;
    bst_node_t prev = (i - 1 + kNumVertices) % kNumVertices;
    test_data.SetAdjacency(i, next, 1, 1.0f);
    test_data.SetAdjacency(i, prev, 1, 1.0f);
  }
  auto dmat = test_data.CreateDMatrix();

  std::vector<GradientPair> gpairs;
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    gpairs.emplace_back(static_cast<float>(i) - 4.5f, 1.0f);
  }
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);

  Args base_args{{"learning_rate", "1.0"},
                 {"max_leaves", "4"},
                 {"min_child_weight", "0.1"},
                 {"reg_lambda", "0.0"},
                 {"min_split_loss", "0.0"},
                 {"max_walk_len", std::to_string(kMaxWalkLen)},
                 {"max_ancestor_distance", "1"},
                 {"num_features", std::to_string(kNumFeatures)},
                 {"num_vertices", std::to_string(kNumVertices)}};

  // Build without colsample_bytree (defaults to 1.0)
  Context ctx_baseline;
  ctx_baseline.seed = 42;
  TreeGUpdater baseline_updater(&ctx_baseline);
  baseline_updater.Configure(base_args);
  RegTreeG baseline_tree;
  baseline_updater.Update(gpairs, graph_data, &baseline_tree);

  // Build with explicit colsample_bytree=1.0
  Context ctx_explicit;
  ctx_explicit.seed = 42;
  TreeGUpdater explicit_updater(&ctx_explicit);
  Args explicit_args = base_args;
  explicit_args.emplace_back("colsample_bytree", "1.0");
  explicit_updater.Configure(explicit_args);
  RegTreeG explicit_tree;
  explicit_updater.Update(gpairs, graph_data, &explicit_tree);

  ASSERT_TRUE(baseline_tree == explicit_tree)
      << "colsample_bytree=1.0 must produce identical tree to default";
}

TEST_F(TreeGUpdaterTest, ColsampleBytreeSubsamplingUsesFewerFeatures) {
  // With colsample_bytree=0.5 and K=10, only ~5 features should be considered.
  // The split feature should always be in [0, K) (original feature space).
  constexpr bst_node_t kNumVertices = 20;
  constexpr bst_feature_t kNumFeatures = 10;
  constexpr bst_node_t kMaxWalkLen = 0;  // d=0 only for simplicity

  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);
  // Give each feature distinct patterns so different subsets produce different trees
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    for (bst_feature_t k = 0; k < kNumFeatures; ++k) {
      test_data.SetFeature(i, k, std::sin(static_cast<float>(i + k * 7)) * 10.0f);
    }
  }
  auto dmat = test_data.CreateDMatrix();

  std::vector<GradientPair> gpairs;
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    gpairs.emplace_back(static_cast<float>(i) - 10.0f, 1.0f);
  }
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);

  Args args{{"learning_rate", "1.0"},
            {"max_leaves", "4"},
            {"min_child_weight", "0.1"},
            {"reg_lambda", "0.0"},
            {"min_split_loss", "0.0"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "0"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)},
            {"colsample_bytree", "0.5"}};

  ctx_->seed = 123;
  updater_->Configure(args);

  RegTreeG tree;
  updater_->Update(gpairs, graph_data, &tree);

  // Tree should have at least one split
  ASSERT_FALSE(tree[0].IsLeaf()) << "Expected at least one split";

  // All split features must be valid original feature indices
  for (bst_node_t nid = 0; nid < tree.NumNodes(); ++nid) {
    if (!tree[nid].IsLeaf()) {
      ASSERT_LT(tree[nid].SplitIndex(), kNumFeatures)
          << "Split feature must be in [0, K) original feature space";
    }
  }
}

TEST_F(TreeGUpdaterTest, ColsampleBytreeDeterministicWithSameSeed) {
  // Same seed should produce the same tree (same feature sample).
  constexpr bst_node_t kNumVertices = 15;
  constexpr bst_feature_t kNumFeatures = 8;
  constexpr bst_node_t kMaxWalkLen = 1;

  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    for (bst_feature_t k = 0; k < kNumFeatures; ++k) {
      test_data.SetFeature(i, k, static_cast<float>((i + 1) * (k + 1)));
    }
    bst_node_t next = (i + 1) % kNumVertices;
    bst_node_t prev = (i - 1 + kNumVertices) % kNumVertices;
    test_data.SetAdjacency(i, next, 1, 1.0f);
    test_data.SetAdjacency(i, prev, 1, 1.0f);
  }
  auto dmat = test_data.CreateDMatrix();

  std::vector<GradientPair> gpairs;
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    gpairs.emplace_back(static_cast<float>(i) - 7.0f, 1.0f);
  }
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);

  Args args{{"learning_rate", "1.0"},
            {"max_leaves", "4"},
            {"min_child_weight", "0.1"},
            {"reg_lambda", "0.0"},
            {"min_split_loss", "0.0"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "1"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)},
            {"colsample_bytree", "0.3"}};

  // Build tree 1 with seed 99
  Context ctx1;
  ctx1.seed = 99;
  TreeGUpdater updater1(&ctx1);
  updater1.Configure(args);
  RegTreeG tree1;
  updater1.Update(gpairs, graph_data, &tree1);

  // Build tree 2 with same seed 99
  Context ctx2;
  ctx2.seed = 99;
  TreeGUpdater updater2(&ctx2);
  updater2.Configure(args);
  RegTreeG tree2;
  updater2.Update(gpairs, graph_data, &tree2);

  ASSERT_TRUE(tree1 == tree2) << "Same seed must produce identical trees with colsample_bytree";
}

TEST_F(TreeGUpdaterTest, ColsampleBytreeDifferentSeedDifferentSample) {
  // Different seeds should (with high probability) produce different trees.
  constexpr bst_node_t kNumVertices = 20;
  constexpr bst_feature_t kNumFeatures = 20;
  constexpr bst_node_t kMaxWalkLen = 0;

  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    for (bst_feature_t k = 0; k < kNumFeatures; ++k) {
      test_data.SetFeature(i, k, std::sin(static_cast<float>(i * 31 + k * 17)) * 5.0f);
    }
  }
  auto dmat = test_data.CreateDMatrix();

  std::vector<GradientPair> gpairs;
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    gpairs.emplace_back(static_cast<float>(i) - 10.0f, 1.0f);
  }
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);

  Args args{{"learning_rate", "1.0"},
            {"max_leaves", "4"},
            {"min_child_weight", "0.1"},
            {"reg_lambda", "0.0"},
            {"min_split_loss", "0.0"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "0"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)},
            {"colsample_bytree", "0.2"}};  // Only 4 out of 20 features

  Context ctx1;
  ctx1.seed = 1;
  TreeGUpdater updater1(&ctx1);
  updater1.Configure(args);
  RegTreeG tree1;
  updater1.Update(gpairs, graph_data, &tree1);

  Context ctx2;
  ctx2.seed = 2;
  TreeGUpdater updater2(&ctx2);
  updater2.Configure(args);
  RegTreeG tree2;
  updater2.Update(gpairs, graph_data, &tree2);

  // With 20 features and colsample=0.2, the probability of picking the same
  // 4 features with different seeds is C(20,4)^{-1} ≈ 0.02%, so this should
  // almost always produce different trees.
  ASSERT_FALSE(tree1 == tree2) << "Different seeds should (very likely) produce different trees";
}

TEST_F(TreeGUpdaterTest, ColsampleBytreeSingleFeatureEdgeCase) {
  // K=1 with any colsample_bytree value should always select the single feature.
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 0;

  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    test_data.SetFeature(i, 0, static_cast<float>(i));
  }
  auto dmat = test_data.CreateDMatrix();

  std::vector<GradientPair> gpairs;
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    gpairs.emplace_back(static_cast<float>(i) - 4.5f, 1.0f);
  }
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);

  // Even with very low colsample_bytree, the single feature should be selected
  Args args{{"learning_rate", "1.0"},
            {"max_leaves", "4"},
            {"min_child_weight", "0.1"},
            {"reg_lambda", "0.0"},
            {"min_split_loss", "0.0"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "0"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)},
            {"colsample_bytree", "0.01"}};  // floor(1 * 0.01) = 0, max(1, 0) = 1

  ctx_->seed = 42;
  updater_->Configure(args);

  RegTreeG tree;
  updater_->Update(gpairs, graph_data, &tree);

  // Should still be able to split on the single feature
  ASSERT_FALSE(tree[0].IsLeaf()) << "Should split even with K=1 and tiny colsample_bytree";
  ASSERT_EQ(tree[0].SplitIndex(), 0) << "Must split on feature 0 (the only feature)";
}

TEST_F(TreeGUpdaterTest, ColsampleBytreeInterceptAlwaysIncluded) {
  // With intercept=true and colsample_bytree<1.0, the intercept feature
  // should still be available for splits at d>=1.
  constexpr bst_node_t kNumVertices = 6;
  constexpr bst_feature_t kNumFeatures = 4;
  constexpr bst_node_t kMaxWalkLen = 1;

  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);
  // Set features to constant values so only intercept (graph connectivity) varies
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    for (bst_feature_t k = 0; k < kNumFeatures; ++k) {
      test_data.SetFeature(i, k, 1.0f);  // All same — no split from raw features
    }
  }
  // Build a graph where half the nodes are connected and half are isolated
  // Connected: 0-1, 1-2 (triangle-ish)
  test_data.SetAdjacency(0, 1, 1, 1.0f);
  test_data.SetAdjacency(1, 0, 1, 1.0f);
  test_data.SetAdjacency(1, 2, 1, 1.0f);
  test_data.SetAdjacency(2, 1, 1, 1.0f);
  // Nodes 3, 4, 5 are isolated (no adjacency entries)

  auto dmat = test_data.CreateDMatrix();

  // Create gradients that favour splitting connected vs isolated
  std::vector<GradientPair> gpairs = {
      GradientPair(-5.0f, 1.0f),  // node 0 (connected)
      GradientPair(-5.0f, 1.0f),  // node 1 (connected)
      GradientPair(-5.0f, 1.0f),  // node 2 (connected)
      GradientPair(5.0f, 1.0f),   // node 3 (isolated)
      GradientPair(5.0f, 1.0f),   // node 4 (isolated)
      GradientPair(5.0f, 1.0f),   // node 5 (isolated)
  };
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);

  // colsample_bytree=0.25 → floor(4*0.25)=1 sampled feature + intercept always included
  Args args{{"learning_rate", "1.0"},
            {"max_leaves", "4"},
            {"min_child_weight", "0.1"},
            {"reg_lambda", "0.0"},
            {"min_split_loss", "0.0"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "0"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)},
            {"intercept", "1"},
            {"colsample_bytree", "0.25"}};

  ctx_->seed = 42;
  updater_->Configure(args);

  RegTreeG tree;
  updater_->Update(gpairs, graph_data, &tree);

  // The tree should be able to split — either on a raw feature or the intercept.
  // The intercept at d=1 captures graph connectivity, which varies.
  // Split feature = kNumFeatures means the intercept feature was chosen.
  bool found_split = false;
  for (bst_node_t nid = 0; nid < tree.NumNodes(); ++nid) {
    if (!tree[nid].IsLeaf()) {
      found_split = true;
      bst_feature_t split_feat = tree[nid].SplitIndex();
      // Valid split feature: either original [0, K) or intercept (K)
      ASSERT_LE(split_feat, kNumFeatures)
          << "Split feature must be in [0, K] (K = intercept index)";
    }
  }
  ASSERT_TRUE(found_split) << "Expected at least one split";
}

// ========== Attention subsets: training vs. definition vs. prediction ==========
//
// Acceptance tests: training generates attention subsets as the Tree-G paper defines them,
// and training and prediction agree on them.
//
// Regression tests for a defect where training partitioned a split-node's *reached* set R_u
// while prediction partitioned its *used* subset S_{*u,rho_u}. The paper mandates the latter
// on both paths. They differ at any node below the root whose used subset is not its
// reached set, which the fixture below exercises deliberately.

namespace {

/**
 * Fixture graph: two triangles joined by a bridge through vertices 3 and 7.
 *
 *   0---1        4---5
 *    \ /          \ /
 *     2 --- 3 ---- 6
 *            \    /
 *             \--7
 *
 * Degrees are mixed, so neighbourhood aggregates separate the two communities and splits
 * with d >= 1 genuinely win. Vertices 3 and 7 are held out (zero gradient and hessian):
 * they never influence a gain, but they do influence their neighbours' aggregates, which
 * is the transductive behaviour the paper requires.
 */
constexpr bst_node_t kNumVertices = 8;
constexpr bst_feature_t kNumFeatures = 2;
constexpr bst_node_t kMaxWalkLen = 2;

/** @brief Edge list of the fixture graph. */
std::vector<std::pair<bst_node_t, bst_node_t>> const& FixtureEdges() {
  static std::vector<std::pair<bst_node_t, bst_node_t>> const kEdges{
      {0, 1}, {1, 2}, {0, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {4, 6}, {6, 7}, {3, 7}};
  return kEdges;
}

/** @brief Node feature k of vertex v. */
float FixtureFeature(bst_node_t v, bst_feature_t k) {
  return k == 0 ? static_cast<float>(v % 4) : static_cast<float>((v * 3) % 5);
}

/** @brief Dense A and A^2 of the fixture graph, each row-major N x N. */
std::vector<std::vector<float>> FixtureAdjacencyPowers() {
  std::vector<float> a(static_cast<std::size_t>(kNumVertices) * kNumVertices, 0.0f);
  for (auto const& edge : FixtureEdges()) {
    a[edge.first * kNumVertices + edge.second] = 1.0f;
    a[edge.second * kNumVertices + edge.first] = 1.0f;
  }
  std::vector<float> a2(static_cast<std::size_t>(kNumVertices) * kNumVertices, 0.0f);
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    for (bst_node_t j = 0; j < kNumVertices; ++j) {
      float sum = 0.0f;
      for (bst_node_t m = 0; m < kNumVertices; ++m) {
        sum += a[i * kNumVertices + m] * a[m * kNumVertices + j];
      }
      a2[i * kNumVertices + j] = sum;
    }
  }
  return {std::move(a), std::move(a2)};
}

/** @brief Build GraphData for the fixture graph directly, bypassing DMatrix. */
GraphData MakeFixtureGraphData() {
  GraphData gd;
  gd.num_vertices = kNumVertices;
  gd.num_features = kNumFeatures;
  gd.max_walk_len = kMaxWalkLen;

  gd.features.assign(static_cast<std::size_t>(kNumVertices) * kNumFeatures, 0.0f);
  for (bst_node_t v = 0; v < kNumVertices; ++v) {
    for (bst_feature_t k = 0; k < kNumFeatures; ++k) {
      gd.features[v * kNumFeatures + k] = FixtureFeature(v, k);
    }
  }

  auto const powers = FixtureAdjacencyPowers();
  gd.adj_powers.resize(kMaxWalkLen);
  gd.adj_diag.resize(kMaxWalkLen);
  for (bst_node_t d = 0; d < kMaxWalkLen; ++d) {
    CsrMatrix csr;
    csr.nrows = kNumVertices;
    csr.ncols = kNumVertices;
    csr.row_ptr.push_back(0);
    for (bst_node_t i = 0; i < kNumVertices; ++i) {
      for (bst_node_t j = 0; j < kNumVertices; ++j) {
        float const value = powers[d][i * kNumVertices + j];
        if (value != 0.0f) {
          csr.col_idx.push_back(j);
          csr.values.push_back(value);
        }
      }
      csr.row_ptr.push_back(static_cast<int>(csr.col_idx.size()));
    }
    gd.adj_powers[d] = std::move(csr);
    gd.adj_diag[d].resize(kNumVertices);
    for (bst_node_t i = 0; i < kNumVertices; ++i) {
      gd.adj_diag[d][i] = powers[d][i * kNumVertices + i];
    }
  }
  return gd;
}

/** @brief Build the same fixture graph as a DMatrix with layout [X | A | A^2]. */
std::shared_ptr<DMatrix> MakeFixtureDMatrix() {
  bst_feature_t const total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;
  std::vector<float> data(static_cast<std::size_t>(kNumVertices) * total_cols, 0.0f);

  for (bst_node_t v = 0; v < kNumVertices; ++v) {
    for (bst_feature_t k = 0; k < kNumFeatures; ++k) {
      data[v * total_cols + k] = FixtureFeature(v, k);
    }
  }
  auto const powers = FixtureAdjacencyPowers();
  for (bst_node_t d = 0; d < kMaxWalkLen; ++d) {
    bst_feature_t const block = kNumFeatures + d * kNumVertices;
    for (bst_node_t i = 0; i < kNumVertices; ++i) {
      for (bst_node_t j = 0; j < kNumVertices; ++j) {
        data[i * total_cols + block + j] = powers[d][i * kNumVertices + j];
      }
    }
  }
  return GetDMatrixFromData(data, kNumVertices, total_cols);
}

/** @brief Gradient pairs; vertices 3 and 7 are held out with zero weight. */
std::vector<GradientPair> FixtureGradients() {
  return {GradientPair(-0.9f, 0.3f), GradientPair(-0.7f, 0.3f), GradientPair(-0.2f, 0.3f),
          GradientPair(0.0f, 0.0f),  GradientPair(0.6f, 0.3f),  GradientPair(0.8f, 0.3f),
          GradientPair(0.4f, 0.3f),  GradientPair(0.0f, 0.0f)};
}

/** @brief Training arguments that make the split search rich enough to be non-trivial. */
Args FixtureTrainArgs() {
  return Args{{"learning_rate", "1.0"},
              {"max_leaves", "6"},
              {"min_child_weight", "0.0"},
              {"reg_lambda", "0.0"},
              {"min_split_loss", "0.0"},
              {"max_walk_len", std::to_string(kMaxWalkLen)},
              {"max_ancestor_distance", "2"},
              {"attention_type_sample_probability", "1.0"},
              {"colsample_bytree", "1.0"},
              {"num_features", std::to_string(kNumFeatures)},
              {"num_vertices", std::to_string(kNumVertices)}};
}

/** @brief Members of an ActiveSet as a sorted vertex list; empty for an unsized set. */
std::vector<bst_node_t> Members(ActiveSet const& set) {
  std::vector<bst_node_t> out;
  if (set.NumVertices() == 0) {
    return out;
  }
  set.ForEach([&](bst_node_t v) { out.push_back(v); });
  return out;
}

/**
 * @brief Brute-force attention subsets for one split node.
 *
 * Thresholds the split-value column directly, so it checks the updater's subset
 * bookkeeping independently of how the updater partitions vertices.
 */
GeneratedSubsets BruteForceAttentionSubsets(RegTreeG const& tree, GraphData const& graph_data,
                                            RegTreeG::TreeGNode const& node,
                                            ActiveSet const& used) {
  GeneratedSubsets expected{ActiveSet{kNumVertices, false}, ActiveSet{kNumVertices, false},
                            /*populated=*/true};
  auto const phi = tree.ComputeFeatureSplitValues(node.SplitIndex(), node.WalkLen(),
                                                  node.RestrictionTypeValue(), used, graph_data);
  for (bst_node_t v = 0; v < kNumVertices; ++v) {
    if (!used.Test(v)) {
      continue;
    }
    if (phi[v] <= node.SplitCond()) {
      expected.lte.Set(v, true);
    } else {
      expected.gte.Set(v, true);
    }
  }
  return expected;
}

/** @brief Route every vertex from the root using @p split_values; returns the leaf per vertex. */
std::vector<bst_node_t> RouteAllVertices(RegTreeG const& tree,
                                         std::vector<std::vector<bst_float>> const& split_values) {
  std::vector<bst_node_t> leaf_of(kNumVertices, RegTreeG::kInvalidNodeId);
  for (bst_node_t v = 0; v < kNumVertices; ++v) {
    bst_node_t nid = RegTreeG::kRoot;
    while (!tree[nid].IsLeaf()) {
      nid = split_values[nid][v] <= tree[nid].SplitCond() ? tree[nid].LeftChild()
                                                          : tree[nid].RightChild();
    }
    leaf_of[v] = nid;
  }
  return leaf_of;
}

/** @brief Reached set per node, obtained by routing all of V down a finished tree. */
std::vector<ActiveSet> ReachedSets(RegTreeG const& tree,
                                   std::vector<std::vector<bst_float>> const& split_values) {
  std::vector<ActiveSet> reached(tree.NumNodes(), ActiveSet{kNumVertices, false});
  for (bst_node_t v = 0; v < kNumVertices; ++v) {
    bst_node_t nid = RegTreeG::kRoot;
    reached[nid].Set(v, true);
    while (!tree[nid].IsLeaf()) {
      nid = split_values[nid][v] <= tree[nid].SplitCond() ? tree[nid].LeftChild()
                                                          : tree[nid].RightChild();
      reached[nid].Set(v, true);
    }
  }
  return reached;
}

/** @brief A tree trained by the updater together with the subsets it generated. */
struct TrainedTree {
  RegTreeG tree;
  std::vector<GeneratedSubsets> subsets;
};

/** @brief Train one tree on the fixture graph with the updater. */
TrainedTree TrainFixtureTree(GraphData const& graph_data) {
  Context ctx;
  ctx.seed = 11;
  TreeGUpdater updater(&ctx);
  updater.Configure(FixtureTrainArgs());

  TrainedTree out;
  updater.Update(FixtureGradients(), graph_data, &out.tree);
  out.subsets = updater.GeneratedSubsetsForTesting();
  return out;
}

}  // anonymous namespace

/**
 * @brief Training's attention subsets equal the paper's definition.
 *
 * Also asserts the scenario is non-trivial, because a fixture in which every node's used
 * subset coincides with its reached set cannot distinguish correct behaviour from the old
 * defect. The pre-existing suite was exactly such a fixture, which is why it stayed green
 * across a change in training semantics.
 */
TEST(TreeGAttentionSubsets, TrainingSubsetsMatchPaperDefinition) {
  GraphData const graph_data = MakeFixtureGraphData();
  auto const trained = TrainFixtureTree(graph_data);
  auto const& tree = trained.tree;

  std::vector<GeneratedSubsets> subsets;
  std::vector<std::vector<bst_float>> split_values;
  ReconstructSubsets(tree, graph_data, &subsets, &split_values);
  auto const reached = ReachedSets(tree, split_values);

  ActiveSet const all_vertices{kNumVertices, true};
  int split_nodes = 0;
  int ancestor_pointer_nodes = 0;
  int used_differs_from_reached = 0;
  int held_out_in_subset = 0;

  for (bst_node_t nid = 0; nid < tree.NumNodes(); ++nid) {
    if (tree[nid].IsLeaf()) {
      continue;
    }
    ++split_nodes;
    auto const& node = tree[nid];
    ASSERT_TRUE(trained.subsets[nid].populated) << "split node " << nid << " has no subsets";

    if (node.AncestorIndex() != RegTreeG::kInvalidNodeId) {
      ++ancestor_pointer_nodes;
    }

    ActiveSet const& used = ResolveUsedSubset(node, trained.subsets, all_vertices);
    auto const expected = BruteForceAttentionSubsets(tree, graph_data, node, used);
    ASSERT_EQ(Members(trained.subsets[nid].lte), Members(expected.lte)) << "node " << nid;
    ASSERT_EQ(Members(trained.subsets[nid].gte), Members(expected.gte)) << "node " << nid;

    if (Members(used) != Members(reached[nid])) {
      ++used_differs_from_reached;
    }
    for (bst_node_t held_out : {3, 7}) {
      if (trained.subsets[nid].lte.Test(held_out) || trained.subsets[nid].gte.Test(held_out)) {
        ++held_out_in_subset;
        break;
      }
    }
  }

  ASSERT_GT(split_nodes, 1) << "fixture produced a stump; nothing below the root to test";
  EXPECT_GT(ancestor_pointer_nodes, 0)
      << "fixture never selected an ancestor attention set, so S_{*u,rho_u} is always V";
  EXPECT_GT(used_differs_from_reached, 0)
      << "fixture is degenerate: the used subset equals the reached set at every node, so "
         "this test cannot detect the defect it exists to guard against";
  EXPECT_GT(held_out_in_subset, 0)
      << "zero-weight vertices never entered a generated subset; behaviour is not transductive";
}

/**
 * @brief prediction's reconstructed subsets equal training's.
 */
TEST(TreeGAttentionSubsets, TrainingSubsetsMatchPrediction) {
  GraphData const graph_data = MakeFixtureGraphData();
  auto const trained = TrainFixtureTree(graph_data);

  std::vector<GeneratedSubsets> predicted;
  std::vector<std::vector<bst_float>> split_values;
  ReconstructSubsets(trained.tree, graph_data, &predicted, &split_values);

  for (bst_node_t nid = 0; nid < trained.tree.NumNodes(); ++nid) {
    if (trained.tree[nid].IsLeaf()) {
      continue;
    }
    EXPECT_EQ(Members(trained.subsets[nid].lte), Members(predicted[nid].lte)) << "node " << nid;
    EXPECT_EQ(Members(trained.subsets[nid].gte), Members(predicted[nid].gte)) << "node " << nid;
  }
}

/**
 * @brief every vertex routes to a well-defined leaf, deterministically.
 *
 * Subsets agreeing is necessary but not sufficient: the split values derived from them must
 * also route identically. This closes the loop from subsets to observable behaviour.
 */
TEST(TreeGAttentionSubsets, TrainingRoutingMatchesPrediction) {
  GraphData const graph_data = MakeFixtureGraphData();
  auto const trained = TrainFixtureTree(graph_data);

  std::vector<GeneratedSubsets> predicted;
  std::vector<std::vector<bst_float>> split_values;
  ReconstructSubsets(trained.tree, graph_data, &predicted, &split_values);
  auto const leaf_of = RouteAllVertices(trained.tree, split_values);

  for (bst_node_t v = 0; v < kNumVertices; ++v) {
    ASSERT_NE(leaf_of[v], RegTreeG::kInvalidNodeId) << "vertex " << v << " did not reach a leaf";
    ASSERT_TRUE(trained.tree[leaf_of[v]].IsLeaf()) << "vertex " << v << " stopped at a split node";
  }

  // Reconstruction must be a pure function of (tree, graph).
  std::vector<GeneratedSubsets> again_subsets;
  std::vector<std::vector<bst_float>> again_values;
  ReconstructSubsets(trained.tree, graph_data, &again_subsets, &again_values);
  EXPECT_EQ(leaf_of, RouteAllVertices(trained.tree, again_values));
}

/**
 * @brief with one round and learning_rate = 1, PredictBatch returns
 *        base_score + the leaf value each vertex routes to.
 *
 * This is the end-to-end statement: the booster's public prediction path agrees with the
 * tree its own training produced. If training and prediction disagreed on subsets, vertices
 * would reach different leaves and this would fail.
 */
TEST(TreeGAttentionSubsets, BoosterPredictionMatchesTrainingLeaves) {
  constexpr float kBaseScore = 0.5f;
  Context ctx;
  ctx.seed = 11;

  auto dmat = MakeFixtureDMatrix();
  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, kBaseScore, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());
  ASSERT_NE(booster, nullptr);
  booster->Configure(FixtureTrainArgs());

  auto const grads = FixtureGradients();
  GradientContainer gpair;
  gpair.gpair.Reshape(kNumVertices, 1);
  auto h_gpair = gpair.gpair.HostView();
  for (bst_node_t v = 0; v < kNumVertices; ++v) {
    h_gpair(v, 0) = grads[v];
  }
  booster->DoBoost(dmat, &gpair, nullptr);

  HostDeviceVector<float> out_preds;
  booster->PredictBatch(dmat, &out_preds, false, 0, 0);
  auto const& preds = out_preds.ConstHostVector();
  ASSERT_EQ(preds.size(), static_cast<std::size_t>(kNumVertices));

  auto const& trees = booster->ModelForTesting().trees;
  ASSERT_EQ(trees.size(), 1U) << "expected exactly one tree after one round";
  auto const& tree = *trees.front();

  GraphData const graph_data = MakeFixtureGraphData();
  std::vector<GeneratedSubsets> subsets;
  std::vector<std::vector<bst_float>> split_values;
  ReconstructSubsets(tree, graph_data, &subsets, &split_values);
  auto const leaf_of = RouteAllVertices(tree, split_values);

  for (bst_node_t v = 0; v < kNumVertices; ++v) {
    float const expected = kBaseScore + tree[leaf_of[v]].LeafValue();
    EXPECT_NEAR(preds[v], expected, 1e-5f) << "vertex " << v << " routes to leaf " << leaf_of[v]
                                           << " but PredictBatch disagrees with that leaf's value";
  }
}

/**
 * @brief regression guard documenting why the old behaviour fails.
 *
 * The defect was that training built S_{u,+-} from the reached set R_u. This test exhibits
 * a concrete split node for which the two differ, and asserts the implementation follows
 * the used subset. Reproducing the old behaviour here would produce different subsets at
 * that node, and therefore a model whose predictions disagree with its own training.
 */
TEST(TreeGAttentionSubsets, SubsetsFollowUsedSubsetNotReachedSet) {
  GraphData const graph_data = MakeFixtureGraphData();
  auto const trained = TrainFixtureTree(graph_data);
  auto const& tree = trained.tree;

  std::vector<GeneratedSubsets> subsets;
  std::vector<std::vector<bst_float>> split_values;
  ReconstructSubsets(tree, graph_data, &subsets, &split_values);
  auto const reached = ReachedSets(tree, split_values);

  ActiveSet const all_vertices{kNumVertices, true};
  bool found_witness = false;

  for (bst_node_t nid = 0; nid < tree.NumNodes() && !found_witness; ++nid) {
    if (tree[nid].IsLeaf()) {
      continue;
    }
    ActiveSet const& used = ResolveUsedSubset(tree[nid], trained.subsets, all_vertices);

    // Look for a vertex that belongs to a generated subset of this node but never reaches
    // it. Under the old reached-set behaviour such a vertex could not appear in S_{u,+-}.
    for (bst_node_t v = 0; v < kNumVertices; ++v) {
      bool const in_subset = trained.subsets[nid].lte.Test(v) || trained.subsets[nid].gte.Test(v);
      if (!in_subset || reached[nid].Test(v)) {
        continue;
      }
      found_witness = true;

      ActiveSet union_set{kNumVertices, false};
      trained.subsets[nid].lte.ForEach([&](bst_node_t j) { union_set.Set(j, true); });
      trained.subsets[nid].gte.ForEach([&](bst_node_t j) { union_set.Set(j, true); });

      EXPECT_EQ(Members(union_set), Members(used))
          << "node " << nid << ": S_{u,+} union S_{u,-} must equal S_{*u,rho_u}";
      EXPECT_NE(Members(union_set), Members(reached[nid]))
          << "node " << nid << ": the subsets coincide with the reached set, so vertex " << v
          << " does not actually distinguish the two behaviours";
      break;
    }
  }

  EXPECT_TRUE(found_witness)
      << "no split node has a subset member outside its reached set; the fixture no longer "
         "distinguishes used-subset partitioning from the old reached-set behaviour and must be "
         "strengthened";
}

// ========== gbtree-compatible gain and leaf weight ==========

namespace {
/** @brief Updater configured with every regulariser that changes the gbtree formulas. */
Args GbtreeFormulaArgs() {
  return Args{{"num_features", "1"},
              {"num_vertices", "2"},
              {"reg_lambda", "1.7"},
              {"reg_alpha", "0.3"},
              {"max_delta_step", "0.8"}};
}
}  // anonymous namespace

TEST(TreeGUpdater, GainMatchesGbtreeFormula) {
  Context ctx;
  for (bool const with_max_delta_step : {false, true}) {
    TreeGUpdater updater(&ctx);
    Args args = GbtreeFormulaArgs();
    if (!with_max_delta_step) {
      args.emplace_back("max_delta_step", "0");
    }
    updater.Configure(args);
    auto const& param = updater.ParamForTesting();

    std::mt19937 rng{7};
    std::uniform_real_distribution<double> grad_dist{-5.0, 5.0};
    std::uniform_real_distribution<double> hess_dist{0.01, 5.0};
    for (int trial = 0; trial < 100; ++trial) {
      GradStats const left{grad_dist(rng), hess_dist(rng)};
      GradStats const right{grad_dist(rng), hess_dist(rng)};
      GradStats const parent = left + right;
      double const expected = tree::CalcGain(param, left.sum_grad, left.sum_hess) +
                              tree::CalcGain(param, right.sum_grad, right.sum_hess) -
                              tree::CalcGain(param, parent.sum_grad, parent.sum_hess);
      EXPECT_DOUBLE_EQ(updater.CalcGain(parent, left, right), expected) << "trial " << trial;
    }
  }
}

TEST(TreeGUpdater, LeafWeightMatchesGbtreeFormula) {
  Context ctx;
  TreeGUpdater updater(&ctx);
  updater.Configure(GbtreeFormulaArgs());
  auto const& param = updater.ParamForTesting();

  std::mt19937 rng{13};
  std::uniform_real_distribution<double> grad_dist{-5.0, 5.0};
  std::uniform_real_distribution<double> hess_dist{0.0, 5.0};
  for (int trial = 0; trial < 100; ++trial) {
    GradStats const stats{grad_dist(rng), hess_dist(rng)};
    EXPECT_DOUBLE_EQ(updater.CalcWeight(stats),
                     tree::CalcWeight(param, stats.sum_grad, stats.sum_hess))
        << "trial " << trial;
  }
}

/**
 * @brief A split whose loss change lies in [gamma, 2 gamma) is accepted, as in gbtree.
 *
 * Two vertices with g = -1, +1 and h = 1 and reg_lambda = 0: the gbtree loss change is
 * 1 + 1 - 0 = 2. With min_split_loss = 1.5 gbtree accepts it; a 0.5-scaled gain (1.0) would not.
 */
TEST(TreeGUpdater, MinSplitLossThresholdMatchesGbtree) {
  constexpr bst_node_t kNumVertices = 2;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 0;

  Context ctx;
  TreeGUpdater updater(&ctx);
  updater.Configure(Args{{"learning_rate", "1.0"},
                         {"max_leaves", "2"},
                         {"min_child_weight", "0.0"},
                         {"reg_lambda", "0.0"},
                         {"min_split_loss", "1.5"},
                         {"max_walk_len", std::to_string(kMaxWalkLen)},
                         {"max_ancestor_distance", "0"},
                         {"num_features", std::to_string(kNumFeatures)},
                         {"num_vertices", std::to_string(kNumVertices)}});

  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);
  test_data.SetFeature(0, 0, 0.0f);
  test_data.SetFeature(1, 0, 1.0f);
  auto dmat = test_data.CreateDMatrix();
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);

  std::vector<GradientPair> const gpairs{GradientPair(-1.0f, 1.0f), GradientPair(1.0f, 1.0f)};
  RegTreeG tree;
  updater.Update(gpairs, graph_data, &tree);

  ASSERT_FALSE(tree[RegTreeG::kRoot].IsLeaf()) << "loss change 2 >= min_split_loss 1.5";
  EXPECT_FLOAT_EQ(tree.Stat(RegTreeG::kRoot).loss_chg, 2.0f);
  // Leaf weights: -G / H = +1 and -1 (learning_rate = 1).
  EXPECT_FLOAT_EQ(tree[tree[RegTreeG::kRoot].LeftChild()].LeafValue(), 1.0f);
  EXPECT_FLOAT_EQ(tree[tree[RegTreeG::kRoot].RightChild()].LeafValue(), -1.0f);
}

/**
 * @brief The split-value cache budget changes speed only: no cache, a one-matrix
 *        cache, and the default (everything cached) grow identical trees.
 */
TEST(TreeGUpdater, SplitCacheBudgetDoesNotChangeModel) {
  constexpr bst_node_t kNumVertices = 8;
  constexpr bst_feature_t kNumFeatures = 2;
  constexpr bst_node_t kMaxWalkLen = 2;

  // Hard-coded irregular graph: ring 0..5 plus chords and a pendant pair.
  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);
  std::vector<std::pair<bst_node_t, bst_node_t>> const edges{
      {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}, {0, 3}, {1, 6}, {6, 7}, {2, 7}};
  std::vector<std::vector<float>> adj(kNumVertices, std::vector<float>(kNumVertices, 0.0f));
  for (auto [u, v] : edges) {
    adj[u][v] = adj[v][u] = 1.0f;
  }
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    // Feature 0 is constant and feature 1 is parity, so at d=0 only weak splits exist;
    // gradients follow vertex degree, which only graph-aware (d >= 1) splits can see.
    test_data.SetFeature(i, 0, 1.0f);
    test_data.SetFeature(i, 1, static_cast<float>(i % 2));
    for (bst_node_t j = 0; j < kNumVertices; ++j) {
      if (adj[i][j] != 0.0f) {
        test_data.SetAdjacency(i, j, 1, adj[i][j]);
      }
      float a2 = 0.0f;
      for (bst_node_t m = 0; m < kNumVertices; ++m) {
        a2 += adj[i][m] * adj[m][j];
      }
      if (a2 != 0.0f) {
        test_data.SetAdjacency(i, j, 2, a2);
      }
    }
  }
  auto dmat = test_data.CreateDMatrix();
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);

  std::vector<GradientPair> gpairs;
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    float degree = std::accumulate(adj[i].cbegin(), adj[i].cend(), 0.0f);
    gpairs.emplace_back(degree >= 3.0f ? -2.0f : 2.0f, 1.0f);
  }

  // S_total = K + intercept = 3 columns.
  std::size_t const one_matrix = kNumVertices * (kNumFeatures + 1) * sizeof(float);
  auto grow = [&](std::size_t budget) {
    Context ctx;
    TreeGUpdater updater(&ctx);
    updater.Configure(Args{{"max_leaves", "6"},
                           {"min_child_weight", "0.0"},
                           {"max_walk_len", std::to_string(kMaxWalkLen)},
                           {"max_ancestor_distance", "2"},
                           {"intercept", "1"},
                           {"num_features", std::to_string(kNumFeatures)},
                           {"num_vertices", std::to_string(kNumVertices)},
                           {"max_cached_split_bytes", std::to_string(budget)}});
    std::vector<RegTreeG> trees(3);
    for (auto& tree : trees) {
      updater.Update(gpairs, graph_data, &tree);
    }
    return trees;
  };

  auto const reference = grow(0);
  ASSERT_FALSE(reference[0][RegTreeG::kRoot].IsLeaf());
  // The cache only matters for d >= 1 splits, so the reference must contain some.
  bool has_graph_split = false;
  for (auto const& tree : reference) {
    for (bst_node_t nid = 0; nid < tree.NumNodes(); ++nid) {
      has_graph_split |= !tree[nid].IsLeaf() && tree[nid].WalkLen() >= 1;
    }
  }
  ASSERT_TRUE(has_graph_split);
  for (std::size_t budget : {one_matrix, static_cast<std::size_t>(1) << 30}) {
    auto const trees = grow(budget);
    for (std::size_t t = 0; t < trees.size(); ++t) {
      EXPECT_TRUE(trees[t] == reference[t]) << "budget=" << budget << " tree=" << t;
    }
  }
}

/**
 * @brief The parallel split search is bit-identical to the serial one. Several feature
 *        columns are exact duplicates, so equal-gain ties across threads must resolve to
 *        the lowest feature index, as in the serial first-wins scan.
 */
TEST(TreeGUpdater, ParallelSplitSearchMatchesSerial) {
  constexpr bst_node_t kNumVertices = 12;
  constexpr bst_feature_t kNumFeatures = 10;
  constexpr bst_node_t kMaxWalkLen = 2;

  // Hard-coded graph: ring 0..11 plus chords {0,6}, {3,9}, {1,4}.
  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);
  std::vector<std::vector<float>> adj(kNumVertices, std::vector<float>(kNumVertices, 0.0f));
  auto add_edge = [&](bst_node_t u, bst_node_t v) {
    adj[u][v] = adj[v][u] = 1.0f;
  };
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    add_edge(i, (i + 1) % kNumVertices);
  }
  add_edge(0, 6);
  add_edge(3, 9);
  add_edge(1, 4);
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    for (bst_node_t k = 0; k < kNumFeatures; ++k) {
      // Columns k and k + 5 are identical: every split has an exact tie partner.
      test_data.SetFeature(i, k, static_cast<float>((i * (k % 5 + 1) + k % 5) % 7));
    }
    for (bst_node_t j = 0; j < kNumVertices; ++j) {
      if (adj[i][j] != 0.0f) {
        test_data.SetAdjacency(i, j, 1, adj[i][j]);
      }
      float a2 = 0.0f;
      for (bst_node_t m = 0; m < kNumVertices; ++m) {
        a2 += adj[i][m] * adj[m][j];
      }
      if (a2 != 0.0f) {
        test_data.SetAdjacency(i, j, 2, a2);
      }
    }
  }
  auto dmat = test_data.CreateDMatrix();
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);

  std::vector<GradientPair> gpairs;
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    gpairs.emplace_back(static_cast<float>((i * 5) % 7) - 3.0f, 1.0f);
  }

  auto grow = [&](std::int32_t n_threads) {
    Context ctx;
    ctx.UpdateAllowUnknown(Args{{"nthread", std::to_string(n_threads)}});
    TreeGUpdater updater(&ctx);
    updater.Configure(Args{{"max_leaves", "8"},
                           {"min_child_weight", "0.0"},
                           {"max_walk_len", std::to_string(kMaxWalkLen)},
                           {"max_ancestor_distance", "2"},
                           {"intercept", "1"},
                           {"num_features", std::to_string(kNumFeatures)},
                           {"num_vertices", std::to_string(kNumVertices)}});
    std::vector<RegTreeG> trees(3);
    for (auto& tree : trees) {
      updater.Update(gpairs, graph_data, &tree);
    }
    return trees;
  };

  auto const serial = grow(1);
  ASSERT_FALSE(serial[0][RegTreeG::kRoot].IsLeaf());
  for (auto const& tree : serial) {
    for (bst_node_t nid = 0; nid < tree.NumNodes(); ++nid) {
      if (!tree[nid].IsLeaf()) {
        // Ties resolve to the lower duplicate; the upper copy is never chosen.
        // (Index kNumFeatures is the virtual intercept, which has no duplicate.)
        auto const k = tree[nid].SplitIndex();
        EXPECT_TRUE(k < 5u || k == kNumFeatures) << "node " << nid << " feature " << k;
      }
    }
  }
  auto const parallel = grow(4);
  for (std::size_t t = 0; t < serial.size(); ++t) {
    EXPECT_TRUE(parallel[t] == serial[t]) << "tree=" << t;
  }
}

/**
 * @brief The filtered walk over pre-sorted orders gives the same models as the per-call
 *        sort. Feature values have many ties, so the (value, vertex) tie order is exercised.
 *        Budget 0 caches nothing (sort path); the default budget caches every matrix and
 *        order (presorted path for large nodes).
 */
TEST(TreeGUpdater, PresortedScanMatchesSort) {
  constexpr bst_node_t kNumVertices = 16;
  constexpr bst_feature_t kNumFeatures = 3;
  constexpr bst_node_t kMaxWalkLen = 2;

  // Hard-coded graph: ring 0..15 plus chords {0,8}, {2,11}, {5,13}, {4,7}.
  TestDMatrix test_data(kNumVertices, kNumFeatures, kMaxWalkLen);
  std::vector<std::vector<float>> adj(kNumVertices, std::vector<float>(kNumVertices, 0.0f));
  auto add_edge = [&](bst_node_t u, bst_node_t v) {
    adj[u][v] = adj[v][u] = 1.0f;
  };
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    add_edge(i, (i + 1) % kNumVertices);
  }
  add_edge(0, 8);
  add_edge(2, 11);
  add_edge(5, 13);
  add_edge(4, 7);
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    // Few distinct values per column: long runs of ties.
    test_data.SetFeature(i, 0, static_cast<float>(i % 3));
    test_data.SetFeature(i, 1, static_cast<float>((i * 7) % 4));
    test_data.SetFeature(i, 2, static_cast<float>(i / 6));
    for (bst_node_t j = 0; j < kNumVertices; ++j) {
      if (adj[i][j] != 0.0f) {
        test_data.SetAdjacency(i, j, 1, adj[i][j]);
      }
      float a2 = 0.0f;
      for (bst_node_t m = 0; m < kNumVertices; ++m) {
        a2 += adj[i][m] * adj[m][j];
      }
      if (a2 != 0.0f) {
        test_data.SetAdjacency(i, j, 2, a2);
      }
    }
  }
  auto dmat = test_data.CreateDMatrix();
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);

  std::vector<GradientPair> gpairs;
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    float degree = std::accumulate(adj[i].cbegin(), adj[i].cend(), 0.0f);
    gpairs.emplace_back((degree >= 3.0f ? -2.0f : 1.0f) + static_cast<float>(i % 3) * 0.5f, 1.0f);
  }

  auto grow = [&](std::size_t budget, std::int32_t n_threads) {
    Context ctx;
    ctx.UpdateAllowUnknown(Args{{"nthread", std::to_string(n_threads)}});
    TreeGUpdater updater(&ctx);
    updater.Configure(Args{{"max_leaves", "8"},
                           {"min_child_weight", "0.0"},
                           {"max_walk_len", std::to_string(kMaxWalkLen)},
                           {"max_ancestor_distance", "2"},
                           {"intercept", "1"},
                           {"num_features", std::to_string(kNumFeatures)},
                           {"num_vertices", std::to_string(kNumVertices)},
                           {"max_cached_split_bytes", std::to_string(budget)}});
    std::vector<RegTreeG> trees(3);
    for (auto& tree : trees) {
      updater.Update(gpairs, graph_data, &tree);
    }
    return trees;
  };

  auto const reference = grow(0, 1);
  ASSERT_FALSE(reference[0][RegTreeG::kRoot].IsLeaf());
  for (std::int32_t n_threads : {1, 4}) {
    auto const trees = grow(static_cast<std::size_t>(1) << 30, n_threads);
    for (std::size_t t = 0; t < trees.size(); ++t) {
      EXPECT_TRUE(trees[t] == reference[t]) << "nthread=" << n_threads << " tree=" << t;
    }
  }
}

}  // namespace treeg
}  // namespace xgboost
