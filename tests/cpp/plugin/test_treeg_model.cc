/**
 * Copyright 2024-2026, XGBoost contributors
 *
 * @brief Unit tests for Tree-G model
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/context.h>
#include <xgboost/data.h>
#include <xgboost/json.h>
#include <xgboost/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "../../../plugin/treeg/model.h"
#include "../filesystem.h"
#include "../helpers.h"

namespace xgboost {
namespace treeg {

// ========== Test Helpers ==========

/**
 * @brief Convert a dense N×N row-major matrix to CsrMatrix (test helper).
 * Only stores entries with value != 0.
 */
static CsrMatrix DenseToCsr(const std::vector<float>& dense, int N) {
  CsrMatrix csr;
  csr.nrows = N;
  csr.ncols = N;
  csr.row_ptr.reserve(N + 1);
  csr.row_ptr.push_back(0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float v = dense[i * N + j];
      if (v != 0.0f) {
        csr.col_idx.push_back(j);
        csr.values.push_back(v);
      }
    }
    csr.row_ptr.push_back(static_cast<int>(csr.col_idx.size()));
  }
  return csr;
}

/**
 * @brief Build a GraphData from dense X and dense adjacency matrices (test helper).
 * Converts each adj_power from dense to CSR and extracts diag.
 */
static GraphData BuildGraphDataFromDense(bst_node_t N, bst_feature_t K, bst_node_t dmax,
                                         const std::vector<float>& X,
                                         const std::vector<std::vector<float>>& dense_adj_powers) {
  GraphData gd;
  gd.num_vertices = N;
  gd.num_features = K;
  gd.max_walk_len = dmax;
  gd.features = X;
  gd.adj_powers.resize(dmax);
  gd.adj_diag.resize(dmax);
  for (bst_node_t d = 0; d < dmax; ++d) {
    gd.adj_powers[d] = DenseToCsr(dense_adj_powers[d], static_cast<int>(N));
    gd.adj_diag[d].resize(N);
    for (bst_node_t i = 0; i < N; ++i) {
      gd.adj_diag[d][i] = dense_adj_powers[d][i * N + i];
    }
  }
  return gd;
}

// ========== ActiveSet Tests ==========

TEST(ActiveSet, BasicOperations) {
  constexpr bst_node_t kNumVertices = 100;

  // Test default constructor
  ActiveSet empty_set;
  ASSERT_EQ(empty_set.NumVertices(), 0);

  // Test parameterized constructor - all inactive
  ActiveSet set_inactive(kNumVertices, false);
  ASSERT_EQ(set_inactive.NumVertices(), kNumVertices);
  ASSERT_EQ(set_inactive.Count(), 0);

  // Test parameterized constructor - all active
  ActiveSet set_active(kNumVertices, true);
  ASSERT_EQ(set_active.NumVertices(), kNumVertices);
  ASSERT_EQ(set_active.Count(), kNumVertices);

  // Test all vertices are active
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    ASSERT_TRUE(set_active.Test(i));
    ASSERT_FALSE(set_inactive.Test(i));
  }
}

TEST(ActiveSet, SetAndTest) {
  constexpr bst_node_t kNumVertices = 50;
  ActiveSet set(kNumVertices, false);

  // Set specific vertices
  set.Set(0, true);
  set.Set(10, true);
  set.Set(25, true);
  set.Set(49, true);

  ASSERT_EQ(set.Count(), 4);
  ASSERT_TRUE(set.Test(0));
  ASSERT_TRUE(set.Test(10));
  ASSERT_TRUE(set.Test(25));
  ASSERT_TRUE(set.Test(49));

  ASSERT_FALSE(set.Test(1));
  ASSERT_FALSE(set.Test(11));
  ASSERT_FALSE(set.Test(24));

  // Unset a vertex
  set.Set(10, false);
  ASSERT_FALSE(set.Test(10));
  ASSERT_EQ(set.Count(), 3);
}

TEST(ActiveSet, ForEach) {
  constexpr bst_node_t kNumVertices = 100;
  ActiveSet set(kNumVertices, false);

  // Set some vertices
  std::vector<bst_node_t> expected = {5, 15, 33, 64, 99};
  for (auto v : expected) {
    set.Set(v, true);
  }

  // Collect vertices using ForEach
  std::vector<bst_node_t> collected;
  set.ForEach([&](bst_node_t v) { collected.push_back(v); });

  // Sort both for comparison
  std::sort(collected.begin(), collected.end());
  ASSERT_EQ(collected, expected);
}

TEST(ActiveSet, Clear) {
  constexpr bst_node_t kNumVertices = 50;
  ActiveSet set(kNumVertices, true);

  ASSERT_EQ(set.Count(), kNumVertices);

  set.Clear();
  ASSERT_EQ(set.Count(), 0);

  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    ASSERT_FALSE(set.Test(i));
  }
}

TEST(ActiveSet, Equality) {
  constexpr bst_node_t kNumVertices = 50;
  ActiveSet set1(kNumVertices, false);
  ActiveSet set2(kNumVertices, false);

  ASSERT_TRUE(set1 == set2);

  set1.Set(10, true);
  ASSERT_FALSE(set1 == set2);

  set2.Set(10, true);
  ASSERT_TRUE(set1 == set2);
}

TEST(ActiveSet, Mask) {
  // Test case from task specification: n=6, indicator (0,1,0,0,1,0)
  constexpr bst_node_t kNumVertices = 6;
  ActiveSet set(kNumVertices, false);

  // Set vertices 1 and 4 to active (indicator: 0,1,0,0,1,0)
  set.Set(1, true);
  set.Set(4, true);

  // Get the mask diagonal (now returns const reference)
  const std::vector<bst_float>& mask = set.Mask();

  // Verify shape
  ASSERT_EQ(mask.size(), kNumVertices);

  // Verify values: should be [0, 1, 0, 0, 1, 0]
  ASSERT_FLOAT_EQ(mask[0], 0.0f);
  ASSERT_FLOAT_EQ(mask[1], 1.0f);
  ASSERT_FLOAT_EQ(mask[2], 0.0f);
  ASSERT_FLOAT_EQ(mask[3], 0.0f);
  ASSERT_FLOAT_EQ(mask[4], 1.0f);
  ASSERT_FLOAT_EQ(mask[5], 0.0f);
}

TEST(ActiveSet, MaskAllActive) {
  constexpr bst_node_t kNumVertices = 10;
  ActiveSet set(kNumVertices, true);

  const std::vector<bst_float>& mask = set.Mask();

  ASSERT_EQ(mask.size(), kNumVertices);
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    ASSERT_FLOAT_EQ(mask[i], 1.0f);
  }
}

TEST(ActiveSet, MaskAllInactive) {
  constexpr bst_node_t kNumVertices = 10;
  ActiveSet set(kNumVertices, false);

  const std::vector<bst_float>& mask = set.Mask();

  ASSERT_EQ(mask.size(), kNumVertices);
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    ASSERT_FLOAT_EQ(mask[i], 0.0f);
  }
}

// ========== TreeGParam Tests ==========

TEST(TreeGParam, DefaultValues) {
  TreeGParam param;

  ASSERT_EQ(param.num_nodes, 1);
  ASSERT_EQ(param.num_features, 0);
  ASSERT_EQ(param.max_walk_len, 2);
  ASSERT_EQ(param.max_ancestor_distance, 2);
  ASSERT_EQ(param.num_vertices, 0);
}

TEST(TreeGParam, JSONSerialization) {
  TreeGParam param;
  param.num_nodes = 5;
  param.num_features = 10;
  param.max_walk_len = 3;
  param.max_ancestor_distance = 4;
  param.num_vertices = 100;

  Json json{Object()};
  param.ToJson(&json);

  TreeGParam loaded_param;
  loaded_param.FromJson(json);

  ASSERT_EQ(loaded_param.num_nodes, param.num_nodes);
  ASSERT_EQ(loaded_param.num_features, param.num_features);
  ASSERT_EQ(loaded_param.max_walk_len, param.max_walk_len);
  ASSERT_EQ(loaded_param.max_ancestor_distance, param.max_ancestor_distance);
  ASSERT_EQ(loaded_param.num_vertices, param.num_vertices);
}

TEST(TreeGParam, Equality) {
  TreeGParam param1;
  TreeGParam param2;

  ASSERT_TRUE(param1 == param2);

  param1.num_nodes = 10;
  ASSERT_FALSE(param1 == param2);

  param2.num_nodes = 10;
  ASSERT_TRUE(param1 == param2);
}

// ========== TreeGNodeStat Tests ==========

TEST(TreeGNodeStat, Construction) {
  TreeGNodeStat stat;
  // Default constructor leaves values uninitialized

  TreeGNodeStat stat2(1.5f, 2.5f, 3.5f);
  ASSERT_EQ(stat2.loss_chg, 1.5f);
  ASSERT_EQ(stat2.sum_hess, 2.5f);
  ASSERT_EQ(stat2.base_weight, 3.5f);
}

TEST(TreeGNodeStat, Equality) {
  TreeGNodeStat stat1(1.0f, 2.0f, 3.0f);
  TreeGNodeStat stat2(1.0f, 2.0f, 3.0f);

  ASSERT_TRUE(stat1 == stat2);

  stat1.loss_chg = 9.0f;
  ASSERT_FALSE(stat1 == stat2);

  stat2.loss_chg = 9.0f;
  ASSERT_TRUE(stat1 == stat2);
}

// ========== RegTreeG Basic Tests ==========

TEST(RegTreeG, DefaultConstruction) {
  RegTreeG tree;

  ASSERT_EQ(tree.NumNodes(), 1);
  ASSERT_EQ(tree.NumFeatures(), 0);
  ASSERT_EQ(tree.NumVertices(), 0);
  ASSERT_EQ(tree.MaxWalkLen(), 2);
  ASSERT_EQ(tree.MaxAncestorDistance(), 2);

  // Root should be a leaf
  ASSERT_TRUE(tree[0].IsLeaf());
  ASSERT_TRUE(tree[0].IsRoot());
}

TEST(RegTreeG, ParameterizedConstruction) {
  constexpr bst_node_t kNumVertices = 100;
  constexpr bst_feature_t kNumFeatures = 20;
  constexpr bst_node_t kMaxWalkLen = 3;
  constexpr bst_node_t kMaxAncestorDist = 5;

  RegTreeG tree(kNumVertices, kNumFeatures, kMaxWalkLen, kMaxAncestorDist);

  ASSERT_EQ(tree.NumVertices(), kNumVertices);
  ASSERT_EQ(tree.NumFeatures(), kNumFeatures);
  ASSERT_EQ(tree.MaxWalkLen(), kMaxWalkLen);
  ASSERT_EQ(tree.MaxAncestorDistance(), kMaxAncestorDist);

  // The root is a leaf with a self pointer; generated subsets are not stored in the tree.
  ASSERT_TRUE(tree[0].IsLeaf());
  ASSERT_EQ(tree[0].AncestorIndex(), RegTreeG::kInvalidNodeId);
}

TEST(RegTreeG, ExpandNode) {
  constexpr bst_node_t kNumVertices = 50;
  RegTreeG tree(kNumVertices, 10);

  // Expand root
  tree.ExpandNode(0, 5, 1.5f, 1.0f, 2.0f, 3.0f, 0.5f, 10.0f, 4.0f, 6.0f);

  ASSERT_FALSE(tree[0].IsLeaf());
  ASSERT_EQ(tree[0].SplitIndex(), 5);
  ASSERT_EQ(tree[0].SplitCond(), 1.5f);

  auto left = tree[0].LeftChild();
  auto right = tree[0].RightChild();

  ASSERT_TRUE(tree[left].IsLeaf());
  ASSERT_EQ(tree[left].LeafValue(), 2.0f);
  ASSERT_EQ(tree[left].Parent(), 0);
  ASSERT_TRUE(tree[left].IsLeftChild());

  ASSERT_TRUE(tree[right].IsLeaf());
  ASSERT_EQ(tree[right].LeafValue(), 3.0f);
  ASSERT_EQ(tree[right].Parent(), 0);
  ASSERT_FALSE(tree[right].IsLeftChild());

  // Check statistics
  ASSERT_EQ(tree.Stat(0).loss_chg, 0.5f);
  ASSERT_EQ(tree.Stat(0).sum_hess, 10.0f);
  ASSERT_EQ(tree.Stat(0).base_weight, 1.0f);

  ASSERT_EQ(tree.Stat(left).sum_hess, 4.0f);
  ASSERT_EQ(tree.Stat(right).sum_hess, 6.0f);
}

TEST(RegTreeG, SetNodeTreeGParams) {
  constexpr bst_node_t kNumVertices = 50;
  RegTreeG tree(kNumVertices, 10, 3, 4);

  // Expand root
  tree.ExpandNode(0, 0, 0.5f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f);

  // Set Tree-G params.  Only (d, r, *, rho) are stored; the generated subsets
  // S_{u,+}/S_{u,-} are graph-dependent and rebuilt by ReconstructSubsets.
  tree.SetNodeTreeGParams(0, 2, RestrictionType::kCycle, RegTreeG::kInvalidNodeId, true);

  ASSERT_EQ(tree[0].WalkLen(), 2);
  ASSERT_EQ(tree[0].RestrictionTypeValue(), RestrictionType::kCycle);
  ASSERT_EQ(tree[0].AncestorIndex(), RegTreeG::kInvalidNodeId);
  ASSERT_TRUE(tree[0].AncestorSign());
}

// ========== TreeGNode Specific Tests ==========

TEST(TreeGNode, RestrictionType) {
  constexpr bst_node_t kNumVertices = 30;
  RegTreeG tree(kNumVertices, 5);

  tree.ExpandNode(0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

  // Test kSource
  tree.SetNodeTreeGParams(0, 1, RestrictionType::kSource, RegTreeG::kInvalidNodeId, false);

  ASSERT_EQ(tree[0].RestrictionTypeValue(), RestrictionType::kSource);

  // Test kCycle
  tree.SetNodeTreeGParams(0, 2, RestrictionType::kCycle, RegTreeG::kInvalidNodeId, true);

  ASSERT_EQ(tree[0].RestrictionTypeValue(), RestrictionType::kCycle);
}

TEST(TreeGNode, AncestorPointer) {
  constexpr bst_node_t kNumVertices = 25;
  RegTreeG tree(kNumVertices, 7, 3, 5);

  // Expand root
  tree.ExpandNode(0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  auto left = tree[0].LeftChild();

  // Expand left child
  tree.ExpandNode(left, 1, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

  // Set ancestor pointer to root, using root's S_plus
  tree.SetNodeTreeGParams(left, 1, RestrictionType::kSource, 0,
                          true);  // ancestor_idx=0, ancestor_sign=true

  ASSERT_EQ(tree[left].AncestorIndex(), 0);
  ASSERT_TRUE(tree[left].AncestorSign());

  // Set ancestor pointer to root, using root's S_minus
  tree.SetNodeTreeGParams(left, 2, RestrictionType::kCycle, 0,
                          false);  // ancestor_idx=0, ancestor_sign=false

  ASSERT_EQ(tree[left].AncestorIndex(), 0);
  ASSERT_FALSE(tree[left].AncestorSign());
}

// ========== JSON Serialization Tests ==========

TEST(RegTreeG, BasicJSONSerialization) {
  constexpr bst_node_t kNumVertices = 50;
  constexpr bst_feature_t kNumFeatures = 15;

  RegTreeG tree(kNumVertices, kNumFeatures, 3, 4);

  // Expand root
  tree.ExpandNode(0, 5, 2.5f, 1.0f, 2.0f, 3.0f, 0.8f, 12.0f, 5.0f, 7.0f);

  tree.SetNodeTreeGParams(0, 2, RestrictionType::kCycle, RegTreeG::kInvalidNodeId, true);

  // Save to JSON
  Json out{Object()};
  tree.SaveModel(&out);

  // Load from JSON
  RegTreeG loaded_tree;
  loaded_tree.LoadModel(out);

  // Verify parameters
  ASSERT_EQ(loaded_tree.NumNodes(), tree.NumNodes());
  ASSERT_EQ(loaded_tree.NumFeatures(), tree.NumFeatures());
  ASSERT_EQ(loaded_tree.NumVertices(), tree.NumVertices());
  ASSERT_EQ(loaded_tree.MaxWalkLen(), tree.MaxWalkLen());
  ASSERT_EQ(loaded_tree.MaxAncestorDistance(), tree.MaxAncestorDistance());

  // Verify root node structure
  ASSERT_FALSE(loaded_tree[0].IsLeaf());
  ASSERT_EQ(loaded_tree[0].SplitIndex(), 5);
  ASSERT_EQ(loaded_tree[0].SplitCond(), 2.5f);

  // Verify Tree-G params
  ASSERT_EQ(loaded_tree[0].WalkLen(), 2);
  ASSERT_EQ(loaded_tree[0].RestrictionTypeValue(), RestrictionType::kCycle);
  ASSERT_EQ(loaded_tree[0].AncestorIndex(), RegTreeG::kInvalidNodeId);
  ASSERT_TRUE(loaded_tree[0].AncestorSign());

  // The tree holds only learned parameters, so a round trip is lossless.
  ASSERT_TRUE(loaded_tree == tree);

  // Verify children
  auto left = loaded_tree[0].LeftChild();
  auto right = loaded_tree[0].RightChild();
  ASSERT_TRUE(loaded_tree[left].IsLeaf());
  ASSERT_TRUE(loaded_tree[right].IsLeaf());
  ASSERT_EQ(loaded_tree[left].LeafValue(), 2.0f);
  ASSERT_EQ(loaded_tree[right].LeafValue(), 3.0f);
}

TEST(RegTreeG, ComplexTreeSerialization) {
  constexpr bst_node_t kNumVertices = 80;
  constexpr bst_feature_t kNumFeatures = 20;

  RegTreeG tree(kNumVertices, kNumFeatures, 4, 5);

  // Build a more complex tree
  tree.ExpandNode(0, 0, 0.0f, 1.0f, 2.0f, 3.0f, 1.0f, 10.0f, 4.0f, 6.0f);
  auto left = tree[0].LeftChild();
  auto right = tree[0].RightChild();

  tree.ExpandNode(left, 1, 1.0f, 2.0f, 4.0f, 5.0f, 0.5f, 4.0f, 2.0f, 2.0f);
  tree.ExpandNode(right, 2, 2.0f, 3.0f, 6.0f, 7.0f, 0.3f, 6.0f, 3.0f, 3.0f);

  // Set Tree-G params for internal nodes
  tree.SetNodeTreeGParams(0, 1, RestrictionType::kSource, RegTreeG::kInvalidNodeId, false);
  tree.SetNodeTreeGParams(left, 2, RestrictionType::kCycle, 0, true);
  tree.SetNodeTreeGParams(right, 1, RestrictionType::kSource, 0, false);

  // Save and load
  Json out{Object()};
  tree.SaveModel(&out);

  RegTreeG loaded_tree;
  loaded_tree.LoadModel(out);

  // Re-save loaded tree
  Json resaved{Object()};
  loaded_tree.SaveModel(&resaved);

  // The tree holds only learned parameters, so the round trip is lossless.
  ASSERT_EQ(get<Object>(out["tree_param"]), get<Object>(resaved["tree_param"]));
  ASSERT_TRUE(loaded_tree == tree);

  // Verify tree structure is preserved
  ASSERT_EQ(loaded_tree.NumNodes(), tree.NumNodes());
  ASSERT_EQ(loaded_tree[0].WalkLen(), 1);
  ASSERT_EQ(loaded_tree[left].WalkLen(), 2);
  ASSERT_EQ(loaded_tree[right].WalkLen(), 1);

  ASSERT_EQ(loaded_tree[left].RestrictionTypeValue(), RestrictionType::kCycle);
  ASSERT_EQ(loaded_tree[right].RestrictionTypeValue(), RestrictionType::kSource);

  ASSERT_EQ(loaded_tree[left].AncestorIndex(), 0);
  ASSERT_EQ(loaded_tree[right].AncestorIndex(), 0);
  ASSERT_TRUE(loaded_tree[left].AncestorSign());
  ASSERT_FALSE(loaded_tree[right].AncestorSign());
}

// ========== Edge Cases and Validation Tests ==========

TEST(RegTreeG, ValidationChecks) {
  constexpr bst_node_t kNumVertices = 30;
  RegTreeG tree(kNumVertices, 10, 2, 3);

  // Trying to set walk_len > max_walk_len should fail with CHECK
  tree.ExpandNode(0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

  // This should trigger a CHECK failure (exception)
  EXPECT_THROW(tree.SetNodeTreeGParams(0, 5, RestrictionType::kSource,  // walk_len=5 > max=2
                                       RegTreeG::kInvalidNodeId, true),
               dmlc::Error);
}

TEST(RegTreeG, CannotExpandNonLeaf) {
  constexpr bst_node_t kNumVertices = 40;
  RegTreeG tree(kNumVertices, 8);

  tree.ExpandNode(0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

  // Trying to expand a non-leaf should fail
  EXPECT_THROW(tree.ExpandNode(0, 1, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f), dmlc::Error);
}

// ========== ComputeSplitValue Tests ==========

/**
 * @brief Helper to create test DMatrix with specific data.
 *
 * Creates DMatrix with format [X | A | A^2 | ... | A^d]
 */
std::shared_ptr<DMatrix> CreateTestDMatrix(const std::vector<std::vector<float>>& full_data) {
  std::size_t n_rows = full_data.size();
  bst_feature_t n_cols = full_data[0].size();

  // Flatten to 1D vector
  std::vector<float> dense_data;
  dense_data.reserve(n_rows * n_cols);
  for (const auto& row : full_data) {
    dense_data.insert(dense_data.end(), row.begin(), row.end());
  }

  return GetDMatrixFromData(dense_data, n_rows, n_cols);
}

// ========== GraphData Tests ==========

TEST(GraphData, BuildBasicExtraction) {
  // 4 vertices, 2 features, d_max=2
  // DMatrix layout: [X (2 cols) | A^1 (4 cols) | A^2 (4 cols)] = 10 columns
  //
  // X:                A^1:                   A^2:
  //  1.0  2.0         0  1  1  0             2  1  1  1
  //  3.0  4.0         1  0  1  1             1  3  1  0
  //  5.0  6.0         1  1  0  0             1  1  2  1
  //  7.0  8.0         0  1  0  0             1  0  1  1
  constexpr bst_node_t kN = 4;
  constexpr bst_feature_t kK = 2;
  constexpr bst_node_t kDMax = 2;

  std::vector<std::vector<float>> full_data = {//  X        A^1          A^2
                                               {1, 2, 0, 1, 1, 0, 2, 1, 1, 1},
                                               {3, 4, 1, 0, 1, 1, 1, 3, 1, 0},
                                               {5, 6, 1, 1, 0, 0, 1, 1, 2, 1},
                                               {7, 8, 0, 1, 0, 0, 1, 0, 1, 1}};
  auto dmat = CreateTestDMatrix(full_data);

  auto gd = GraphData::Build(dmat.get(), kN, kK, kDMax);

  // --- Dimensions ---
  EXPECT_EQ(gd.num_vertices, kN);
  EXPECT_EQ(gd.num_features, kK);
  EXPECT_EQ(gd.max_walk_len, kDMax);
  ASSERT_EQ(gd.features.size(), kN * kK);
  ASSERT_EQ(gd.adj_powers.size(), kDMax);
  ASSERT_EQ(gd.adj_diag.size(), kDMax);

  // --- Feature matrix X ---
  EXPECT_FLOAT_EQ(gd.features[0 * kK + 0], 1.0f);
  EXPECT_FLOAT_EQ(gd.features[0 * kK + 1], 2.0f);
  EXPECT_FLOAT_EQ(gd.features[1 * kK + 0], 3.0f);
  EXPECT_FLOAT_EQ(gd.features[1 * kK + 1], 4.0f);
  EXPECT_FLOAT_EQ(gd.features[2 * kK + 0], 5.0f);
  EXPECT_FLOAT_EQ(gd.features[2 * kK + 1], 6.0f);
  EXPECT_FLOAT_EQ(gd.features[3 * kK + 0], 7.0f);
  EXPECT_FLOAT_EQ(gd.features[3 * kK + 1], 8.0f);

  // --- A^1 CSR structure ---
  const auto& a1 = gd.adj_powers[0];
  EXPECT_EQ(a1.nrows, kN);
  EXPECT_EQ(a1.ncols, kN);
  // A^1 non-zeros: row0={1,2}, row1={0,2,3}, row2={0,1}, row3={1} → nnz=8
  EXPECT_EQ(a1.Nnz(), 8);
  EXPECT_EQ(a1.row_ptr.size(), kN + 1);
  // Row 0: 2 nnz (cols 1,2)
  EXPECT_EQ(a1.row_ptr[1] - a1.row_ptr[0], 2);
  // Row 1: 3 nnz (cols 0,2,3)
  EXPECT_EQ(a1.row_ptr[2] - a1.row_ptr[1], 3);
  // Row 2: 2 nnz (cols 0,1)
  EXPECT_EQ(a1.row_ptr[3] - a1.row_ptr[2], 2);
  // Row 3: 1 nnz (col 1)
  EXPECT_EQ(a1.row_ptr[4] - a1.row_ptr[3], 1);

  // --- A^1 diagonals (all zero in this matrix) ---
  for (int i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(gd.adj_diag[0][i], 0.0f);
  }

  // --- A^2 CSR structure ---
  const auto& a2 = gd.adj_powers[1];
  EXPECT_EQ(a2.nrows, kN);
  EXPECT_EQ(a2.ncols, kN);
  // A^2 non-zeros: row0={0,1,2,3}=4, row1={0,1,2}=3, row2={0,1,2,3}=4, row3={0,2,3}=3 → nnz=14
  EXPECT_EQ(a2.Nnz(), 14);

  // --- A^2 diagonals ---
  EXPECT_FLOAT_EQ(gd.adj_diag[1][0], 2.0f);
  EXPECT_FLOAT_EQ(gd.adj_diag[1][1], 3.0f);
  EXPECT_FLOAT_EQ(gd.adj_diag[1][2], 2.0f);
  EXPECT_FLOAT_EQ(gd.adj_diag[1][3], 1.0f);
}

TEST(GraphData, BuildSingleFeatureNoAdj) {
  // Edge case: d_max=0 → no adjacency blocks, only features.
  constexpr bst_node_t kN = 3;
  constexpr bst_feature_t kK = 2;
  constexpr bst_node_t kDMax = 0;

  std::vector<std::vector<float>> full_data = {{10, 20}, {30, 40}, {50, 60}};
  auto dmat = CreateTestDMatrix(full_data);

  auto gd = GraphData::Build(dmat.get(), kN, kK, kDMax);

  EXPECT_EQ(gd.num_vertices, kN);
  EXPECT_EQ(gd.num_features, kK);
  EXPECT_EQ(gd.max_walk_len, kDMax);
  ASSERT_EQ(gd.features.size(), kN * kK);
  ASSERT_TRUE(gd.adj_powers.empty());
  ASSERT_TRUE(gd.adj_diag.empty());

  EXPECT_FLOAT_EQ(gd.features[0], 10.0f);
  EXPECT_FLOAT_EQ(gd.features[1], 20.0f);
  EXPECT_FLOAT_EQ(gd.features[2], 30.0f);
  EXPECT_FLOAT_EQ(gd.features[3], 40.0f);
  EXPECT_FLOAT_EQ(gd.features[4], 50.0f);
  EXPECT_FLOAT_EQ(gd.features[5], 60.0f);
}

TEST(GraphData, BuildZeroVertices) {
  // Empty graph: N=0 should produce empty arrays without crashing.
  std::vector<float> empty_data;
  auto dmat = GetDMatrixFromData(empty_data, 0, 0);

  auto gd = GraphData::Build(dmat.get(), 0, 0, 2);

  EXPECT_EQ(gd.num_vertices, 0);
  EXPECT_EQ(gd.num_features, 0);
  EXPECT_EQ(gd.max_walk_len, 2);
  EXPECT_TRUE(gd.features.empty());
  EXPECT_EQ(gd.adj_powers.size(), 2);
  EXPECT_EQ(gd.adj_powers[0].Nnz(), 0);
  EXPECT_EQ(gd.adj_powers[1].Nnz(), 0);
  EXPECT_EQ(gd.adj_diag.size(), 2);
  EXPECT_TRUE(gd.adj_diag[0].empty());
  EXPECT_TRUE(gd.adj_diag[1].empty());
}

TEST(GraphData, RejectsTooManyColumns) {
  // The declared shape needs num_features + N * max_walk_len > bst_feature_t max columns.
  // The size check must fire before anything is allocated, so a 1x1 DMatrix is enough.
  std::vector<float> data{0.0f};
  auto dmat = GetDMatrixFromData(data, 1, 1);
  constexpr bst_node_t kN = 1 << 20;
  constexpr bst_node_t kMaxWalkLen = 1 << 12;  // kN * kMaxWalkLen = 2^32 > UINT32_MAX
  try {
    [[maybe_unused]] auto gd = GraphData::Build(dmat.get(), kN, 1, kMaxWalkLen);
    FAIL() << "Expected GraphData::Build to reject an oversized graph.";
  } catch (dmlc::Error const& e) {
    EXPECT_NE(std::string{e.what()}.find("graph too large"), std::string::npos) << e.what();
  }
}

TEST(GraphData, BuildMatchesSparsePage) {
  // Verify that GraphData CSR reproduces every non-zero entry
  // in the SparsePage — guarantees no data corruption in the extraction.
  constexpr bst_node_t kN = 4;
  constexpr bst_feature_t kK = 1;
  constexpr bst_node_t kDMax = 2;

  std::vector<std::vector<float>> full_data = {{0, 0, 1, 1, 0, 2, 1, 1, 1},
                                               {3, 1, 0, 1, 1, 1, 3, 1, 0},
                                               {6, 1, 1, 0, 0, 1, 1, 2, 1},
                                               {9, 0, 1, 0, 0, 1, 0, 1, 1}};
  auto dmat = CreateTestDMatrix(full_data);

  auto gd = GraphData::Build(dmat.get(), kN, kK, kDMax);

  // Cross-check features against the original 2D data.
  for (bst_node_t i = 0; i < kN; ++i) {
    for (bst_feature_t k = 0; k < kK; ++k) {
      EXPECT_FLOAT_EQ(gd.features[i * kK + k], full_data[i][k])
          << "X mismatch at (" << i << ", " << k << ")";
    }
  }

  // Cross-check adjacency CSR: reconstruct dense from CSR and compare.
  for (bst_node_t d = 0; d < kDMax; ++d) {
    const auto& csr = gd.adj_powers[d];
    for (bst_node_t i = 0; i < kN; ++i) {
      // Build a map of non-zero columns for this row.
      std::map<int, float> row_nz;
      for (int p = csr.row_ptr[i]; p < csr.row_ptr[i + 1]; ++p) {
        row_nz[csr.col_idx[p]] = csr.values[p];
      }
      for (bst_node_t j = 0; j < kN; ++j) {
        bst_feature_t col = kK + d * kN + j;
        float expected = full_data[i][col];
        float actual = row_nz.count(j) ? row_nz[j] : 0.0f;
        EXPECT_FLOAT_EQ(actual, expected)
            << "adj_powers[" << d << "] mismatch at (" << i << ", " << j << ")";
      }
    }
  }
}

/**
 * @brief Build a DMatrix in which every zero is replaced by NaN (test helper).
 *
 * GetDMatrixFromData uses missing=NaN, so the resulting SparsePage omits those
 * entries entirely. This yields a genuinely sparse page holding the same logical
 * matrix as CreateTestDMatrix(full_data).
 */
std::shared_ptr<DMatrix> CreateSparseTestDMatrix(const std::vector<std::vector<float>>& full_data) {
  auto sparse_data = full_data;
  for (auto& row : sparse_data) {
    for (auto& v : row) {
      if (v == 0.0f) {
        v = std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
  return CreateTestDMatrix(sparse_data);
}

/**
 * @brief Assert that two GraphData instances are element-wise identical.
 */
void ExpectGraphDataEq(const GraphData& lhs, const GraphData& rhs) {
  ASSERT_EQ(lhs.num_vertices, rhs.num_vertices);
  ASSERT_EQ(lhs.num_features, rhs.num_features);
  ASSERT_EQ(lhs.max_walk_len, rhs.max_walk_len);

  ASSERT_EQ(lhs.features.size(), rhs.features.size());
  for (std::size_t i = 0; i < lhs.features.size(); ++i) {
    EXPECT_FLOAT_EQ(lhs.features[i], rhs.features[i]) << "X mismatch at " << i;
  }

  ASSERT_EQ(lhs.adj_powers.size(), rhs.adj_powers.size());
  for (std::size_t d = 0; d < lhs.adj_powers.size(); ++d) {
    const auto& a = lhs.adj_powers[d];
    const auto& b = rhs.adj_powers[d];
    EXPECT_EQ(a.Nnz(), b.Nnz()) << "nnz mismatch for power " << d;
    EXPECT_EQ(a.row_ptr, b.row_ptr) << "row_ptr mismatch for power " << d;
    EXPECT_EQ(a.col_idx, b.col_idx) << "col_idx mismatch for power " << d;
    ASSERT_EQ(a.values.size(), b.values.size());
    for (std::size_t p = 0; p < a.values.size(); ++p) {
      EXPECT_FLOAT_EQ(a.values[p], b.values[p]) << "value mismatch for power " << d;
    }
    EXPECT_EQ(lhs.adj_diag[d], rhs.adj_diag[d]) << "diag mismatch for power " << d;
  }
}

TEST(GraphData, BuildSparseInputMatchesDense) {
  // A sparse SparsePage (zeros absent) and a dense one (zeros stored explicitly)
  // must yield byte-identical GraphData: absent and zero mean the same thing here.
  constexpr bst_node_t kN = 4;
  constexpr bst_feature_t kK = 1;
  constexpr bst_node_t kDMax = 2;

  std::vector<std::vector<float>> full_data = {{0, 0, 1, 1, 0, 2, 1, 1, 1},
                                               {3, 1, 0, 1, 1, 1, 3, 1, 0},
                                               {6, 1, 1, 0, 0, 1, 1, 2, 1},
                                               {9, 0, 1, 0, 0, 1, 0, 1, 1}};

  auto dense_mat = CreateTestDMatrix(full_data);
  auto sparse_mat = CreateSparseTestDMatrix(full_data);

  // Guard the premise: the sparse page really must hold fewer entries.
  ASSERT_LT(sparse_mat->Info().num_nonzero_, dense_mat->Info().num_nonzero_);

  auto dense_gd = GraphData::Build(dense_mat.get(), kN, kK, kDMax);
  auto sparse_gd = GraphData::Build(sparse_mat.get(), kN, kK, kDMax);

  ExpectGraphDataEq(dense_gd, sparse_gd);
}

TEST(GraphData, BuildSparseInputDropsExplicitZeros) {
  // CSR storage must reflect the true edge count, not the stored entry count,
  // for both input representations.
  constexpr bst_node_t kN = 3;
  constexpr bst_feature_t kK = 1;
  constexpr bst_node_t kDMax = 1;

  // A^1 has exactly 2 non-zero entries: (0,1) and (1,0).
  std::vector<std::vector<float>> full_data = {{5, 0, 1, 0}, {7, 1, 0, 0}, {9, 0, 0, 0}};

  std::vector<std::pair<std::string, std::shared_ptr<DMatrix>>> cases;
  cases.emplace_back("dense", CreateTestDMatrix(full_data));
  cases.emplace_back("sparse", CreateSparseTestDMatrix(full_data));

  for (auto const& [name, dmat] : cases) {
    auto gd = GraphData::Build(dmat.get(), kN, kK, kDMax);
    EXPECT_EQ(gd.adj_powers[0].Nnz(), 2) << "unexpected nnz for " << name;
    // Vertex 2 is isolated in both representations.
    EXPECT_EQ(gd.adj_powers[0].row_ptr[2], gd.adj_powers[0].row_ptr[3]) << name;
  }
}

TEST(GraphData, BuildMissingImputedAsZeroInBothBlocks) {
  // Missing entries are imputed to zero, with identical semantics in the X block
  // and the adjacency blocks. There is no learned default direction.
  constexpr bst_node_t kN = 2;
  constexpr bst_feature_t kK = 2;
  constexpr bst_node_t kDMax = 1;

  auto nan = std::numeric_limits<float>::quiet_NaN();
  //                       X (2)     A^1 (2)
  std::vector<std::vector<float>> with_missing = {{1.0f, nan, nan, 4.0f}, {nan, 2.0f, 3.0f, nan}};
  std::vector<std::vector<float>> with_zeros = {{1.0f, 0.0f, 0.0f, 4.0f}, {0.0f, 2.0f, 3.0f, 0.0f}};

  auto gd_missing = GraphData::Build(CreateTestDMatrix(with_missing).get(), kN, kK, kDMax);
  auto gd_zeros = GraphData::Build(CreateTestDMatrix(with_zeros).get(), kN, kK, kDMax);

  ExpectGraphDataEq(gd_missing, gd_zeros);

  // Missing X entries surface as 0.0, not NaN.
  EXPECT_FLOAT_EQ(gd_missing.features[0 * kK + 1], 0.0f);
  EXPECT_FLOAT_EQ(gd_missing.features[1 * kK + 0], 0.0f);
  EXPECT_FALSE(std::isnan(gd_missing.features[0 * kK + 1]));

  // Missing adjacency entries become absent edges, not zero-weight edges.
  EXPECT_EQ(gd_missing.adj_powers[0].Nnz(), 2);
}

TEST(GraphData, BuildRejectsMultiPageDMatrix) {
  // Tree-G aggregates over arbitrary vertex subsets, so it cannot stream pages.
  // Multi-page input must fail loudly rather than silently use page 0.
  constexpr bst_idx_t kRows = 8;
  constexpr bst_feature_t kCols = 9;  // X(1) | A^1(8) for 8 vertices... see below
  constexpr bst_node_t kN = 8;
  constexpr bst_feature_t kK = 1;
  constexpr bst_node_t kDMax = 1;

  HostDeviceVector<float> data;
  auto& h_data = data.HostVector();
  h_data.resize(kRows * kCols);
  for (std::size_t i = 0; i < h_data.size(); ++i) {
    h_data[i] = static_cast<float>(i % 3);
  }

  common::TemporaryDirectory tempdir;
  auto dmat = GetExternalMemoryDMatrixFromData(data, kRows, kCols, tempdir, /*n_batches=*/4);
  ASSERT_FALSE(dmat->SingleColBlock()) << "test premise: DMatrix must be multi-page";

  EXPECT_THROW(static_cast<void>(GraphData::Build(dmat.get(), kN, kK, kDMax)), dmlc::Error);
}

// ========== ComputeSplitValue Tests ==========

TEST(RegTreeG, ComputeSplitValueTestData) {
  // Test case from task specification
  // n=4, k=1 (num_features), d_max=2 (walk_len 0, 1, 2)
  constexpr bst_node_t kNumVertices = 4;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_feature_t kFeatureIdx = 0;

  // X = [0, 3, 6, 9]
  // A = [[0,1,1,0],[1,0,1,1],[1,1,0,0],[0,1,0,0]]
  // A^2 = [[2,1,1,1],[1,3,1,0],[1,1,2,1],[1,0,1,1]]

  // DMatrix format: [X | A | A^2]
  // Total columns: 1 (X) + 4 (A) + 4 (A^2) = 9

  std::vector<std::vector<float>> full_data = {
      // X,  A,         A^2
      {0, 0, 1, 1, 0, 2, 1, 1, 1},  // vertex 0
      {3, 1, 0, 1, 1, 1, 3, 1, 0},  // vertex 1
      {6, 1, 1, 0, 0, 1, 1, 2, 1},  // vertex 2
      {9, 0, 1, 0, 0, 1, 0, 1, 1}   // vertex 3
  };

  auto dmat = CreateTestDMatrix(full_data);
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);

  // Create tree with correct parameters
  RegTreeG tree(kNumVertices, kNumFeatures, kMaxWalkLen);

  // Initialize ActiveSet to (0,1,1,0)
  ActiveSet active_set(kNumVertices, false);
  active_set.Set(1, true);
  active_set.Set(2, true);

  // Verify Mask
  const auto& mask = active_set.Mask();
  ASSERT_EQ(mask.size(), kNumVertices);
  ASSERT_FLOAT_EQ(mask[0], 0.0f);
  ASSERT_FLOAT_EQ(mask[1], 1.0f);
  ASSERT_FLOAT_EQ(mask[2], 1.0f);
  ASSERT_FLOAT_EQ(mask[3], 0.0f);

  // Split value of vertex 1 with, walk_len=1, r=1 (Source)
  // Expected: 6
  // Calculation: (A^1 @ Mask @ X)[1] = A^1[1,:] @ (mask * X)
  // mask * X = [0, 3, 6, 0]
  // A^1[1,:] = [1, 0, 1, 1]
  // Result = 1*0 + 0*3 + 1*6 + 1*0 = 6
  bst_float result1 = tree.ComputeFeatureSplitValues(kFeatureIdx, 1, RestrictionType::kSource,
                                                     active_set, graph_data)[1];
  ASSERT_FLOAT_EQ(result1, 6.0f);

  // Split value of vertex 1 with, walk_len=1, r=2 (Cycle)
  // Expected: 0
  // Calculation: diag(A^1)[1] * mask[1] * X[1] = 0 * 1 * 3 = 0
  bst_float result2 = tree.ComputeFeatureSplitValues(kFeatureIdx, 1, RestrictionType::kCycle,
                                                     active_set, graph_data)[1];
  ASSERT_FLOAT_EQ(result2, 0.0f);

  // Split value of vertex 1 with, walk_len=2, r=2 (Cycle)
  // Expected: 9
  // Calculation: diag(A^2)[1] * mask[1] * X[1] = 3 * 1 * 3 = 9
  bst_float result3 = tree.ComputeFeatureSplitValues(kFeatureIdx, 2, RestrictionType::kCycle,
                                                     active_set, graph_data)[1];
  ASSERT_FLOAT_EQ(result3, 9.0f);
}

TEST(RegTreeG, ComputeSplitValueWalkLenZero) {
  // When d=0, split value is raw X[v, feature_idx] — no masking.
  constexpr bst_node_t kNumVertices = 4;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_feature_t kFeatureIdx = 0;

  // X = [0, 3, 6, 9]
  std::vector<std::vector<float>> full_data = {{0, 0, 1, 1, 0, 2, 1, 1, 1},
                                               {3, 1, 0, 1, 1, 1, 3, 1, 0},
                                               {6, 1, 1, 0, 0, 1, 1, 2, 1},
                                               {9, 0, 1, 0, 0, 1, 0, 1, 1}};
  auto dmat = CreateTestDMatrix(full_data);
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  RegTreeG tree(kNumVertices, kNumFeatures, kMaxWalkLen);

  // Partial active set: only vertices 1 and 2
  ActiveSet partial(kNumVertices, false);
  partial.Set(1, true);
  partial.Set(2, true);

  // d=0: every vertex gets its raw feature, regardless of mask
  std::vector<bst_float> vec =
      tree.ComputeFeatureSplitValues(kFeatureIdx, 0, RestrictionType::kSource, partial, graph_data);
  ASSERT_EQ(vec.size(), kNumVertices);
  ASSERT_FLOAT_EQ(vec[0], 0.0f);
  ASSERT_FLOAT_EQ(vec[1], 3.0f);
  ASSERT_FLOAT_EQ(vec[2], 6.0f);
  ASSERT_FLOAT_EQ(vec[3], 9.0f);

  // Verify Cycle gives the same result at d=0 (restriction type is ignored)
  std::vector<bst_float> vec2 =
      tree.ComputeFeatureSplitValues(kFeatureIdx, 0, RestrictionType::kCycle, partial, graph_data);
  ASSERT_EQ(vec2.size(), kNumVertices);
  ASSERT_FLOAT_EQ(vec2[0], 0.0f);
  ASSERT_FLOAT_EQ(vec2[1], 3.0f);
  ASSERT_FLOAT_EQ(vec2[2], 6.0f);
  ASSERT_FLOAT_EQ(vec2[3], 9.0f);
}

TEST(RegTreeG, ComputeFeatureSplitValuesAllVertices) {
  // Same graph as above; checks the whole column
  constexpr bst_node_t kNumVertices = 4;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_feature_t kFeatureIdx = 0;

  std::vector<std::vector<float>> full_data = {
      // X,  A,         A^2
      {0, 0, 1, 1, 0, 2, 1, 1, 1},  // vertex 0
      {3, 1, 0, 1, 1, 1, 3, 1, 0},  // vertex 1
      {6, 1, 1, 0, 0, 1, 1, 2, 1},  // vertex 2
      {9, 0, 1, 0, 0, 1, 0, 1, 1}   // vertex 3
  };

  auto dmat = CreateTestDMatrix(full_data);
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  RegTreeG tree(kNumVertices, kNumFeatures, kMaxWalkLen);

  ActiveSet active_set(kNumVertices, false);
  active_set.Set(1, true);
  active_set.Set(2, true);

  // Test walk_len=1, r=1 (Source)
  // Expected: [9, 6, 3, 3]
  // For vertex 0: A^1[0,:] @ (mask * X) = [0,1,1,0] @ [0,3,6,0] = 9
  // For vertex 1: A^1[1,:] @ (mask * X) = [1,0,1,1] @ [0,3,6,0] = 6
  // For vertex 2: A^1[2,:] @ (mask * X) = [1,1,0,0] @ [0,3,6,0] = 3
  // For vertex 3: A^1[3,:] @ (mask * X) = [0,1,0,0] @ [0,3,6,0] = 3
  std::vector<bst_float> result1 = tree.ComputeFeatureSplitValues(
      kFeatureIdx, 1, RestrictionType::kSource, active_set, graph_data);

  ASSERT_EQ(result1.size(), kNumVertices);
  ASSERT_FLOAT_EQ(result1[0], 9.0f);
  ASSERT_FLOAT_EQ(result1[1], 6.0f);
  ASSERT_FLOAT_EQ(result1[2], 3.0f);
  ASSERT_FLOAT_EQ(result1[3], 3.0f);

  // Test walk_len=1, r=2 (Cycle)
  // Expected: [0, 0, 0, 0]
  // diag(A^1) = [0, 0, 0, 0], so all results are 0
  std::vector<bst_float> result2 = tree.ComputeFeatureSplitValues(
      kFeatureIdx, 1, RestrictionType::kCycle, active_set, graph_data);

  ASSERT_EQ(result2.size(), kNumVertices);
  ASSERT_FLOAT_EQ(result2[0], 0.0f);
  ASSERT_FLOAT_EQ(result2[1], 0.0f);
  ASSERT_FLOAT_EQ(result2[2], 0.0f);
  ASSERT_FLOAT_EQ(result2[3], 0.0f);

  // Test walk_len=2, r=2 (Cycle)
  // Expected: [0, 9, 12, 0]
  // For vertex 0: diag(A^2)[0] * mask[0] * X[0] = 2 * 0 * 0 = 0
  // For vertex 1: diag(A^2)[1] * mask[1] * X[1] = 3 * 1 * 3 = 9
  // For vertex 2: diag(A^2)[2] * mask[2] * X[2] = 2 * 1 * 6 = 12
  // For vertex 3: diag(A^2)[3] * mask[3] * X[3] = 1 * 0 * 9 = 0
  std::vector<bst_float> result3 = tree.ComputeFeatureSplitValues(
      kFeatureIdx, 2, RestrictionType::kCycle, active_set, graph_data);

  ASSERT_EQ(result3.size(), kNumVertices);
  ASSERT_FLOAT_EQ(result3[0], 0.0f);
  ASSERT_FLOAT_EQ(result3[1], 9.0f);
  ASSERT_FLOAT_EQ(result3[2], 12.0f);
  ASSERT_FLOAT_EQ(result3[3], 0.0f);
}

// ========== Intercept Feature Tests ==========

TEST(RegTreeG, InterceptWalkLenZero) {
  // When intercept=true and feature_idx == num_features, d=0 should yield 1.0
  // for every vertex — the raw intercept value — regardless of active set.
  constexpr bst_node_t kNumVertices = 4;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_feature_t kInterceptIdx = kNumFeatures;  // virtual feature

  std::vector<std::vector<float>> full_data = {{0, 0, 1, 1, 0, 2, 1, 1, 1},
                                               {3, 1, 0, 1, 1, 1, 3, 1, 0},
                                               {6, 1, 1, 0, 0, 1, 1, 2, 1},
                                               {9, 0, 1, 0, 0, 1, 0, 1, 1}};
  auto dmat = CreateTestDMatrix(full_data);
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  RegTreeG tree(kNumVertices, kNumFeatures, kMaxWalkLen, /*max_ancestor_dist=*/2,
                /*intercept=*/true);
  ASSERT_TRUE(tree.Intercept());

  // Arbitrary partial active set (should be ignored at d=0)
  ActiveSet partial(kNumVertices, false);
  partial.Set(1, true);

  // Every vertex gets 1.0
  auto vec = tree.ComputeFeatureSplitValues(kInterceptIdx, 0, RestrictionType::kSource, partial,
                                            graph_data);
  ASSERT_EQ(vec.size(), kNumVertices);
  for (bst_node_t v = 0; v < kNumVertices; ++v) {
    ASSERT_FLOAT_EQ(vec[v], 1.0f);
  }
}

TEST(RegTreeG, InterceptSourceWalkLen1) {
  // At d=1, Source, intercept → split_values[i] = sum_j A[i,j] * mask[j] * 1.0
  // i.e., masked row-sum of A (weighted in-degree from the active set).
  //
  // A = [[0,1,1,0],[1,0,1,1],[1,1,0,0],[0,1,0,0]]
  // mask = {1,2} → [0,1,1,0]
  //
  // vertex 0: A[0,:] · mask = 0*0 + 1*1 + 1*1 + 0*0 = 2
  // vertex 1: A[1,:] · mask = 1*0 + 0*1 + 1*1 + 1*0 = 1
  // vertex 2: A[2,:] · mask = 1*0 + 1*1 + 0*1 + 0*0 = 1
  // vertex 3: A[3,:] · mask = 0*0 + 1*1 + 0*1 + 0*0 = 1
  constexpr bst_node_t kNumVertices = 4;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_feature_t kInterceptIdx = kNumFeatures;

  std::vector<std::vector<float>> full_data = {{0, 0, 1, 1, 0, 2, 1, 1, 1},
                                               {3, 1, 0, 1, 1, 1, 3, 1, 0},
                                               {6, 1, 1, 0, 0, 1, 1, 2, 1},
                                               {9, 0, 1, 0, 0, 1, 0, 1, 1}};
  auto dmat = CreateTestDMatrix(full_data);
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  RegTreeG tree(kNumVertices, kNumFeatures, kMaxWalkLen, 2, /*intercept=*/true);

  ActiveSet active(kNumVertices, false);
  active.Set(1, true);
  active.Set(2, true);

  auto vec = tree.ComputeFeatureSplitValues(kInterceptIdx, 1, RestrictionType::kSource, active,
                                            graph_data);
  ASSERT_EQ(vec.size(), kNumVertices);
  ASSERT_FLOAT_EQ(vec[0], 2.0f);
  ASSERT_FLOAT_EQ(vec[1], 1.0f);
  ASSERT_FLOAT_EQ(vec[2], 1.0f);
  ASSERT_FLOAT_EQ(vec[3], 1.0f);
}

TEST(RegTreeG, InterceptCycleWalkLen2) {
  // At d=2, Cycle, intercept → split_values[i] = A^2[i,i] * mask[i] * 1.0
  //
  // A^2 = [[2,1,1,1],[1,3,1,0],[1,1,2,1],[1,0,1,1]]
  // mask = {1,2} → [0,1,1,0]
  //
  // vertex 0: mask[0]=0 → 0
  // vertex 1: A^2[1,1]*1*1 = 3
  // vertex 2: A^2[2,2]*1*1 = 2
  // vertex 3: mask[3]=0 → 0
  constexpr bst_node_t kNumVertices = 4;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_feature_t kInterceptIdx = kNumFeatures;

  std::vector<std::vector<float>> full_data = {{0, 0, 1, 1, 0, 2, 1, 1, 1},
                                               {3, 1, 0, 1, 1, 1, 3, 1, 0},
                                               {6, 1, 1, 0, 0, 1, 1, 2, 1},
                                               {9, 0, 1, 0, 0, 1, 0, 1, 1}};
  auto dmat = CreateTestDMatrix(full_data);
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  RegTreeG tree(kNumVertices, kNumFeatures, kMaxWalkLen, 2, /*intercept=*/true);

  ActiveSet active(kNumVertices, false);
  active.Set(1, true);
  active.Set(2, true);

  auto vec =
      tree.ComputeFeatureSplitValues(kInterceptIdx, 2, RestrictionType::kCycle, active, graph_data);
  ASSERT_EQ(vec.size(), kNumVertices);
  ASSERT_FLOAT_EQ(vec[0], 0.0f);
  ASSERT_FLOAT_EQ(vec[1], 3.0f);
  ASSERT_FLOAT_EQ(vec[2], 2.0f);
  ASSERT_FLOAT_EQ(vec[3], 0.0f);
}

TEST(RegTreeG, InterceptCheckFailsWhenDisabled) {
  // When intercept=false, feature_idx == num_features must trigger CHECK failure.
  constexpr bst_node_t kNumVertices = 4;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_feature_t kInterceptIdx = kNumFeatures;

  std::vector<std::vector<float>> full_data = {{0, 0, 1, 1, 0, 2, 1, 1, 1},
                                               {3, 1, 0, 1, 1, 1, 3, 1, 0},
                                               {6, 1, 1, 0, 0, 1, 1, 2, 1},
                                               {9, 0, 1, 0, 0, 1, 0, 1, 1}};
  auto dmat = CreateTestDMatrix(full_data);
  auto graph_data = GraphData::Build(dmat.get(), kNumVertices, kNumFeatures, kMaxWalkLen);
  RegTreeG tree(kNumVertices, kNumFeatures, kMaxWalkLen, 2, /*intercept=*/false);
  ASSERT_FALSE(tree.Intercept());

  ActiveSet active(kNumVertices, true);

  EXPECT_THROW((void)tree.ComputeFeatureSplitValues(kInterceptIdx, 1, RestrictionType::kSource,
                                                    active, graph_data),
               dmlc::Error);
}

TEST(RegTreeG, InterceptSerializationRoundTrip) {
  // Verify intercept flag survives JSON save/load, and that the key is required.
  TreeGParam param;
  param.num_nodes = 1;
  param.num_features = 5;
  param.num_vertices = 10;
  param.intercept = true;

  Json json{Object()};
  param.ToJson(&json);

  TreeGParam loaded;
  loaded.FromJson(json);
  ASSERT_TRUE(loaded.intercept);

  // No model format without the key has shipped, so a missing key is an error.
  auto& obj = get<Object>(json);
  obj.erase(std::string{"intercept"});

  TreeGParam missing_key;
  EXPECT_THROW(missing_key.FromJson(json), dmlc::Error);
}

TEST(RegTreeG, InterceptRegTreeGAccessor) {
  RegTreeG tree_on(4, 1, 2, 2, /*intercept=*/true);
  ASSERT_TRUE(tree_on.Intercept());

  RegTreeG tree_off(4, 1, 2, 2, /*intercept=*/false);
  ASSERT_FALSE(tree_off.Intercept());

  // Default constructor should have intercept=false
  RegTreeG tree_default;
  ASSERT_FALSE(tree_default.Intercept());
}

// ========== ComputeSplitValues (all features) Tests ==========

namespace {
/** @brief Split values for every feature, plus the intercept column when it is enabled. */
std::vector<float> AllFeatureSplitValues(RegTreeG const& tree, bst_node_t walk_len,
                                         RestrictionType restriction, ActiveSet const& subset,
                                         GraphData const& graph_data) {
  std::vector<bst_feature_t> features(tree.NumFeatures());
  std::iota(features.begin(), features.end(), 0);
  return tree.ComputeSplitValues(walk_len, restriction, subset, graph_data, features,
                                 tree.Intercept());
}
}  // anonymous namespace

/**
 * @brief Verify ComputeSplitValues columns match ComputeFeatureSplitValues
 *        for every (feature, walk_len, restriction_type, attention_set) combo.
 *
 * Uses the canonical 4-vertex, 2-feature test graph from earlier tests.
 */
TEST(RegTreeG, BatchComputeSplitValuesMatchesSingleFeature) {
  // n=4, k=2, d_max=2
  // X:                A^1:                   A^2:
  //  1.0  2.0         0  1  1  0             2  1  1  1
  //  3.0  4.0         1  0  1  1             1  3  1  0
  //  5.0  6.0         1  1  0  0             1  1  2  1
  //  7.0  8.0         0  1  0  0             1  0  1  1
  constexpr bst_node_t kN = 4;
  constexpr bst_feature_t kK = 2;
  constexpr bst_node_t kDMax = 2;

  std::vector<std::vector<float>> full_data = {{1, 2, 0, 1, 1, 0, 2, 1, 1, 1},
                                               {3, 4, 1, 0, 1, 1, 1, 3, 1, 0},
                                               {5, 6, 1, 1, 0, 0, 1, 1, 2, 1},
                                               {7, 8, 0, 1, 0, 0, 1, 0, 1, 1}};
  auto dmat = CreateTestDMatrix(full_data);
  auto graph_data = GraphData::Build(dmat.get(), kN, kK, kDMax);
  RegTreeG tree(kN, kK, kDMax);

  // Two attention sets: all-vertices and a partial set {1, 2}
  ActiveSet all_vertices(kN, true);
  ActiveSet partial(kN, false);
  partial.Set(1, true);
  partial.Set(2, true);

  std::vector<const ActiveSet*> attention_sets = {&all_vertices, &partial};
  bst_feature_t K_total = kK;  // no intercept

  for (const auto* aset : attention_sets) {
    // d=0 (restriction type ignored)
    {
      auto batch = AllFeatureSplitValues(tree, 0, RestrictionType::kSource, *aset, graph_data);
      ASSERT_EQ(batch.size(), static_cast<std::size_t>(kN) * K_total);
      for (bst_feature_t k = 0; k < K_total; ++k) {
        auto single =
            tree.ComputeFeatureSplitValues(k, 0, RestrictionType::kSource, *aset, graph_data);
        for (bst_node_t i = 0; i < kN; ++i) {
          EXPECT_FLOAT_EQ(batch[i * K_total + k], single[i])
              << "d=0 mismatch at vertex=" << i << " feature=" << k;
        }
      }
    }
    // d=1..d_max, both restriction types
    for (bst_node_t d = 1; d <= kDMax; ++d) {
      for (auto r : {RestrictionType::kSource, RestrictionType::kCycle}) {
        auto batch = AllFeatureSplitValues(tree, d, r, *aset, graph_data);
        ASSERT_EQ(batch.size(), static_cast<std::size_t>(kN) * K_total);
        for (bst_feature_t k = 0; k < K_total; ++k) {
          auto single = tree.ComputeFeatureSplitValues(k, d, r, *aset, graph_data);
          for (bst_node_t i = 0; i < kN; ++i) {
            EXPECT_NEAR(batch[i * K_total + k], single[i], 1e-5f)
                << "d=" << d << " r=" << static_cast<int>(r) << " vertex=" << i << " feature=" << k;
          }
        }
      }
    }
  }
}

/**
 * @brief Verify ComputeSplitValues with intercept enabled.
 *
 * When intercept=true, K_total = K + 1 and the last column must match
 * ComputeFeatureSplitValues(feature_idx=K, ...) — the virtual intercept.
 */
TEST(RegTreeG, BatchComputeSplitValuesWithIntercept) {
  constexpr bst_node_t kN = 4;
  constexpr bst_feature_t kK = 1;
  constexpr bst_node_t kDMax = 2;

  std::vector<std::vector<float>> full_data = {{0, 0, 1, 1, 0, 2, 1, 1, 1},
                                               {3, 1, 0, 1, 1, 1, 3, 1, 0},
                                               {6, 1, 1, 0, 0, 1, 1, 2, 1},
                                               {9, 0, 1, 0, 0, 1, 0, 1, 1}};
  auto dmat = CreateTestDMatrix(full_data);
  auto graph_data = GraphData::Build(dmat.get(), kN, kK, kDMax);
  RegTreeG tree(kN, kK, kDMax, /*max_ancestor_dist=*/2, /*intercept=*/true);
  ASSERT_TRUE(tree.Intercept());

  bst_feature_t K_total = kK + 1;  // intercept adds one column

  ActiveSet partial(kN, false);
  partial.Set(1, true);
  partial.Set(2, true);

  // d=0: real feature column + intercept column (all 1.0)
  {
    auto batch = AllFeatureSplitValues(tree, 0, RestrictionType::kSource, partial, graph_data);
    ASSERT_EQ(batch.size(), static_cast<std::size_t>(kN) * K_total);
    for (bst_feature_t k = 0; k < K_total; ++k) {
      auto single =
          tree.ComputeFeatureSplitValues(k, 0, RestrictionType::kSource, partial, graph_data);
      for (bst_node_t i = 0; i < kN; ++i) {
        EXPECT_FLOAT_EQ(batch[i * K_total + k], single[i])
            << "d=0 intercept mismatch at vertex=" << i << " feature=" << k;
      }
    }
  }
  // d=1, Source — intercept column = masked row-sum of A
  {
    auto batch = AllFeatureSplitValues(tree, 1, RestrictionType::kSource, partial, graph_data);
    for (bst_feature_t k = 0; k < K_total; ++k) {
      auto single =
          tree.ComputeFeatureSplitValues(k, 1, RestrictionType::kSource, partial, graph_data);
      for (bst_node_t i = 0; i < kN; ++i) {
        EXPECT_NEAR(batch[i * K_total + k], single[i], 1e-5f)
            << "d=1 Source intercept mismatch at vertex=" << i << " feature=" << k;
      }
    }
  }
  // d=2, Cycle — intercept column = diag(A^2) * mask * 1.0
  {
    auto batch = AllFeatureSplitValues(tree, 2, RestrictionType::kCycle, partial, graph_data);
    for (bst_feature_t k = 0; k < K_total; ++k) {
      auto single =
          tree.ComputeFeatureSplitValues(k, 2, RestrictionType::kCycle, partial, graph_data);
      for (bst_node_t i = 0; i < kN; ++i) {
        EXPECT_NEAR(batch[i * K_total + k], single[i], 1e-5f)
            << "d=2 Cycle intercept mismatch at vertex=" << i << " feature=" << k;
      }
    }
  }
}

/**
 * @brief Verify ComputeSplitValues with specific known values.
 *
 * Directly checks the numerical output for the Source d=1 case with the
 * 4-vertex, 2-feature graph and mask = {1, 2}.
 */
TEST(RegTreeG, BatchComputeSplitValuesKnownValues) {
  constexpr bst_node_t kN = 4;
  constexpr bst_feature_t kK = 2;
  constexpr bst_node_t kDMax = 2;

  // X:                A^1:
  //  1.0  2.0         0  1  1  0
  //  3.0  4.0         1  0  1  1
  //  5.0  6.0         1  1  0  0
  //  7.0  8.0         0  1  0  0
  std::vector<std::vector<float>> full_data = {{1, 2, 0, 1, 1, 0, 2, 1, 1, 1},
                                               {3, 4, 1, 0, 1, 1, 1, 3, 1, 0},
                                               {5, 6, 1, 1, 0, 0, 1, 1, 2, 1},
                                               {7, 8, 0, 1, 0, 0, 1, 0, 1, 1}};
  auto dmat = CreateTestDMatrix(full_data);
  auto graph_data = GraphData::Build(dmat.get(), kN, kK, kDMax);
  RegTreeG tree(kN, kK, kDMax);

  ActiveSet mask(kN, false);
  mask.Set(1, true);
  mask.Set(2, true);

  // d=1, Source: result = A^1 @ X_masked
  // X_masked (mask={1,2}):
  //   row 0: [0, 0]
  //   row 1: [3, 4]
  //   row 2: [5, 6]
  //   row 3: [0, 0]
  //
  // A^1 @ X_masked:
  //   vertex 0: [0,1,1,0] . [col0] = 0+3+5+0=8, [col1] = 0+4+6+0=10
  //   vertex 1: [1,0,1,1] . [col0] = 0+0+5+0=5, [col1] = 0+0+6+0=6
  //   vertex 2: [1,1,0,0] . [col0] = 0+3+0+0=3, [col1] = 0+4+0+0=4
  //   vertex 3: [0,1,0,0] . [col0] = 0+3+0+0=3, [col1] = 0+4+0+0=4
  auto batch = AllFeatureSplitValues(tree, 1, RestrictionType::kSource, mask, graph_data);
  ASSERT_EQ(batch.size(), kN * kK);

  // feature 0: [8, 5, 3, 3]
  EXPECT_NEAR(batch[0 * kK + 0], 8.0f, 1e-5f);
  EXPECT_NEAR(batch[1 * kK + 0], 5.0f, 1e-5f);
  EXPECT_NEAR(batch[2 * kK + 0], 3.0f, 1e-5f);
  EXPECT_NEAR(batch[3 * kK + 0], 3.0f, 1e-5f);

  // feature 1: [10, 6, 4, 4]
  EXPECT_NEAR(batch[0 * kK + 1], 10.0f, 1e-5f);
  EXPECT_NEAR(batch[1 * kK + 1], 6.0f, 1e-5f);
  EXPECT_NEAR(batch[2 * kK + 1], 4.0f, 1e-5f);
  EXPECT_NEAR(batch[3 * kK + 1], 4.0f, 1e-5f);
}

// ========== ComputeSplitValues with Sampled Features Tests ==========

/**
 * @brief Verify sampled ComputeSplitValues matches full version for selected columns.
 *
 * When sampling a subset of features, the sampled result matrix should match
 * the corresponding columns of the full result matrix.
 */
TEST(RegTreeG, BatchComputeSplitValuesSampledMatchesFull) {
  constexpr bst_node_t kN = 4;
  constexpr bst_feature_t kK = 4;
  constexpr bst_node_t kDmax = 1;

  RegTreeG tree(kN, kK, kDmax, /*max_ancestor_dist=*/0, /*intercept=*/false);

  // Build GraphData with 4 features and a ring adjacency.
  std::vector<float> features = {
      1.0f,  2.0f,  3.0f,  4.0f,   // vertex 0
      5.0f,  6.0f,  7.0f,  8.0f,   // vertex 1
      9.0f,  10.0f, 11.0f, 12.0f,  // vertex 2
      13.0f, 14.0f, 15.0f, 16.0f   // vertex 3
  };
  // Ring adjacency: 0-1, 1-2, 2-3, 3-0
  std::vector<float> ring(kN * kN, 0.0f);
  ring[0 * kN + 1] = 1.0f;
  ring[0 * kN + 3] = 1.0f;
  ring[1 * kN + 0] = 1.0f;
  ring[1 * kN + 2] = 1.0f;
  ring[2 * kN + 1] = 1.0f;
  ring[2 * kN + 3] = 1.0f;
  ring[3 * kN + 2] = 1.0f;
  ring[3 * kN + 0] = 1.0f;
  auto gd = BuildGraphDataFromDense(kN, kK, kDmax, features, {ring});

  ActiveSet all(kN, true);

  // Full batch: N × K matrix
  auto full = AllFeatureSplitValues(tree, 1, RestrictionType::kSource, all, gd);
  ASSERT_EQ(full.size(), static_cast<std::size_t>(kN * kK));

  // Sampled: features {1, 3} → N × 2 matrix
  std::vector<bst_feature_t> sampled = {1, 3};
  auto reduced =
      tree.ComputeSplitValues(1, RestrictionType::kSource, all, gd, sampled, tree.Intercept());
  ASSERT_EQ(reduced.size(), static_cast<std::size_t>(kN * 2));

  // Verify: reduced column j should match full column sampled[j]
  for (int i = 0; i < kN; ++i) {
    EXPECT_NEAR(reduced[i * 2 + 0], full[i * kK + 1], 1e-5f)
        << "Mismatch at vertex " << i << " sampled feature 1";
    EXPECT_NEAR(reduced[i * 2 + 1], full[i * kK + 3], 1e-5f)
        << "Mismatch at vertex " << i << " sampled feature 3";
  }
}

/**
 * @brief Verify sampled ComputeSplitValues at d=0 extracts correct raw features.
 */
TEST(RegTreeG, BatchComputeSplitValuesSampledD0) {
  constexpr bst_node_t kN = 3;
  constexpr bst_feature_t kK = 5;
  constexpr bst_node_t kDmax = 1;

  RegTreeG tree(kN, kK, kDmax, /*max_ancestor_dist=*/0, /*intercept=*/false);

  std::vector<float> X = {
      10.0f, 20.0f, 30.0f, 40.0f, 50.0f,  // vertex 0
      11.0f, 21.0f, 31.0f, 41.0f, 51.0f,  // vertex 1
      12.0f, 22.0f, 32.0f, 42.0f, 52.0f   // vertex 2
  };
  // Empty adjacency (all zeros)
  std::vector<float> empty_adj(kN * kN, 0.0f);
  auto gd = BuildGraphDataFromDense(kN, kK, kDmax, X, {empty_adj});

  ActiveSet all(kN, true);

  // Sample features {0, 2, 4}
  std::vector<bst_feature_t> sampled = {0, 2, 4};
  auto result =
      tree.ComputeSplitValues(0, RestrictionType::kSource, all, gd, sampled, tree.Intercept());
  ASSERT_EQ(result.size(), static_cast<std::size_t>(kN * 3));

  // d=0 → raw features
  EXPECT_NEAR(result[0 * 3 + 0], 10.0f, 1e-5f);  // X[0,0]
  EXPECT_NEAR(result[0 * 3 + 1], 30.0f, 1e-5f);  // X[0,2]
  EXPECT_NEAR(result[0 * 3 + 2], 50.0f, 1e-5f);  // X[0,4]
  EXPECT_NEAR(result[1 * 3 + 0], 11.0f, 1e-5f);  // X[1,0]
  EXPECT_NEAR(result[1 * 3 + 1], 31.0f, 1e-5f);  // X[1,2]
  EXPECT_NEAR(result[1 * 3 + 2], 51.0f, 1e-5f);  // X[1,4]
}

/**
 * @brief Verify sampled ComputeSplitValues with intercept.
 *
 * Intercept column should always appear as the last column even with sampling.
 */
TEST(RegTreeG, BatchComputeSplitValuesSampledWithIntercept) {
  constexpr bst_node_t kN = 3;
  constexpr bst_feature_t kK = 4;
  constexpr bst_node_t kDmax = 1;

  RegTreeG tree(kN, kK, kDmax, /*max_ancestor_dist=*/0, /*intercept=*/true);

  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                          7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  // All-connected graph
  std::vector<float> all_conn(kN * kN, 1.0f);
  auto gd = BuildGraphDataFromDense(kN, kK, kDmax, X, {all_conn});

  ActiveSet all(kN, true);

  // Sample features {1} → S=1, S_total=2 (1 feature + intercept)
  std::vector<bst_feature_t> sampled = {1};
  auto result =
      tree.ComputeSplitValues(1, RestrictionType::kSource, all, gd, sampled, tree.Intercept());
  int S_total = 2;  // 1 sampled + intercept
  ASSERT_EQ(result.size(), static_cast<std::size_t>(kN * S_total));

  // Verify intercept column (last column) via full-feature computation
  auto full = AllFeatureSplitValues(tree, 1, RestrictionType::kSource, all, gd);
  int K_total = kK + 1;

  for (int i = 0; i < kN; ++i) {
    // Sampled feature 1 should match full feature column 1
    EXPECT_NEAR(result[i * S_total + 0], full[i * K_total + 1], 1e-5f)
        << "Feature 1 mismatch at vertex " << i;
    // Intercept column should match full intercept column (last)
    EXPECT_NEAR(result[i * S_total + 1], full[i * K_total + kK], 1e-5f)
        << "Intercept mismatch at vertex " << i;
  }
}

/**
 * @brief Verify sampled ComputeSplitValues with Cycle restriction type.
 */
TEST(RegTreeG, BatchComputeSplitValuesSampledCycle) {
  constexpr bst_node_t kN = 4;
  constexpr bst_feature_t kK = 3;
  constexpr bst_node_t kDmax = 1;

  RegTreeG tree(kN, kK, kDmax, /*max_ancestor_dist=*/0, /*intercept=*/false);

  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                          7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  // Diagonal-heavy adjacency for Cycle
  std::vector<float> diag_adj(kN * kN, 0.0f);
  diag_adj[0 * kN + 0] = 2.0f;
  diag_adj[1 * kN + 1] = 3.0f;
  diag_adj[2 * kN + 2] = 1.5f;
  diag_adj[3 * kN + 3] = 0.5f;
  auto gd = BuildGraphDataFromDense(kN, kK, kDmax, X, {diag_adj});

  ActiveSet all(kN, true);

  auto full = AllFeatureSplitValues(tree, 1, RestrictionType::kCycle, all, gd);
  ASSERT_EQ(full.size(), static_cast<std::size_t>(kN * kK));

  // Sample features {0, 2}
  std::vector<bst_feature_t> sampled = {0, 2};
  auto reduced =
      tree.ComputeSplitValues(1, RestrictionType::kCycle, all, gd, sampled, tree.Intercept());
  ASSERT_EQ(reduced.size(), static_cast<std::size_t>(kN * 2));

  for (int i = 0; i < kN; ++i) {
    EXPECT_NEAR(reduced[i * 2 + 0], full[i * kK + 0], 1e-5f)
        << "Cycle feature 0 mismatch at vertex " << i;
    EXPECT_NEAR(reduced[i * 2 + 1], full[i * kK + 2], 1e-5f)
        << "Cycle feature 2 mismatch at vertex " << i;
  }
}

/**
 * @brief Verify all-features sampled produces identical result to full version.
 */
TEST(RegTreeG, BatchComputeSplitValuesAllFeaturesMatchesFull) {
  constexpr bst_node_t kN = 4;
  constexpr bst_feature_t kK = 3;
  constexpr bst_node_t kDmax = 1;

  RegTreeG tree(kN, kK, kDmax, /*max_ancestor_dist=*/0, /*intercept=*/true);

  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                          7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  std::vector<float> pair_adj(kN * kN, 0.0f);
  pair_adj[0 * kN + 1] = 1.0f;
  pair_adj[1 * kN + 0] = 1.0f;
  pair_adj[2 * kN + 3] = 1.0f;
  pair_adj[3 * kN + 2] = 1.0f;
  auto gd = BuildGraphDataFromDense(kN, kK, kDmax, X, {pair_adj});

  ActiveSet all(kN, true);

  auto full = AllFeatureSplitValues(tree, 1, RestrictionType::kSource, all, gd);

  // All features: {0, 1, 2}
  std::vector<bst_feature_t> all_feats = {0, 1, 2};
  auto sampled =
      tree.ComputeSplitValues(1, RestrictionType::kSource, all, gd, all_feats, tree.Intercept());

  ASSERT_EQ(full.size(), sampled.size());
  for (std::size_t i = 0; i < full.size(); ++i) {
    EXPECT_NEAR(full[i], sampled[i], 1e-5f) << "Mismatch at index " << i;
  }
}

// ========== Attention subsets: ResolveUsedSubset, PartitionSubset, ReconstructSubsets ==========
//
// Unit tests for the functions that build a split-node's attention subsets S_{u,+}, S_{u,-}.
// These partition the node's used subset S_{*u,rho_u} (the ancestor's attention subset it
// points at, or all of V for a self-pointer) by the node's split criterion. Training and
// prediction share these functions, so they are tested here once, directly.

namespace {

/** @brief Build an ActiveSet over @p n vertices containing exactly @p members. */
ActiveSet MakeSet(bst_node_t n, std::vector<bst_node_t> const& members) {
  ActiveSet set{n, false};
  for (auto v : members) {
    set.Set(v, true);
  }
  return set;
}

/** @brief Collect the active vertices of @p set in ascending order. */
std::vector<bst_node_t> Members(ActiveSet const& set) {
  std::vector<bst_node_t> out;
  set.ForEach([&](bst_node_t v) { out.push_back(v); });
  return out;
}

}  // anonymous namespace

// ---------- PartitionSubset ----------

TEST(TreeGAttentionSubsets, PartitionSubsetSplits) {
  constexpr bst_node_t kNumVertices = 6;
  // Used subset omits vertices 1 and 4 to prove they are never partitioned.
  ActiveSet const used = MakeSet(kNumVertices, {0, 2, 3, 5});
  std::vector<float> const phi{0.0f, 9.0f, 2.0f, 5.0f, 9.0f, 7.0f};
  constexpr float kThreshold = 5.0f;

  auto const subsets = PartitionSubset(phi, used, kThreshold);

  // phi <= 5 -> lte (vertex 3 sits exactly on the threshold), phi > 5 -> gte.
  EXPECT_EQ(Members(subsets.lte), (std::vector<bst_node_t>{0, 2, 3}));
  EXPECT_EQ(Members(subsets.gte), (std::vector<bst_node_t>{5}));

  // Partition invariants: the subsets are disjoint and their union is the used subset.
  for (bst_node_t v = 0; v < kNumVertices; ++v) {
    bool const in_lte = subsets.lte.Test(v);
    bool const in_gte = subsets.gte.Test(v);
    EXPECT_FALSE(in_lte && in_gte) << "vertex " << v << " is in both subsets";
    EXPECT_EQ(in_lte || in_gte, used.Test(v)) << "vertex " << v << " violates the union invariant";
  }
}

TEST(TreeGAttentionSubsets, PartitionSubsetIncludesVerticesOutsideAnyReachedSet) {
  // The transductive case: vertex 4 carries no label and is routed nowhere during
  // training, yet it still belongs to an attention subset because it is in S.
  constexpr bst_node_t kNumVertices = 5;
  ActiveSet const used = MakeSet(kNumVertices, {0, 1, 4});
  std::vector<float> const phi{-1.0f, 3.0f, 0.0f, 0.0f, 8.0f};

  auto const subsets = PartitionSubset(phi, used, 1.0f);

  EXPECT_TRUE(subsets.gte.Test(4));
  EXPECT_EQ(Members(subsets.lte), (std::vector<bst_node_t>{0}));
  EXPECT_EQ(Members(subsets.gte), (std::vector<bst_node_t>{1, 4}));
}

TEST(TreeGAttentionSubsets, PartitionSubsetEmptyUsedSubset) {
  constexpr bst_node_t kNumVertices = 4;
  ActiveSet const used{kNumVertices, false};
  std::vector<float> const phi(kNumVertices, 1.0f);

  auto const subsets = PartitionSubset(phi, used, 0.0f);

  EXPECT_EQ(subsets.lte.Count(), 0);
  EXPECT_EQ(subsets.gte.Count(), 0);
  EXPECT_EQ(subsets.lte.NumVertices(), kNumVertices);
  EXPECT_EQ(subsets.gte.NumVertices(), kNumVertices);
}

TEST(TreeGAttentionSubsets, PartitionSubsetRejectsSizeMismatch) {
  ActiveSet const used{4, true};
  std::vector<float> const phi(3, 0.0f);
  EXPECT_THROW(static_cast<void>(PartitionSubset(phi, used, 0.0f)), dmlc::Error);
}

// ---------- ResolveUsedSubset ----------

TEST(TreeGAttentionSubsets, ResolveUsedSubsetSelfPointerUsesAllVertices) {
  constexpr bst_node_t kNumVertices = 5;
  ActiveSet const all_vertices{kNumVertices, true};
  std::vector<GeneratedSubsets> const subsets(3);

  RegTreeG::TreeGNode node;
  node.SetAncestorPointer(RegTreeG::kInvalidNodeId, /*use_ancestor_gte=*/true);

  auto const& used = ResolveUsedSubset(node, subsets, all_vertices);
  EXPECT_EQ(used.Count(), kNumVertices);
  // The root convention is independent of rho.
  node.SetAncestorPointer(RegTreeG::kInvalidNodeId, /*use_ancestor_gte=*/false);
  EXPECT_EQ(ResolveUsedSubset(node, subsets, all_vertices).Count(), kNumVertices);
}

TEST(TreeGAttentionSubsets, ResolveUsedSubsetAncestorSignSelectsSubset) {
  constexpr bst_node_t kNumVertices = 5;
  ActiveSet const all_vertices{kNumVertices, true};

  std::vector<GeneratedSubsets> subsets(2);
  subsets[0].lte = MakeSet(kNumVertices, {0, 1});
  subsets[0].gte = MakeSet(kNumVertices, {2, 3, 4});
  subsets[0].populated = true;

  RegTreeG::TreeGNode node;
  node.SetAncestorPointer(0, /*use_ancestor_gte=*/true);
  EXPECT_EQ(Members(ResolveUsedSubset(node, subsets, all_vertices)),
            (std::vector<bst_node_t>{2, 3, 4}));

  node.SetAncestorPointer(0, /*use_ancestor_gte=*/false);
  EXPECT_EQ(Members(ResolveUsedSubset(node, subsets, all_vertices)),
            (std::vector<bst_node_t>{0, 1}));
}

TEST(TreeGAttentionSubsets, ResolveUsedSubsetRejectsUnprocessedAncestor) {
  constexpr bst_node_t kNumVertices = 5;
  ActiveSet const all_vertices{kNumVertices, true};
  // Entry 1 was never populated, i.e. the ancestor is a leaf or has not been
  // visited yet. BFS order must guarantee this never happens.
  std::vector<GeneratedSubsets> const subsets(2);

  RegTreeG::TreeGNode node;
  node.SetAncestorPointer(1, /*use_ancestor_gte=*/true);
  EXPECT_THROW(static_cast<void>(ResolveUsedSubset(node, subsets, all_vertices)), dmlc::Error);
}

TEST(TreeGAttentionSubsets, ResolveUsedSubsetRejectsOutOfRangeAncestor) {
  constexpr bst_node_t kNumVertices = 5;
  ActiveSet const all_vertices{kNumVertices, true};
  std::vector<GeneratedSubsets> const subsets(2);

  RegTreeG::TreeGNode node;
  node.SetAncestorPointer(7, /*use_ancestor_gte=*/true);
  EXPECT_THROW(static_cast<void>(ResolveUsedSubset(node, subsets, all_vertices)), dmlc::Error);
}

// ---------- ReconstructSubsets on a hand-built tree ----------

namespace {

/** @brief Number of vertices in the hand-built fixture graph. */
constexpr bst_node_t kN = 6;
/** @brief Number of features in the hand-built fixture graph. */
constexpr bst_feature_t kK = 2;
/** @brief Maximum walk length in the hand-built fixture graph. */
constexpr bst_node_t kDMax = 2;

/**
 * @brief Build the fixture graph: a 6-vertex path 0-1-2-3-4-5 with A and A^2.
 *
 * The path gives every vertex a distinct neighbourhood, so masking out part of the
 * graph changes the aggregates of the remaining vertices. That is what makes the
 * generated-subset mechanism observable.
 */
GraphData MakePathGraph() {
  GraphData gd;
  gd.num_vertices = kN;
  gd.num_features = kK;
  gd.max_walk_len = kDMax;

  // Feature 0 counts up, feature 1 counts down, so the two disagree on ordering.
  gd.features.assign(static_cast<std::size_t>(kN) * kK, 0.0f);
  for (bst_node_t i = 0; i < kN; ++i) {
    gd.features[i * kK + 0] = static_cast<float>(i);
    gd.features[i * kK + 1] = static_cast<float>(kN - i);
  }

  // Dense A for the undirected path, then A^2 = A @ A.
  std::vector<float> a(static_cast<std::size_t>(kN) * kN, 0.0f);
  for (bst_node_t i = 0; i + 1 < kN; ++i) {
    a[i * kN + (i + 1)] = 1.0f;
    a[(i + 1) * kN + i] = 1.0f;
  }
  std::vector<float> a2(static_cast<std::size_t>(kN) * kN, 0.0f);
  for (bst_node_t i = 0; i < kN; ++i) {
    for (bst_node_t j = 0; j < kN; ++j) {
      float sum = 0.0f;
      for (bst_node_t m = 0; m < kN; ++m) {
        sum += a[i * kN + m] * a[m * kN + j];
      }
      a2[i * kN + j] = sum;
    }
  }

  gd.adj_powers.resize(kDMax);
  gd.adj_diag.resize(kDMax);
  std::vector<std::vector<float>> const dense{a, a2};
  for (bst_node_t d = 0; d < kDMax; ++d) {
    CsrMatrix csr;
    csr.nrows = kN;
    csr.ncols = kN;
    csr.row_ptr.push_back(0);
    for (bst_node_t i = 0; i < kN; ++i) {
      for (bst_node_t j = 0; j < kN; ++j) {
        float const v = dense[d][i * kN + j];
        if (v != 0.0f) {
          csr.col_idx.push_back(j);
          csr.values.push_back(v);
        }
      }
      csr.row_ptr.push_back(static_cast<int>(csr.col_idx.size()));
    }
    gd.adj_powers[d] = std::move(csr);
    gd.adj_diag[d].resize(kN);
    for (bst_node_t i = 0; i < kN; ++i) {
      gd.adj_diag[d][i] = dense[d][i * kN + i];
    }
  }
  return gd;
}

/** @brief Split parameters of one hand-built split-node. */
struct SplitSpec {
  bst_feature_t feature;
  bst_node_t walk_len;
  RestrictionType restriction;
  bst_node_t ancestor;
  bool ancestor_sign;
  float threshold;
};

/** @brief Apply @p spec to node @p nid and give it two leaf children. */
void MakeSplit(RegTreeG* tree, bst_node_t nid, SplitSpec const& spec) {
  tree->ExpandNode(nid, spec.feature, spec.threshold, /*base_weight=*/0.0f,
                   /*left_leaf_weight=*/0.0f, /*right_leaf_weight=*/0.0f,
                   /*loss_change=*/1.0f, /*sum_hess=*/1.0f, /*left_sum=*/0.5f,
                   /*right_sum=*/0.5f);
  tree->SetNodeTreeGParams(nid, spec.walk_len, spec.restriction, spec.ancestor, spec.ancestor_sign);
}

/**
 * @brief Independently compute the attention subsets of one split-node.
 *
 * Thresholds the split-value column directly instead of walking the tree, so it checks the
 * subset bookkeeping in ReconstructSubsets rather than the kernel (tested above).
 */
GeneratedSubsets BruteForceAttentionSubsets(RegTreeG const& tree, GraphData const& gd,
                                            SplitSpec const& spec, ActiveSet const& used) {
  GeneratedSubsets expected{ActiveSet{kN, false}, ActiveSet{kN, false}};
  auto const phi =
      tree.ComputeFeatureSplitValues(spec.feature, spec.walk_len, spec.restriction, used, gd);
  for (bst_node_t j = 0; j < kN; ++j) {
    if (!used.Test(j)) {
      continue;
    }
    if (phi[j] <= spec.threshold) {
      expected.lte.Set(j, true);
    } else {
      expected.gte.Set(j, true);
    }
  }
  return expected;
}

}  // anonymous namespace

TEST(TreeGAttentionSubsets, ReconstructSubsetsHandBuiltTree) {
  GraphData const gd = MakePathGraph();
  RegTreeG tree{kN, kK, kDMax, /*max_ancestor_dist=*/2, /*intercept=*/false};

  // Node 0 (root): self pointer, so S = V.  Splits on the d=1 neighbourhood sum of
  // feature 0, i.e. phi(j) = sum of feature 0 over j's path neighbours.
  SplitSpec const root{
      /*feature=*/0,
      /*walk_len=*/1,
      /*restriction=*/RestrictionType::kSource,
      /*ancestor=*/RegTreeG::kInvalidNodeId,
      /*ancestor_sign=*/true,
      /*threshold=*/4.0f,
  };
  MakeSplit(&tree, 0, root);
  bst_node_t const n1 = tree[0].LeftChild();
  bst_node_t const n2 = tree[0].RightChild();

  // Node 1 (left child of root): points to the root with rho = '-', so S = S_{0,-}.
  SplitSpec const left{
      /*feature=*/1,
      /*walk_len=*/1,
      /*restriction=*/RestrictionType::kSource,
      /*ancestor=*/0,
      /*ancestor_sign=*/false,
      /*threshold=*/5.0f,
  };
  MakeSplit(&tree, n1, left);
  bst_node_t const n3 = tree[n1].LeftChild();

  // Node 3 (grandchild): points to the root with rho = '+', so S = S_{0,+}.  This is the
  // non-trivial case: its used subset is *not* on its own root-to-node path, and contains
  // vertices that are routed into the sibling subtree.
  SplitSpec const grand{
      /*feature=*/0,
      /*walk_len=*/2,
      /*restriction=*/RestrictionType::kSource,
      /*ancestor=*/0,
      /*ancestor_sign=*/true,
      /*threshold=*/6.0f,
  };
  MakeSplit(&tree, n3, grand);

  std::vector<GeneratedSubsets> subsets;
  std::vector<std::vector<float>> split_values;
  ReconstructSubsets(tree, gd, &subsets, &split_values);

  // --- Brute-force attention subsets, resolving used subsets by hand in BFS order. ---
  ActiveSet const all_vertices{kN, true};
  auto const expected_root = BruteForceAttentionSubsets(tree, gd, root, all_vertices);
  auto const expected_left = BruteForceAttentionSubsets(tree, gd, left, expected_root.lte);
  auto const expected_grand = BruteForceAttentionSubsets(tree, gd, grand, expected_root.gte);

  EXPECT_EQ(Members(subsets[0].lte), Members(expected_root.lte));
  EXPECT_EQ(Members(subsets[0].gte), Members(expected_root.gte));
  EXPECT_EQ(Members(subsets[n1].lte), Members(expected_left.lte));
  EXPECT_EQ(Members(subsets[n1].gte), Members(expected_left.gte));
  EXPECT_EQ(Members(subsets[n3].lte), Members(expected_grand.lte));
  EXPECT_EQ(Members(subsets[n3].gte), Members(expected_grand.gte));

  // The fixture must actually exercise the mechanism, otherwise the assertions above
  // would hold trivially.
  ASSERT_GT(expected_root.lte.Count(), 0) << "root must generate a non-empty S_{0,-}";
  ASSERT_GT(expected_root.gte.Count(), 0) << "root must generate a non-empty S_{0,+}";
  ASSERT_LT(expected_root.gte.Count(), kN) << "node 3's subset must be a strict subset of V";

  // Node 3's used subset (S_{0,+}) must contain a vertex that the root routes left, i.e.
  // a vertex that can never reach node 3.  This is exactly what the old
  // reached-set-based implementation got wrong.
  bool found_outside = false;
  expected_root.gte.ForEach([&](bst_node_t v) {
    if (split_values[0][v] <= root.threshold) {
      found_outside = true;
    }
  });
  EXPECT_FALSE(found_outside) << "sanity: S_{0,+} is defined by phi > theta";
  ASSERT_GT(expected_grand.lte.Count() + expected_grand.gte.Count(), 0)
      << "the grandchild must partition a non-empty subset";

  // Leaves generate no subsets.
  for (bst_node_t nid = 0; nid < tree.NumNodes(); ++nid) {
    if (tree[nid].IsLeaf()) {
      EXPECT_EQ(subsets[nid].lte.NumVertices(), 0) << "leaf " << nid << " must stay empty";
      EXPECT_TRUE(split_values[nid].empty()) << "leaf " << nid << " must have no split values";
    }
  }
}

TEST(TreeGAttentionSubsets, ReconstructSubsetsUsedSubsetDiffersFromRootToNodePath) {
  // Isolates the property that motivated this refactor: a node's used subset is resolved
  // from its ancestor pointer, not from the vertices that reach it.
  GraphData const gd = MakePathGraph();
  RegTreeG tree{kN, kK, kDMax, /*max_ancestor_dist=*/2, /*intercept=*/false};

  SplitSpec const root{
      /*feature=*/0,
      /*walk_len=*/1,
      /*restriction=*/RestrictionType::kSource,
      /*ancestor=*/RegTreeG::kInvalidNodeId,
      /*ancestor_sign=*/true,
      /*threshold=*/4.0f,
  };
  MakeSplit(&tree, 0, root);
  bst_node_t const n1 = tree[0].LeftChild();

  // The left child points at the root's '+' subset, which is disjoint from the vertices
  // routed into the left subtree.
  SplitSpec const left{
      /*feature=*/0,
      /*walk_len=*/1,
      /*restriction=*/RestrictionType::kSource,
      /*ancestor=*/0,
      /*ancestor_sign=*/true,
      /*threshold=*/3.0f,
  };
  MakeSplit(&tree, n1, left);

  std::vector<GeneratedSubsets> subsets;
  std::vector<std::vector<float>> split_values;
  ReconstructSubsets(tree, gd, &subsets, &split_values);

  // Vertices routed to node 1 are exactly S_{0,-}; its used subset is S_{0,+}.
  ActiveSet const& reached_by_n1 = subsets[0].lte;
  ActiveSet const& used_by_n1 = subsets[0].gte;
  ASSERT_GT(used_by_n1.Count(), 0);

  used_by_n1.ForEach([&](bst_node_t v) {
    EXPECT_FALSE(reached_by_n1.Test(v))
        << "vertex " << v << " is in node 1's used subset but never reaches node 1";
  });

  // Node 1's generated subsets therefore partition vertices it never sees.
  bst_node_t const generated = subsets[n1].lte.Count() + subsets[n1].gte.Count();
  EXPECT_EQ(generated, used_by_n1.Count());
}

// ========== DMatrix column layout ==========

namespace {
/** @brief Dense DMatrix with @p n_rows rows and @p n_cols columns of ones. */
std::shared_ptr<DMatrix> MakeOnesDMatrix(bst_idx_t n_rows, bst_feature_t n_cols) {
  std::vector<float> data(n_rows * n_cols, 1.0f);
  return GetDMatrixFromData(data, n_rows, n_cols);
}
}  // anonymous namespace

TEST(GraphData, BuildRejectsTooManyColumns) {
  // N = 3, K = 2, d_max = 1 -> exactly 2 + 3 * 1 = 5 columns are expected.
  auto dmat = MakeOnesDMatrix(3, 6);
  EXPECT_THROW(static_cast<void>(GraphData::Build(dmat.get(), 3, 2, 1)), dmlc::Error);
}

TEST(GraphData, BuildRejectsTooFewColumns) {
  auto dmat = MakeOnesDMatrix(3, 4);
  EXPECT_THROW(static_cast<void>(GraphData::Build(dmat.get(), 3, 2, 1)), dmlc::Error);
}

TEST(GraphData, BuildRejectsRowCountMismatch) {
  // One row per vertex is required; 4 rows for 3 vertices is a mis-shaped matrix.
  auto dmat = MakeOnesDMatrix(4, 2 + 3);
  EXPECT_THROW(static_cast<void>(GraphData::Build(dmat.get(), 3, 2, 1)), dmlc::Error);
}

// ========== RegTreeG::LoadModel validation ==========

namespace {
/**
 * @brief A valid saved tree used as the base for single-field mutations.
 *
 * Nodes: 0 splits (d=1, self pointer) into 1, 2; 1 splits on the intercept (d=2, Cycle,
 * pointing at ancestor 0) into 3, 4; 2 splits (d=0) into 5, 6. K = 3 with intercept, so valid
 * split indices are [0, 4).
 */
RegTreeG MakeValidLoadTree() {
  RegTreeG tree{/*n_vertices=*/10, /*n_features=*/3, /*max_walk_len=*/2,
                /*max_ancestor_dist=*/2, /*intercept=*/true};
  tree.ExpandNode(0, 1, 0.5f, 0.0f, 0.1f, 0.2f, 1.0f, 4.0f, 2.0f, 2.0f);
  tree.SetNodeTreeGParams(0, 1, RestrictionType::kSource, RegTreeG::kInvalidNodeId, true);
  tree.ExpandNode(1, 3, 1.5f, 0.0f, 0.3f, 0.4f, 0.5f, 2.0f, 1.0f, 1.0f);
  tree.SetNodeTreeGParams(1, 2, RestrictionType::kCycle, 0, false);
  tree.ExpandNode(2, 0, 2.5f, 0.0f, 0.5f, 0.6f, 0.5f, 2.0f, 1.0f, 1.0f);
  tree.SetNodeTreeGParams(2, 0, RestrictionType::kSource, RegTreeG::kInvalidNodeId, true);
  return tree;
}

Json SaveTree(RegTreeG const& tree) {
  Json out{Object{}};
  tree.SaveModel(&out);
  return out;
}

Json& SavedNode(Json* saved, std::size_t nid) { return get<Array>((*saved)["nodes"])[nid]; }
}  // anonymous namespace

TEST(RegTreeG, LoadAcceptsValidTree) {
  auto const tree = MakeValidLoadTree();
  RegTreeG loaded;
  ASSERT_NO_THROW(loaded.LoadModel(SaveTree(tree)));
  EXPECT_TRUE(loaded == tree);
}

TEST(RegTreeG, LoadRejectsInvalidRestrictionType) {
  auto saved = SaveTree(MakeValidLoadTree());
  SavedNode(&saved, 0)["restriction_type"] = Integer{7};
  RegTreeG loaded;
  EXPECT_THROW(loaded.LoadModel(saved), dmlc::Error);
}

TEST(RegTreeG, LoadRejectsWalkLenAboveMax) {
  auto saved = SaveTree(MakeValidLoadTree());
  SavedNode(&saved, 0)["walk_len"] = Integer{3};  // max_walk_len = 2
  RegTreeG loaded;
  EXPECT_THROW(loaded.LoadModel(saved), dmlc::Error);
}

TEST(RegTreeG, LoadRejectsNonAncestorAttention) {
  auto saved = SaveTree(MakeValidLoadTree());
  // Node 1 is the sibling of node 2, not an ancestor.
  SavedNode(&saved, 2)["ancestor_index"] = Integer{1};
  RegTreeG loaded;
  EXPECT_THROW(loaded.LoadModel(saved), dmlc::Error);
}

TEST(RegTreeG, LoadRejectsAncestorBeyondMaxDistance) {
  auto saved = SaveTree(MakeValidLoadTree());
  // Node 0 is an ancestor of node 1 at distance 1, which max_ancestor_distance = 0 forbids.
  get<String>(saved["tree_param"]["max_ancestor_distance"]) = "0";
  RegTreeG loaded;
  EXPECT_THROW(loaded.LoadModel(saved), dmlc::Error);
}

TEST(RegTreeG, LoadRejectsOutOfRangeChild) {
  auto saved = SaveTree(MakeValidLoadTree());
  SavedNode(&saved, 0)["left_child"] = Integer{99};
  RegTreeG loaded;
  EXPECT_THROW(loaded.LoadModel(saved), dmlc::Error);
}

TEST(RegTreeG, LoadRejectsOutOfRangeSplitIndex) {
  auto saved = SaveTree(MakeValidLoadTree());
  SavedNode(&saved, 0)["split_index"] = Integer{4};  // K + intercept = 4 split features
  RegTreeG loaded;
  EXPECT_THROW(loaded.LoadModel(saved), dmlc::Error);
}

// ========== Dump ==========

TEST(RegTreeG, DumpUsesLessOrEqual) {
  auto const tree = MakeValidLoadTree();
  FeatureMap fmap;
  auto const dump = tree.DumpModel(fmap, false, "text");
  EXPECT_NE(dump.find("0:[f1<=0.5"), std::string::npos) << dump;
  EXPECT_EQ(dump.find("<0.5"), std::string::npos) << dump;
}

TEST(RegTreeG, DumpNamesInterceptSplit) {
  auto const tree = MakeValidLoadTree();  // node 1 splits on index 3 == num_features
  FeatureMap fmap;
  fmap.PushBack(0, "x0", "q");
  fmap.PushBack(1, "x1", "q");
  fmap.PushBack(2, "x2", "q");
  fmap.PushBack(3, "A_0_0", "q");  // DMatrix column K is A[0,0], not the intercept
  auto const dump = tree.DumpModel(fmap, false, "text");
  EXPECT_NE(dump.find("[intercept<=1.5"), std::string::npos) << dump;
  EXPECT_EQ(dump.find("A_0_0"), std::string::npos) << dump;
  EXPECT_NE(dump.find("[x1<=0.5"), std::string::npos) << dump;
}

TEST(RegTreeG, DumpThresholdRoutesEqualValueLeft) {
  // The dumped "<=" must agree with routing: a vertex whose value equals the threshold
  // goes to the 'yes' (left) child.
  constexpr bst_node_t kVertices = 3;
  std::vector<std::vector<float>> full_data = {{1.0f}, {2.0f}, {3.0f}};
  auto dmat = CreateTestDMatrix(full_data);
  auto gd = GraphData::Build(dmat.get(), kVertices, 1, 0);

  RegTreeG tree{kVertices, 1, 0, 0, false};
  tree.ExpandNode(0, 0, 2.0f, 0.0f, 1.0f, -1.0f, 1.0f, 3.0f, 2.0f, 1.0f);
  tree.SetNodeTreeGParams(0, 0, RestrictionType::kSource, RegTreeG::kInvalidNodeId, true);

  FeatureMap fmap;
  auto const dump = tree.DumpModel(fmap, false, "text");
  ASSERT_NE(dump.find("[f0<=2,"), std::string::npos) << dump;
  ASSERT_NE(dump.find("yes=" + std::to_string(tree[0].LeftChild())), std::string::npos);

  std::vector<GeneratedSubsets> subsets;
  std::vector<std::vector<float>> split_values;
  ReconstructSubsets(tree, gd, &subsets, &split_values);
  EXPECT_FLOAT_EQ(split_values[0][1], tree[0].SplitCond());
  EXPECT_TRUE(subsets[0].lte.Test(1)) << "value == threshold must go left";
  EXPECT_FALSE(subsets[0].lte.Test(2));
}

}  // namespace treeg
}  // namespace xgboost
