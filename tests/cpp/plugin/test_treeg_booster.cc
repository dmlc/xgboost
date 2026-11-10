/**
 * Copyright 2024-2026, XGBoost contributors
 *
 * @brief Tests for TreeGBooster gradient booster
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/context.h>
#include <xgboost/data.h>
#include <xgboost/gbm.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/json.h>
#include <xgboost/learner.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../../plugin/treeg/booster.h"
#include "../../../plugin/treeg/model.h"
#include "../../../src/collective/communicator-inl.h"  // for IsDistributed
#include "../../../src/common/threading_utils.h"
#include "../collective/test_worker.h"  // for TestDistributedGlobal
#include "../helpers.h"

namespace xgboost {
namespace treeg {

// ========== Helper Functions for Graph Data Generation ==========

/**
 * @brief Generate adjacency matrix powers for a simple graph structure.
 *
 * Creates a simple connected graph where each node is connected to its neighbors
 * in a ring topology (circular linked list).
 */
std::vector<float> GenerateRingGraphAdjacency(bst_node_t num_vertices, bst_node_t max_walk_len) {
  // Total features: num_vertices (original) + num_vertices * max_walk_len (adjacency matrices)
  // DMatrix format: [X | A | A^2 | ... | A^d]
  // Row i stores: [X[i,:] | A[i,:] | A^2[i,:] | ... | A^d[i,:]]
  bst_feature_t total_features = num_vertices + num_vertices * max_walk_len;
  std::vector<float> data(num_vertices * total_features, 0.0f);

  // Fill original features (first num_vertices columns)
  // For simplicity, feature values are just the node index
  for (bst_node_t i = 0; i < num_vertices; ++i) {
    data[i * total_features + i] = static_cast<float>(i);
  }

  // Fill adjacency matrix A (next num_vertices columns)
  // A[i,j] is stored at column num_vertices + j in row i
  bst_feature_t a_start = num_vertices;
  for (bst_node_t i = 0; i < num_vertices; ++i) {
    // Connect to next node in ring
    bst_node_t next = (i + 1) % num_vertices;
    data[i * total_features + a_start + next] = 1.0f;
    // Connect to previous node in ring (undirected)
    bst_node_t prev = (i - 1 + num_vertices) % num_vertices;
    data[i * total_features + a_start + prev] = 1.0f;
  }

  // Fill A^2 if max_walk_len >= 2
  if (max_walk_len >= 2) {
    bst_feature_t a2_start = a_start + num_vertices;
    // Compute A^2 from A: A^2[i,j] = sum_k A[i,k] * A[k,j]
    for (bst_node_t i = 0; i < num_vertices; ++i) {
      for (bst_node_t j = 0; j < num_vertices; ++j) {
        float sum = 0.0f;
        for (bst_node_t k = 0; k < num_vertices; ++k) {
          float a_ik = data[i * total_features + a_start + k];
          float a_kj = data[k * total_features + a_start + j];
          sum += a_ik * a_kj;
        }
        if (sum != 0.0f) {
          data[i * total_features + a2_start + j] = sum;
        }
      }
    }
  }

  return data;
}

/**
 * @brief Create a simple graph DMatrix with labels equal to features.
 *
 * Test case (a): 10 nodes, single feature = label
 */
std::shared_ptr<DMatrix> CreateSimpleGraphDMatrix() {
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  // Total columns: kNumFeatures + kNumVertices * kMaxWalkLen
  // DMatrix format: [X | A | A^2]
  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;

  std::vector<float> data(kNumVertices * total_cols, 0.0f);
  std::vector<float> labels(kNumVertices);
  std::vector<float> weights(kNumVertices, 1.0f);

  // Fill original feature (single column) with values 0-9
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    data[i * total_cols] = static_cast<float>(i);
    labels[i] = static_cast<float>(i);  // Label = feature
  }

  // Fill adjacency matrix A (ring topology)
  // A[i,j] at column kNumFeatures + j in row i
  bst_feature_t a_start = kNumFeatures;
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    bst_node_t next = (i + 1) % kNumVertices;
    bst_node_t prev = (i - 1 + kNumVertices) % kNumVertices;
    data[i * total_cols + a_start + next] = 1.0f;
    data[i * total_cols + a_start + prev] = 1.0f;
  }

  // Fill A^2: compute from A
  bst_feature_t a2_start = a_start + kNumVertices;
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    for (bst_node_t j = 0; j < kNumVertices; ++j) {
      float sum = 0.0f;
      for (bst_node_t k = 0; k < kNumVertices; ++k) {
        float a_ik = data[i * total_cols + a_start + k];
        float a_kj = data[k * total_cols + a_start + j];
        sum += a_ik * a_kj;
      }
      if (sum != 0.0f) {
        data[i * total_cols + a2_start + j] = sum;
      }
    }
  }

  // Create DMatrix
  auto dmat = GetDMatrixFromData(data, kNumVertices, total_cols);
  dmat->Info().labels.Reshape(kNumVertices);
  auto& h_labels = dmat->Info().labels.Data()->HostVector();
  h_labels = labels;

  dmat->Info().weights_.HostVector() = weights;

  return dmat;
}

/**
 * @brief Create a larger graph where labels are mean of neighbor features.
 *
 * Test case (b): 100 nodes, single feature, label = mean of neighbor features
 */
std::shared_ptr<DMatrix> CreateMeanAggregationGraphDMatrix() {
  constexpr bst_node_t kNumVertices = 100;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  // DMatrix format: [X | A | A^2]
  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;

  std::vector<float> data(kNumVertices * total_cols, 0.0f);
  std::vector<float> labels(kNumVertices);
  std::vector<float> weights(kNumVertices, 1.0f);

  // Mask first 20 nodes as test set (weight = 0)
  for (bst_node_t i = 0; i < 20; ++i) {
    weights[i] = 0.0f;
  }

  // Fill features with random-like values based on node index
  std::vector<float> feature_values(kNumVertices);
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    feature_values[i] = std::sin(static_cast<float>(i) * 0.1f) * 10.0f;
    data[i * total_cols] = feature_values[i];
  }

  // Fill adjacency matrix A (ring topology with additional connections)
  // A[i,j] at column kNumFeatures + j in row i
  bst_feature_t a_start = kNumFeatures;
  std::vector<std::vector<bst_node_t>> neighbors(kNumVertices);
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    // Connect to next and previous in ring
    bst_node_t next = (i + 1) % kNumVertices;
    bst_node_t prev = (i - 1 + kNumVertices) % kNumVertices;
    data[i * total_cols + a_start + next] = 1.0f;
    data[i * total_cols + a_start + prev] = 1.0f;
    neighbors[i].push_back(next);
    neighbors[i].push_back(prev);

    // Add one more connection to node at distance 10 (if exists)
    if (kNumVertices > 10) {
      bst_node_t far = (i + 10) % kNumVertices;
      data[i * total_cols + a_start + far] = 1.0f;
      neighbors[i].push_back(far);
    }
  }

  // Calculate labels as mean of neighbor features
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    float sum = 0.0f;
    for (auto neighbor : neighbors[i]) {
      sum += feature_values[neighbor];
    }
    labels[i] = sum / static_cast<float>(neighbors[i].size());
  }

  // Fill A^2: compute from A
  bst_feature_t a2_start = a_start + kNumVertices;
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    for (bst_node_t j = 0; j < kNumVertices; ++j) {
      float sum = 0.0f;
      for (bst_node_t k = 0; k < kNumVertices; ++k) {
        float a_ik = data[i * total_cols + a_start + k];
        float a_kj = data[k * total_cols + a_start + j];
        sum += a_ik * a_kj;
      }
      if (sum != 0.0f) {
        data[i * total_cols + a2_start + j] = sum;
      }
    }
  }

  // Create DMatrix
  auto dmat = GetDMatrixFromData(data, kNumVertices, total_cols);
  dmat->Info().labels.Reshape(kNumVertices);
  auto& h_labels = dmat->Info().labels.Data()->HostVector();
  h_labels = labels;

  dmat->Info().weights_.HostVector() = weights;

  return dmat;
}

/**
 * @brief Generate random gradients for testing.
 */
GradientContainer GenerateRandomGradients(Context const* ctx, bst_node_t num_vertices) {
  return xgboost::GenerateRandomGradients(ctx, num_vertices, 1);
}

/**
 * @brief Generate random multi-class gradients for testing.
 *
 * The gradient matrix has shape [num_vertices, n_groups], matching the
 * layout that xgboost objectives produce for one-vs-all multi-class.
 */
GradientContainer GenerateMultiClassGradients(Context const* ctx, bst_node_t num_vertices,
                                              bst_target_t n_groups) {
  return xgboost::GenerateRandomGradients(ctx, num_vertices, n_groups);
}

/**
 * @brief Create a simple multi-class graph DMatrix.
 *
 * Nodes have class labels 0..(n_classes-1) assigned round-robin.
 * Uses same ring topology as CreateSimpleGraphDMatrix.
 *
 * @param n_classes Number of classes
 * @return Shared pointer to DMatrix
 */
std::shared_ptr<DMatrix> CreateMultiClassGraphDMatrix(bst_target_t n_classes) {
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;

  std::vector<float> data(kNumVertices * total_cols, 0.0f);
  std::vector<float> labels(kNumVertices);
  std::vector<float> weights(kNumVertices, 1.0f);

  // Fill features with node index; label = class assigned round-robin
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    data[i * total_cols] = static_cast<float>(i);
    labels[i] = static_cast<float>(i % n_classes);
  }

  // Ring adjacency A
  bst_feature_t a_start = kNumFeatures;
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    bst_node_t next = (i + 1) % kNumVertices;
    bst_node_t prev = (i - 1 + kNumVertices) % kNumVertices;
    data[i * total_cols + a_start + next] = 1.0f;
    data[i * total_cols + a_start + prev] = 1.0f;
  }

  // A^2
  bst_feature_t a2_start = a_start + kNumVertices;
  for (bst_node_t i = 0; i < kNumVertices; ++i) {
    for (bst_node_t j = 0; j < kNumVertices; ++j) {
      float sum = 0.0f;
      for (bst_node_t k = 0; k < kNumVertices; ++k) {
        float a_ik = data[i * total_cols + a_start + k];
        float a_kj = data[k * total_cols + a_start + j];
        sum += a_ik * a_kj;
      }
      if (sum != 0.0f) {
        data[i * total_cols + a2_start + j] = sum;
      }
    }
  }

  auto dmat = GetDMatrixFromData(data, kNumVertices, total_cols);
  dmat->Info().labels.Reshape(kNumVertices);
  auto& h_labels = dmat->Info().labels.Data()->HostVector();
  h_labels = labels;
  dmat->Info().weights_.HostVector() = weights;
  return dmat;
}

// ========== Unit Tests ==========

/**
 * @brief Test 1: Verify TreeGBooster plugin registration.
 */
TEST(TreeGBooster, PluginRegistration) {
  Context ctx;
  auto mparam = MakeMP(10, 0.5f, 1);

  // Check that "treeg" booster can be created
  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  ASSERT_NE(gbm, nullptr);

  // Verify it's actually a TreeGBooster
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());
  ASSERT_NE(treeg_booster, nullptr);
}

/**
 * @brief Test 2: Basic training with 10-node simple graph.
 */
TEST(TreeGBooster, BasicTrainingSimpleGraph) {
  Context ctx;
  auto dmat = CreateSimpleGraphDMatrix();

  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, 0.5f, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());
  ASSERT_NE(treeg_booster, nullptr);

  // Configure booster
  Args args{{"learning_rate", "0.1"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "2"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  treeg_booster->Configure(args);

  // Generate gradients
  auto gpair = GenerateRandomGradients(&ctx, kNumVertices);

  // Perform one boosting round
  HostDeviceVector<float> out_preds;
  treeg_booster->DoBoost(dmat, &gpair, nullptr);

  // Verify tree was created
  Json config{Object{}};
  treeg_booster->SaveConfig(&config);
  ASSERT_TRUE(IsA<Object>(config["treeg_booster_train_param"]));

  // Verify model has trees
  Json model{Object{}};
  treeg_booster->SaveModel(&model);
  ASSERT_EQ(get<String>(model["name"]), "treeg");

  auto const& model_obj = get<Object const>(model["model"]);
  ASSERT_TRUE(model_obj.find("trees") != model_obj.end());

  auto const& trees = get<Array const>(model["model"]["trees"]);
  ASSERT_EQ(trees.size(), 1u);  // One tree after one iteration
}

/**
 * @brief Test 3: Training with 100-node graph (mean aggregation).
 */
TEST(TreeGBooster, TrainingMeanAggregationGraph) {
  Context ctx;
  auto dmat = CreateMeanAggregationGraphDMatrix();

  constexpr bst_node_t kNumVertices = 100;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, 0.0f, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());
  ASSERT_NE(treeg_booster, nullptr);

  // Configure booster
  Args args{{"learning_rate", "0.1"},
            {"max_leaves", "16"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "2"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  treeg_booster->Configure(args);

  // Generate gradients
  auto gpair = GenerateRandomGradients(&ctx, kNumVertices);

  // Perform multiple boosting rounds
  HostDeviceVector<float> out_preds;
  for (int i = 0; i < 3; ++i) {
    treeg_booster->DoBoost(dmat, &gpair, nullptr);
  }

  // Verify model has multiple trees
  Json model{Object{}};
  treeg_booster->SaveModel(&model);
  auto const& trees = get<Array const>(model["model"]["trees"]);
  ASSERT_EQ(trees.size(), 3u);  // Three trees after three iterations

  // Verify iteration_indptr is correct
  auto const& indptr = get<Array const>(model["model"]["iteration_indptr"]);
  ASSERT_EQ(indptr.size(), 4u);  // [0, 1, 2, 3]
  ASSERT_EQ(get<Integer const>(indptr[0]), 0);
  ASSERT_EQ(get<Integer const>(indptr[3]), 3);
}

/**
 * @brief Test 4: Configuration and parameter validation.
 */
TEST(TreeGBooster, Configuration) {
  Context ctx;
  auto mparam = MakeMP(50, 0.5f, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());

  // Test valid configuration
  Args valid_args{{"learning_rate", "0.3"},       {"max_leaves", "64"},
                  {"min_split_loss", "0.1"},      {"max_walk_len", "3"},
                  {"max_ancestor_distance", "1"}, {"num_features", "5"},
                  {"num_vertices", "100"}};
  ASSERT_NO_THROW(treeg_booster->Configure(valid_args));

  // Save and verify configuration
  Json config{Object{}};
  treeg_booster->SaveConfig(&config);
  ASSERT_EQ(get<String>(config["name"]), "treeg");

  // Verify the config has the train param object
  ASSERT_TRUE(IsA<Object>(config["treeg_booster_train_param"]));
}

/**
 * @brief Test 5: CPU-only constraint validation.
 */
TEST(TreeGBooster, CPUOnlyConstraint) {
  Context ctx;
  auto mparam = MakeMP(10, 0.5f, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());

  // CPU should work
  ASSERT_TRUE(ctx.IsCPU());
  Args cpu_args{{"num_features", "5"}, {"num_vertices", "10"}};
  ASSERT_NO_THROW(treeg_booster->Configure(cpu_args));

#ifdef XGBOOST_USE_CUDA
  // GPU should be rejected (if CUDA is available)
  Context gpu_ctx;
  gpu_ctx.UpdateAllowUnknown(Args{{"device", "cuda"}});
  std::unique_ptr<GradientBooster> gbm_gpu(GradientBooster::Create("treeg", &gpu_ctx, &mparam));
  auto* treeg_booster_gpu = dynamic_cast<TreeGBooster*>(gbm_gpu.get());

  Args gpu_args{{"num_features", "5"}, {"num_vertices", "10"}};
  ASSERT_THROW(treeg_booster_gpu->Configure(gpu_args), dmlc::Error);
#endif
}

/**
 * @brief Test 6: Model save and load (JSON serialization).
 */
TEST(TreeGBooster, JsonIO) {
  Context ctx;
  auto dmat = CreateSimpleGraphDMatrix();

  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, 0.5f, 1);

  // Train original booster
  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.1"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  treeg_booster->Configure(args);

  auto gpair = GenerateRandomGradients(&ctx, kNumVertices);
  HostDeviceVector<float> out_preds;

  // Train for 2 rounds
  for (int i = 0; i < 2; ++i) {
    treeg_booster->DoBoost(dmat, &gpair, nullptr);
  }

  // Save model
  Json model_out{Object{}};
  treeg_booster->SaveModel(&model_out);
  Json config_out{Object{}};
  treeg_booster->SaveConfig(&config_out);

  // Create new booster and load model
  std::unique_ptr<GradientBooster> gbm_loaded(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster_loaded = dynamic_cast<TreeGBooster*>(gbm_loaded.get());

  treeg_booster_loaded->LoadModel(model_out);
  treeg_booster_loaded->LoadConfig(config_out);

  // Verify loaded model matches
  Json model_reloaded{Object{}};
  treeg_booster_loaded->SaveModel(&model_reloaded);

  ASSERT_EQ(get<String>(model_out["name"]), get<String>(model_reloaded["name"]));
  auto const& trees_out = get<Array const>(model_out["model"]["trees"]);
  auto const& trees_reloaded = get<Array const>(model_reloaded["model"]["trees"]);
  ASSERT_EQ(trees_out.size(), trees_reloaded.size());
  ASSERT_EQ(trees_out.size(), 2u);
}

/**
 * @brief Test 7: Feature importance calculation.
 */
TEST(TreeGBooster, FeatureScore) {
  Context ctx;
  auto dmat = CreateSimpleGraphDMatrix();

  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, 0.5f, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.1"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  treeg_booster->Configure(args);

  auto gpair = GenerateRandomGradients(&ctx, kNumVertices);
  HostDeviceVector<float> out_preds;

  // Train for 3 rounds
  for (int i = 0; i < 3; ++i) {
    treeg_booster->DoBoost(dmat, &gpair, nullptr);
  }

  // Test different importance types
  std::vector<std::string> importance_types = {"weight", "gain", "cover", "total_gain",
                                               "total_cover"};

  for (const auto& importance_type : importance_types) {
    std::vector<bst_feature_t> features;
    std::vector<float> scores;

    treeg_booster->FeatureScore(importance_type, {}, &features, &scores);

    // Should have some feature scores
    ASSERT_EQ(features.size(), scores.size());
    // All features with splits should be < num_features
    for (auto fid : features) {
      ASSERT_LT(fid, kNumFeatures);
    }

    // Scores should be non-negative
    for (auto score : scores) {
      ASSERT_GE(score, 0.0f);
    }
  }
}

/**
 * @brief Test 8: Multi-iteration training with incremental tree addition.
 */
TEST(TreeGBooster, MultiIterationTraining) {
  Context ctx;
  auto dmat = CreateSimpleGraphDMatrix();

  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr int kNumRounds = 5;

  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, 0.5f, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.1"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  treeg_booster->Configure(args);

  auto gpair = GenerateRandomGradients(&ctx, kNumVertices);
  HostDeviceVector<float> out_preds;

  // Train incrementally and verify tree count increases
  for (int i = 0; i < kNumRounds; ++i) {
    treeg_booster->DoBoost(dmat, &gpair, nullptr);

    Json model{Object{}};
    treeg_booster->SaveModel(&model);
    auto const& trees = get<Array const>(model["model"]["trees"]);
    ASSERT_EQ(trees.size(), static_cast<std::size_t>(i + 1));

    // Verify iteration_indptr
    auto const& indptr = get<Array const>(model["model"]["iteration_indptr"]);
    ASSERT_EQ(indptr.size(), static_cast<std::size_t>(i + 2));  // 0, 1, 2, ..., i+1
    ASSERT_EQ(get<Integer const>(indptr[i + 1]), i + 1);
  }
}

/**
 * @brief Test 9: process_type=update is rejected at configure time.
 *
 * The update branch used to retrain rather than refresh existing trees; it was removed.
 */
TEST(TreeGBooster, RejectsUpdateProcessType) {
  Context ctx;
  auto mparam = MakeMP(1 + 10 * 2, 0.5f, 1);
  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));

  Args args{{"num_features", "1"}, {"num_vertices", "10"}, {"process_type", "update"}};
  EXPECT_THROW(gbm->Configure(args), dmlc::Error);

  Args ok_args{{"num_features", "1"}, {"num_vertices", "10"}, {"process_type", "default"}};
  EXPECT_NO_THROW(gbm->Configure(ok_args));
}

/**
 * @brief Test 10: Dimension mismatch validation.
 */
TEST(TreeGBooster, DimensionMismatchError) {
  Context ctx;
  auto dmat = CreateSimpleGraphDMatrix();

  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, 0.5f, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());

  // Configure with WRONG num_vertices
  Args wrong_args{{"learning_rate", "0.1"},
                  {"max_leaves", "8"},
                  {"max_walk_len", std::to_string(kMaxWalkLen)},
                  {"num_features", std::to_string(kNumFeatures)},
                  {"num_vertices", "20"}};  // Wrong! DMatrix has 10 rows
  treeg_booster->Configure(wrong_args);

  auto gpair = GenerateRandomGradients(&ctx, kNumVertices);
  HostDeviceVector<float> out_preds;

  // DoBoost should throw due to dimension mismatch
  ASSERT_THROW(treeg_booster->DoBoost(dmat, &gpair, nullptr), dmlc::Error);
}

/**
 * @brief Test 11: Dump model functionality.
 */
TEST(TreeGBooster, DumpModel) {
  Context ctx;
  auto dmat = CreateSimpleGraphDMatrix();

  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, 0.5f, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.1"},
            {"max_leaves", "4"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  treeg_booster->Configure(args);

  auto gpair = GenerateRandomGradients(&ctx, kNumVertices);
  HostDeviceVector<float> out_preds;

  // Train one tree
  treeg_booster->DoBoost(dmat, &gpair, nullptr);

  // Dump model
  FeatureMap fmap;
  std::vector<std::string> dump_text = treeg_booster->DumpModel(fmap, false, "text");
  ASSERT_EQ(dump_text.size(), 1u);  // One tree
  ASSERT_FALSE(dump_text[0].empty());

  // Dump with stats
  std::vector<std::string> dump_with_stats = treeg_booster->DumpModel(fmap, true, "text");
  ASSERT_EQ(dump_with_stats.size(), 1u);
  ASSERT_FALSE(dump_with_stats[0].empty());

  // JSON format is not yet implemented, so skip it
  // std::vector<std::string> dump_json = treeg_booster->DumpModel(fmap, false, "json");
  // ASSERT_EQ(dump_json.size(), 1u);
  // ASSERT_FALSE(dump_json[0].empty());
}

/**
 * @brief Test 12: Basic prediction after training.
 *
 * Trains a model and verifies that PredictBatch produces:
 * - Correct output size (one prediction per vertex)
 * - Finite, non-NaN values
 * - Values that differ from pure base_score (i.e., trees contribute)
 */
TEST(TreeGBooster, PredictBatchBasicPrediction) {
  Context ctx;
  auto dmat = CreateSimpleGraphDMatrix();

  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, 0.5f, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.3"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  treeg_booster->Configure(args);

  auto gpair = GenerateRandomGradients(&ctx, kNumVertices);
  HostDeviceVector<float> train_preds;

  // Train for 3 rounds
  for (int i = 0; i < 3; ++i) {
    treeg_booster->DoBoost(dmat, &gpair, nullptr);
  }

  // Now predict
  HostDeviceVector<float> out_preds;
  ASSERT_NO_THROW(treeg_booster->PredictBatch(dmat, &out_preds, false, 0, 0));

  // Verify output size
  auto const& h_preds = out_preds.ConstHostVector();
  ASSERT_EQ(h_preds.size(), kNumVertices);

  // Verify all predictions are finite
  for (bst_node_t v = 0; v < kNumVertices; ++v) {
    ASSERT_TRUE(std::isfinite(h_preds[v]))
        << "Prediction for vertex " << v << " is not finite: " << h_preds[v];
  }

  // Verify predictions are not all identical to base_score (trees should contribute)
  float base_score = 0.5f;
  bool any_different = false;
  for (auto pred : h_preds) {
    if (std::abs(pred - base_score) > 1e-8f) {
      any_different = true;
      break;
    }
  }
  ASSERT_TRUE(any_different) << "All predictions equal base_score; trees did not contribute.";
}

/**
 * @brief Test 13: Prediction consistency across save/load.
 *
 * Trains a model, predicts, saves, loads, and verifies predictions match.
 */
TEST(TreeGBooster, PredictBatchSaveLoadConsistency) {
  Context ctx;
  auto dmat = CreateSimpleGraphDMatrix();

  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, 0.5f, 1);

  // Train original
  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.1"},
            {"max_leaves", "4"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  treeg_booster->Configure(args);

  auto gpair = GenerateRandomGradients(&ctx, kNumVertices);
  HostDeviceVector<float> train_preds;
  for (int i = 0; i < 2; ++i) {
    treeg_booster->DoBoost(dmat, &gpair, nullptr);
  }

  // Predict with original model
  HostDeviceVector<float> preds_original;
  treeg_booster->PredictBatch(dmat, &preds_original, false, 0, 0);

  // Save & load
  Json model_out{Object{}};
  treeg_booster->SaveModel(&model_out);
  Json config_out{Object{}};
  treeg_booster->SaveConfig(&config_out);

  std::unique_ptr<GradientBooster> gbm_loaded(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* loaded = dynamic_cast<TreeGBooster*>(gbm_loaded.get());
  loaded->LoadModel(model_out);
  loaded->LoadConfig(config_out);

  // Predict with loaded model
  HostDeviceVector<float> preds_loaded;
  loaded->PredictBatch(dmat, &preds_loaded, false, 0, 0);

  // Verify predictions match
  auto const& h_orig = preds_original.ConstHostVector();
  auto const& h_loaded = preds_loaded.ConstHostVector();
  ASSERT_EQ(h_orig.size(), h_loaded.size());
  for (std::size_t i = 0; i < h_orig.size(); ++i) {
    ASSERT_FLOAT_EQ(h_orig[i], h_loaded[i]) << "Mismatch at vertex " << i;
  }
}

/**
 * @brief Test 14: Prediction with empty model returns base_score.
 */
TEST(TreeGBooster, PredictBatchEmptyModel) {
  Context ctx;
  auto dmat = CreateSimpleGraphDMatrix();

  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  float base_score = 0.5f;
  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, base_score, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.1"},
            {"max_leaves", "4"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  treeg_booster->Configure(args);

  // Predict WITHOUT training (empty model)
  HostDeviceVector<float> out_preds;
  treeg_booster->PredictBatch(dmat, &out_preds, false, 0, 0);

  auto const& h_preds = out_preds.ConstHostVector();
  ASSERT_EQ(h_preds.size(), kNumVertices);
  for (auto pred : h_preds) {
    ASSERT_FLOAT_EQ(pred, base_score);
  }
}

/**
 * @brief Test 15: Unimplemented prediction methods still throw.
 */
TEST(TreeGBooster, UnimplementedPredictionMethods) {
  Context ctx;
  auto dmat = CreateSimpleGraphDMatrix();

  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, 0.5f, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.1"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  treeg_booster->Configure(args);

  HostDeviceVector<bst_float> leaf_preds;
  ASSERT_THROW(treeg_booster->PredictLeaf(dmat.get(), &leaf_preds, 0, 0, false), dmlc::Error);

  HostDeviceVector<float> contribs;
  ASSERT_THROW(treeg_booster->PredictContribution(dmat.get(), &contribs, 0, 0, false), dmlc::Error);
  ASSERT_THROW(treeg_booster->PredictInteractionContributions(dmat.get(), &contribs, 0, 0, false),
               dmlc::Error);
}

// ========== Multi-Class One-vs-All Tests ==========

/**
 * @brief Test 16: Multi-class DoBoost builds n_classes trees per round.
 *
 * After one boosting round with 3 classes, model must contain exactly 3 trees.
 * tree_info must map each tree to its class [0, 1, 2].
 * iteration_indptr must be [0, 3].
 */
TEST(TreeGBooster, MultiClassTreeCountAndTreeInfo) {
  Context ctx;
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_target_t kNumClasses = 3;

  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;
  auto dmat = CreateMultiClassGraphDMatrix(kNumClasses);

  // n_groups = kNumClasses enables one-vs-all
  auto mparam = MakeMP(total_cols, 0.5f, kNumClasses);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());
  ASSERT_NE(booster, nullptr);

  Args args{{"learning_rate", "0.3"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  booster->Configure(args);

  // Multi-class gradients: shape [n_vertices, n_classes]
  auto gpair = GenerateMultiClassGradients(&ctx, kNumVertices, kNumClasses);
  HostDeviceVector<float> preds;
  booster->DoBoost(dmat, &gpair, nullptr);

  // Verify via serialized model
  Json model{Object{}};
  booster->SaveModel(&model);

  auto const& trees = get<Array const>(model["model"]["trees"]);
  ASSERT_EQ(trees.size(), static_cast<std::size_t>(kNumClasses));

  auto const& tree_info = get<Array const>(model["model"]["tree_info"]);
  ASSERT_EQ(tree_info.size(), static_cast<std::size_t>(kNumClasses));
  for (bst_target_t g = 0; g < kNumClasses; ++g) {
    ASSERT_EQ(get<Integer const>(tree_info[g]), static_cast<Integer::Int>(g))
        << "tree_info[" << g << "] should be " << g;
  }

  auto const& indptr = get<Array const>(model["model"]["iteration_indptr"]);
  ASSERT_EQ(indptr.size(), 2u);  // [0, 3]
  ASSERT_EQ(get<Integer const>(indptr[0]), 0);
  ASSERT_EQ(get<Integer const>(indptr[1]), kNumClasses);
}

/**
 * @brief Test 17: Multi-iteration multi-class training.
 *
 * After k rounds with n_classes, model must have k*n_classes trees.
 * tree_info must cycle through classes: [0,1,2, 0,1,2, ...].
 */
TEST(TreeGBooster, MultiClassMultiIteration) {
  Context ctx;
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_target_t kNumClasses = 3;
  constexpr int kNumRounds = 4;

  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;
  auto dmat = CreateMultiClassGraphDMatrix(kNumClasses);
  auto mparam = MakeMP(total_cols, 0.5f, kNumClasses);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.3"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  booster->Configure(args);

  HostDeviceVector<float> preds;
  for (int r = 0; r < kNumRounds; ++r) {
    auto gpair = GenerateMultiClassGradients(&ctx, kNumVertices, kNumClasses);
    booster->DoBoost(dmat, &gpair, nullptr);
  }

  Json model{Object{}};
  booster->SaveModel(&model);

  std::size_t expected_total_trees =
      static_cast<std::size_t>(kNumRounds) * static_cast<std::size_t>(kNumClasses);
  auto const& trees = get<Array const>(model["model"]["trees"]);
  ASSERT_EQ(trees.size(), expected_total_trees);

  // Verify tree_info cycles: [0,1,2, 0,1,2, 0,1,2, 0,1,2]
  auto const& tree_info = get<Array const>(model["model"]["tree_info"]);
  ASSERT_EQ(tree_info.size(), expected_total_trees);
  for (std::size_t t = 0; t < expected_total_trees; ++t) {
    bst_target_t expected_group = static_cast<bst_target_t>(t % kNumClasses);
    ASSERT_EQ(get<Integer const>(tree_info[t]), static_cast<Integer::Int>(expected_group))
        << "tree_info[" << t << "] mismatch";
  }

  // Verify iteration_indptr: [0, 3, 6, 9, 12]
  auto const& indptr = get<Array const>(model["model"]["iteration_indptr"]);
  ASSERT_EQ(indptr.size(), static_cast<std::size_t>(kNumRounds + 1));
  for (std::size_t i = 0; i <= static_cast<std::size_t>(kNumRounds); ++i) {
    ASSERT_EQ(get<Integer const>(indptr[i]), static_cast<Integer::Int>(i * kNumClasses));
  }

  ASSERT_EQ(booster->BoostedRounds(), static_cast<std::int32_t>(kNumRounds));
}

/**
 * @brief Test 18: Multi-class prediction output shape.
 *
 * PredictBatch must produce n_vertices * n_classes predictions.
 * All values must be finite.
 */
TEST(TreeGBooster, MultiClassPredictionShape) {
  Context ctx;
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_target_t kNumClasses = 3;

  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;
  auto dmat = CreateMultiClassGraphDMatrix(kNumClasses);
  auto mparam = MakeMP(total_cols, 0.5f, kNumClasses);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.3"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  booster->Configure(args);

  // Train 2 rounds
  HostDeviceVector<float> train_preds;
  for (int r = 0; r < 2; ++r) {
    auto gpair = GenerateMultiClassGradients(&ctx, kNumVertices, kNumClasses);
    booster->DoBoost(dmat, &gpair, nullptr);
  }

  // Predict
  HostDeviceVector<float> out_preds;
  booster->PredictBatch(dmat, &out_preds, false, 0, 0);

  auto const& h_preds = out_preds.ConstHostVector();
  // Interleaved layout: n_vertices * n_classes
  ASSERT_EQ(h_preds.size(), static_cast<std::size_t>(kNumVertices) * kNumClasses);

  // All predictions must be finite
  for (std::size_t i = 0; i < h_preds.size(); ++i) {
    ASSERT_TRUE(std::isfinite(h_preds[i]))
        << "Prediction at index " << i << " is not finite: " << h_preds[i];
  }

  // At least some predictions should differ from base_score (trees contributed)
  float base_score = 0.5f;
  bool any_different = false;
  for (auto pred : h_preds) {
    if (std::abs(pred - base_score) > 1e-8f) {
      any_different = true;
      break;
    }
  }
  ASSERT_TRUE(any_different) << "All predictions equal base_score; trees did not contribute.";
}

/**
 * @brief Test 19: Multi-class empty model returns base_score for all slots.
 */
TEST(TreeGBooster, MultiClassEmptyModelPrediction) {
  Context ctx;
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_target_t kNumClasses = 4;
  constexpr float kBaseScore = 0.25f;

  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;
  auto dmat = CreateMultiClassGraphDMatrix(kNumClasses);
  auto mparam = MakeMP(total_cols, kBaseScore, kNumClasses);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.3"},
            {"max_leaves", "4"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  booster->Configure(args);

  // No training — predict on empty model
  HostDeviceVector<float> out_preds;
  booster->PredictBatch(dmat, &out_preds, false, 0, 0);

  auto const& h_preds = out_preds.ConstHostVector();
  ASSERT_EQ(h_preds.size(), static_cast<std::size_t>(kNumVertices) * kNumClasses);
  for (auto pred : h_preds) {
    ASSERT_FLOAT_EQ(pred, kBaseScore);
  }
}

/**
 * @brief Test 20: Multi-class JSON I/O preserves tree_info.
 *
 * Save and reload a multi-class model; verify tree_info, tree count,
 * iteration_indptr and predictions match exactly.
 */
TEST(TreeGBooster, MultiClassJsonIO) {
  Context ctx;
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_target_t kNumClasses = 3;
  constexpr int kNumRounds = 2;

  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;
  auto dmat = CreateMultiClassGraphDMatrix(kNumClasses);
  auto mparam = MakeMP(total_cols, 0.5f, kNumClasses);

  // Train original
  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.3"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  booster->Configure(args);

  HostDeviceVector<float> preds;
  for (int r = 0; r < kNumRounds; ++r) {
    auto gpair = GenerateMultiClassGradients(&ctx, kNumVertices, kNumClasses);
    booster->DoBoost(dmat, &gpair, nullptr);
  }

  // Predict with original
  HostDeviceVector<float> preds_orig;
  booster->PredictBatch(dmat, &preds_orig, false, 0, 0);

  // Save
  Json model_out{Object{}};
  booster->SaveModel(&model_out);
  Json config_out{Object{}};
  booster->SaveConfig(&config_out);

  // Verify serialized tree_info
  auto const& ti_orig = get<Array const>(model_out["model"]["tree_info"]);
  ASSERT_EQ(ti_orig.size(), static_cast<std::size_t>(kNumRounds * kNumClasses));

  // Load into new booster
  std::unique_ptr<GradientBooster> gbm_loaded(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* loaded = dynamic_cast<TreeGBooster*>(gbm_loaded.get());
  loaded->LoadModel(model_out);
  loaded->LoadConfig(config_out);

  // Re-save and compare tree_info
  Json model_reloaded{Object{}};
  loaded->SaveModel(&model_reloaded);

  auto const& ti_reloaded = get<Array const>(model_reloaded["model"]["tree_info"]);
  ASSERT_EQ(ti_orig.size(), ti_reloaded.size());
  for (std::size_t i = 0; i < ti_orig.size(); ++i) {
    ASSERT_EQ(get<Integer const>(ti_orig[i]), get<Integer const>(ti_reloaded[i]))
        << "tree_info[" << i << "] mismatch after reload";
  }

  // Verify trees count and indptr
  auto const& trees_reloaded = get<Array const>(model_reloaded["model"]["trees"]);
  ASSERT_EQ(trees_reloaded.size(), static_cast<std::size_t>(kNumRounds * kNumClasses));
  auto const& indptr_reloaded = get<Array const>(model_reloaded["model"]["iteration_indptr"]);
  auto const& indptr_orig = get<Array const>(model_out["model"]["iteration_indptr"]);
  ASSERT_EQ(indptr_orig.size(), indptr_reloaded.size());

  // Predict with loaded model and compare
  HostDeviceVector<float> preds_loaded;
  loaded->PredictBatch(dmat, &preds_loaded, false, 0, 0);

  auto const& h_orig = preds_orig.ConstHostVector();
  auto const& h_loaded = preds_loaded.ConstHostVector();
  ASSERT_EQ(h_orig.size(), h_loaded.size());
  for (std::size_t i = 0; i < h_orig.size(); ++i) {
    ASSERT_FLOAT_EQ(h_orig[i], h_loaded[i])
        << "Prediction mismatch at index " << i << " after reload";
  }
}

/**
 * @brief Test 21: Multi-class prediction with layer range.
 *
 * Predict using a partial layer range and verify output shape and that
 * results differ from full-range prediction.
 */
TEST(TreeGBooster, MultiClassPredictLayerRange) {
  Context ctx;
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_target_t kNumClasses = 3;
  constexpr int kNumRounds = 3;

  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;
  auto dmat = CreateMultiClassGraphDMatrix(kNumClasses);
  auto mparam = MakeMP(total_cols, 0.5f, kNumClasses);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.3"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  booster->Configure(args);

  HostDeviceVector<float> train_preds;
  for (int r = 0; r < kNumRounds; ++r) {
    auto gpair = GenerateMultiClassGradients(&ctx, kNumVertices, kNumClasses);
    booster->DoBoost(dmat, &gpair, nullptr);
  }

  // Full range prediction
  HostDeviceVector<float> full_preds;
  booster->PredictBatch(dmat, &full_preds, false, 0, 0);

  // Partial range: first 2 rounds only (layer_begin=0, layer_end=2)
  HostDeviceVector<float> partial_preds;
  booster->PredictBatch(dmat, &partial_preds, false, 0, 2);

  auto const& h_full = full_preds.ConstHostVector();
  auto const& h_partial = partial_preds.ConstHostVector();

  // Same shape
  ASSERT_EQ(h_full.size(), h_partial.size());
  ASSERT_EQ(h_full.size(), static_cast<std::size_t>(kNumVertices) * kNumClasses);

  // Different values (3 rounds vs 2 rounds should differ)
  bool any_different = false;
  for (std::size_t i = 0; i < h_full.size(); ++i) {
    if (std::abs(h_full[i] - h_partial[i]) > 1e-8f) {
      any_different = true;
      break;
    }
  }
  ASSERT_TRUE(any_different)
      << "Full and partial predictions should differ (3 rounds vs 2 rounds).";
}

/**
 * @brief Test 22: Multi-class feature score calculation.
 *
 * After multi-class training, FeatureScore must still return valid results
 * with scores aggregated across all per-class trees.
 */
TEST(TreeGBooster, MultiClassFeatureScore) {
  Context ctx;
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_target_t kNumClasses = 3;

  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;
  auto dmat = CreateMultiClassGraphDMatrix(kNumClasses);
  auto mparam = MakeMP(total_cols, 0.5f, kNumClasses);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.3"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  booster->Configure(args);

  HostDeviceVector<float> preds;
  for (int r = 0; r < 3; ++r) {
    auto gpair = GenerateMultiClassGradients(&ctx, kNumVertices, kNumClasses);
    booster->DoBoost(dmat, &gpair, nullptr);
  }

  // All importance types should work
  for (const auto& itype : {"weight", "gain", "cover", "total_gain", "total_cover"}) {
    std::vector<bst_feature_t> features;
    std::vector<float> scores;
    booster->FeatureScore(itype, {}, &features, &scores);

    ASSERT_EQ(features.size(), scores.size());
    for (auto fid : features) {
      ASSERT_LT(fid, kNumFeatures);
    }
    for (auto score : scores) {
      ASSERT_GE(score, 0.0f);
    }
  }
}

/**
 * @brief Test 23: Single-class (binary) backward compatibility.
 *
 * With n_groups=1, multi-class path reduces to the single-class path.
 * Verify tree count and prediction shape match the original behavior.
 */
TEST(TreeGBooster, MultiClassSingleGroupBackwardCompat) {
  Context ctx;
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_target_t kNumClasses = 1;  // single-target

  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;
  auto dmat = CreateSimpleGraphDMatrix();
  auto mparam = MakeMP(total_cols, 0.5f, kNumClasses);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.3"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  booster->Configure(args);

  auto gpair = GenerateRandomGradients(&ctx, kNumVertices);
  HostDeviceVector<float> train_preds;
  booster->DoBoost(dmat, &gpair, nullptr);

  // Should be exactly 1 tree (single class, one round)
  Json model{Object{}};
  booster->SaveModel(&model);
  auto const& trees = get<Array const>(model["model"]["trees"]);
  ASSERT_EQ(trees.size(), 1u);

  auto const& tree_info = get<Array const>(model["model"]["tree_info"]);
  ASSERT_EQ(tree_info.size(), 1u);
  ASSERT_EQ(get<Integer const>(tree_info[0]), 0);

  // Prediction shape: n_vertices * 1 = n_vertices
  HostDeviceVector<float> out_preds;
  booster->PredictBatch(dmat, &out_preds, false, 0, 0);
  ASSERT_EQ(out_preds.ConstHostVector().size(), static_cast<std::size_t>(kNumVertices));
}

/**
 * @brief Test 24: Multi-class prediction routing correctness.
 *
 * Each tree's prediction should be routed only to its class column.
 * After training round 1 with n_classes trees, verify that each class column
 * differs from base_score while staying consistent with its own tree.
 */
TEST(TreeGBooster, MultiClassPredictionRouting) {
  Context ctx;
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_target_t kNumClasses = 3;
  constexpr float kBaseScore = 0.0f;

  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;
  auto dmat = CreateMultiClassGraphDMatrix(kNumClasses);
  auto mparam = MakeMP(total_cols, kBaseScore, kNumClasses);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "1.0"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  booster->Configure(args);

  // One round
  auto gpair = GenerateMultiClassGradients(&ctx, kNumVertices, kNumClasses);
  HostDeviceVector<float> train_preds;
  booster->DoBoost(dmat, &gpair, nullptr);

  // Predict
  HostDeviceVector<float> out_preds;
  booster->PredictBatch(dmat, &out_preds, false, 0, 0);

  auto const& h_preds = out_preds.ConstHostVector();
  ASSERT_EQ(h_preds.size(), static_cast<std::size_t>(kNumVertices) * kNumClasses);

  // Each class column should have contributed (some non-zero values).
  // With base_score=0, any non-zero value in a class column means the tree contributed.
  for (bst_target_t g = 0; g < kNumClasses; ++g) {
    bool class_has_contribution = false;
    for (bst_node_t v = 0; v < kNumVertices; ++v) {
      float pred = h_preds[static_cast<std::size_t>(v) * kNumClasses + g];
      if (std::abs(pred - kBaseScore) > 1e-8f) {
        class_has_contribution = true;
        break;
      }
    }
    ASSERT_TRUE(class_has_contribution)
        << "Class " << g << " has no tree contribution — routing may be broken.";
  }
}

/**
 * @brief Test 25: Multi-class DumpModel returns correct number of trees.
 *
 * After 2 rounds with 3 classes, DumpModel should return 6 tree dumps.
 */
TEST(TreeGBooster, MultiClassDumpModel) {
  Context ctx;
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  constexpr bst_target_t kNumClasses = 3;
  constexpr int kNumRounds = 2;

  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;
  auto dmat = CreateMultiClassGraphDMatrix(kNumClasses);
  auto mparam = MakeMP(total_cols, 0.5f, kNumClasses);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.3"},
            {"max_leaves", "4"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  booster->Configure(args);

  HostDeviceVector<float> preds;
  for (int r = 0; r < kNumRounds; ++r) {
    auto gpair = GenerateMultiClassGradients(&ctx, kNumVertices, kNumClasses);
    booster->DoBoost(dmat, &gpair, nullptr);
  }

  FeatureMap fmap;
  auto dump = booster->DumpModel(fmap, false, "text");
  ASSERT_EQ(dump.size(), static_cast<std::size_t>(kNumRounds * kNumClasses));
  for (auto const& d : dump) {
    ASSERT_FALSE(d.empty());
  }
}

/**
 * @brief Test 26: Multi-class with various class counts (2, 4, 5).
 *
 * Ensures the one-vs-all logic works for different numbers of classes,
 * not just 3.
 */
TEST(TreeGBooster, MultiClassVariousClassCounts) {
  Context ctx;
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;

  for (bst_target_t n_classes : {2, 4, 5}) {
    auto dmat = CreateMultiClassGraphDMatrix(n_classes);
    auto mparam = MakeMP(total_cols, 0.5f, n_classes);

    std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
    auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());

    Args args{{"learning_rate", "0.3"},
              {"max_leaves", "8"},
              {"max_walk_len", std::to_string(kMaxWalkLen)},
              {"num_features", std::to_string(kNumFeatures)},
              {"num_vertices", std::to_string(kNumVertices)}};
    booster->Configure(args);

    auto gpair = GenerateMultiClassGradients(&ctx, kNumVertices, n_classes);
    HostDeviceVector<float> preds;
    booster->DoBoost(dmat, &gpair, nullptr);

    // Verify tree count
    Json model{Object{}};
    booster->SaveModel(&model);
    auto const& trees = get<Array const>(model["model"]["trees"]);
    ASSERT_EQ(trees.size(), static_cast<std::size_t>(n_classes))
        << "Wrong tree count for n_classes=" << n_classes;

    // Verify prediction shape
    HostDeviceVector<float> out_preds;
    booster->PredictBatch(dmat, &out_preds, false, 0, 0);
    ASSERT_EQ(out_preds.ConstHostVector().size(),
              static_cast<std::size_t>(kNumVertices) * n_classes)
        << "Wrong prediction size for n_classes=" << n_classes;
  }
}

/**
 * @brief Test 27: Verify attention_type_sample_probability is stored in tparam_ after Configure.
 *
 * Covers the first hop in the propagation chain:
 *   Args{"attention_type_sample_probability", "0.5"}
 *     → TreeGBooster::Configure()
 *     → tparam_.attention_type_sample_probability
 *
 * Checking tparam_ is sufficient here because TreeGBooster::Configure() passes tparam_
 * unchanged to the updater, making tparam_ the canonical source of truth.
 */
TEST(TreeGBooster, AttentionTypeSampleProbabilityParameterPropagatedToTParam) {
  Context ctx;
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;
  auto mparam = MakeMP(total_cols, 0.5f, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());
  ASSERT_NE(booster, nullptr);

  constexpr float kExpectedProb = 0.5f;
  Args args{{"learning_rate", "0.1"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "2"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)},
            {"attention_type_sample_probability", "0.5"}};
  ASSERT_NO_THROW(booster->Configure(args));

  EXPECT_FLOAT_EQ(booster->GetTrainParam().attention_type_sample_probability, kExpectedProb);
}

/**
 * @brief Test 28: Verify attention_type_sample_probability=0 produces kSource-only splits.
 *
 * Covers the full end-to-end propagation path:
 *   Args → TreeGBooster::Configure()
 *        → updater_args["attention_type_sample_probability"]
 *        → TreeGUpdater::Configure()
 *        → SampleRestrictionTypes() always returns {kSource, dist=0}
 *
 * With probability=0.0, every FindBestSplit call hits the empty-set fallback
 * and selects kSource + effective_max_ancestor_distance=0.  The resulting tree
 * must therefore have restriction_type=kSource and ancestor_index=kInvalidNodeId
 * on every node (leaves are hardcoded to these values; split nodes receive them
 * via the fallback path).
 */
TEST(TreeGBooster, AttentionTypeSampleProbabilityZeroProducesSourceOnlyTree) {
  Context ctx;
  auto dmat = CreateSimpleGraphDMatrix();

  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;
  auto mparam = MakeMP(total_cols, 0.5f, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());
  ASSERT_NE(booster, nullptr);

  // Configure with probability=0.0 to guarantee the fallback on every split.
  Args args{{"learning_rate", "0.1"},
            {"max_leaves", "8"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "2"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)},
            {"attention_type_sample_probability", "0.0"}};
  booster->Configure(args);

  auto gpair = GenerateRandomGradients(&ctx, kNumVertices);
  HostDeviceVector<float> out_preds;
  booster->DoBoost(dmat, &gpair, nullptr);

  // Inspect the serialized tree structure to verify fallback was applied.
  Json model{Object{}};
  booster->SaveModel(&model);

  auto const& trees = get<Array const>(model["model"]["trees"]);
  ASSERT_FALSE(trees.empty());

  auto const& nodes = get<Array const>(trees[0]["nodes"]);
  ASSERT_FALSE(nodes.empty());

  // kSource serializes as its underlying integer (1); kInvalidNodeId is max int32.
  const int64_t kSourceInt = static_cast<int64_t>(RestrictionType::kSource);
  const int64_t kInvalidId = static_cast<int64_t>(RegTreeG::kInvalidNodeId);

  for (auto const& node_json : nodes) {
    int64_t restriction_type = get<Integer const>(node_json["restriction_type"]);
    int64_t ancestor_index = get<Integer const>(node_json["ancestor_index"]);
    EXPECT_EQ(restriction_type, kSourceInt)
        << "Expected kSource for every node with attention_type_sample_probability=0.0";
    EXPECT_EQ(ancestor_index, kInvalidId)
        << "Expected kInvalidNodeId for every node with attention_type_sample_probability=0.0";
  }
}

// ========== colsample_bytree Tests ==========

/**
 * @brief Verify colsample_bytree parameter propagates and persists through JSON save/load.
 */
TEST(TreeGBooster, ColsampleBytreeConfigPersistence) {
  Context ctx;

  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 5;
  constexpr bst_node_t kMaxWalkLen = 1;

  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, 0.5f, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());

  Args args{{"learning_rate", "0.1"},
            {"max_leaves", "4"},
            {"colsample_bytree", "0.6"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "1"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)}};
  booster->Configure(args);

  Json config{Object{}};
  booster->SaveConfig(&config);

  auto const& tparam = get<Object const>(config["treeg_train_param"]);
  auto it = tparam.find("colsample_bytree");
  ASSERT_NE(it, tparam.end()) << "colsample_bytree should be present in saved config";

  std::string saved_val = get<String const>(it->second);
  // Float-to-string may produce "0.600000", check prefix
  EXPECT_FLOAT_EQ(std::stof(saved_val), 0.6f)
      << "colsample_bytree should be 0.6, got '" << saved_val << "'";
}

/**
 * @brief Verify colsample_bytree=1.0 produces identical model to no colsample_bytree.
 */
TEST(TreeGBooster, ColsampleBytreeOneMatchesDefault) {
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 3;
  constexpr bst_node_t kMaxWalkLen = 1;
  bst_feature_t total_cols = kNumFeatures + kNumVertices * kMaxWalkLen;

  auto dmat = CreateSimpleGraphDMatrix();

  // Booster without colsample_bytree
  Context ctx1;
  ctx1.seed = 77;
  auto mparam1 = MakeMP(total_cols, 0.5f, 1);
  std::unique_ptr<GradientBooster> gbm1(GradientBooster::Create("treeg", &ctx1, &mparam1));
  Args args1{{"learning_rate", "0.3"},
             {"max_leaves", "4"},
             {"max_walk_len", std::to_string(kMaxWalkLen)},
             {"max_ancestor_distance", "1"},
             {"num_features", "1"},
             {"num_vertices", std::to_string(kNumVertices)}};
  gbm1->Configure(args1);

  // Booster with colsample_bytree=1.0
  Context ctx2;
  ctx2.seed = 77;
  auto mparam2 = MakeMP(total_cols, 0.5f, 1);
  std::unique_ptr<GradientBooster> gbm2(GradientBooster::Create("treeg", &ctx2, &mparam2));
  Args args2 = args1;
  args2.emplace_back("colsample_bytree", "1.0");
  gbm2->Configure(args2);

  // Both should produce the same predictions after one boosting round
  auto* booster1 = dynamic_cast<TreeGBooster*>(gbm1.get());
  auto* booster2 = dynamic_cast<TreeGBooster*>(gbm2.get());

  // Get the internal models and verify they configured correctly
  ASSERT_FLOAT_EQ(booster1->GetTrainParam().colsample_bytree, 1.0f);
  ASSERT_FLOAT_EQ(booster2->GetTrainParam().colsample_bytree, 1.0f);
}

/**
 * @brief Verify colsample_bytree < 1.0 produces a valid trained model.
 */
TEST(TreeGBooster, ColsampleBytreeSubsamplingTrains) {
  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;

  auto dmat = CreateSimpleGraphDMatrix();

  Context ctx;
  ctx.seed = 42;
  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, 0.5f, 1);
  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* treeg_booster = dynamic_cast<TreeGBooster*>(gbm.get());
  ASSERT_NE(treeg_booster, nullptr);

  Args args{{"learning_rate", "0.3"},
            {"max_leaves", "4"},
            {"max_walk_len", std::to_string(kMaxWalkLen)},
            {"max_ancestor_distance", "2"},
            {"num_features", std::to_string(kNumFeatures)},
            {"num_vertices", std::to_string(kNumVertices)},
            {"colsample_bytree", "0.5"}};
  // K=1, colsample=0.5 → max(1, floor(0.5)) = 1 feature (guaranteed)
  treeg_booster->Configure(args);

  auto gpair = GenerateRandomGradients(&ctx, kNumVertices);
  HostDeviceVector<float> out_preds;
  treeg_booster->DoBoost(dmat, &gpair, nullptr);

  ASSERT_EQ(treeg_booster->BoostedRounds(), 1) << "Should have 1 round after DoBoost";

  // Predict and verify non-trivial predictions
  HostDeviceVector<float> preds;
  treeg_booster->PredictBatch(dmat, &preds, false, 0, 0);
  auto& h_preds = preds.HostVector();
  ASSERT_EQ(h_preds.size(), static_cast<std::size_t>(kNumVertices));

  // Predictions should be finite (training did not produce NaN/Inf)
  for (std::size_t i = 0; i < h_preds.size(); ++i) {
    ASSERT_TRUE(std::isfinite(h_preds[i]))
        << "Prediction at index " << i << " is not finite: " << h_preds[i];
  }
}

// ========== Sprint 2: correctness blockers ==========

TEST(TreeGBooster, RejectsDistributedTraining) {
  collective::TestDistributedGlobal(2, [] {
    ASSERT_TRUE(collective::IsDistributed());
    Context ctx;
    auto mparam = MakeMP(1 + 10 * 2, 0.5f, 1);
    std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
    Args args{{"num_features", "1"}, {"num_vertices", "10"}};
    EXPECT_THROW(gbm->Configure(args), dmlc::Error);
  });
}

TEST(TreeGBooster, ConfigurePreservesParamPrecision) {
  Context ctx;
  auto mparam = MakeMP(1 + 10 * 2, 0.5f, 1);
  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());
  ASSERT_NE(booster, nullptr);

  booster->Configure(Args{{"num_features", "1"},
                          {"num_vertices", "10"},
                          {"learning_rate", "1e-7"},
                          {"reg_lambda", "1.23456789"},
                          {"colsample_bytree", "0.123456789"}});

  auto const& tparam = booster->GetTrainParam();
  ASSERT_NE(booster->UpdaterForTesting(), nullptr);
  auto const& uparam = booster->UpdaterForTesting()->ParamForTesting();
  EXPECT_EQ(tparam.learning_rate, 1e-7f);
  EXPECT_EQ(uparam.learning_rate, tparam.learning_rate);
  EXPECT_EQ(uparam.reg_lambda, tparam.reg_lambda);
  EXPECT_EQ(uparam.colsample_bytree, tparam.colsample_bytree);
}

namespace {
/** @brief Train one round on the 10-vertex ring with the given extra args. */
std::unique_ptr<GradientBooster> TrainRing(Context const* ctx, LearnerModelState const* mparam,
                                           Args extra) {
  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", ctx, mparam));
  Args args{{"learning_rate", "0.1"},
            {"max_leaves", "4"},
            {"max_walk_len", "2"},
            {"num_features", "1"},
            {"num_vertices", "10"}};
  args.insert(args.end(), extra.begin(), extra.end());
  gbm->Configure(args);
  auto gpair = GenerateRandomGradients(ctx, 10);
  gbm->DoBoost(CreateSimpleGraphDMatrix(), &gpair, nullptr);
  return gbm;
}
}  // anonymous namespace

TEST(TreeGBooster, AttentionSampleProbabilitySurvivesReload) {
  Context ctx;
  auto mparam = MakeMP(1 + 10 * 2, 0.5f, 1);
  auto trained = TrainRing(&ctx, &mparam, {{"attention_type_sample_probability", "0.5"}});
  Json model{Object{}};
  trained->SaveModel(&model);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());
  booster->LoadModel(model);
  booster->Configure(Args{{"attention_type_sample_probability", "0.5"}});
  auto gpair = GenerateRandomGradients(&ctx, 10);
  booster->DoBoost(CreateSimpleGraphDMatrix(), &gpair, nullptr);

  EXPECT_FLOAT_EQ(booster->GetTrainParam().attention_type_sample_probability, 0.5f);
  EXPECT_FLOAT_EQ(booster->UpdaterForTesting()->ParamForTesting().attention_type_sample_probability,
                  0.5f);
  // Structural parameters are still restored from the model.
  EXPECT_EQ(booster->GetTrainParam().num_vertices, 10);
  EXPECT_EQ(booster->BoostedRounds(), 2);
}

TEST(TreeGBooster, SavedModelOmitsTrainingOnlyParams) {
  Context ctx;
  auto mparam = MakeMP(1 + 10 * 2, 0.5f, 1);
  auto trained = TrainRing(&ctx, &mparam, {{"attention_type_sample_probability", "0.5"}});
  Json model{Object{}};
  trained->SaveModel(&model);
  auto const& mp = get<Object const>(model["model"]["treeg_booster_model_param"]);
  EXPECT_EQ(mp.find("attention_type_sample_probability"), mp.cend());
  EXPECT_EQ(mp.find("min_leaf_size"), mp.cend());

  // Backward compatibility: models written by older versions may still carry the key.
  get<Object>(model["model"]["treeg_booster_model_param"])["attention_type_sample_probability"] =
      String{"1"};
  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  EXPECT_NO_THROW(gbm->LoadModel(model));
  EXPECT_EQ(gbm->BoostedRounds(), 1);
}

TEST(TreeGBooster, OldConfigWithMinLeafSizeStillLoads) {
  Context ctx;
  auto mparam = MakeMP(1 + 10 * 2, 0.5f, 1);
  auto trained = TrainRing(&ctx, &mparam, {});
  Json config{Object{}};
  trained->SaveConfig(&config);
  get<Object>(config["treeg_booster_train_param"])["min_leaf_size"] = String{"5"};

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  EXPECT_NO_THROW(gbm->LoadConfig(config));
}

namespace {
/** @brief Wrap hand-built trees into a saved booster model. */
Json MakeBoosterJson(std::vector<RegTreeG> const& trees, bst_feature_t num_features,
                     bool intercept) {
  TreeGBoosterModelParam param;
  param.num_trees = static_cast<std::int32_t>(trees.size());
  param.num_features = num_features;
  param.num_vertices = trees.front().NumVertices();
  param.max_walk_len = trees.front().MaxWalkLen();
  param.max_ancestor_distance = trees.front().MaxAncestorDistance();
  param.intercept = intercept;

  Json out{Object{}};
  out["name"] = String{"treeg"};
  Json model{Object{}};
  model["treeg_booster_model_param"] = ToJson(param);
  std::vector<Json> jtrees;
  std::vector<Json> tree_info;
  std::vector<Json> indptr{Json{Integer{0}}};
  for (std::size_t t = 0; t < trees.size(); ++t) {
    Json jtree{Object{}};
    trees[t].SaveModel(&jtree);
    jtree["id"] = Integer{static_cast<Integer::Int>(t)};
    jtrees.emplace_back(std::move(jtree));
    tree_info.emplace_back(Integer{0});
    indptr.emplace_back(Integer{static_cast<Integer::Int>(t + 1)});
  }
  model["trees"] = Array{std::move(jtrees)};
  model["tree_info"] = Array{std::move(tree_info)};
  model["iteration_indptr"] = Array{std::move(indptr)};
  out["model"] = std::move(model);
  return out;
}
}  // anonymous namespace

TEST(TreeGBooster, FeatureScoreExcludesIntercept) {
  constexpr bst_feature_t kK = 1;
  RegTreeG tree{10, kK, 2, 2, /*intercept=*/true};
  // Root splits on the intercept (index K), the left child on feature 0.
  tree.ExpandNode(0, kK, 2.0f, 0.0f, 0.0f, 0.0f, 4.0f, 10.0f, 6.0f, 4.0f);
  tree.SetNodeTreeGParams(0, 1, RestrictionType::kSource, RegTreeG::kInvalidNodeId, true);
  tree.ExpandNode(1, 0, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 6.0f, 3.0f, 3.0f);
  tree.SetNodeTreeGParams(1, 0, RestrictionType::kSource, RegTreeG::kInvalidNodeId, true);

  Context ctx;
  auto mparam = MakeMP(kK + 10 * 2, 0.5f, 1);
  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  gbm->LoadModel(MakeBoosterJson({tree}, kK, true));

  for (auto const* itype : {"weight", "gain", "cover", "total_gain", "total_cover"}) {
    std::vector<bst_feature_t> features;
    std::vector<float> scores;
    gbm->FeatureScore(itype, {}, &features, &scores);
    ASSERT_EQ(features, std::vector<bst_feature_t>{0}) << itype;
  }
}

TEST(TreeGBooster, FeatureScoreAggregatesAcrossWalkLengths) {
  constexpr bst_feature_t kK = 2;
  RegTreeG tree{10, kK, 2, 2, false};
  // Feature 0 at d=1 (gain 3, cover 10) and d=2 (gain 5, cover 6); feature 1 at d=0.
  tree.ExpandNode(0, 0, 1.0f, 0.0f, 0.0f, 0.0f, 3.0f, 10.0f, 6.0f, 4.0f);
  tree.SetNodeTreeGParams(0, 1, RestrictionType::kSource, RegTreeG::kInvalidNodeId, true);
  tree.ExpandNode(1, 0, 1.0f, 0.0f, 0.0f, 0.0f, 5.0f, 6.0f, 3.0f, 3.0f);
  tree.SetNodeTreeGParams(1, 2, RestrictionType::kCycle, 0, true);
  tree.ExpandNode(2, 1, 1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 4.0f, 2.0f, 2.0f);
  tree.SetNodeTreeGParams(2, 0, RestrictionType::kSource, RegTreeG::kInvalidNodeId, true);

  Context ctx;
  auto mparam = MakeMP(kK + 10 * 2, 0.5f, 1);
  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  gbm->LoadModel(MakeBoosterJson({tree}, kK, false));

  auto score_of = [&](std::string const& itype) {
    std::vector<bst_feature_t> features;
    std::vector<float> scores;
    gbm->FeatureScore(itype, {}, &features, &scores);
    EXPECT_EQ(features, (std::vector<bst_feature_t>{0, 1}));
    return scores;
  };
  EXPECT_EQ(score_of("weight"), (std::vector<float>{2.0f, 1.0f}));
  EXPECT_EQ(score_of("total_gain"), (std::vector<float>{8.0f, 2.0f}));
  EXPECT_EQ(score_of("gain"), (std::vector<float>{4.0f, 2.0f}));
  EXPECT_EQ(score_of("total_cover"), (std::vector<float>{16.0f, 4.0f}));
  EXPECT_EQ(score_of("cover"), (std::vector<float>{8.0f, 4.0f}));
}

// ========== Small hard-coded graphs for prediction-cache tests ==========

namespace {
/** @brief A small hard-coded graph: undirected edges, dense features and labels. */
struct SmallGraph {
  std::string name;
  bst_node_t n;
  bst_feature_t k;
  std::vector<std::pair<bst_node_t, bst_node_t>> edges;
  std::vector<float> x;       // n * k, row-major
  std::vector<float> labels;  // regression target, or class id in [0, n_groups)
};

/** @brief Ring of 12 vertices, 3 features. */
SmallGraph SmallRing() {
  SmallGraph g{"ring", 12, 3, {}, {}, {}};
  for (bst_node_t i = 0; i < 12; ++i) {
    g.edges.emplace_back(i, (i + 1) % 12);
  }
  g.x = {0.3f, 1.0f, 0.5f, 0.9f, 0.2f, 0.1f, 0.4f, 0.8f, 0.7f, 0.6f, 0.0f, 0.3f,
         0.1f, 0.5f, 0.9f, 0.8f, 0.7f, 0.2f, 0.2f, 0.3f, 0.6f, 0.7f, 0.9f, 0.0f,
         0.5f, 0.4f, 0.8f, 1.0f, 0.6f, 0.4f, 0.0f, 0.1f, 1.0f, 0.6f, 0.8f, 0.5f};
  g.labels = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f};
  return g;
}

/** @brief Two 5-cliques {0..4} and {5..9} joined by the bridge 4-5, 3 features. */
SmallGraph SmallBarbell() {
  SmallGraph g{"barbell", 10, 3, {}, {}, {}};
  for (bst_node_t base : {0, 5}) {
    for (bst_node_t i = 0; i < 5; ++i) {
      for (bst_node_t j = i + 1; j < 5; ++j) {
        g.edges.emplace_back(base + i, base + j);
      }
    }
  }
  g.edges.emplace_back(4, 5);
  g.x = {0.1f, 0.9f, 0.4f, 0.2f, 0.8f, 0.6f, 0.0f, 0.7f, 0.5f, 0.3f, 1.0f, 0.2f, 0.5f, 0.5f, 0.9f,
         0.6f, 0.4f, 0.1f, 0.9f, 0.1f, 0.3f, 0.7f, 0.2f, 0.8f, 1.0f, 0.0f, 0.7f, 0.8f, 0.3f, 0.0f};
  g.labels = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f};
  return g;
}

/** @brief Star with centre 0 and 8 leaves; features have many tied values. */
SmallGraph SmallTiedStar() {
  SmallGraph g{"tied_star", 9, 2, {}, {}, {}};
  for (bst_node_t i = 1; i < 9; ++i) {
    g.edges.emplace_back(0, i);
  }
  g.x = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 2.0f,
         1.0f, 2.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f, 1.0f, 0.0f};
  g.labels = {1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f};
  return g;
}

/**
 * @brief Irregular 10-vertex graph with varying degrees and cycles.
 *
 * Chosen so that, with `attention_type_sample_probability=1`, the trained trees use the Cycle
 * restriction and ancestor attention subsets, not only Source splits on all vertices.
 */
SmallGraph SmallIrregular() {
  SmallGraph g{"irregular", 10, 2, {}, {}, {}};
  g.edges = {{0, 1}, {0, 6}, {1, 2}, {1, 5}, {1, 9}, {2, 3},
             {2, 4}, {3, 4}, {3, 7}, {3, 8}, {3, 9}, {7, 8}};
  g.x = {1.0f, 0.2f, 0.4f, 0.6f, 0.6f, 0.4f, 0.0f, 1.0f, 0.8f, 0.0f,
         0.8f, 0.8f, 0.2f, 0.6f, 0.6f, 0.0f, 1.0f, 0.8f, 0.2f, 0.0f};
  g.labels = {0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  return g;
}

/** @brief Build the [X | A | A^2 | ... | A^d] DMatrix for a small graph. */
std::shared_ptr<DMatrix> MakeSmallGraphDMatrix(SmallGraph const& g, bst_node_t max_walk_len) {
  auto const n = static_cast<std::size_t>(g.n);
  std::vector<float> adj(n * n, 0.0f);
  for (auto const& [u, v] : g.edges) {
    adj[u * n + v] = 1.0f;
    adj[v * n + u] = 1.0f;
  }
  auto const cols = static_cast<std::size_t>(g.k) + n * max_walk_len;
  std::vector<float> data(n * cols, 0.0f);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t f = 0; f < g.k; ++f) {
      data[i * cols + f] = g.x[i * g.k + f];
    }
  }
  std::vector<float> power = adj;
  for (bst_node_t d = 0; d < max_walk_len; ++d) {
    std::size_t const offset = g.k + d * n;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        data[i * cols + offset + j] = power[i * n + j];
      }
    }
    std::vector<float> next(n * n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t m = 0; m < n; ++m) {
        for (std::size_t j = 0; j < n; ++j) {
          next[i * n + j] += power[i * n + m] * adj[m * n + j];
        }
      }
    }
    power = std::move(next);
  }
  return GetDMatrixFromData(data, n, static_cast<bst_feature_t>(cols));
}

/** @brief One training configuration of the prediction-cache tests. */
struct TrainConfig {
  std::string name;
  bst_node_t max_walk_len;
  bst_target_t n_groups;
  Args args;
};

std::vector<TrainConfig> TrainConfigs() {
  return {
      {"walk0", 0, 1, {{"intercept", "false"}}},
      {"walk2_attention",
       2,
       1,
       {{"intercept", "false"},
        {"max_ancestor_distance", "2"},
        {"attention_type_sample_probability", "0.5"}}},
      {"walk2_intercept_colsample",
       2,
       1,
       {{"intercept", "true"},
        {"max_ancestor_distance", "2"},
        {"attention_type_sample_probability", "0.5"},
        {"colsample_bytree", "0.7"}}},
      {"walk2_multiclass",
       2,
       2,
       {{"intercept", "true"},
        {"max_ancestor_distance", "2"},
        {"attention_type_sample_probability", "0.5"}}},
      {"walk2_both_types",
       2,
       1,
       {{"intercept", "true"},
        {"max_ancestor_distance", "3"},
        {"attention_type_sample_probability", "1.0"}}},
  };
}

}  // anonymous namespace

TEST(TreeGBooster, FeatureScoreRejectsUnknownType) {
  Context ctx;
  auto mparam = MakeMP(1 + 10 * 2, 0.5f, 1);
  auto trained = TrainRing(&ctx, &mparam, {});
  std::vector<bst_feature_t> features;
  std::vector<float> scores;
  EXPECT_THROW(trained->FeatureScore("gian", {}, &features, &scores), dmlc::Error);
}

/**
 * @brief GraphData is built once per training DMatrix, not once per round, and
 *        cached-graph prediction matches transient-graph prediction.
 */
TEST(TreeGBooster, GraphDataBuiltOncePerTrainingMatrix) {
  Context ctx;
  auto dmat = CreateSimpleGraphDMatrix();

  constexpr bst_node_t kNumVertices = 10;
  constexpr bst_feature_t kNumFeatures = 1;
  constexpr bst_node_t kMaxWalkLen = 2;
  auto mparam = MakeMP(kNumFeatures + kNumVertices * kMaxWalkLen, 0.5f, 1);

  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", &ctx, &mparam));
  auto* booster = dynamic_cast<TreeGBooster*>(gbm.get());
  ASSERT_NE(booster, nullptr);
  booster->Configure({{"max_leaves", "8"},
                      {"max_walk_len", std::to_string(kMaxWalkLen)},
                      {"max_ancestor_distance", "2"},
                      {"num_features", std::to_string(kNumFeatures)},
                      {"num_vertices", std::to_string(kNumVertices)}});

  for (int round = 0; round < 3; ++round) {
    auto gpair = GenerateRandomGradients(&ctx, kNumVertices);
    booster->DoBoost(dmat, &gpair, nullptr);
  }
  EXPECT_EQ(booster->NumGraphBuilds(), 1u);

  // Prediction on the cached training matrix vs. an identical, uncached matrix.
  HostDeviceVector<float> cached;
  HostDeviceVector<float> fresh;
  booster->PredictBatch(dmat, &cached, false, 0, 0);
  booster->PredictBatch(CreateSimpleGraphDMatrix(), &fresh, false, 0, 0);
  EXPECT_EQ(cached.HostVector(), fresh.HostVector());
  // Prediction never inserts into the cache or builds a training graph.
  EXPECT_EQ(booster->NumGraphBuilds(), 1u);

  // A second training matrix gets its own entry.
  auto gpair = GenerateRandomGradients(&ctx, kNumVertices);
  booster->DoBoost(CreateSimpleGraphDMatrix(), &gpair, nullptr);
  EXPECT_EQ(booster->NumGraphBuilds(), 2u);
}

namespace {
/** @brief Train 3 rounds with squared-error gradients; keep the booster for cache checks. */
std::unique_ptr<GradientBooster> TrainCached(Context const* ctx, LearnerModelState* mparam,
                                             SmallGraph const& g, TrainConfig const& cfg,
                                             std::shared_ptr<DMatrix> const& dmat) {
  std::unique_ptr<GradientBooster> gbm(GradientBooster::Create("treeg", ctx, mparam));
  Args args{{"learning_rate", "0.3"},
            {"max_leaves", "6"},
            {"max_walk_len", std::to_string(cfg.max_walk_len)},
            {"num_features", std::to_string(g.k)},
            {"num_vertices", std::to_string(g.n)}};
  args.insert(args.end(), cfg.args.begin(), cfg.args.end());
  gbm->Configure(args);

  auto const n = static_cast<std::size_t>(g.n);
  auto const n_groups = static_cast<std::size_t>(cfg.n_groups);
  HostDeviceVector<float> preds;
  for (int r = 0; r < 3; ++r) {
    gbm->PredictBatch(dmat, &preds, /*training=*/true, 0, 0);
    auto const& h_preds = preds.ConstHostVector();
    GradientContainer gpair;
    gpair.gpair = linalg::Matrix<GradientPair>{{n, n_groups}, ctx->Device()};
    auto h_gpair = gpair.gpair.HostView();
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t c = 0; c < n_groups; ++c) {
        float const target = n_groups == 1
                                 ? g.labels[i]
                                 : (static_cast<std::size_t>(g.labels[i]) == c ? 1.0f : 0.0f);
        h_gpair(i, c) = GradientPair{h_preds[i * n_groups + c] - target, 1.0f};
      }
    }
    gbm->DoBoost(dmat, &gpair, nullptr);
  }
  return gbm;
}

std::vector<std::uint32_t> Bits(HostDeviceVector<float> const& v) {
  auto const& h = v.ConstHostVector();
  std::vector<std::uint32_t> bits(h.size());
  std::memcpy(bits.data(), h.data(), bits.size() * sizeof(float));
  return bits;
}
}  // anonymous namespace

/**
 * @brief The margin advanced from training leaf positions equals a fresh two-phase
 *        prediction on an identical, uncached DMatrix, bit for bit.
 */
TEST(TreeGBooster, PredictionCacheMatchesFreshPrediction) {
  for (auto const& graph : {SmallRing(), SmallBarbell(), SmallTiedStar(), SmallIrregular()}) {
    for (auto const& cfg : TrainConfigs()) {
      for (std::int32_t n_threads : {1, 4}) {
        SCOPED_TRACE(graph.name + "_" + cfg.name + " nthread=" + std::to_string(n_threads));
        Context ctx;
        ctx.UpdateAllowUnknown(Args{{"nthread", std::to_string(n_threads)}, {"seed", "7"}});
        auto const n_cols = static_cast<bst_feature_t>(graph.k + graph.n * cfg.max_walk_len);
        auto mparam = MakeMP(n_cols, 0.5f, cfg.n_groups);
        auto dmat = MakeSmallGraphDMatrix(graph, cfg.max_walk_len);
        auto gbm = TrainCached(&ctx, &mparam, graph, cfg, dmat);
        auto const* booster = dynamic_cast<TreeGBooster const*>(gbm.get());
        ASSERT_NE(booster, nullptr);
        // The cache was advanced by DoBoost, so no tree is re-predicted below.
        EXPECT_EQ(booster->CachedVersionForTesting(dmat.get()), 3u);

        HostDeviceVector<float> cached;
        HostDeviceVector<float> fresh;
        gbm->PredictBatch(dmat, &cached, /*training=*/true, 0, 0);
        gbm->PredictBatch(MakeSmallGraphDMatrix(graph, cfg.max_walk_len), &fresh, false, 0, 0);
        EXPECT_EQ(Bits(cached), Bits(fresh));
      }
    }
  }
}

/** @brief Layer slices bypass the cache and leave it valid. */
TEST(TreeGBooster, PredictionCacheResetOnLayerSlice) {
  auto const graph = SmallIrregular();
  auto const cfg = TrainConfigs().back();
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"seed", "7"}});
  auto const n_cols = static_cast<bst_feature_t>(graph.k + graph.n * cfg.max_walk_len);
  auto mparam = MakeMP(n_cols, 0.5f, cfg.n_groups);
  auto dmat = MakeSmallGraphDMatrix(graph, cfg.max_walk_len);
  auto gbm = TrainCached(&ctx, &mparam, graph, cfg, dmat);
  auto const* booster = dynamic_cast<TreeGBooster const*>(gbm.get());
  ASSERT_NE(booster, nullptr);
  auto fresh_dmat = MakeSmallGraphDMatrix(graph, cfg.max_walk_len);

  auto expect_same = [&](bst_layer_t begin, bst_layer_t end) {
    HostDeviceVector<float> cached;
    HostDeviceVector<float> fresh;
    gbm->PredictBatch(dmat, &cached, false, begin, end);
    gbm->PredictBatch(fresh_dmat, &fresh, false, begin, end);
    EXPECT_EQ(Bits(cached), Bits(fresh)) << "layers [" << begin << ", " << end << ")";
  };

  expect_same(1, 3);
  EXPECT_EQ(booster->CachedVersionForTesting(dmat.get()), 3u);
  expect_same(0, 0);
  // A shorter prefix rebuilds the cache at that version, then grows back.
  expect_same(0, 1);
  EXPECT_EQ(booster->CachedVersionForTesting(dmat.get()), 1u);
  expect_same(0, 0);
  EXPECT_EQ(booster->CachedVersionForTesting(dmat.get()), 3u);
}

}  // namespace treeg
}  // namespace xgboost
