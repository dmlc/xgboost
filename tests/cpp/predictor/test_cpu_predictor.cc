/**
 * Copyright 2017-2026, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/predictor.h>

#include <algorithm>  // for fill, min
#include <array>      // for array
#include <cmath>      // for isnan
#include <cstddef>    // for size_t
#include <limits>     // for numeric_limits
#include <utility>    // for pair
#include <vector>     // for vector

#include "../../../src/collective/communicator-inl.h"
#include "../../../src/common/kernel.h"
#include "../../../src/data/adapter.h"
#include "../../../src/data/proxy_dmatrix.h"
#include "../../../src/gbm/gbtree.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../../../src/predictor/array_tree_layout.h"
#include "../../../src/predictor/prediction_kernel.h"
#include "../../../src/tree/tree_view.h"
#include "../collective/test_worker.h"  // for TestDistributedGlobal
#include "../helpers.h"
#include "test_predictor.h"
#include "test_shap.h"

namespace xgboost {
TEST(CpuPredictor, Basic) {
  Context ctx;
  size_t constexpr kRows = 5;
  size_t constexpr kCols = 5;
  auto dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
  TestBasic(dmat.get(), &ctx);
}

TEST(CpuPredictor, PredictLeafKernel) {
  Context ctx;
  LearnerModelState mparam{MakeMP(1, 0.0f, 2, ctx.Device())};
  auto model = CreateTestModel(&mparam, &ctx, 2);
  model->trees.front()->ExpandNode(0, 0, 0.5f, true, 0.0f, 1.0f, 2.0f, 0.0f, 2.0f, 1.0f, 1.0f);
  auto dmat = GetDMatrixFromData({0.0f, 1.0f, std::numeric_limits<float>::quiet_NaN()}, 3, 1);
  HostDeviceVector<float> leaves;
  common::DispatchKernel<predictor::PredictLeafKernel>(&ctx, dmat.get(), &leaves, *model, 0);
  ASSERT_EQ(leaves.ConstHostVector(), (std::vector<float>{1, 0, 2, 0, 1, 0}));

  // Reusing the output with a tree limit must resize it and preserve row-major order.
  common::DispatchKernel<predictor::PredictLeafKernel>(&ctx, dmat.get(), &leaves, *model, 1);
  ASSERT_EQ(leaves.ConstHostVector(), (std::vector<float>{1, 2, 1}));
}

TEST(CpuPredictor, BatchPredictionWithWeights) {
  Context ctx;
  TestBatchPredictionWithWeights(&ctx);
}

TEST(CpuPredictor, InplacePredictionWithWeights) {
  Context ctx;
  TestInplacePredictionWithWeights(&ctx);
}

template <typename ArrayLayoutT>
void CheckArrayLayout(const RegTree& tree, ArrayLayoutT buffer, int max_depth, int depth,
                      size_t nid, size_t nid_array) {
  const auto& split_idx = buffer.SplitIndex();
  const auto& split_cond = buffer.SplitCond();
  const auto& default_left = buffer.DefaultLeft();
  const auto& nidx_in_tree = buffer.NidxInTree();
  const auto& nodes = tree.GetNodes(DeviceOrd::CPU());

  if (depth == max_depth) {
    ASSERT_EQ(nidx_in_tree[nid_array - (1u << max_depth) + 1], nid);
    return;
  }

  if (nodes[nid].IsLeaf()) {
    ASSERT_EQ(default_left[nid_array], 0);
    ASSERT_TRUE(std::isnan(split_cond[nid_array]));

    CheckArrayLayout(tree, buffer, max_depth, depth + 1, nid, 2 * nid_array + 2);
  } else {
    ASSERT_EQ(nodes[nid].SplitIndex(), split_idx[nid_array]);
    ASSERT_EQ(nodes[nid].SplitCond(), split_cond[nid_array]);
    ASSERT_EQ(nodes[nid].DefaultLeft(), default_left[nid_array]);

    if (nodes[nid].LeftChild() != RegTree::kInvalidNodeId) {
      CheckArrayLayout(tree, buffer, max_depth, depth + 1, nodes[nid].LeftChild(),
                       2 * nid_array + 1);
    }
    if (nodes[nid].RightChild() != RegTree::kInvalidNodeId) {
      CheckArrayLayout(tree, buffer, max_depth, depth + 1, nodes[nid].RightChild(),
                       2 * nid_array + 2);
    }
  }
}

TEST(CpuPredictor, ArrayTreeLayout) {
  Context ctx;

  RegTree tree;
  size_t n_nodes = 15;  // 2^4 - 1
  for (size_t nid = 0; nid < n_nodes; ++nid) {
    // Some place-holders
    size_t split_index = nid + 1;
    bst_float split_cond = nid + 2;
    bool default_left = nid % 2 == 0;

    tree.ExpandNode(nid, split_index, split_cond, default_left, 0, 0, 0, 0, 0, 0, 0);
  }

  auto sc_tree = tree::ScalarTreeView{ctx.Device(), false, &tree};
  {
    constexpr bst_node_t kDepth = 1;
    predictor::ArrayTreeLayout buffer(sc_tree, kDepth);
    ASSERT_EQ(buffer.NumLevels(), kDepth);
    ASSERT_EQ(buffer.TreeDepth(), 4);
    ASSERT_FALSE(buffer.IsComplete());
    CheckArrayLayout(tree, buffer, kDepth, 0, 0, 0);
  }
  {
    constexpr bst_node_t kDepth = 2;
    predictor::ArrayTreeLayout buffer{sc_tree, kDepth};
    CheckArrayLayout(tree, buffer, kDepth, 0, 0, 0);
  }
  {
    constexpr bst_node_t kDepth = 3;
    predictor::ArrayTreeLayout buffer{sc_tree, kDepth};
    CheckArrayLayout(tree, buffer, kDepth, 0, 0, 0);
  }
  {
    constexpr bst_node_t kDepth = 4;
    predictor::ArrayTreeLayout buffer{sc_tree, kDepth};
    ASSERT_EQ(buffer.NumLevels(), kDepth);
    ASSERT_TRUE(buffer.IsComplete());
    CheckArrayLayout(tree, buffer, kDepth, 0, 0, 0);
  }
  {
    // The number of unrolled levels is capped by the depth of the tree.
    constexpr bst_node_t kDepth = 5;
    predictor::ArrayTreeLayout buffer{sc_tree, kDepth};
    ASSERT_EQ(buffer.NumLevels(), 4);
    ASSERT_TRUE(buffer.IsComplete());
    CheckArrayLayout(tree, buffer, 4, 0, 0, 0);
  }
}

/**
 * Process must agree with a walk of the original tree for every number of unrolled
 * levels: layouts of the default depth (and of deeper trees) traverse with the level
 * count fixed at compile time, shallower layouts with the runtime level loop.  Both the
 * dense and the missing-aware traversal are checked on trees whose inner levels are fully
 * populated (every node offset of every level is used) and that contain early leaves.
 */
TEST(CpuPredictor, ArrayTreeLayoutProcessDepths) {
  Context ctx;
  constexpr int kMaxLevels = predictor::ArrayTreeLayout::kMaxNumDeepLevels;
  constexpr bst_feature_t kFeatures = 8;
  constexpr std::size_t kRows = 64;
  auto const nan = std::numeric_limits<float>::quiet_NaN();

  std::array<RegTree::FVec, kRows> dense;
  std::array<RegTree::FVec, kRows> missing;
  for (std::size_t i = 0; i < kRows; ++i) {
    dense[i].Init(kFeatures);
    missing[i].Init(kFeatures);
    auto d = dense[i].Data();
    auto m = missing[i].Data();
    for (bst_feature_t f = 0; f < kFeatures; ++f) {
      // Values in [-2, 2] in steps of 0.5, never equal to a split condition.
      d[f] = m[f] = static_cast<float>(static_cast<int>((i * 7 + f * 3) % 9) - 4) * 0.5f;
    }
    dense[i].HasMissing(false);
    bool const row_missing = i % 3 == 0;
    if (row_missing) {
      m[i % kFeatures] = nan;
      m[(i / 3) % kFeatures] = nan;
    }
    missing[i].HasMissing(row_missing);
  }

  for (bst_node_t depth : {0, 1, 2, 3, 4, 5, 6, 7, 8, 10}) {
    RegTree tree{1, kFeatures};
    // Breadth-first expansion; some positions below level 1 stay leaves so that the
    // padding of the layout (NaN split conditions, always to the right) is exercised.
    std::vector<std::pair<bst_node_t, bst_node_t>> frontier{{RegTree::kRoot, 0}};
    for (std::size_t k = 0; k < frontier.size(); ++k) {
      auto const [nidx, level] = frontier[k];
      if (level >= depth || (level >= 2 && nidx % 7 == 3)) {
        continue;
      }
      auto const split = static_cast<bst_feature_t>((nidx * 5 + level) % kFeatures);
      auto const cond = static_cast<float>(static_cast<int>(nidx % 5) - 2) + 0.25f;
      tree.ExpandNode(nidx, split, cond, nidx % 3 != 0, 0, 0, 0, 0, 0, 0, 0);
      frontier.emplace_back(tree[nidx].LeftChild(), level + 1);
      frontier.emplace_back(tree[nidx].RightChild(), level + 1);
    }
    auto view = tree::ScalarTreeView{ctx.Device(), false, &tree};
    predictor::ArrayTreeLayout const layout{view};
    int const n_levels = std::min(static_cast<int>(depth), kMaxLevels);
    ASSERT_EQ(layout.TreeDepth(), depth);
    ASSERT_EQ(layout.NumLevels(), n_levels);
    ASSERT_EQ(layout.IsComplete(), depth <= kMaxLevels);
    auto const nodes = tree.GetNodes(ctx.Device());

    auto expect = [&](RegTree::FVec const& feat) {
      bst_node_t nidx = RegTree::kRoot;
      for (int level = 0; level < n_levels && !nodes[nidx].IsLeaf(); ++level) {
        auto const& node = nodes[nidx];
        auto const value = feat.GetFvalue(node.SplitIndex());
        if (std::isnan(value)) {
          nidx = node.DefaultChild();
        } else if (value < node.SplitCond()) {
          nidx = node.LeftChild();
        } else {
          nidx = node.RightChild();
        }
      }
      return nidx;
    };

    for (std::size_t block_size : {1, 2, 3, 8, 16, 31, 32, 63, 64}) {
      SCOPED_TRACE(testing::Message() << "depth=" << depth << ", block_size=" << block_size);
      std::vector<bst_node_t> expected_dense(block_size);
      std::vector<bst_node_t> expected_missing(block_size);
      for (std::size_t i = 0; i < block_size; ++i) {
        expected_dense[i] = expect(dense[i]);
        expected_missing[i] = expect(missing[i]);
      }
      common::Span<RegTree::FVec> dense_block{dense.data(), block_size};
      common::Span<RegTree::FVec> missing_block{missing.data(), block_size};
      std::vector<bst_node_t> actual(block_size, RegTree::kInvalidNodeId);

      layout.Process<false, false>(dense_block, block_size, actual.data());
      ASSERT_EQ(actual, expected_dense);
      // The missing-aware traversal of the same layout must agree on a dense block.
      std::fill(actual.begin(), actual.end(), RegTree::kInvalidNodeId);
      layout.Process<false, true>(dense_block, block_size, actual.data());
      ASSERT_EQ(actual, expected_dense);
      std::fill(actual.begin(), actual.end(), RegTree::kInvalidNodeId);
      layout.Process<false, true>(missing_block, block_size, actual.data());
      ASSERT_EQ(actual, expected_missing);
    }
  }
}

TEST(CpuPredictor, IterationRange) {
  Context ctx;
  TestIterationRange(&ctx);
}

TEST(CpuPredictor, ExternalMemory) {
  Context ctx;
  bst_idx_t constexpr kRows{64};
  bst_feature_t constexpr kCols{12};
  auto dmat =
      RandomDataGenerator{kRows, kCols, 0.5f}.Batches(3).GenerateSparsePageDMatrix("temp", true);
  TestBasic(dmat.get(), &ctx);
}

TEST(CpuPredictor, InplacePredict) {
  bst_idx_t constexpr kRows{128};
  bst_feature_t constexpr kCols{64};
  Context ctx;
  auto gen = RandomDataGenerator{kRows, kCols, 0.5}.Device(ctx.Device());
  {
    HostDeviceVector<float> data;
    gen.GenerateDense(&data);
    ASSERT_EQ(data.Size(), kRows * kCols);
    std::shared_ptr<data::DMatrixProxy> x{new data::DMatrixProxy{}};
    auto array_interface = GetArrayInterface(&data, kRows, kCols);
    std::string arr_str;
    Json::Dump(array_interface, &arr_str);
    x->SetArray(arr_str.data());
    TestInplacePrediction(&ctx, x, kRows, kCols);
  }

  {
    HostDeviceVector<float> data;
    HostDeviceVector<std::size_t> rptrs;
    HostDeviceVector<bst_feature_t> columns;
    gen.GenerateCSR(&data, &rptrs, &columns);
    auto data_interface = GetArrayInterface(&data, kRows * kCols, 1);
    auto rptr_interface = GetArrayInterface(&rptrs, kRows + 1, 1);
    auto col_interface = GetArrayInterface(&columns, kRows * kCols, 1);
    std::string data_str, rptr_str, col_str;
    Json::Dump(data_interface, &data_str);
    Json::Dump(rptr_interface, &rptr_str);
    Json::Dump(col_interface, &col_str);
    std::shared_ptr<data::DMatrixProxy> x{new data::DMatrixProxy};
    x->SetCsr(rptr_str.data(), col_str.data(), data_str.data(), kCols, true);
    TestInplacePrediction(&ctx, x, kRows, kCols);
  }
}

namespace {
void TestTrainingPredictionCache(bool use_subsampling) {
  std::size_t constexpr kRows = 64, kCols = 16, kClasses = 4;
  LearnerModelState mparam{MakeMP(kCols, .0, kClasses)};
  Context ctx;

  std::unique_ptr<gbm::GBTree> gbm;
  gbm.reset(static_cast<gbm::GBTree*>(GradientBooster::Create("gbtree", &ctx, &mparam)));
  Args args{{"tree_method", "hist"}};
  if (use_subsampling) {
    args.emplace_back("subsample", "0.5");
  }
  gbm->Configure(args);

  auto dmat = RandomDataGenerator(kRows, kCols, 0).Classes(kClasses).GenerateDMatrix(true);

  GradientContainer gpair;
  gpair.gpair = linalg::Matrix<GradientPair>({kRows, kClasses}, ctx.Device());
  auto h_gpair = gpair.gpair.HostView();
  for (size_t i = 0; i < kRows * kClasses; ++i) {
    std::apply(h_gpair, linalg::UnravelIndex(i, kRows, kClasses)) = {static_cast<float>(i), 1};
  }

  // After one training iteration, GBTree's prediction cache contains cached predictions.
  gbm->DoBoost(dmat, &gpair, nullptr);
  auto const& prediction_cache = gbm->PredictionCache(dmat.get());

  HostDeviceVector<float> out_predictions;
  // perform prediction from scratch on the same input data, should be equal to cached result
  gbm->PredictBatch(dmat, &out_predictions, false, 0, 0);

  std::vector<float>& out_predictions_h = out_predictions.HostVector();
  auto const& prediction_cache_from_train = prediction_cache.predictions.ConstHostVector();
  for (size_t i = 0; i < out_predictions_h.size(); ++i) {
    ASSERT_NEAR(out_predictions_h[i], prediction_cache_from_train[i], kRtEps);
  }
}
}  // namespace

TEST(CPUPredictor, GHistIndexTraining) {
  size_t constexpr kRows{128}, kCols{16}, kBins{64};
  Context ctx;
  auto p_hist = RandomDataGenerator{kRows, kCols, 0.0}.Bins(kBins).GenerateQuantileDMatrix(false);
  HostDeviceVector<float> storage(kRows * kCols);
  auto columnar = RandomDataGenerator{kRows, kCols, 0.0}.GenerateArrayInterface(&storage);
  auto adapter = data::ArrayAdapter(columnar.c_str());
  std::shared_ptr<DMatrix> p_full{
      DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 1)};
  TestTrainingPrediction(&ctx, kRows, kBins, p_full, p_hist);
}

TEST(CPUPredictor, CategoricalPrediction) { TestCategoricalPrediction(false); }

TEST(CPUPredictor, CategoricalPredictLeaf) {
  Context ctx;
  TestCategoricalPredictLeaf(&ctx);
}

TEST(CpuPredictor, TrainingPredictionCache) {
  TestTrainingPredictionCache(false);
  TestTrainingPredictionCache(true);
}

TEST(CpuPredictor, LesserFeatures) {
  Context ctx;
  TestPredictionWithLesserFeatures(&ctx);
}

TEST(CpuPredictor, Sparse) {
  Context ctx;
  TestSparsePrediction(&ctx, 0.2);
  TestSparsePrediction(&ctx, 0.8);
}

TEST(CpuPredictor, Multi) {
  Context ctx;
  TestVectorLeafPrediction(&ctx);
}

TEST(CpuPredictor, Access) { TestPredictionDeviceAccess(); }
}  // namespace xgboost
