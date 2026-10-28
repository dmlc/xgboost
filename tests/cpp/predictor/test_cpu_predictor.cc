/**
 * Copyright 2017-2026, XGBoost contributors
 */
#include <gtest/gtest.h>

#include <algorithm>  // for generate
#include <cmath>      // for isnan
#include <limits>
#include <memory>  // for make_unique
#include <random>  // for mt19937
#include <set>     // for set
#include <vector>  // for vector

#include "../../../src/collective/communicator-inl.h"
#include "../../../src/common/categorical.h"  // for CatBitField
#include "../../../src/common/kernel.h"
#include "../../../src/data/adapter.h"
#include "../../../src/data/proxy_dmatrix.h"
#include "../../../src/gbm/gbtree.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../../../src/predictor/array_tree_layout.h"
#include "../../../src/predictor/predict_fn.h"
#include "../../../src/predictor/prediction_kernel.h"
#include "../../../src/tree/tree_view.h"
#include "../collective/test_worker.h"  // for TestDistributedGlobal
#include "../helpers.h"
#include "test_predictor.h"
#include "test_shap.h"

namespace xgboost {
TEST(CpuPredictor, InitOutPredictions) {
  Context ctx;
  TestInitOutPredictions(&ctx);
}

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
  model->trees.front()->Expand(
      {{0, 0, 0.5f, true}, {0.0f, 2.0f}, {1.0f, 1.0f}, {2.0f, 1.0f}, 0.0f});
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

namespace {
template <bst_node_t kDepth>
using LayoutForTest = predictor::ArrayTreeLayout<false, true, kDepth, tree::ScalarTreeView>;
}

TEST(CpuPredictor, ArrayTreeLayout) {
  Context ctx;

  RegTree tree;
  bst_node_t n_nodes = 15;  // 2^4 - 1
  for (bst_node_t nid = 0; nid < n_nodes; ++nid) {
    // Some place-holders
    bst_feature_t split_index = nid + 1;
    bst_float split_cond = nid + 2;
    bool default_left = nid % 2 == 0;

    tree.Expand({{nid, split_index, split_cond, default_left}, {0, 0}, {0, 0}, {0, 0}, 0});
  }

  auto sc_tree = tree::ScalarTreeView{ctx.Device(), false, &tree};
  {
    constexpr bst_node_t kDepth = 1;
    LayoutForTest<kDepth> buffer(sc_tree, sc_tree.GetCategoriesMatrix());
    CheckArrayLayout(tree, buffer, kDepth, 0, 0, 0);
  }
  {
    constexpr bst_node_t kDepth = 2;
    LayoutForTest<kDepth> buffer{sc_tree, sc_tree.GetCategoriesMatrix()};
    CheckArrayLayout(tree, buffer, kDepth, 0, 0, 0);
  }
  {
    constexpr bst_node_t kDepth = 3;
    LayoutForTest<kDepth> buffer{sc_tree, sc_tree.GetCategoriesMatrix()};
    CheckArrayLayout(tree, buffer, kDepth, 0, 0, 0);
  }
  {
    constexpr bst_node_t kDepth = 4;
    LayoutForTest<kDepth> buffer{sc_tree, sc_tree.GetCategoriesMatrix()};
    CheckArrayLayout(tree, buffer, kDepth, 0, 0, 0);
  }
  {
    constexpr bst_node_t kDepth = 5;
    LayoutForTest<kDepth> buffer{sc_tree, sc_tree.GetCategoriesMatrix()};
    CheckArrayLayout(tree, buffer, kDepth, 0, 0, 0);
  }
}

namespace {
bst_feature_t constexpr kMixedCols = 4;

// Alternate categorical and numerical splits, with some shallow leaves for the layout to pad.
void GrowMixedTree(RegTree* tree, bst_node_t nidx, bst_node_t depth, bst_node_t max_depth) {
  if (depth == max_depth || (depth > 1 && nidx % 5 == 0)) {
    return;
  }
  bst_feature_t fidx = depth % kMixedCols;
  std::vector<std::uint32_t> cats;
  if (fidx % 2 == 0) {
    // Two words, so that larger categories are outside of the bit field.
    cats.resize(2);
    common::CatBitField bits{common::Span<std::uint32_t>{cats}};
    bits.Set(nidx % 8);
    bits.Set(33);
  }
  auto left_weight = static_cast<float>(2 * nidx + 1);
  auto right_weight = static_cast<float>(2 * nidx + 2);
  tree->Expand({{nidx, fidx, 3.5f, nidx % 2 == 0, cats},
                {0.0f, 1.0f},
                {left_weight, 1.0f},
                {right_weight, 1.0f},
                0.0f});
  bst_node_t left = (*tree)[nidx].LeftChild();
  bst_node_t right = (*tree)[nidx].RightChild();
  GrowMixedTree(tree, left, depth + 1, max_depth);
  GrowMixedTree(tree, right, depth + 1, max_depth);
}

// Valid categories in and out of the bit field, invalid categories, and optionally missing values.
std::vector<float> MixedRows(std::size_t n_rows, bool any_missing) {
  std::vector<float> values{0.0f,  1.0f,  2.0f,  3.5f,    4.0f,  5.0f,
                            7.0f,  33.0f, 34.0f, 63.0f,   64.0f, 1000.0f,
                            -1.0f, 1e8f};
  if (any_missing) {
    values.push_back(std::numeric_limits<float>::quiet_NaN());
  }
  std::mt19937 rng{0};
  std::vector<float> rows(n_rows * kMixedCols);
  std::generate(rows.begin(), rows.end(), [&] { return values[rng() % values.size()]; });
  return rows;
}

template <int kDepth, bool any_missing>
void CheckLayoutMatchesWalk(tree::ScalarTreeView const& tree, std::vector<float> const& rows) {
  auto const& cats = tree.GetCategoriesMatrix();
  std::size_t constexpr kBlockSize = 64;
  std::vector<RegTree::FVec> feats(kBlockSize);
  for (std::size_t i = 0; i < kBlockSize; ++i) {
    std::vector<Entry> entries;
    for (bst_feature_t f = 0; f < kMixedCols; ++f) {
      auto fvalue = rows[i * kMixedCols + f];
      if (!std::isnan(fvalue)) {
        entries.emplace_back(f, fvalue);
      }
    }
    feats[i].Init(kMixedCols);
    feats[i].Fill(common::Span<Entry const>{entries});
  }

  predictor::ArrayTreeLayout<true, any_missing, kDepth, tree::ScalarTreeView> layout{tree, cats};
  std::vector<bst_node_t> nidx(kBlockSize, 0);
  layout.Process(common::Span{feats}, kBlockSize, nidx.data());

  for (std::size_t i = 0; i < kBlockSize; ++i) {
    bst_node_t expected = RegTree::kRoot;
    for (int depth = 0; depth < kDepth && !tree.IsLeaf(expected); ++depth) {
      auto fvalue = feats[i].GetFvalue(tree.SplitIndex(expected));
      expected = predictor::GetNextNode<any_missing, true>(tree, expected, fvalue, cats);
    }
    ASSERT_EQ(nidx[i], expected) << "depth: " << kDepth << ", row: " << i;
  }
}

template <bool any_missing>
void CheckMixedLayout(tree::ScalarTreeView const& tree) {
  auto rows = MixedRows(64, any_missing);
  CheckLayoutMatchesWalk<1, any_missing>(tree, rows);
  CheckLayoutMatchesWalk<2, any_missing>(tree, rows);
  CheckLayoutMatchesWalk<3, any_missing>(tree, rows);
  CheckLayoutMatchesWalk<4, any_missing>(tree, rows);
  CheckLayoutMatchesWalk<5, any_missing>(tree, rows);
  CheckLayoutMatchesWalk<6, any_missing>(tree, rows);
}
}  // namespace

TEST(CpuPredictor, ArrayTreeLayoutCategorical) {
  Context ctx;
  RegTree tree;
  GrowMixedTree(&tree, RegTree::kRoot, 0, 8);
  tree.FinalizeLeaves(1.0f);
  auto sc_tree = tree::ScalarTreeView{ctx.Device(), false, &tree};
  CheckMixedLayout<false>(sc_tree);
  CheckMixedLayout<true>(sc_tree);
}

TEST(CpuPredictor, CategoricalBlockPrediction) {
  // More than one block of rows, for any number of threads.
  std::size_t constexpr kRows = 1024;
  for (bool any_missing : {false, true}) {
    Context ctx;
    LearnerModelState mparam{MakeMP(kMixedCols, .5, 1, ctx.Device())};
    gbm::GBTreeModel model(&mparam, &ctx);
    std::vector<std::unique_ptr<RegTree>> trees;
    trees.push_back(std::make_unique<RegTree>());
    // Deeper than the array layout, so that traversal also continues in the original tree.
    GrowMixedTree(trees.front().get(), RegTree::kRoot, 0, 8);
    trees.front()->FinalizeLeaves(1.0f);
    model.CommitModelGroup(std::move(trees), 0);
    auto const tree = model.trees.front()->HostScView();
    ASSERT_GT(tree.MaxDepth(), 6);

    auto m = GetDMatrixFromData(MixedRows(kRows, any_missing), kRows, kMixedCols);
    ASSERT_EQ(m->IsDense(), !any_missing);

    HostDeviceVector<float> predt;
    predictor::InitOutPredictions(&ctx, m->Info(), &predt, model);
    common::DispatchKernel<predictor::PredictBatchKernel>(&ctx, m.get(), &predt, model, 0, 0,
                                                          nullptr);
    // Leaf prediction always walks the tree one row at a time.
    HostDeviceVector<float> leaves;
    common::DispatchKernel<predictor::PredictLeafKernel>(&ctx, m.get(), &leaves, model, 0);

    auto score = mparam.BaseScore(DeviceOrd::CPU())(0);
    auto const& h_predt = predt.ConstHostVector();
    auto const& h_leaves = leaves.ConstHostVector();
    ASSERT_EQ(h_predt.size(), kRows);
    ASSERT_EQ(h_leaves.size(), kRows);
    for (std::size_t i = 0; i < kRows; ++i) {
      auto leaf = static_cast<bst_node_t>(h_leaves[i]);
      ASSERT_FLOAT_EQ(h_predt[i], tree.LeafValue(leaf) + score) << "row: " << i;
    }
    ASSERT_GT(std::set<float>(h_leaves.cbegin(), h_leaves.cend()).size(), 16);
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
