/**
 * Copyright 2017-2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/c_api.h>
#include <xgboost/learner.h>
#include <xgboost/logging.h>
#include <xgboost/predictor.h>

#include <string>

#include "../../../src/data/device_adapter.cuh"
#include "../../../src/data/proxy_dmatrix.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../collective/test_worker.h"  // for TestDistributedGlobal, BaseMGPUTest
#include "../helpers.h"
#include "test_predictor.h"

namespace xgboost::predictor {
TEST(GPUPredictor, Basic) {
  auto cpu_lparam = MakeCUDACtx(-1);
  auto gpu_lparam = MakeCUDACtx(0);

  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", &gpu_lparam));
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor", &cpu_lparam));

  gpu_predictor->Configure({});
  cpu_predictor->Configure({});

  for (size_t i = 1; i < 33; i *= 2) {
    int n_row = i, n_col = i;
    auto dmat = RandomDataGenerator(n_row, n_col, 0).GenerateDMatrix();

    auto ctx = MakeCUDACtx(0);
    LearnerModelParam mparam{MakeMP(n_col, .5, 1, ctx.Device())};
    gbm::GBTreeModel model = CreateTestModel(&mparam, &ctx);

    // Test predict batch
    PredictionCacheEntry gpu_out_predictions;
    PredictionCacheEntry cpu_out_predictions;

    gpu_predictor->InitOutPredictions(dmat->Info(), &gpu_out_predictions.predictions, model);
    gpu_predictor->PredictBatch(dmat.get(), &gpu_out_predictions, model, 0);
    cpu_predictor->InitOutPredictions(dmat->Info(), &cpu_out_predictions.predictions, model);
    cpu_predictor->PredictBatch(dmat.get(), &cpu_out_predictions, model, 0);

    std::vector<float>& gpu_out_predictions_h = gpu_out_predictions.predictions.HostVector();
    std::vector<float>& cpu_out_predictions_h = cpu_out_predictions.predictions.HostVector();
    float abs_tolerance = 0.001;
    for (size_t j = 0; j < gpu_out_predictions.predictions.Size(); j++) {
      ASSERT_NEAR(gpu_out_predictions_h[j], cpu_out_predictions_h[j], abs_tolerance);
    }
  }
}

namespace {
void VerifyBasicColumnSplit(std::array<std::vector<float>, 32> const& expected_result) {
  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();

  auto ctx = MakeCUDACtx(GPUIDX);
  std::unique_ptr<Predictor> predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", &ctx));
  predictor->Configure({});

  for (size_t i = 1; i < 33; i *= 2) {
    size_t n_row = i, n_col = i;
    auto dmat = RandomDataGenerator(n_row, n_col, 0).GenerateDMatrix();
    std::unique_ptr<DMatrix> sliced{dmat->SliceCol(world_size, rank)};

    LearnerModelParam mparam{MakeMP(n_col, .5, 1, ctx.Device())};
    gbm::GBTreeModel model = CreateTestModel(&mparam, &ctx);

    // Test predict batch
    PredictionCacheEntry out_predictions;

    predictor->InitOutPredictions(sliced->Info(), &out_predictions.predictions, model);
    predictor->PredictBatch(sliced.get(), &out_predictions, model, 0);

    std::vector<float>& out_predictions_h = out_predictions.predictions.HostVector();
    EXPECT_EQ(out_predictions_h, expected_result[i - 1]);
  }
}
}  // anonymous namespace

class MGPUPredictorTest : public collective::BaseMGPUTest {};

TEST_F(MGPUPredictorTest, BasicColumnSplit) {
  auto ctx = MakeCUDACtx(0);
  std::unique_ptr<Predictor> predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", &ctx));
  predictor->Configure({});

  std::array<std::vector<float>, 32> result{};
  for (size_t i = 1; i < 33; i *= 2) {
    size_t n_row = i, n_col = i;
    auto dmat = RandomDataGenerator(n_row, n_col, 0).GenerateDMatrix();

    LearnerModelParam mparam{MakeMP(n_col, .5, 1, ctx.Device())};
    gbm::GBTreeModel model = CreateTestModel(&mparam, &ctx);

    // Test predict batch
    PredictionCacheEntry out_predictions;

    predictor->InitOutPredictions(dmat->Info(), &out_predictions.predictions, model);
    predictor->PredictBatch(dmat.get(), &out_predictions, model, 0);

    std::vector<float>& out_predictions_h = out_predictions.predictions.HostVector();
    result[i - 1] = out_predictions_h;
  }

  this->DoTest([&] { VerifyBasicColumnSplit(result); }, true);
  this->DoTest([&] { VerifyBasicColumnSplit(result); }, false);
}

TEST(GPUPredictor, EllpackBasic) {
  size_t constexpr kCols{8};
  auto ctx = MakeCUDACtx(0);
  for (size_t bins = 2; bins < 258; bins += 16) {
    size_t rows = bins * 16;
    auto p_m = RandomDataGenerator{rows, kCols, 0.0}
                   .Bins(bins)
                   .Device(ctx.Device())
                   .GenerateQuantileDMatrix(false);
    ASSERT_FALSE(p_m->PageExists<SparsePage>());
    TestPredictionFromGradientIndex<EllpackPage>(&ctx, rows, kCols, p_m);
    TestPredictionFromGradientIndex<EllpackPage>(&ctx, bins, kCols, p_m);
  }
}

TEST(GPUPredictor, EllpackTraining) {
  auto ctx = MakeCUDACtx(0);
  size_t constexpr kRows{128}, kCols{16}, kBins{64};
  auto p_ellpack = RandomDataGenerator{kRows, kCols, 0.0}
                       .Bins(kBins)
                       .Device(ctx.Device())
                       .GenerateQuantileDMatrix(false);
  HostDeviceVector<float> storage(kRows * kCols);
  auto columnar =
      RandomDataGenerator{kRows, kCols, 0.0}.Device(ctx.Device()).GenerateArrayInterface(&storage);
  auto adapter = data::CupyAdapter(columnar);
  std::shared_ptr<DMatrix> p_full{
      DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 1)};
  TestTrainingPrediction(&ctx, kRows, kBins, p_full, p_ellpack);
}

namespace {
template <typename Create>
void TestDecisionStumpExternalMemory(Context const* ctx, bst_feature_t n_features,
                                     Create create_fn) {
  std::int32_t n_classes = 3;
  LearnerModelParam mparam{MakeMP(n_features, .5, n_classes, ctx->Device())};
  auto model = CreateTestModel(&mparam, ctx, n_classes);
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", ctx));
  gpu_predictor->Configure({});

  for (auto p_fmat : {create_fn(400), create_fn(800), create_fn(2048)}) {
    p_fmat->Info().base_margin_ = linalg::Constant(ctx, 0.5f, p_fmat->Info().num_row_, n_classes);
    PredictionCacheEntry out_predictions;
    gpu_predictor->InitOutPredictions(p_fmat->Info(), &out_predictions.predictions, model);
    gpu_predictor->PredictBatch(p_fmat.get(), &out_predictions, model, 0);
    ASSERT_EQ(out_predictions.predictions.Size(), p_fmat->Info().num_row_ * n_classes);
    auto const& h_predt = out_predictions.predictions.ConstHostVector();
    for (size_t i = 0; i < h_predt.size() / n_classes; i++) {
      ASSERT_EQ(h_predt[i * n_classes], 2.0);
      ASSERT_EQ(h_predt[i * n_classes + 1], 0.5);
      ASSERT_EQ(h_predt[i * n_classes + 2], 0.5);
    }
  }
}
}  // namespace

TEST(GPUPredictor, ExternalMemory) {
  auto ctx = MakeCUDACtx(0);

  bst_bin_t max_bin = 128;
  bst_feature_t n_features = 32;

  TestDecisionStumpExternalMemory(&ctx, n_features, [&](bst_idx_t n_samples) {
    return RandomDataGenerator{n_samples, n_features, 0.0f}
        .Batches(4)
        .Device(ctx.Device())
        .Bins(max_bin)
        .GenerateSparsePageDMatrix("temp", false);
  });
  TestDecisionStumpExternalMemory(&ctx, n_features, [&](bst_idx_t n_samples) {
    return RandomDataGenerator{n_samples, n_features, 0.0f}
        .Batches(4)
        .Device(ctx.Device())
        .Bins(max_bin)
        .GenerateExtMemQuantileDMatrix("temp", false);
  });
}

TEST(GPUPredictor, InplacePredictCupy) {
  auto ctx = MakeCUDACtx(0);
  size_t constexpr kRows{128}, kCols{64};
  RandomDataGenerator gen(kRows, kCols, 0.5);
  gen.Device(ctx.Device());
  HostDeviceVector<float> data;
  std::string interface_str = gen.GenerateArrayInterface(&data);
  std::shared_ptr<DMatrix> p_fmat{new data::DMatrixProxy};
  dynamic_cast<data::DMatrixProxy*>(p_fmat.get())->SetCUDAArray(interface_str.c_str());
  TestInplacePrediction(&ctx, p_fmat, kRows, kCols);
}

TEST(GPUPredictor, InplacePredictCuDF) {
  auto ctx = MakeCUDACtx(0);
  size_t constexpr kRows{128}, kCols{64};
  RandomDataGenerator gen(kRows, kCols, 0.5);
  gen.Device(ctx.Device());
  std::vector<HostDeviceVector<float>> storage(kCols);
  auto interface_str = gen.GenerateColumnarArrayInterface(&storage);
  std::shared_ptr<DMatrix> p_fmat{new data::DMatrixProxy};
  dynamic_cast<data::DMatrixProxy*>(p_fmat.get())->SetCUDAArray(interface_str.c_str());
  TestInplacePrediction(&ctx, p_fmat, kRows, kCols);
}

TEST(GpuPredictor, LesserFeatures) {
  auto ctx = MakeCUDACtx(0);
  TestPredictionWithLesserFeatures(&ctx);
}

TEST_F(MGPUPredictorTest, LesserFeaturesColumnSplit) {
  this->DoTest([] { TestPredictionWithLesserFeaturesColumnSplit(true); }, true);
  this->DoTest([] { TestPredictionWithLesserFeaturesColumnSplit(true); }, false);
}

// Very basic test of empty model
TEST(GPUPredictor, ShapStump) {
  cudaSetDevice(0);

  auto ctx = MakeCUDACtx(0);
  LearnerModelParam mparam{MakeMP(1, .5, 1, ctx.Device())};
  gbm::GBTreeModel model(&mparam, &ctx);

  std::vector<std::unique_ptr<RegTree>> trees;
  trees.push_back(std::make_unique<RegTree>());
  model.CommitModelGroup(std::move(trees), 0);

  auto gpu_lparam = MakeCUDACtx(0);
  std::unique_ptr<Predictor> gpu_predictor = std::unique_ptr<Predictor>(
      Predictor::Create("gpu_predictor", &gpu_lparam));
  gpu_predictor->Configure({});
  HostDeviceVector<float> predictions;
  auto dmat = RandomDataGenerator(3, 1, 0).GenerateDMatrix();
  gpu_predictor->PredictContribution(dmat.get(), &predictions, model);
  auto& phis = predictions.HostVector();
  auto base_score = mparam.BaseScore(DeviceOrd::CPU())(0);
  EXPECT_EQ(phis[0], 0.0);
  EXPECT_EQ(phis[1], base_score);
  EXPECT_EQ(phis[2], 0.0);
  EXPECT_EQ(phis[3], base_score);
  EXPECT_EQ(phis[4], 0.0);
  EXPECT_EQ(phis[5], base_score);
}

TEST(GPUPredictor, Shap) {
  auto ctx = MakeCUDACtx(0);
  LearnerModelParam mparam{MakeMP(1, .5, 1, ctx.Device())};
  gbm::GBTreeModel model(&mparam, &ctx);

  std::vector<std::unique_ptr<RegTree>> trees;
  trees.push_back(std::make_unique<RegTree>());
  trees[0]->ExpandNode(0, 0, 0.5, true, 1.0, -1.0, 1.0, 0.0, 5.0, 2.0, 3.0);
  model.CommitModelGroup(std::move(trees), 0);

  auto cpu_lparam = MakeCUDACtx(-1);
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", &ctx));
  std::unique_ptr<Predictor> cpu_predictor = std::unique_ptr<Predictor>(
      Predictor::Create("cpu_predictor", &cpu_lparam));
  gpu_predictor->Configure({});
  cpu_predictor->Configure({});
  HostDeviceVector<float> predictions;
  HostDeviceVector<float> cpu_predictions;
  auto dmat = RandomDataGenerator(3, 1, 0).GenerateDMatrix();
  gpu_predictor->PredictContribution(dmat.get(), &predictions, model);
  cpu_predictor->PredictContribution(dmat.get(), &cpu_predictions, model);
  auto& phis = predictions.HostVector();
  auto& cpu_phis = cpu_predictions.HostVector();
  for (auto i = 0ull; i < phis.size(); i++) {
    EXPECT_NEAR(cpu_phis[i], phis[i], 1e-3);
  }
}

TEST_P(ShapExternalMemoryTest, GPUPredictor) {
  auto ctx = MakeCUDACtx(0);
  auto [is_qdm, is_interaction] = this->GetParam();
  this->Run(&ctx, is_qdm, is_interaction);
}

TEST(GPUPredictor, IterationRange) {
  auto ctx = MakeCUDACtx(0);
  TestIterationRange(&ctx);
}

TEST_F(MGPUPredictorTest, IterationRangeColumnSplit) {
  TestIterationRangeColumnSplit(curt::AllVisibleGPUs(), true);
}

TEST(GPUPredictor, CategoricalPrediction) {
  TestCategoricalPrediction(true, false);
}

TEST_F(MGPUPredictorTest, CategoricalPredictionColumnSplit) {
  this->DoTest([] { TestCategoricalPrediction(true, true); }, true);
  this->DoTest([] { TestCategoricalPrediction(true, true); }, false);
}

TEST(GPUPredictor, CategoricalPredictLeaf) {
  auto ctx = MakeCUDACtx(curt::AllVisibleGPUs() == 1 ? 0 : collective::GetRank());
  TestCategoricalPredictLeaf(&ctx, false);
}

TEST_F(MGPUPredictorTest, CategoricalPredictionLeafColumnSplit) {
  this->DoTest(
      [&] {
        auto ctx = MakeCUDACtx(collective::GetRank());
        TestCategoricalPredictLeaf(&ctx, true);
      },
      true);
  this->DoTest(
      [&] {
        auto ctx = MakeCUDACtx(collective::GetRank());
        TestCategoricalPredictLeaf(&ctx, true);
      },
      false);
}

TEST(GPUPredictor, PredictLeafBasic) {
  size_t constexpr kRows = 5, kCols = 5;
  auto dmat = RandomDataGenerator(kRows, kCols, 0).Device(DeviceOrd::CUDA(0)).GenerateDMatrix();
  auto lparam = MakeCUDACtx(GPUIDX);
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", &lparam));
  gpu_predictor->Configure({});

  LearnerModelParam mparam{MakeMP(kCols, .0, 1)};
  Context ctx;
  gbm::GBTreeModel model = CreateTestModel(&mparam, &ctx);

  HostDeviceVector<float> leaf_out_predictions;
  gpu_predictor->PredictLeaf(dmat.get(), &leaf_out_predictions, model);
  auto const& h_leaf_out_predictions = leaf_out_predictions.ConstHostVector();
  for (auto v : h_leaf_out_predictions) {
    ASSERT_EQ(v, 0);
  }
}

TEST(GPUPredictor, Sparse) {
  auto ctx = MakeCUDACtx(0);
  TestSparsePrediction(&ctx, 0.2);
  TestSparsePrediction(&ctx, 0.8);
}

TEST_F(MGPUPredictorTest, SparseColumnSplit) {
  TestSparsePredictionColumnSplit(curt::AllVisibleGPUs(), true, 0.2);
  TestSparsePredictionColumnSplit(curt::AllVisibleGPUs(), true, 0.8);
}
}  // namespace xgboost::predictor
