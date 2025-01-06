/**
 * Copyright 2020-2025, XGBoost Contributors
 */
#include "test_predictor.h"

#include <gtest/gtest.h>
#include <xgboost/context.h>             // for Context
#include <xgboost/data.h>                // for DMatrix, BatchIterator, BatchSet, MetaInfo
#include <xgboost/host_device_vector.h>  // for HostDeviceVector
#include <xgboost/json.h>                // for Json
#include <xgboost/predictor.h>           // for PredictionCacheEntry, Predictor, Predic...
#include <xgboost/string_view.h>         // for StringView

#include <limits>         // for numeric_limits
#include <memory>         // for shared_ptr
#include <unordered_map>  // for unordered_map

#include "../../../src/common/bitfield.h"         // for LBitField32
#include "../../../src/data/iterative_dmatrix.h"  // for IterativeDMatrix
#include "../../../src/data/proxy_dmatrix.h"      // for DMatrixProxy
#include "../collective/test_worker.h"            // for TestDistributedGlobal
#include "../helpers.h"                           // for GetDMatrixFromData, RandomDataGenerator
#include "xgboost/json.h"                         // for Json, Object, get, String
#include "xgboost/linalg.h"                       // for MakeVec, Tensor, TensorView, Vector
#include "xgboost/logging.h"                      // for CHECK
#include "xgboost/span.h"                         // for operator!=, SpanIterator, Span
#include "xgboost/tree_model.h"                   // for RegTree

namespace xgboost {
void TestBasic(DMatrix* dmat, Context const *ctx) {
  auto predictor = std::unique_ptr<Predictor>(CreatePredictorForTest(ctx));

  size_t const kRows = dmat->Info().num_row_;
  size_t const kCols = dmat->Info().num_col_;

  LearnerModelParam mparam{MakeMP(kCols, .0, 1)};

  gbm::GBTreeModel model = CreateTestModel(&mparam, ctx);

  // Test predict batch
  PredictionCacheEntry out_predictions;
  predictor->InitOutPredictions(dmat->Info(), &out_predictions.predictions, model);
  predictor->PredictBatch(dmat, &out_predictions, model, 0);

  std::vector<float>& out_predictions_h = out_predictions.predictions.HostVector();
  for (size_t i = 0; i < out_predictions.predictions.Size(); i++) {
    ASSERT_EQ(out_predictions_h[i], 1.5);
  }

  // Test predict leaf
  HostDeviceVector<float> leaf_out_predictions;
  predictor->PredictLeaf(dmat, &leaf_out_predictions, model);
  auto const& h_leaf_out_predictions = leaf_out_predictions.ConstHostVector();
  for (auto v : h_leaf_out_predictions) {
    ASSERT_EQ(v, 0);
  }

  if (dmat->Info().IsColumnSplit()) {
    // Predict contribution is not supported for column split.
    return;
  }

  // Test predict contribution
  HostDeviceVector<float> out_contribution_hdv;
  auto& out_contribution = out_contribution_hdv.HostVector();
  predictor->PredictContribution(dmat, &out_contribution_hdv, model);
  ASSERT_EQ(out_contribution.size(), kRows * (kCols + 1));
  for (size_t i = 0; i < out_contribution.size(); ++i) {
    auto const& contri = out_contribution[i];
    // shift 1 for bias, as test tree is a decision dump, only global bias is
    // filled with LeafValue().
    if ((i + 1) % (kCols + 1) == 0) {
      ASSERT_EQ(out_contribution.back(), 1.5f);
    } else {
      ASSERT_EQ(contri, 0);
    }
  }
  // Test predict contribution (approximate method)
  predictor->PredictContribution(dmat, &out_contribution_hdv, model, 0, nullptr, true);
  for (size_t i = 0; i < out_contribution.size(); ++i) {
    auto const& contri = out_contribution[i];
    // shift 1 for bias, as test tree is a decision dump, only global bias is
    // filled with LeafValue().
    if ((i + 1) % (kCols + 1) == 0) {
      ASSERT_EQ(out_contribution.back(), 1.5f);
    } else {
      ASSERT_EQ(contri, 0);
    }
  }
}

TEST(Predictor, PredictionCache) {
  size_t constexpr kRows = 16, kCols = 4;

  PredictionContainer container;
  DMatrix *m;
  // Add a cache that is immediately expired.
  auto add_cache = [&]() {
    auto p_dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
    container.Cache(p_dmat, DeviceOrd::CPU());
    m = p_dmat.get();
  };

  add_cache();
  ASSERT_EQ(container.Container().size(), 0ul);
  add_cache();
  EXPECT_ANY_THROW(container.Entry(m));
}

void TestTrainingPrediction(Context const *ctx, size_t rows, size_t bins,
                            std::shared_ptr<DMatrix> p_full, std::shared_ptr<DMatrix> p_hist) {
  size_t constexpr kCols = 16;
  size_t constexpr kClasses = 3;
  size_t constexpr kIters = 3;

  std::unique_ptr<Learner> learner;

  p_hist->Info().labels.Reshape(rows, 1);
  auto &h_label = p_hist->Info().labels.Data()->HostVector();

  for (size_t i = 0; i < rows; ++i) {
    h_label[i] = i % kClasses;
  }

  learner.reset(Learner::Create({}));
  learner->SetParams(Args{{"objective", "multi:softprob"},
                          {"num_feature", std::to_string(kCols)},
                          {"num_class", std::to_string(kClasses)},
                          {"max_bin", std::to_string(bins)},
                          {"device", ctx->IsSycl() ? "cpu" : ctx->DeviceName()}});
  learner->Configure();

  for (size_t i = 0; i < kIters; ++i) {
    learner->UpdateOneIter(i, p_hist);
  }

  Json model{Object{}};
  learner->SaveModel(&model);

  learner.reset(Learner::Create({}));
  learner->LoadModel(model);
  learner->SetParam("device", ctx->DeviceName());
  learner->Configure();

  HostDeviceVector<float> from_full;
  learner->Predict(p_full, false, &from_full, 0, 0);

  HostDeviceVector<float> from_hist;
  learner->Predict(p_hist, false, &from_hist, 0, 0);

  for (size_t i = 0; i < rows; ++i) {
    EXPECT_NEAR(from_hist.ConstHostVector()[i], from_full.ConstHostVector()[i], kRtEps);
  }

  // Contributions
  HostDeviceVector<float> from_full_contribs;
  learner->Predict(p_full, false, &from_full_contribs, 0, 0, false, false, true);
  HostDeviceVector<float> from_hist_contribs;
  learner->Predict(p_hist, false, &from_hist_contribs, 0, 0, false, false, true);
  for (size_t i = 0; i < from_full_contribs.ConstHostVector().size(); ++i) {
    EXPECT_NEAR(from_hist_contribs.ConstHostVector()[i], from_full_contribs.ConstHostVector()[i],
                kRtEps);
  }

  // Contributions (approximate method)
  HostDeviceVector<float> from_full_approx_contribs;
  learner->Predict(p_full, false, &from_full_approx_contribs, 0, 0, false, false, false, true);
  HostDeviceVector<float> from_hist_approx_contribs;
  learner->Predict(p_hist, false, &from_hist_approx_contribs, 0, 0, false, false, false, true);
  for (size_t i = 0; i < from_full_approx_contribs.ConstHostVector().size(); ++i) {
    EXPECT_NEAR(from_hist_approx_contribs.ConstHostVector()[i],
                from_full_approx_contribs.ConstHostVector()[i], kRtEps);
  }
}

void TestInplacePrediction(Context const *ctx, std::shared_ptr<DMatrix> x, bst_idx_t rows,
                           bst_feature_t cols) {
  std::size_t constexpr kClasses { 4 };
  auto gen = RandomDataGenerator{rows, cols, 0.5}.Device(ctx->Device()).Classes(kClasses);
  std::shared_ptr<DMatrix> m = gen.GenerateDMatrix(true);

  std::unique_ptr<Learner> learner {
    Learner::Create({m})
  };

  learner->SetParam("num_parallel_tree", "4");
  learner->SetParam("num_class", std::to_string(kClasses));
  learner->SetParam("seed", "0");
  learner->SetParam("subsample", "0.5");
  learner->SetParam("tree_method", "hist");
  for (int32_t it = 0; it < 4; ++it) {
    learner->UpdateOneIter(it, m);
  }

  learner->SetParam("device", ctx->DeviceName());
  learner->Configure();

  HostDeviceVector<float> *p_out_predictions_0{nullptr};
  learner->InplacePredict(x, PredictionType::kMargin, std::numeric_limits<float>::quiet_NaN(),
                          &p_out_predictions_0, 0, 2);
  CHECK(p_out_predictions_0);
  HostDeviceVector<float> predict_0 (p_out_predictions_0->Size());
  predict_0.Copy(*p_out_predictions_0);

  HostDeviceVector<float> *p_out_predictions_1{nullptr};
  learner->InplacePredict(x, PredictionType::kMargin, std::numeric_limits<float>::quiet_NaN(),
                          &p_out_predictions_1, 2, 4);
  CHECK(p_out_predictions_1);
  HostDeviceVector<float> predict_1 (p_out_predictions_1->Size());
  predict_1.Copy(*p_out_predictions_1);

  HostDeviceVector<float>* p_out_predictions{nullptr};
  learner->InplacePredict(x, PredictionType::kMargin, std::numeric_limits<float>::quiet_NaN(),
                          &p_out_predictions, 0, 4);

  auto& h_pred = p_out_predictions->HostVector();
  auto& h_pred_0 = predict_0.HostVector();
  auto& h_pred_1 = predict_1.HostVector();

  ASSERT_EQ(h_pred.size(), rows * kClasses);
  ASSERT_EQ(h_pred.size(), h_pred_0.size());
  ASSERT_EQ(h_pred.size(), h_pred_1.size());
  for (size_t i = 0; i < h_pred.size(); ++i) {
    // Need to remove the global bias here.
    ASSERT_NEAR(h_pred[i], h_pred_0[i] + h_pred_1[i] - 0.5f, kRtEps);
  }

  learner->SetParam("device", "cpu");
  learner->Configure();
}

namespace {
std::unique_ptr<Learner> LearnerForTest(Context const *ctx, std::shared_ptr<DMatrix> dmat,
                                        size_t iters, size_t forest = 1) {
  std::unique_ptr<Learner> learner{Learner::Create({dmat})};
  learner->SetParams(
      Args{{"num_parallel_tree", std::to_string(forest)}, {"device", ctx->IsSycl() ? "cpu" : ctx->DeviceName()}});
  for (size_t i = 0; i < iters; ++i) {
    learner->UpdateOneIter(i, dmat);
  }

  return learner;
}

void VerifyPredictionWithLesserFeatures(Learner *learner, bst_idx_t kRows,
                                        std::shared_ptr<DMatrix> m_test,
                                        std::shared_ptr<DMatrix> m_invalid) {
  HostDeviceVector<float> prediction;
  Json config{Object()};
  learner->SaveConfig(&config);

  learner->Predict(m_test, false, &prediction, 0, 0);
  ASSERT_EQ(prediction.Size(), kRows);

  ASSERT_THROW({ learner->Predict(m_invalid, false, &prediction, 0, 0); }, dmlc::Error);
}

}  // anonymous namespace

void TestPredictionWithLesserFeatures(Context const *ctx) {
  size_t constexpr kRows = 256, kTrainCols = 256, kTestCols = 4, kIters = 4;
  auto m_train = RandomDataGenerator(kRows, kTrainCols, 0.5).GenerateDMatrix(true);
  auto learner = LearnerForTest(ctx, m_train, kIters);
  auto m_test = RandomDataGenerator(kRows, kTestCols, 0.5).GenerateDMatrix(false);
  auto m_invalid = RandomDataGenerator(kRows, kTrainCols + 1, 0.5).GenerateDMatrix(false);
  VerifyPredictionWithLesserFeatures(learner.get(), kRows, m_test, m_invalid);
}

void TestPredictionDeviceAccess() {
  Context ctx;
  size_t constexpr kRows = 256, kTrainCols = 256, kTestCols = 4, kIters = 4;
  auto m_train = RandomDataGenerator(kRows, kTrainCols, 0.5).GenerateDMatrix(true);
  auto m_test = RandomDataGenerator(kRows, kTestCols, 0.5).GenerateDMatrix(false);
  auto learner = LearnerForTest(&ctx, m_train, kIters);

  HostDeviceVector<float> from_cpu;
  {
    ASSERT_TRUE(from_cpu.Device().IsCPU());
    Context cpu_ctx;
    learner->SetParam("device", cpu_ctx.DeviceName());
    learner->Predict(m_test, false, &from_cpu, 0, 0);
    ASSERT_TRUE(from_cpu.HostCanWrite());
    ASSERT_FALSE(from_cpu.DeviceCanRead());
  }

#if defined(XGBOOST_USE_CUDA)
  HostDeviceVector<float> from_cuda;
  {
    Context cuda_ctx = MakeCUDACtx(0);
    learner->SetParam("device", cuda_ctx.DeviceName());
    learner->Predict(m_test, false, &from_cuda, 0, 0);
    ASSERT_EQ(from_cuda.Device(), DeviceOrd::CUDA(0));
    ASSERT_TRUE(from_cuda.DeviceCanWrite());
    ASSERT_FALSE(from_cuda.HostCanRead());
  }

  auto const &h_cpu = from_cpu.ConstHostVector();
  auto const &h_gpu = from_cuda.ConstHostVector();
  for (size_t i = 0; i < h_cpu.size(); ++i) {
    ASSERT_NEAR(h_cpu[i], h_gpu[i], kRtEps);
  }
#endif  // defined(XGBOOST_USE_CUDA)
}

void TestPredictionWithLesserFeaturesColumnSplit(bool use_gpu) {
  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();

  std::size_t constexpr kRows = 256, kTrainCols = 256, kTestCols = 4, kIters = 4;
  auto m_train = RandomDataGenerator(kRows, kTrainCols, 0.5).Seed(rank).GenerateDMatrix(true);
  Context ctx;
  if (use_gpu) {
    ctx = MakeCUDACtx(curt::AllVisibleGPUs() == 1 ? 0 : rank);
  }
  auto learner = LearnerForTest(&ctx, m_train, kIters);
  auto m_test = RandomDataGenerator(kRows, kTestCols, 0.5).GenerateDMatrix(false);
  auto m_invalid = RandomDataGenerator(kRows, kTrainCols + 1, 0.5).GenerateDMatrix(false);

  std::shared_ptr<DMatrix> sliced_test{m_test->SliceCol(world_size, rank)};
  std::shared_ptr<DMatrix> sliced_invalid{m_invalid->SliceCol(world_size, rank)};

  VerifyPredictionWithLesserFeatures(learner.get(), kRows, sliced_test, sliced_invalid);
}

void GBTreeModelForTest(gbm::GBTreeModel *model, uint32_t split_ind,
                        bst_cat_t split_cat, float left_weight,
                        float right_weight) {
  PredictionCacheEntry out_predictions;

  std::vector<std::unique_ptr<RegTree>> trees;
  trees.push_back(std::unique_ptr<RegTree>(new RegTree));
  auto& p_tree = trees.front();

  std::vector<uint32_t> split_cats(LBitField32::ComputeStorageSize(split_cat));
  LBitField32 cats_bits(split_cats);
  cats_bits.Set(split_cat);

  p_tree->ExpandCategorical(0, split_ind, split_cats, true, 1.5f,
                            left_weight, right_weight,
                            3.0f, 2.2f, 7.0f, 9.0f);
  model->CommitModelGroup(std::move(trees), 0);
}

void TestCategoricalPrediction(bool use_gpu, bool is_column_split) {
  Context ctx;
  if (use_gpu) {
    ctx = MakeCUDACtx(curt::AllVisibleGPUs() == 1 ? 0 : collective::GetRank());
  }
  size_t constexpr kCols = 10;
  PredictionCacheEntry out_predictions;

  LearnerModelParam mparam{MakeMP(kCols, .5, 1)};
  uint32_t split_ind = 3;
  bst_cat_t split_cat = 4;
  float left_weight = 1.3f;
  float right_weight = 1.7f;

  gbm::GBTreeModel model(&mparam, &ctx);
  GBTreeModelForTest(&model, split_ind, split_cat, left_weight, right_weight);

  std::unique_ptr<Predictor> predictor{CreatePredictorForTest(&ctx)};

  std::vector<float> row(kCols);
  row[split_ind] = split_cat;
  auto m = GetDMatrixFromData(row, 1, kCols);

  std::vector<FeatureType> types(10, FeatureType::kCategorical);
  m->Info().feature_types.HostVector() = types;
  if (is_column_split) {
    m = std::shared_ptr<DMatrix>{m->SliceCol(collective::GetWorldSize(), collective::GetRank())};
  }

  predictor->InitOutPredictions(m->Info(), &out_predictions.predictions, model);
  predictor->PredictBatch(m.get(), &out_predictions, model, 0);
  auto score = mparam.BaseScore(DeviceOrd::CPU())(0);
  ASSERT_EQ(out_predictions.predictions.Size(), 1ul);
  ASSERT_EQ(out_predictions.predictions.HostVector()[0],
            right_weight + score);  // go to right for matching cat

  row[split_ind] = split_cat + 1;
  m = GetDMatrixFromData(row, 1, kCols);
  if (is_column_split) {
    m = std::shared_ptr<DMatrix>{m->SliceCol(collective::GetWorldSize(), collective::GetRank())};
  }
  out_predictions.version = 0;
  predictor->InitOutPredictions(m->Info(), &out_predictions.predictions, model);
  predictor->PredictBatch(m.get(), &out_predictions, model, 0);
  ASSERT_EQ(out_predictions.predictions.HostVector()[0], left_weight + score);
}

void TestCategoricalPredictLeaf(Context const *ctx, bool is_column_split) {
  size_t constexpr kCols = 10;
  PredictionCacheEntry out_predictions;

  LearnerModelParam mparam{MakeMP(kCols, .5, 1)};

  uint32_t split_ind = 3;
  bst_cat_t split_cat = 4;
  float left_weight = 1.3f;
  float right_weight = 1.7f;

  gbm::GBTreeModel model(&mparam, ctx);
  GBTreeModelForTest(&model, split_ind, split_cat, left_weight, right_weight);

  std::unique_ptr<Predictor> predictor{CreatePredictorForTest(ctx)};

  std::vector<float> row(kCols);
  row[split_ind] = split_cat;
  auto m = GetDMatrixFromData(row, 1, kCols);
  if (is_column_split) {
    m = std::shared_ptr<DMatrix>{m->SliceCol(collective::GetWorldSize(), collective::GetRank())};
  }

  predictor->PredictLeaf(m.get(), &out_predictions.predictions, model);
  CHECK_EQ(out_predictions.predictions.Size(), 1);
  // go to left if it doesn't match the category, otherwise right.
  ASSERT_EQ(out_predictions.predictions.HostVector()[0], 2);

  row[split_ind] = split_cat + 1;
  m = GetDMatrixFromData(row, 1, kCols);
  if (is_column_split) {
    m = std::shared_ptr<DMatrix>{m->SliceCol(collective::GetWorldSize(), collective::GetRank())};
  }
  out_predictions.version = 0;
  predictor->InitOutPredictions(m->Info(), &out_predictions.predictions, model);
  predictor->PredictLeaf(m.get(), &out_predictions.predictions, model);
  ASSERT_EQ(out_predictions.predictions.HostVector()[0], 1);
}

void TestIterationRange(Context const* ctx) {
  size_t constexpr kRows = 1000, kCols = 20, kClasses = 4, kForest = 3, kIters = 10;
  auto dmat = RandomDataGenerator(kRows, kCols, 0)
                  .Device(ctx->Device())
                  .Classes(kClasses)
                  .GenerateDMatrix(true);
  auto learner = LearnerForTest(ctx, dmat, kIters, kForest);

  bool bound = false;
  bst_layer_t lend{3};
  std::unique_ptr<Learner> sliced{learner->Slice(0, lend, 1, &bound)};
  ASSERT_FALSE(bound);

  HostDeviceVector<float> out_predt_sliced;
  HostDeviceVector<float> out_predt_ranged;

  {
    sliced->Predict(dmat, true, &out_predt_sliced, 0, 0, false, false, false, false, false);
    learner->Predict(dmat, true, &out_predt_ranged, 0, lend, false, false, false, false, false);

    auto const &h_sliced = out_predt_sliced.HostVector();
    auto const &h_range = out_predt_ranged.HostVector();
    ASSERT_EQ(h_sliced.size(), h_range.size());
    ASSERT_EQ(h_sliced, h_range);
  }

  // SHAP
  {
    sliced->Predict(dmat, false, &out_predt_sliced, 0, 0, false, false, true, false, false);
    learner->Predict(dmat, false, &out_predt_ranged, 0, lend, false, false, true, false, false);

    auto const &h_sliced = out_predt_sliced.HostVector();
    auto const &h_range = out_predt_ranged.HostVector();
    ASSERT_EQ(h_sliced.size(), h_range.size());
    ASSERT_EQ(h_sliced, h_range);
  }

  // SHAP interaction
  {
    sliced->Predict(dmat, false, &out_predt_sliced, 0, 0, false, false, false, false, true);
    learner->Predict(dmat, false, &out_predt_ranged, 0, lend, false, false, false, false, true);
    auto const &h_sliced = out_predt_sliced.HostVector();
    auto const &h_range = out_predt_ranged.HostVector();
    ASSERT_EQ(h_sliced.size(), h_range.size());
    ASSERT_EQ(h_sliced, h_range);
  }

  // Leaf
  {
    sliced->Predict(dmat, false, &out_predt_sliced, 0, 0, false, true, false, false, false);
    learner->Predict(dmat, false, &out_predt_ranged, 0, lend, false, true, false, false, false);
    auto const &h_sliced = out_predt_sliced.HostVector();
    auto const &h_range = out_predt_ranged.HostVector();
    ASSERT_EQ(h_sliced.size(), h_range.size());
    ASSERT_EQ(h_sliced, h_range);
  }
}

namespace {
void VerifyIterationRangeColumnSplit(bool use_gpu, Json const &ranged_model,
                                     Json const &sliced_model, std::size_t rows, std::size_t cols,
                                     std::size_t classes,
                                     std::vector<float> const &expected_margin_ranged,
                                     std::vector<float> const &expected_margin_sliced,
                                     std::vector<float> const &expected_leaf_ranged,
                                     std::vector<float> const &expected_leaf_sliced) {
  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  Context ctx;
  if (use_gpu) {
    ctx = MakeCUDACtx(curt::AllVisibleGPUs() == 1 ? 0 : rank);
  }
  collective::GetWorkerLocalThreads(world_size, &ctx);

  auto dmat = RandomDataGenerator(rows, cols, 0).Classes(classes).GenerateDMatrix(true);
  std::shared_ptr<DMatrix> Xy{dmat->SliceCol(world_size, rank)};

  std::unique_ptr<Learner> learner{Learner::Create({Xy})};
  auto args = Args{{"device", ctx.DeviceName()}, {"nthread", std::to_string(ctx.Threads())}};
  learner->SetParams(args);
  learner->LoadModel(ranged_model);

  std::unique_ptr<Learner> sliced{Learner::Create({Xy})};
  sliced->SetParams(args);
  sliced->LoadModel(sliced_model);

  HostDeviceVector<float> out_predt_sliced;
  HostDeviceVector<float> out_predt_ranged;

  // margin
  {
    sliced->Predict(Xy, true, &out_predt_sliced, 0, 0, false, false, false, false, false);
    learner->Predict(Xy, true, &out_predt_ranged, 0, 3, false, false, false, false, false);
    auto const &h_sliced = out_predt_sliced.HostVector();
    auto const &h_ranged = out_predt_ranged.HostVector();
    EXPECT_EQ(h_sliced.size(), expected_margin_sliced.size());
    for (std::size_t i = 0; i < expected_margin_sliced.size(); ++i) {
      ASSERT_FLOAT_EQ(h_sliced[i], expected_margin_sliced[i]) << "rank " << rank << ", i " << i;
    }
    EXPECT_EQ(h_ranged.size(), expected_margin_ranged.size());
    for (std::size_t i = 0; i < expected_margin_ranged.size(); ++i) {
      ASSERT_FLOAT_EQ(h_ranged[i], expected_margin_ranged[i]) << "rank " << rank << ", i " << i;
    }
  }

  // Leaf
  {
    sliced->Predict(Xy, false, &out_predt_sliced, 0, 0, false, true, false, false, false);
    learner->Predict(Xy, false, &out_predt_ranged, 0, 3, false, true, false, false, false);
    auto const &h_sliced = out_predt_sliced.HostVector();
    auto const &h_ranged = out_predt_ranged.HostVector();
    EXPECT_EQ(h_sliced.size(), expected_leaf_sliced.size());
    for (std::size_t i = 0; i < expected_leaf_sliced.size(); ++i) {
      ASSERT_FLOAT_EQ(h_sliced[i], expected_leaf_sliced[i]) << "rank " << rank << ", i " << i;
    }
    EXPECT_EQ(h_ranged.size(), expected_leaf_ranged.size());
    for (std::size_t i = 0; i < expected_leaf_ranged.size(); ++i) {
      ASSERT_FLOAT_EQ(h_ranged[i], expected_leaf_ranged[i]) << "rank " << rank << ", i " << i;
    }
  }
}
}  // anonymous namespace

void TestIterationRangeColumnSplit(int world_size, bool use_gpu) {
  std::size_t constexpr kRows = 1000, kCols = 20, kClasses = 4, kForest = 3, kIters = 10;
  auto dmat = RandomDataGenerator(kRows, kCols, 0).Classes(kClasses).GenerateDMatrix(true);
  Context ctx;
  if (use_gpu) {
    ctx = MakeCUDACtx(0);
  }
  auto learner = LearnerForTest(&ctx, dmat, kIters, kForest);

  bool bound = false;
  std::unique_ptr<Learner> sliced{learner->Slice(0, 3, 1, &bound)};
  ASSERT_FALSE(bound);

  // margin
  HostDeviceVector<float> margin_predt_sliced;
  HostDeviceVector<float> margin_predt_ranged;
  sliced->Predict(dmat, true, &margin_predt_sliced, 0, 0, false, false, false, false, false);
  learner->Predict(dmat, true, &margin_predt_ranged, 0, 3, false, false, false, false, false);
  auto const &margin_sliced = margin_predt_sliced.HostVector();
  auto const &margin_ranged = margin_predt_ranged.HostVector();

  // Leaf
  HostDeviceVector<float> leaf_predt_sliced;
  HostDeviceVector<float> leaf_predt_ranged;
  sliced->Predict(dmat, false, &leaf_predt_sliced, 0, 0, false, true, false, false, false);
  learner->Predict(dmat, false, &leaf_predt_ranged, 0, 3, false, true, false, false, false);
  auto const &leaf_sliced = leaf_predt_sliced.HostVector();
  auto const &leaf_ranged = leaf_predt_ranged.HostVector();

  Json ranged_model{Object{}};
  learner->SaveModel(&ranged_model);
  Json sliced_model{Object{}};
  sliced->SaveModel(&sliced_model);

#if !defined(XGBOOST_USE_NCCL)
  if (use_gpu) {
    GTEST_SKIP_("Not compiled with NCCL");
    return;
  }
#endif  // defined(XGBOOST_USE_NCCL)
  collective::TestDistributedGlobal(world_size, [&] {
    VerifyIterationRangeColumnSplit(use_gpu, ranged_model, sliced_model, kRows, kCols, kClasses,
                                    margin_ranged, margin_sliced, leaf_ranged, leaf_sliced);
  });

#if defined(XGBOOST_USE_FEDERATED)
  collective::TestFederatedGlobal(world_size, [&] {
    VerifyIterationRangeColumnSplit(use_gpu, ranged_model, sliced_model, kRows, kCols, kClasses,
                                    margin_ranged, margin_sliced, leaf_ranged, leaf_sliced);
  });
#endif  // defined(XGBOOST_USE_FEDERATED)
}

void TestSparsePrediction(Context const *ctx, float sparsity) {
  size_t constexpr kRows = 512, kCols = 128, kIters = 4;
  auto Xy = RandomDataGenerator(kRows, kCols, sparsity).GenerateDMatrix(true);
  auto learner = LearnerForTest(ctx, Xy, kIters);

  HostDeviceVector<float> sparse_predt;

  Json model{Object{}};
  learner->SaveModel(&model);

  learner.reset(Learner::Create({Xy}));
  learner->LoadModel(model);
  learner->SetParam("device", ctx->DeviceName());
  learner->Configure();

  if (ctx->IsCUDA()) {
    learner->SetParam("tree_method", "gpu_hist");
    learner->SetParam("device", ctx->Device().Name());
  }
  learner->Predict(Xy, false, &sparse_predt, 0, 0);

  HostDeviceVector<float> with_nan(kRows * kCols, std::numeric_limits<float>::quiet_NaN());
  auto &h_with_nan = with_nan.HostVector();
  for (auto const &page : Xy->GetBatches<SparsePage>()) {
    auto batch = page.GetView();
    for (size_t i = 0; i < batch.Size(); ++i) {
      auto row = batch[i];
      for (auto e : row) {
        h_with_nan[i * kCols + e.index] = e.fvalue;
      }
    }
  }

  learner->SetParam("tree_method", "hist");
  learner->SetParam("gpu_id", "-1");
  // Xcode_12.4 doesn't compile with `std::make_shared`.
  auto dense = std::shared_ptr<DMatrix>(new data::DMatrixProxy{});
  auto array_interface = GetArrayInterface(&with_nan, kRows, kCols);
  std::string arr_str;
  Json::Dump(array_interface, &arr_str);
  dynamic_cast<data::DMatrixProxy *>(dense.get())->SetArrayData(arr_str.data());
  HostDeviceVector<float> *p_dense_predt;
  learner->InplacePredict(dense, PredictionType::kValue, std::numeric_limits<float>::quiet_NaN(),
                          &p_dense_predt, 0, 0);

  auto const &dense_predt = *p_dense_predt;
  if (ctx->IsCPU()) {
    ASSERT_EQ(dense_predt.HostVector(), sparse_predt.HostVector());
  } else {
    auto const &h_dense = dense_predt.HostVector();
    auto const &h_sparse = sparse_predt.HostVector();
    ASSERT_EQ(h_dense.size(), h_sparse.size());
    for (size_t i = 0; i < h_dense.size(); ++i) {
      ASSERT_FLOAT_EQ(h_dense[i], h_sparse[i]);
    }
  }
}

namespace {
void VerifySparsePredictionColumnSplit(bool use_gpu, Json const &model, std::size_t rows,
                                       std::size_t cols, float sparsity,
                                       std::vector<float> const &expected_predt) {
  Context ctx;
  if (use_gpu) {
    ctx = MakeCUDACtx(curt::AllVisibleGPUs() == 1 ? 0 : collective::GetRank());
  }
  auto Xy = RandomDataGenerator(rows, cols, sparsity).GenerateDMatrix(true);
  std::shared_ptr<DMatrix> sliced{Xy->SliceCol(collective::GetWorldSize(), collective::GetRank())};
  HostDeviceVector<float> sparse_predt;

  std::unique_ptr<Learner> learner{Learner::Create({sliced})};
  learner->SetParam("device", ctx.DeviceName());
  learner->LoadModel(model);
  learner->Predict(sliced, false, &sparse_predt, 0, 0);

  auto const &predt = sparse_predt.HostVector();
  ASSERT_EQ(predt.size(), expected_predt.size());
  for (size_t i = 0; i < predt.size(); ++i) {
    ASSERT_FLOAT_EQ(predt[i], expected_predt[i]);
  }
}
}  // anonymous namespace

void TestSparsePredictionColumnSplit(int world_size, bool use_gpu, float sparsity) {
  Context ctx;
  if (use_gpu) {
    ctx = MakeCUDACtx(0);
  }
  size_t constexpr kRows = 512, kCols = 128, kIters = 4;
  auto Xy = RandomDataGenerator(kRows, kCols, sparsity).GenerateDMatrix(true);
  auto learner = LearnerForTest(&ctx, Xy, kIters);

  HostDeviceVector<float> sparse_predt;

  Json model{Object{}};
  learner->SaveModel(&model);

  learner.reset(Learner::Create({Xy}));
  learner->LoadModel(model);

  learner->SetParam("device", ctx.DeviceName());
  learner->Predict(Xy, false, &sparse_predt, 0, 0);

#if !defined(XGBOOST_USE_NCCL)
  if (use_gpu) {
    GTEST_SKIP_("Not compiled with NCCL.");
    return;
  }
#endif  // defined(XGBOOST_USE_CUDA)
  collective::TestDistributedGlobal(world_size, [&] {
    VerifySparsePredictionColumnSplit(use_gpu, model, kRows, kCols, sparsity,
                                      sparse_predt.HostVector());
  });

#if defined(XGBOOST_USE_FEDERATED)
  collective::TestFederatedGlobal(world_size, [&] {
    VerifySparsePredictionColumnSplit(use_gpu, model, kRows, kCols, sparsity,
                                      sparse_predt.HostVector());
  });
#endif  // defined(XGBOOST_USE_FEDERATED)
}

void TestVectorLeafPrediction(Context const *ctx) {
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor", ctx));

  size_t constexpr kRows = 5;
  size_t constexpr kCols = 5;

  LearnerModelParam mparam{static_cast<bst_feature_t>(kCols),
                           linalg::Vector<float>{{0.5}, {1}, DeviceOrd::CPU()}, 1, 3,
                           MultiStrategy::kMultiOutputTree};

  std::vector<std::unique_ptr<RegTree>> trees;
  trees.emplace_back(new RegTree{mparam.LeafLength(), mparam.num_feature});

  std::vector<float> p_w(mparam.LeafLength(), 0.0f);
  std::vector<float> l_w(mparam.LeafLength(), 1.0f);
  std::vector<float> r_w(mparam.LeafLength(), 2.0f);

  auto &tree = trees.front();
  tree->ExpandNode(0, static_cast<bst_feature_t>(1), 2.0, true,
                   linalg::MakeVec(p_w.data(), p_w.size()), linalg::MakeVec(l_w.data(), l_w.size()),
                   linalg::MakeVec(r_w.data(), r_w.size()));
  ASSERT_TRUE(tree->IsMultiTarget());
  ASSERT_TRUE(mparam.IsVectorLeaf());

  gbm::GBTreeModel model{&mparam, ctx};
  model.CommitModelGroup(std::move(trees), 0);

  auto run_test = [&](float expected, HostDeviceVector<float> *p_data) {
    {
      auto p_fmat = GetDMatrixFromData(p_data->ConstHostVector(), kRows, kCols);
      PredictionCacheEntry predt_cache;
      cpu_predictor->InitOutPredictions(p_fmat->Info(), &predt_cache.predictions, model);
      ASSERT_EQ(predt_cache.predictions.Size(), kRows * mparam.LeafLength());
      cpu_predictor->PredictBatch(p_fmat.get(), &predt_cache, model, 0, 1);
      auto const &h_predt = predt_cache.predictions.HostVector();
      for (auto v : h_predt) {
        ASSERT_EQ(v, expected);
      }
    }

    {
      // inplace
      PredictionCacheEntry predt_cache;
      auto p_fmat = GetDMatrixFromData(p_data->ConstHostVector(), kRows, kCols);
      cpu_predictor->InitOutPredictions(p_fmat->Info(), &predt_cache.predictions, model);
      auto arr = GetArrayInterface(p_data, kRows, kCols);
      std::string str;
      Json::Dump(arr, &str);
      auto proxy = std::shared_ptr<DMatrix>(new data::DMatrixProxy{});
      dynamic_cast<data::DMatrixProxy *>(proxy.get())->SetArrayData(str.data());
      cpu_predictor->InplacePredict(proxy, model, std::numeric_limits<float>::quiet_NaN(),
                                    &predt_cache, 0, 1);
      auto const &h_predt = predt_cache.predictions.HostVector();
      for (auto v : h_predt) {
        ASSERT_EQ(v, expected);
      }
    }

    {
      // ghist
      PredictionCacheEntry predt_cache;
      auto &h_data = p_data->HostVector();
      // give it at least two bins, otherwise the histogram cuts only have min and max values.
      for (std::size_t i = 0; i < 5; ++i) {
        h_data[i] = 1.0;
      }
      auto p_fmat = GetDMatrixFromData(p_data->ConstHostVector(), kRows, kCols);

      cpu_predictor->InitOutPredictions(p_fmat->Info(), &predt_cache.predictions, model);

      auto iter = NumpyArrayIterForTest{ctx, *p_data, kRows, static_cast<bst_feature_t>(kCols),
                                        static_cast<std::size_t>(1)};
      p_fmat = std::make_shared<data::IterativeDMatrix>(
          &iter, iter.Proxy(), nullptr, Reset, Next, std::numeric_limits<float>::quiet_NaN(), 0,
          256, std::numeric_limits<std::int64_t>::max());

      cpu_predictor->InitOutPredictions(p_fmat->Info(), &predt_cache.predictions, model);
      cpu_predictor->PredictBatch(p_fmat.get(), &predt_cache, model, 0, 1);
      auto const &h_predt = predt_cache.predictions.HostVector();
      // the smallest v uses the min_value from histogram cuts, which leads to a left leaf
      // during prediction.
      for (std::size_t i = 5; i < h_predt.size(); ++i) {
        ASSERT_EQ(h_predt[i], expected) << i;
      }
    }
  };

  // go to right
  HostDeviceVector<float> data(kRows * kCols, model.trees.front()->SplitCond(RegTree::kRoot) + 1.0);
  run_test(2.5, &data);

  // go to left
  data.HostVector().assign(data.Size(), model.trees.front()->SplitCond(RegTree::kRoot) - 1.0);
  run_test(1.5, &data);
}

void ShapExternalMemoryTest::Run(Context const *ctx, bool is_qdm, bool is_interaction) {
  bst_idx_t n_samples{2048};
  bst_feature_t n_features{16};
  bst_target_t n_classes{3};
  bst_bin_t max_bin{64};
  auto create_pfmat = [&](RandomDataGenerator &rng) {
    if (is_qdm) {
      return rng.Bins(max_bin).GenerateExtMemQuantileDMatrix("temp", true);
    }
    return rng.GenerateSparsePageDMatrix("temp", true);
  };
  auto p_fmat = create_pfmat(RandomDataGenerator(n_samples, n_features, 0)
                                 .Batches(1)
                                 .Device(ctx->Device())
                                 .Classes(n_classes));
  std::unique_ptr<Learner> learner{Learner::Create({p_fmat})};
  learner->SetParam("device", ctx->DeviceName());
  learner->SetParam("base_score", "0.5");
  learner->SetParam("num_parallel_tree", "3");
  learner->SetParam("max_bin", std::to_string(max_bin));
  for (std::int32_t i = 0; i < 4; ++i) {
    learner->UpdateOneIter(i, p_fmat);
  }
  Json model{Object{}};
  learner->SaveModel(&model);
  auto j_booster = model["learner"]["gradient_booster"]["model"];
  auto model_param = MakeMP(n_features, 0.0, n_classes, ctx->Device());

  gbm::GBTreeModel gbtree{&model_param, ctx};
  gbtree.LoadModel(j_booster);

  std::unique_ptr<Predictor> predictor{
      Predictor::Create(ctx->IsCPU() ? "cpu_predictor" : "gpu_predictor", ctx)};
  predictor->Configure({});
  HostDeviceVector<float> contrib;
  if (is_interaction) {
    predictor->PredictInteractionContributions(p_fmat.get(), &contrib, gbtree);
  } else {
    predictor->PredictContribution(p_fmat.get(), &contrib, gbtree);
  }

  auto p_fmat_ext = create_pfmat(RandomDataGenerator(n_samples, n_features, 0)
                                     .Batches(4)
                                     .Device(ctx->Device())
                                     .Classes(n_classes));

  HostDeviceVector<float> contrib_ext;
  if (is_interaction) {
    predictor->PredictInteractionContributions(p_fmat_ext.get(), &contrib_ext, gbtree);
  } else {
    predictor->PredictContribution(p_fmat_ext.get(), &contrib_ext, gbtree);
  }

  ASSERT_EQ(contrib_ext.Size(), contrib.Size());

  auto h_contrib = contrib.ConstHostSpan();
  auto h_contrib_ext = contrib_ext.ConstHostSpan();
  for (std::size_t i = 0; i < h_contrib.size(); ++i) {
    ASSERT_EQ(h_contrib[i], h_contrib_ext[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(Predictor, ShapExternalMemoryTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool()));
}  // namespace xgboost
