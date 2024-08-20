/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_TEST_PREDICTOR_H_
#define XGBOOST_TEST_PREDICTOR_H_

#include <xgboost/context.h>  // for Context
#include <xgboost/predictor.h>

#include <cstddef>
#include <string>

#include "../../../src/gbm/gbtree_model.h"  // for GBTreeModel
#include "../helpers.h"

namespace xgboost {
inline gbm::GBTreeModel CreateTestModel(LearnerModelParam const* param, Context const* ctx,
                                        size_t n_classes = 1) {
  gbm::GBTreeModel model(param, ctx);

  for (size_t i = 0; i < n_classes; ++i) {
    std::vector<std::unique_ptr<RegTree>> trees;
    trees.push_back(std::unique_ptr<RegTree>(new RegTree));
    if (i == 0) {
      (*trees.back())[0].SetLeaf(1.5f);
      (*trees.back()).Stat(0).sum_hess = 1.0f;
    }
    model.CommitModelGroup(std::move(trees), i);
  }

  return model;
}

inline auto CreatePredictorForTest(Context const* ctx) {
  if (ctx->IsCPU()) {
    return Predictor::Create("cpu_predictor", ctx);
  } else if (ctx->IsSycl()) {
    return Predictor::Create("sycl_predictor", ctx);
  } else {
    return Predictor::Create("gpu_predictor", ctx);
  }
}

// fixme: cpu test
template <typename Page>
void TestPredictionFromGradientIndex(Context const* ctx, size_t rows, size_t cols,
                                     std::shared_ptr<DMatrix> p_hist) {
  constexpr size_t kClasses { 3 };

  LearnerModelParam mparam{MakeMP(cols, .5, kClasses)};
  auto cuda_ctx = MakeCUDACtx(0);

  std::unique_ptr<Predictor> predictor =
      std::unique_ptr<Predictor>(CreatePredictorForTest(&cuda_ctx));
  predictor->Configure({});

  gbm::GBTreeModel model = CreateTestModel(&mparam, ctx, kClasses);

  {
    auto p_precise = RandomDataGenerator(rows, cols, 0).GenerateDMatrix();

    PredictionCacheEntry approx_out_predictions;
    predictor->InitOutPredictions(p_hist->Info(), &approx_out_predictions.predictions, model);
    predictor->PredictBatch(p_hist.get(), &approx_out_predictions, model, 0);

    PredictionCacheEntry precise_out_predictions;
    predictor->InitOutPredictions(p_precise->Info(), &precise_out_predictions.predictions, model);
    predictor->PredictBatch(p_precise.get(), &precise_out_predictions, model, 0);

    for (size_t i = 0; i < rows; ++i) {
      CHECK_EQ(approx_out_predictions.predictions.HostVector()[i],
               precise_out_predictions.predictions.HostVector()[i]);
    }
  }

  {
    // Predictor should never try to create the histogram index by itself.  As only
    // histogram index from training data is valid and predictor doesn't known which
    // matrix is used for training.
    auto p_dmat = RandomDataGenerator(rows, cols, 0).GenerateDMatrix();
    PredictionCacheEntry precise_out_predictions;
    predictor->InitOutPredictions(p_dmat->Info(), &precise_out_predictions.predictions, model);
    predictor->PredictBatch(p_dmat.get(), &precise_out_predictions, model, 0);
    CHECK(!p_dmat->PageExists<Page>());
  }
}

void TestBasic(DMatrix* dmat, Context const * ctx);

// p_full and p_hist should come from the same data set.
void TestTrainingPrediction(Context const* ctx, size_t rows, size_t bins,
                            std::shared_ptr<DMatrix> p_full, std::shared_ptr<DMatrix> p_hist);

void TestInplacePrediction(Context const* ctx, std::shared_ptr<DMatrix> x, bst_idx_t rows,
                           bst_feature_t cols);

void TestPredictionWithLesserFeatures(Context const* ctx);

void TestPredictionDeviceAccess();

void TestCategoricalPrediction(bool use_gpu, bool is_column_split);

void TestPredictionWithLesserFeaturesColumnSplit(bool use_gpu);

void TestCategoricalPredictLeaf(Context const *ctx, bool is_column_split);

void TestIterationRange(Context const* ctx);

void TestIterationRangeColumnSplit(int world_size, bool use_gpu);

void TestSparsePrediction(Context const* ctx, float sparsity);

void TestSparsePredictionColumnSplit(int world_size, bool use_gpu, float sparsity);

void TestVectorLeafPrediction(Context const* ctx);

class ShapExternalMemoryTest : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 public:
  void Run(Context const* ctx, bool is_qdm, bool is_interaction);
};
}  // namespace xgboost

#endif  // XGBOOST_TEST_PREDICTOR_H_
