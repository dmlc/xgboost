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

template <typename Page>
void TestPredictionFromGradientIndex(std::string name, size_t rows, size_t cols,
                                     std::shared_ptr<DMatrix> p_hist) {
  constexpr size_t kClasses { 3 };

  LearnerModelParam mparam{MakeMP(cols, .5, kClasses)};
  auto lparam = CreateEmptyGenericParam(0);

  std::unique_ptr<Predictor> predictor =
      std::unique_ptr<Predictor>(Predictor::Create(name, &lparam));
  predictor->Configure({});

  Context ctx;
  ctx.UpdateAllowUnknown(Args{});
  gbm::GBTreeModel model = CreateTestModel(&mparam, &ctx, kClasses);

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

// p_full and p_hist should come from the same data set.
void TestTrainingPrediction(size_t rows, size_t bins, std::string tree_method,
                            std::shared_ptr<DMatrix> p_full,
                            std::shared_ptr<DMatrix> p_hist);

void TestInplacePrediction(std::shared_ptr<DMatrix> x, std::string predictor, bst_row_t rows,
                           bst_feature_t cols, int32_t device = -1);

void TestPredictionWithLesserFeatures(std::string preditor_name);

void TestCategoricalPrediction(std::string name);

void TestCategoricalPredictLeaf(StringView name);

void TestIterationRange(std::string name);

void TestSparsePrediction(float sparsity, std::string predictor);

void TestVectorLeafPrediction(Context const* ctx);
}  // namespace xgboost

#endif  // XGBOOST_TEST_PREDICTOR_H_
