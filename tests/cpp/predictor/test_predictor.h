#ifndef XGBOOST_TEST_PREDICTOR_H_
#define XGBOOST_TEST_PREDICTOR_H_

#include <xgboost/predictor.h>
#include <string>
#include <cstddef>
#include "../helpers.h"

namespace xgboost {
template <typename Page>
void TestPredictionFromGradientIndex(std::string name, size_t rows, size_t cols,
                                     std::shared_ptr<DMatrix> p_hist) {
  constexpr size_t kClasses { 3 };

  LearnerModelParam param;
  param.num_feature = cols;
  param.num_output_group = kClasses;
  param.base_score = 0.5;

  auto lparam = CreateEmptyGenericParam(0);

  std::unique_ptr<Predictor> predictor =
      std::unique_ptr<Predictor>(Predictor::Create(name, &lparam));
  predictor->Configure({});

  gbm::GBTreeModel model = CreateTestModel(&param, kClasses);

  {
    auto p_precise = RandomDataGenerator(rows, cols, 0).GenerateDMatrix();

    PredictionCacheEntry approx_out_predictions;
    predictor->PredictBatch(p_hist.get(), &approx_out_predictions, model, 0);

    PredictionCacheEntry precise_out_predictions;
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
    predictor->PredictBatch(p_dmat.get(), &precise_out_predictions, model, 0);
    ASSERT_FALSE(p_dmat->PageExists<Page>());
  }
}

// p_full and p_hist should come from the same data set.
void TestTrainingPrediction(size_t rows, std::string tree_method,
                            std::shared_ptr<DMatrix> p_full,
                            std::shared_ptr<DMatrix> p_hist);

void TestInplacePrediction(dmlc::any x, std::string predictor,
                           bst_row_t rows, bst_feature_t cols,
                           int32_t device = -1);
}  // namespace xgboost

#endif  // XGBOOST_TEST_PREDICTOR_H_
