#ifndef XGBOOST_TEST_PREDICTOR_H_
#define XGBOOST_TEST_PREDICTOR_H_

#include <xgboost/predictor.h>
#include <string>
#include <cstddef>
#include "../helpers.h"

namespace xgboost {
template <typename Page>
void TestPredictionFromGradientIndex(std::string name, size_t rows, int32_t bins) {
  constexpr size_t kCols { 8 }, kClasses { 3 };

  LearnerModelParam param;
  param.num_feature = kCols;
  param.num_output_group = kClasses;
  param.base_score = 0.5;

  auto lparam = CreateEmptyGenericParam(0);

  std::unique_ptr<Predictor> predictor =
      std::unique_ptr<Predictor>(Predictor::Create(name, &lparam));
  predictor->Configure({});

  gbm::GBTreeModel model = CreateTestModel(&param, kClasses);

  {
    auto p_ellpack = RandomDataGenerator(rows, kCols, 0).GenerateDMatix();
    // Use same number of bins as rows.
    for (auto const &page DMLC_ATTRIBUTE_UNUSED :
         p_ellpack->GetBatches<Page>({0, static_cast<int32_t>(bins), 0})) {
    }

    auto p_precise = RandomDataGenerator(rows, kCols, 0).GenerateDMatix();

    PredictionCacheEntry approx_out_predictions;
    predictor->PredictBatch(p_ellpack.get(), &approx_out_predictions, model, 0);

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
    auto p_dmat = RandomDataGenerator(rows, kCols, 0).GenerateDMatix();
    PredictionCacheEntry precise_out_predictions;
    predictor->PredictBatch(p_dmat.get(), &precise_out_predictions, model, 0);
    ASSERT_FALSE(p_dmat->PageExists<Page>());
  }
}

void TestTrainingPrediction(size_t rows, std::string tree_method);

}  // namespace xgboost

#endif  // XGBOOST_TEST_PREDICTOR_H_
