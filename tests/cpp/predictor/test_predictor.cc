/*!
 * Copyright 2020 by Contributors
 */

#include <gtest/gtest.h>
#include <xgboost/predictor.h>
#include <xgboost/data.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/generic_parameters.h>

#include "test_predictor.h"

#include "../helpers.h"
#include "../../../src/common/io.h"

namespace xgboost {
TEST(Predictor, PredictionCache) {
  size_t constexpr kRows = 16, kCols = 4;

  PredictionContainer container;
  DMatrix* m;
  // Add a cache that is immediately expired.
  auto add_cache = [&]() {
    auto p_dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
    container.Cache(p_dmat, GenericParameter::kCpuId);
    m = p_dmat.get();
  };

  add_cache();
  ASSERT_EQ(container.Container().size(), 0);
  add_cache();
  EXPECT_ANY_THROW(container.Entry(m));
}

void TestTrainingPrediction(size_t rows, std::string tree_method,
                            std::shared_ptr<DMatrix> p_full,
                            std::shared_ptr<DMatrix> p_hist) {
  size_t constexpr kCols = 16;
  size_t constexpr kClasses = 3;
  size_t constexpr kIters = 3;

  std::unique_ptr<Learner> learner;
  auto train = [&](std::string predictor, HostDeviceVector<float> *out) {
    auto &h_label = p_hist->Info().labels_.HostVector();
    h_label.resize(rows);

    for (size_t i = 0; i < rows; ++i) {
      h_label[i] = i % kClasses;
    }

    learner.reset(Learner::Create({}));
    learner->SetParam("tree_method", tree_method);
    learner->SetParam("objective", "multi:softprob");
    learner->SetParam("num_feature", std::to_string(kCols));
    learner->SetParam("num_class", std::to_string(kClasses));
    learner->Configure();

    for (size_t i = 0; i < kIters; ++i) {
      learner->UpdateOneIter(i, p_hist);
    }

    HostDeviceVector<float> from_full;
    learner->Predict(p_full, false, &from_full);

    HostDeviceVector<float> from_hist;
    learner->Predict(p_hist, false, &from_hist);

    for (size_t i = 0; i < rows; ++i) {
      EXPECT_NEAR(from_hist.ConstHostVector()[i],
                  from_full.ConstHostVector()[i], kRtEps);
    }
  };

  HostDeviceVector<float> predictions_0;
  train("cpu_predictor", &predictions_0);

  HostDeviceVector<float> predictions_1;
  train("gpu_predictor", &predictions_1);
}

void TestInplacePrediction(dmlc::any x, std::string predictor,
                           bst_row_t rows, bst_feature_t cols,
                           int32_t device) {
  size_t constexpr kClasses { 4 };
  auto gen = RandomDataGenerator{rows, cols, 0.5}.Device(device);
  std::shared_ptr<DMatrix> m = gen.GenerateDMatrix(true, false, kClasses);

  std::unique_ptr<Learner> learner {
    Learner::Create({m})
  };

  learner->SetParam("num_parallel_tree", "4");
  learner->SetParam("num_class", std::to_string(kClasses));
  learner->SetParam("seed", "0");
  learner->SetParam("subsample", "0.5");
  learner->SetParam("gpu_id", std::to_string(device));
  learner->SetParam("predictor", predictor);
  for (int32_t it = 0; it < 4; ++it) {
    learner->UpdateOneIter(it, m);
  }

  HostDeviceVector<float> *p_out_predictions_0{nullptr};
  learner->InplacePredict(x, "margin", std::numeric_limits<float>::quiet_NaN(),
                          &p_out_predictions_0, 0, 2);
  CHECK(p_out_predictions_0);
  HostDeviceVector<float> predict_0 (p_out_predictions_0->Size());
  predict_0.Copy(*p_out_predictions_0);

  HostDeviceVector<float> *p_out_predictions_1{nullptr};
  learner->InplacePredict(x, "margin", std::numeric_limits<float>::quiet_NaN(),
                          &p_out_predictions_1, 2, 4);
  CHECK(p_out_predictions_1);
  HostDeviceVector<float> predict_1 (p_out_predictions_1->Size());
  predict_1.Copy(*p_out_predictions_1);

  HostDeviceVector<float>* p_out_predictions{nullptr};
  learner->InplacePredict(x, "margin", std::numeric_limits<float>::quiet_NaN(),
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

  learner->SetParam("gpu_id", "-1");
  learner->Configure();
}
}  // namespace xgboost
