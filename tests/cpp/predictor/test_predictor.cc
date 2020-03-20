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
    auto *pp_dmat = CreateDMatrix(kRows, kCols, 0);
    auto p_dmat = *pp_dmat;
    container.Cache(p_dmat, GenericParameter::kCpuId);
    m = p_dmat.get();
    delete pp_dmat;
  };

  add_cache();
  ASSERT_EQ(container.Container().size(), 0);
  add_cache();
  EXPECT_ANY_THROW(container.Entry(m));
}

// Only run this test when CUDA is enabled.
void TestTrainingPrediction(size_t rows, std::string tree_method) {
  size_t constexpr kCols = 16;
  size_t constexpr kClasses = 3;
  size_t constexpr kIters = 3;

  std::unique_ptr<Learner> learner;
  auto train = [&](std::string predictor, HostDeviceVector<float>* out) {
    auto pp_m = CreateDMatrix(rows, kCols, 0);
    auto p_m = *pp_m;

    auto &h_label = p_m->Info().labels_.HostVector();
    h_label.resize(rows);

    for (size_t i = 0; i < rows; ++i) {
      h_label[i] = i % kClasses;
    }

    learner.reset(Learner::Create({}));
    learner->SetParam("tree_method", tree_method);
    learner->SetParam("objective", "multi:softprob");
    learner->SetParam("predictor", predictor);
    learner->SetParam("num_feature", std::to_string(kCols));
    learner->SetParam("num_class", std::to_string(kClasses));
    learner->Configure();

    for (size_t i = 0; i < kIters; ++i) {
      learner->UpdateOneIter(i, p_m);
    }
    learner->Predict(p_m, false, out);
    delete pp_m;
  };
  // Alternate the predictor, CPU predictor can not use ellpack while GPU predictor can
  // not use CPU histogram index.  So it's guaranteed one of the following is not
  // predicting from histogram index.  Note: As of writing only GPU supports predicting
  // from gradient index, the test is written for future portability.
  HostDeviceVector<float> predictions_0;
  train("cpu_predictor", &predictions_0);

  HostDeviceVector<float> predictions_1;
  train("gpu_predictor", &predictions_1);

  for (size_t i = 0; i < rows; ++i) {
    EXPECT_NEAR(predictions_1.ConstHostVector()[i],
                predictions_0.ConstHostVector()[i], kRtEps);
  }
}
}  // namespace xgboost
