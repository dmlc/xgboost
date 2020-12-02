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
#include "../../../src/common/categorical.h"
#include "../../../src/common/bitfield.h"

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
  ASSERT_EQ(container.Container().size(), 0ul);
  add_cache();
  EXPECT_ANY_THROW(container.Entry(m));
}

void TestTrainingPrediction(size_t rows, size_t bins,
                            std::string tree_method,
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
    learner->SetParam("max_bin", std::to_string(bins));
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

void TestPredictionWithLesserFeatures(std::string predictor_name) {
  size_t constexpr kRows = 256, kTrainCols = 256, kTestCols = 4, kIters = 4;
  auto m_train = RandomDataGenerator(kRows, kTrainCols, 0.5).GenerateDMatrix(true);
  auto m_test = RandomDataGenerator(kRows, kTestCols, 0.5).GenerateDMatrix(false);
  std::unique_ptr<Learner> learner{Learner::Create({m_train})};

  for (size_t i = 0; i < kIters; ++i) {
    learner->UpdateOneIter(i, m_train);
  }

  HostDeviceVector<float> prediction;
  learner->SetParam("predictor", predictor_name);
  learner->Configure();
  Json config{Object()};
  learner->SaveConfig(&config);
  ASSERT_EQ(get<String>(config["learner"]["gradient_booster"]["gbtree_train_param"]["predictor"]), predictor_name);

  learner->Predict(m_test, false, &prediction);
  ASSERT_EQ(prediction.Size(), kRows);

  auto m_invalid = RandomDataGenerator(kRows, kTrainCols + 1, 0.5).GenerateDMatrix(false);
  ASSERT_THROW({learner->Predict(m_invalid, false, &prediction);}, dmlc::Error);

#if defined(XGBOOST_USE_CUDA)
  HostDeviceVector<float> from_cpu;
  learner->SetParam("predictor", "cpu_predictor");
  learner->Predict(m_test, false, &from_cpu);

  HostDeviceVector<float> from_cuda;
  learner->SetParam("predictor", "gpu_predictor");
  learner->Predict(m_test, false, &from_cuda);

  auto const& h_cpu = from_cpu.ConstHostVector();
  auto const& h_gpu = from_cuda.ConstHostVector();
  for (size_t i = 0; i < h_cpu.size(); ++i) {
    ASSERT_NEAR(h_cpu[i], h_gpu[i], kRtEps);
  }
#endif  // defined(XGBOOST_USE_CUDA)
}

void TestCategoricalPrediction(std::string name) {
  size_t constexpr kCols = 10;
  PredictionCacheEntry out_predictions;

  LearnerModelParam param;
  param.num_feature = kCols;
  param.num_output_group = 1;
  param.base_score = 0.5;

  gbm::GBTreeModel model(&param);

  std::vector<std::unique_ptr<RegTree>> trees;
  trees.push_back(std::unique_ptr<RegTree>(new RegTree));
  auto& p_tree = trees.front();

  uint32_t split_ind = 3;
  bst_cat_t split_cat = 4;
  float left_weight = 1.3f;
  float right_weight = 1.7f;

  std::vector<uint32_t> split_cats(LBitField32::ComputeStorageSize(split_cat));
  LBitField32 cats_bits(split_cats);
  cats_bits.Set(split_cat);

  p_tree->ExpandCategorical(0, split_ind, split_cats, true, 1.5f,
                            left_weight, right_weight,
                            3.0f, 2.2f, 7.0f, 9.0f);
  model.CommitModel(std::move(trees), 0);

  GenericParameter runtime;
  runtime.gpu_id = 0;
  std::unique_ptr<Predictor> predictor{
      Predictor::Create(name.c_str(), &runtime)};

  std::vector<float> row(kCols);
  row[split_ind] = split_cat;
  auto m = GetDMatrixFromData(row, 1, kCols);

  predictor->PredictBatch(m.get(), &out_predictions, model, 0);
  ASSERT_EQ(out_predictions.predictions.Size(), 1ul);
  ASSERT_EQ(out_predictions.predictions.HostVector()[0],
            right_weight + param.base_score);  // go to right for matching cat

  row[split_ind] = split_cat + 1;
  m = GetDMatrixFromData(row, 1, kCols);
  out_predictions.version = 0;
  predictor->PredictBatch(m.get(), &out_predictions, model, 0);
  ASSERT_EQ(out_predictions.predictions.HostVector()[0],
            left_weight + param.base_score);
}
}  // namespace xgboost
