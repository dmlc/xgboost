/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#include "test_predictor.h"

#include <gtest/gtest.h>
#include <xgboost/context.h>                      // for Context
#include <xgboost/data.h>                         // for DMatrix, BatchIterator, BatchSet, MetaInfo
#include <xgboost/host_device_vector.h>           // for HostDeviceVector
#include <xgboost/predictor.h>                    // for PredictionCacheEntry, Predictor, Predic...

#include <algorithm>                              // for max
#include <limits>                                 // for numeric_limits
#include <unordered_map>                          // for unordered_map

#include "../../../src/common/bitfield.h"         // for LBitField32
#include "../../../src/data/iterative_dmatrix.h"  // for IterativeDMatrix
#include "../../../src/data/proxy_dmatrix.h"      // for DMatrixProxy
#include "../helpers.h"                           // for GetDMatrixFromData, RandomDataGenerator
#include "xgboost/json.h"                         // for Json, Object, get, String
#include "xgboost/linalg.h"                       // for MakeVec, Tensor, TensorView, Vector
#include "xgboost/logging.h"                      // for CHECK
#include "xgboost/span.h"                         // for operator!=, SpanIterator, Span
#include "xgboost/tree_model.h"                   // for RegTree

namespace xgboost {
TEST(Predictor, PredictionCache) {
  size_t constexpr kRows = 16, kCols = 4;

  PredictionContainer container;
  DMatrix *m;
  // Add a cache that is immediately expired.
  auto add_cache = [&]() {
    auto p_dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
    container.Cache(p_dmat, Context::kCpuId);
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
  auto train = [&](std::string predictor) {
    p_hist->Info().labels.Reshape(rows, 1);
    auto &h_label = p_hist->Info().labels.Data()->HostVector();

    for (size_t i = 0; i < rows; ++i) {
      h_label[i] = i % kClasses;
    }

    learner.reset(Learner::Create({}));
    learner->SetParam("tree_method", tree_method);
    learner->SetParam("objective", "multi:softprob");
    learner->SetParam("num_feature", std::to_string(kCols));
    learner->SetParam("num_class", std::to_string(kClasses));
    learner->SetParam("max_bin", std::to_string(bins));
    learner->SetParam("predictor", predictor);
    learner->Configure();

    for (size_t i = 0; i < kIters; ++i) {
      learner->UpdateOneIter(i, p_hist);
    }

    Json model{Object{}};
    learner->SaveModel(&model);

    learner.reset(Learner::Create({}));
    learner->LoadModel(model);
    learner->SetParam("predictor", predictor);
    learner->Configure();

    HostDeviceVector<float> from_full;
    learner->Predict(p_full, false, &from_full, 0, 0);

    HostDeviceVector<float> from_hist;
    learner->Predict(p_hist, false, &from_hist, 0, 0);

    for (size_t i = 0; i < rows; ++i) {
      EXPECT_NEAR(from_hist.ConstHostVector()[i],
                  from_full.ConstHostVector()[i], kRtEps);
    }
  };

  if (tree_method == "gpu_hist") {
    train("gpu_predictor");
  } else {
    train("cpu_predictor");
  }
}

void TestInplacePrediction(std::shared_ptr<DMatrix> x, std::string predictor, bst_row_t rows,
                           bst_feature_t cols, int32_t device) {
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

  learner->Predict(m_test, false, &prediction, 0, 0);
  ASSERT_EQ(prediction.Size(), kRows);

  auto m_invalid = RandomDataGenerator(kRows, kTrainCols + 1, 0.5).GenerateDMatrix(false);
  ASSERT_THROW({learner->Predict(m_invalid, false, &prediction, 0, 0);}, dmlc::Error);

#if defined(XGBOOST_USE_CUDA)
  HostDeviceVector<float> from_cpu;
  learner->SetParam("predictor", "cpu_predictor");
  learner->Predict(m_test, false, &from_cpu, 0, 0);

  HostDeviceVector<float> from_cuda;
  learner->SetParam("predictor", "gpu_predictor");
  learner->Predict(m_test, false, &from_cuda, 0, 0);

  auto const& h_cpu = from_cpu.ConstHostVector();
  auto const& h_gpu = from_cuda.ConstHostVector();
  for (size_t i = 0; i < h_cpu.size(); ++i) {
    ASSERT_NEAR(h_cpu[i], h_gpu[i], kRtEps);
  }
#endif  // defined(XGBOOST_USE_CUDA)
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

void TestCategoricalPrediction(std::string name) {
  size_t constexpr kCols = 10;
  PredictionCacheEntry out_predictions;

  LearnerModelParam mparam{MakeMP(kCols, .5, 1)};
  uint32_t split_ind = 3;
  bst_cat_t split_cat = 4;
  float left_weight = 1.3f;
  float right_weight = 1.7f;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{});
  gbm::GBTreeModel model(&mparam, &ctx);
  GBTreeModelForTest(&model, split_ind, split_cat, left_weight, right_weight);

  ctx.UpdateAllowUnknown(Args{{"gpu_id", "0"}});
  std::unique_ptr<Predictor> predictor{Predictor::Create(name.c_str(), &ctx)};

  std::vector<float> row(kCols);
  row[split_ind] = split_cat;
  auto m = GetDMatrixFromData(row, 1, kCols);

  std::vector<FeatureType> types(10, FeatureType::kCategorical);
  m->Info().feature_types.HostVector() = types;

  predictor->InitOutPredictions(m->Info(), &out_predictions.predictions, model);
  predictor->PredictBatch(m.get(), &out_predictions, model, 0);
  auto score = mparam.BaseScore(Context::kCpuId)(0);
  ASSERT_EQ(out_predictions.predictions.Size(), 1ul);
  ASSERT_EQ(out_predictions.predictions.HostVector()[0],
            right_weight + score);  // go to right for matching cat

  row[split_ind] = split_cat + 1;
  m = GetDMatrixFromData(row, 1, kCols);
  out_predictions.version = 0;
  predictor->InitOutPredictions(m->Info(), &out_predictions.predictions, model);
  predictor->PredictBatch(m.get(), &out_predictions, model, 0);
  ASSERT_EQ(out_predictions.predictions.HostVector()[0], left_weight + score);
}

void TestCategoricalPredictLeaf(StringView name) {
  size_t constexpr kCols = 10;
  PredictionCacheEntry out_predictions;

  LearnerModelParam mparam{MakeMP(kCols, .5, 1)};

  uint32_t split_ind = 3;
  bst_cat_t split_cat = 4;
  float left_weight = 1.3f;
  float right_weight = 1.7f;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{});

  gbm::GBTreeModel model(&mparam, &ctx);
  GBTreeModelForTest(&model, split_ind, split_cat, left_weight, right_weight);

  ctx.gpu_id = 0;
  std::unique_ptr<Predictor> predictor{Predictor::Create(name.c_str(), &ctx)};

  std::vector<float> row(kCols);
  row[split_ind] = split_cat;
  auto m = GetDMatrixFromData(row, 1, kCols);

  predictor->PredictLeaf(m.get(), &out_predictions.predictions, model);
  CHECK_EQ(out_predictions.predictions.Size(), 1);
  // go to left if it doesn't match the category, otherwise right.
  ASSERT_EQ(out_predictions.predictions.HostVector()[0], 2);

  row[split_ind] = split_cat + 1;
  m = GetDMatrixFromData(row, 1, kCols);
  out_predictions.version = 0;
  predictor->InitOutPredictions(m->Info(), &out_predictions.predictions, model);
  predictor->PredictLeaf(m.get(), &out_predictions.predictions, model);
  ASSERT_EQ(out_predictions.predictions.HostVector()[0], 1);
}


void TestIterationRange(std::string name) {
  size_t constexpr kRows = 1000, kCols = 20, kClasses = 4, kForest = 3;
  auto dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix(true, true, kClasses);
  std::unique_ptr<Learner> learner{Learner::Create({dmat})};

  learner->SetParams(Args{{"num_parallel_tree", std::to_string(kForest)},
                          {"predictor", name}});

  size_t kIters = 10;
  for (size_t i = 0; i < kIters; ++i) {
    learner->UpdateOneIter(i, dmat);
  }

  bool bound = false;
  std::unique_ptr<Learner> sliced {learner->Slice(0, 3, 1, &bound)};
  ASSERT_FALSE(bound);

  HostDeviceVector<float> out_predt_sliced;
  HostDeviceVector<float> out_predt_ranged;

  // margin
  {
    sliced->Predict(dmat, true, &out_predt_sliced, 0, 0, false, false, false,
                    false, false);

    learner->Predict(dmat, true, &out_predt_ranged, 0, 3, false, false, false,
                     false, false);

    auto const &h_sliced = out_predt_sliced.HostVector();
    auto const &h_range = out_predt_ranged.HostVector();
    ASSERT_EQ(h_sliced.size(), h_range.size());
    ASSERT_EQ(h_sliced, h_range);
  }

  // SHAP
  {
    sliced->Predict(dmat, false, &out_predt_sliced, 0, 0, false, false,
                    true, false, false);

    learner->Predict(dmat, false, &out_predt_ranged, 0, 3, false, false, true,
                     false, false);

    auto const &h_sliced = out_predt_sliced.HostVector();
    auto const &h_range = out_predt_ranged.HostVector();
    ASSERT_EQ(h_sliced.size(), h_range.size());
    ASSERT_EQ(h_sliced, h_range);
  }

  // SHAP interaction
  {
    sliced->Predict(dmat, false, &out_predt_sliced, 0, 0, false, false,
                    false, false, true);
    learner->Predict(dmat, false, &out_predt_ranged, 0, 3, false, false, false,
                     false, true);
    auto const &h_sliced = out_predt_sliced.HostVector();
    auto const &h_range = out_predt_ranged.HostVector();
    ASSERT_EQ(h_sliced.size(), h_range.size());
    ASSERT_EQ(h_sliced, h_range);
  }

  // Leaf
  {
    sliced->Predict(dmat, false, &out_predt_sliced, 0, 0, false, true,
                    false, false, false);
    learner->Predict(dmat, false, &out_predt_ranged, 0, 3, false, true, false,
                     false, false);
    auto const &h_sliced = out_predt_sliced.HostVector();
    auto const &h_range = out_predt_ranged.HostVector();
    ASSERT_EQ(h_sliced.size(), h_range.size());
    ASSERT_EQ(h_sliced, h_range);
  }
}

void TestSparsePrediction(float sparsity, std::string predictor) {
  size_t constexpr kRows = 512, kCols = 128;
  auto Xy = RandomDataGenerator(kRows, kCols, sparsity).GenerateDMatrix(true);
  std::unique_ptr<Learner> learner{Learner::Create({Xy})};
  learner->Configure();
  for (size_t i = 0; i < 4; ++i) {
    learner->UpdateOneIter(i, Xy);
  }

  HostDeviceVector<float> sparse_predt;

  Json model{Object{}};
  learner->SaveModel(&model);

  learner.reset(Learner::Create({Xy}));
  learner->LoadModel(model);

  learner->SetParam("predictor", predictor);
  learner->Predict(Xy, false, &sparse_predt, 0, 0);

  HostDeviceVector<float> with_nan(kRows * kCols, std::numeric_limits<float>::quiet_NaN());
  auto& h_with_nan = with_nan.HostVector();
  for (auto const &page : Xy->GetBatches<SparsePage>()) {
    auto batch = page.GetView();
    for (size_t i = 0; i < batch.Size(); ++i) {
      auto row = batch[i];
      for (auto e : row) {
        h_with_nan[i * kCols + e.index] = e.fvalue;
      }
    }
  }

  learner->SetParam("predictor", "cpu_predictor");
  // Xcode_12.4 doesn't compile with `std::make_shared`.
  auto dense = std::shared_ptr<DMatrix>(new data::DMatrixProxy{});
  auto array_interface = GetArrayInterface(&with_nan, kRows, kCols);
  std::string arr_str;
  Json::Dump(array_interface, &arr_str);
  dynamic_cast<data::DMatrixProxy *>(dense.get())->SetArrayData(arr_str.data());
  HostDeviceVector<float> *p_dense_predt;
  learner->InplacePredict(dense, PredictionType::kValue, std::numeric_limits<float>::quiet_NaN(),
                          &p_dense_predt, 0, 0);

  auto const& dense_predt = *p_dense_predt;
  if (predictor == "cpu_predictor") {
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

void TestVectorLeafPrediction(Context const *ctx) {
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor", ctx));

  size_t constexpr kRows = 5;
  size_t constexpr kCols = 5;

  LearnerModelParam mparam{static_cast<bst_feature_t>(kCols),
                           linalg::Vector<float>{{0.5}, {1}, Context::kCpuId}, 1, 3,
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
      p_fmat =
          std::make_shared<data::IterativeDMatrix>(&iter, iter.Proxy(), nullptr, Reset, Next,
                                                   std::numeric_limits<float>::quiet_NaN(), 0, 256);

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
}  // namespace xgboost
