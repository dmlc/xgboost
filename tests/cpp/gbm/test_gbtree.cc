/*!
 * Copyright 2019-2022 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>

#include "../../../src/data/adapter.h"
#include "../../../src/data/proxy_dmatrix.h"
#include "../../../src/gbm/gbtree.h"
#include "../filesystem.h"  // dmlc::TemporaryDirectory
#include "../helpers.h"
#include "xgboost/base.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/learner.h"
#include "xgboost/predictor.h"

namespace xgboost {
TEST(GBTree, SelectTreeMethod) {
  size_t constexpr kCols = 10;

  Context ctx;
  LearnerModelParam mparam{MakeMP(kCols, .5, 1)};

  std::unique_ptr<GradientBooster> p_gbm {
    GradientBooster::Create("gbtree", &ctx, &mparam)};
  auto& gbtree = dynamic_cast<gbm::GBTree&> (*p_gbm);

  // Test if `tree_method` can be set
  Args args {{"tree_method", "approx"}};
  gbtree.Configure({args.cbegin(), args.cend()});

  gbtree.Configure(args);
  auto const& tparam = gbtree.GetTrainParam();
  gbtree.Configure({{"tree_method", "approx"}});
  ASSERT_EQ(tparam.updater_seq, "grow_histmaker");
  gbtree.Configure({{"tree_method", "exact"}});
  ASSERT_EQ(tparam.updater_seq, "grow_colmaker,prune");
  gbtree.Configure({{"tree_method", "hist"}});
  ASSERT_EQ(tparam.updater_seq, "grow_quantile_histmaker");
  gbtree.Configure({{"booster", "dart"}, {"tree_method", "hist"}});
  ASSERT_EQ(tparam.updater_seq, "grow_quantile_histmaker");

#ifdef XGBOOST_USE_CUDA
  ctx.UpdateAllowUnknown(Args{{"gpu_id", "0"}});
  gbtree.Configure({{"tree_method", "gpu_hist"}});
  ASSERT_EQ(tparam.updater_seq, "grow_gpu_hist");
  gbtree.Configure({{"booster", "dart"}, {"tree_method", "gpu_hist"}});
  ASSERT_EQ(tparam.updater_seq, "grow_gpu_hist");
#endif  // XGBOOST_USE_CUDA
}

TEST(GBTree, PredictionCache) {
  size_t constexpr kRows = 100, kCols = 10;
  Context ctx;
  LearnerModelParam mparam{MakeMP(kCols, .5, 1)};

  std::unique_ptr<GradientBooster> p_gbm {
    GradientBooster::Create("gbtree", &ctx, &mparam)};
  auto& gbtree = dynamic_cast<gbm::GBTree&> (*p_gbm);

  gbtree.Configure({{"tree_method", "hist"}});
  auto p_m = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();
  auto gpair = GenerateRandomGradients(kRows);
  PredictionCacheEntry out_predictions;
  gbtree.DoBoost(p_m.get(), &gpair, &out_predictions, nullptr);

  gbtree.PredictBatch(p_m.get(), &out_predictions, false, 0, 0);
  ASSERT_EQ(1, out_predictions.version);
  std::vector<float> first_iter = out_predictions.predictions.HostVector();
  // Add 1 more boosted round
  gbtree.DoBoost(p_m.get(), &gpair, &out_predictions, nullptr);
  gbtree.PredictBatch(p_m.get(), &out_predictions, false, 0, 0);
  ASSERT_EQ(2, out_predictions.version);
  // Update the cache for all rounds
  out_predictions.version = 0;
  gbtree.PredictBatch(p_m.get(), &out_predictions, false, 0, 0);
  ASSERT_EQ(2, out_predictions.version);

  gbtree.DoBoost(p_m.get(), &gpair, &out_predictions, nullptr);
  // drop the cache.
  gbtree.PredictBatch(p_m.get(), &out_predictions, false, 1, 2);
  ASSERT_EQ(0, out_predictions.version);
  // half open set [1, 3)
  gbtree.PredictBatch(p_m.get(), &out_predictions, false, 1, 3);
  ASSERT_EQ(0, out_predictions.version);
  // iteration end
  gbtree.PredictBatch(p_m.get(), &out_predictions, false, 0, 2);
  ASSERT_EQ(2, out_predictions.version);
  // restart the cache when end iteration is smaller than cache version
  gbtree.PredictBatch(p_m.get(), &out_predictions, false, 0, 1);
  ASSERT_EQ(1, out_predictions.version);
  ASSERT_EQ(out_predictions.predictions.HostVector(), first_iter);
}

TEST(GBTree, WrongUpdater) {
  size_t constexpr kRows = 17;
  size_t constexpr kCols = 15;

  auto p_dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  p_dmat->Info().labels.Reshape(kRows);

  auto learner = std::unique_ptr<Learner>(Learner::Create({p_dmat}));
  // Hist can not be used for updating tree.
  learner->SetParams(Args{{"tree_method", "hist"}, {"process_type", "update"}});
  ASSERT_THROW(learner->UpdateOneIter(0, p_dmat), dmlc::Error);
  // Prune can not be used for learning new tree.
  learner->SetParams(
      Args{{"tree_method", "prune"}, {"process_type", "default"}});
  ASSERT_THROW(learner->UpdateOneIter(0, p_dmat), dmlc::Error);
}

#ifdef XGBOOST_USE_CUDA
TEST(GBTree, ChoosePredictor) {
  // The test ensures data don't get pulled into device.
  size_t constexpr kRows = 17;
  size_t constexpr kCols = 15;

  auto p_dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  auto& data = (*(p_dmat->GetBatches<SparsePage>().begin())).data;
  p_dmat->Info().labels.Reshape(kRows);

  auto learner = std::unique_ptr<Learner>(Learner::Create({p_dmat}));
  learner->SetParams(Args{{"tree_method", "gpu_hist"}, {"gpu_id", "0"}});
  for (size_t i = 0; i < 4; ++i) {
    learner->UpdateOneIter(i, p_dmat);
  }
  ASSERT_TRUE(data.HostCanWrite());
  dmlc::TemporaryDirectory tempdir;
  const std::string fname = tempdir.path + "/model_param.bst";

  {
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname.c_str(), "w"));
    learner->Save(fo.get());
  }

  // a new learner
  learner = std::unique_ptr<Learner>(Learner::Create({p_dmat}));
  {
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r"));
    learner->Load(fi.get());
  }
  learner->SetParams(Args{{"tree_method", "gpu_hist"}, {"gpu_id", "0"}});
  for (size_t i = 0; i < 4; ++i) {
    learner->UpdateOneIter(i, p_dmat);
  }
  ASSERT_TRUE(data.HostCanWrite());

  // pull data into device.
  data.HostVector();
  data.SetDevice(0);
  data.DeviceSpan();
  ASSERT_FALSE(data.HostCanWrite());

  // another new learner
  learner = std::unique_ptr<Learner>(Learner::Create({p_dmat}));
  learner->SetParams(Args{{"tree_method", "gpu_hist"}, {"gpu_id", "0"}});
  for (size_t i = 0; i < 4; ++i) {
    learner->UpdateOneIter(i, p_dmat);
  }
  // data is not pulled back into host
  ASSERT_FALSE(data.HostCanWrite());
}
#endif  // XGBOOST_USE_CUDA

// Some other parts of test are in `Tree.JsonIO'.
TEST(GBTree, JsonIO) {
  size_t constexpr kRows = 16, kCols = 16;

  Context ctx;
  LearnerModelParam mparam{MakeMP(kCols, .5, 1)};

  std::unique_ptr<GradientBooster> gbm {
    CreateTrainedGBM("gbtree", Args{}, kRows, kCols, &mparam, &ctx) };

  Json model {Object()};
  model["model"] = Object();
  auto& j_model = model["model"];

  model["config"] = Object();
  auto& j_param = model["config"];

  gbm->SaveModel(&j_model);
  gbm->SaveConfig(&j_param);

  std::string model_str;
  Json::Dump(model, &model_str);

  model = Json::Load({model_str.c_str(), model_str.size()});
  ASSERT_EQ(get<String>(model["model"]["name"]), "gbtree");

  auto const& gbtree_model = model["model"]["model"];
  ASSERT_EQ(get<Array>(gbtree_model["trees"]).size(), 1ul);
  ASSERT_EQ(get<Integer>(get<Object>(get<Array>(gbtree_model["trees"]).front()).at("id")), 0);
  ASSERT_EQ(get<Array>(gbtree_model["tree_info"]).size(), 1ul);

  auto j_train_param = model["config"]["gbtree_model_param"];
  ASSERT_EQ(get<String>(j_train_param["num_parallel_tree"]), "1");
}

TEST(Dart, JsonIO) {
  size_t constexpr kRows = 16, kCols = 16;

  Context ctx;
  LearnerModelParam mparam{MakeMP(kCols, .5, 1)};

  std::unique_ptr<GradientBooster> gbm{
      CreateTrainedGBM("dart", Args{}, kRows, kCols, &mparam, &ctx)};

  Json model {Object()};
  model["model"] = Object();
  auto& j_model = model["model"];
  model["config"] = Object();

  auto& j_param = model["config"];

  gbm->SaveModel(&j_model);
  gbm->SaveConfig(&j_param);

  std::string model_str;
  Json::Dump(model, &model_str);

  model = Json::Load({model_str.c_str(), model_str.size()});

  ASSERT_EQ(get<String>(model["model"]["name"]), "dart") << model;
  ASSERT_EQ(get<String>(model["config"]["name"]), "dart");
  ASSERT_TRUE(IsA<Object>(model["model"]["gbtree"]));
  ASSERT_NE(get<Array>(model["model"]["weight_drop"]).size(), 0ul);
}

namespace {
class Dart : public testing::TestWithParam<char const*> {
 public:
  void Run(std::string predictor) {
    size_t constexpr kRows = 16, kCols = 10;

    HostDeviceVector<float> data;
    auto rng = RandomDataGenerator(kRows, kCols, 0);
    if (predictor == "gpu_predictor") {
      rng.Device(0);
    }
    auto array_str = rng.GenerateArrayInterface(&data);
    auto p_mat = GetDMatrixFromData(data.HostVector(), kRows, kCols);

    std::vector<bst_float> labels(kRows);
    for (size_t i = 0; i < kRows; ++i) {
      labels[i] = i % 2;
    }
    p_mat->SetInfo("label", labels.data(), DataType::kFloat32, kRows);

    auto learner = std::unique_ptr<Learner>(Learner::Create({p_mat}));
    learner->SetParam("booster", "dart");
    learner->SetParam("rate_drop", "0.5");
    learner->Configure();

    for (size_t i = 0; i < 16; ++i) {
      learner->UpdateOneIter(i, p_mat);
    }

    learner->SetParam("predictor", predictor);

    HostDeviceVector<float> predts_training;
    learner->Predict(p_mat, false, &predts_training, 0, 0, true);

    HostDeviceVector<float>* inplace_predts;
    std::shared_ptr<data::DMatrixProxy> x{new data::DMatrixProxy{}};
    if (predictor == "gpu_predictor") {
      x->SetCUDAArray(array_str.c_str());
    } else {
      x->SetArrayData(array_str.c_str());
    }
    learner->InplacePredict(x, PredictionType::kValue, std::numeric_limits<float>::quiet_NaN(),
                            &inplace_predts, 0, 0);
    CHECK(inplace_predts);

    HostDeviceVector<float> predts_inference;
    learner->Predict(p_mat, false, &predts_inference, 0, 0, false);

    auto const& h_predts_training = predts_training.ConstHostVector();
    auto const& h_predts_inference = predts_inference.ConstHostVector();
    auto const& h_inplace_predts = inplace_predts->HostVector();
    ASSERT_EQ(h_predts_training.size(), h_predts_inference.size());
    ASSERT_EQ(h_inplace_predts.size(), h_predts_inference.size());
    for (size_t i = 0; i < predts_inference.Size(); ++i) {
      // Inference doesn't drop tree.
      ASSERT_GT(std::abs(h_predts_training[i] - h_predts_inference[i]), kRtEps * 10);
      // Inplace prediction is inference.
      ASSERT_LT(h_inplace_predts[i] - h_predts_inference[i], kRtEps / 10);
    }
  }
};
}  // anonymous namespace

TEST_P(Dart, Prediction) { this->Run(GetParam()); }

#if defined(XGBOOST_USE_CUDA)
INSTANTIATE_TEST_SUITE_P(PredictorTypes, Dart,
                         testing::Values("auto", "cpu_predictor", "gpu_predictor"));
#else
INSTANTIATE_TEST_SUITE_P(PredictorTypes, Dart, testing::Values("auto", "cpu_predictor"));
#endif  // defined(XGBOOST_USE_CUDA)


std::pair<Json, Json> TestModelSlice(std::string booster) {
  size_t constexpr kRows = 1000, kCols = 100, kForest = 2, kClasses = 3;
  auto m = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true, false, kClasses);

  int32_t kIters = 10;
  std::unique_ptr<Learner> learner {
    Learner::Create({m})
  };
  learner->SetParams(Args{{"booster", booster},
                          {"tree_method", "hist"},
                          {"num_parallel_tree", std::to_string(kForest)},
                          {"num_class", std::to_string(kClasses)},
                          {"subsample", "0.5"},
                          {"max_depth", "2"}});

  for (auto i = 0; i < kIters; ++i) {
    learner->UpdateOneIter(i, m);
  }

  Json model{Object()};
  Json config{Object()};
  learner->SaveModel(&model);
  learner->SaveConfig(&config);
  bool out_of_bound = false;

  size_t constexpr kSliceStart = 2, kSliceEnd = 8, kStep = 3;
  std::unique_ptr<Learner> sliced {learner->Slice(kSliceStart, kSliceEnd, kStep, &out_of_bound)};
  Json sliced_model{Object()};
  sliced->SaveModel(&sliced_model);

  auto get_shape = [&](Json const& model) {
    if (booster == "gbtree") {
      return get<Object const>(model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]);
    } else {
      return get<Object const>(model["learner"]["gradient_booster"]["gbtree"]["model"]["gbtree_model_param"]);
    }
  };

  auto const& model_shape = get_shape(sliced_model);
  CHECK_EQ(get<String const>(model_shape.at("num_trees")), std::to_string(2 * kClasses * kForest));

  Json sliced_config {Object()};
  sliced->SaveConfig(&sliced_config);
  // Only num trees is changed
  if (booster == "gbtree") {
    sliced_config["learner"]["gradient_booster"]["gbtree_model_param"]["num_trees"] = String("60");
  } else {
    sliced_config["learner"]["gradient_booster"]["gbtree"]["gbtree_model_param"]["num_trees"] =
        String("60");
  }
  CHECK_EQ(sliced_config, config);

  auto get_trees = [&](Json const& model) {
    if (booster == "gbtree") {
      return get<Array const>(model["learner"]["gradient_booster"]["model"]["trees"]);
    } else {
      return get<Array const>(model["learner"]["gradient_booster"]["gbtree"]["model"]["trees"]);
    }
  };

  auto get_info = [&](Json const& model) {
    if (booster == "gbtree") {
      return get<Array const>(model["learner"]["gradient_booster"]["model"]["tree_info"]);
    } else {
      return get<Array const>(model["learner"]["gradient_booster"]["gbtree"]["model"]["tree_info"]);
    }
  };

  auto const &sliced_trees = get_trees(sliced_model);
  CHECK_EQ(sliced_trees.size(), 2 * kClasses * kForest);

  auto constexpr kLayerSize = kClasses * kForest;
  auto const &sliced_info = get_info(sliced_model);

  for (size_t layer = 0; layer < 2; ++layer) {
    for (size_t j = 0; j < kClasses; ++j) {
      for (size_t k = 0; k < kForest; ++k) {
        auto idx = layer * kLayerSize + j * kForest + k;
        auto const &group = get<Integer const>(sliced_info.at(idx));
        CHECK_EQ(static_cast<size_t>(group), j);
      }
    }
  }

  auto const& trees = get_trees(model);

  // Sliced layers are [2, 5]
  auto begin = kLayerSize * kSliceStart;
  auto end = begin + kLayerSize;
  auto j = 0;
  for (size_t i = begin; i < end; ++i) {
    Json tree = trees[i];
    tree["id"] = Integer(0);  // id is different, we set it to 0 to allow comparison.
    auto sliced_tree = sliced_trees[j];
    sliced_tree["id"] = Integer(0);
    CHECK_EQ(tree, sliced_tree);
    j++;
  }

  begin = kLayerSize * (kSliceStart + kStep);
  end = begin + kLayerSize;
  for (size_t i = begin; i < end; ++i) {
    Json tree = trees[i];
    tree["id"] = Integer(0);
    auto sliced_tree = sliced_trees[j];
    sliced_tree["id"] = Integer(0);
    CHECK_EQ(tree, sliced_tree);
    j++;
  }

  // CHECK sliced model doesn't have dependency on the old one
  learner.reset();
  CHECK_EQ(sliced->GetNumFeature(), kCols);

  return std::make_pair(model, sliced_model);
}

TEST(GBTree, Slice) {
  TestModelSlice("gbtree");
}

TEST(Dart, Slice) {
  Json model, sliced_model;
  std::tie(model, sliced_model) = TestModelSlice("dart");
  auto const& weights = get<Array const>(model["learner"]["gradient_booster"]["weight_drop"]);
  auto const& trees = get<Array const>(model["learner"]["gradient_booster"]["gbtree"]["model"]["trees"]);
  ASSERT_EQ(weights.size(), trees.size());
}

TEST(GBTree, FeatureScore) {
  size_t n_samples = 1000, n_features = 10, n_classes = 4;
  auto m = RandomDataGenerator{n_samples, n_features, 0.5}.GenerateDMatrix(true, false, n_classes);

  std::unique_ptr<Learner> learner{ Learner::Create({m}) };
  learner->SetParam("num_class", std::to_string(n_classes));

  learner->Configure();
  for (size_t i = 0; i < 2; ++i) {
    learner->UpdateOneIter(i, m);
  }

  std::vector<bst_feature_t> features_weight;
  std::vector<float> scores_weight;
  learner->CalcFeatureScore("weight", {}, &features_weight, &scores_weight);
  ASSERT_EQ(features_weight.size(), scores_weight.size());
  ASSERT_LE(features_weight.size(), learner->GetNumFeature());
  ASSERT_TRUE(std::is_sorted(features_weight.begin(), features_weight.end()));

  auto test_eq = [&learner, &scores_weight](std::string type) {
    std::vector<bst_feature_t> features;
    std::vector<float> scores;
    learner->CalcFeatureScore(type, {}, &features, &scores);

    std::vector<bst_feature_t> features_total;
    std::vector<float> scores_total;
    learner->CalcFeatureScore("total_" + type, {}, &features_total, &scores_total);

    for (size_t i = 0; i < scores_weight.size(); ++i) {
      ASSERT_LE(RelError(scores_total[i] / scores[i], scores_weight[i]), kRtEps);
    }
  };

  test_eq("gain");
  test_eq("cover");
}

TEST(GBTree, PredictRange) {
  size_t n_samples = 1000, n_features = 10, n_classes = 4;
  auto m = RandomDataGenerator{n_samples, n_features, 0.5}.GenerateDMatrix(true, false, n_classes);

  std::unique_ptr<Learner> learner{Learner::Create({m})};
  learner->SetParam("num_class", std::to_string(n_classes));

  learner->Configure();
  for (size_t i = 0; i < 2; ++i) {
    learner->UpdateOneIter(i, m);
  }
  HostDeviceVector<float> out_predt;
  ASSERT_THROW(learner->Predict(m, false, &out_predt, 0, 3), dmlc::Error);

  auto m_1 =
      RandomDataGenerator{n_samples, n_features, 0.5}.GenerateDMatrix(true, false, n_classes);
  HostDeviceVector<float> out_predt_full;
  learner->Predict(m_1, false, &out_predt_full, 0, 0);
  ASSERT_TRUE(std::equal(out_predt.HostVector().begin(), out_predt.HostVector().end(),
                         out_predt_full.HostVector().begin()));

  {
    // inplace predict
    HostDeviceVector<float> raw_storage;
    auto raw = RandomDataGenerator{n_samples, n_features, 0.5}.GenerateArrayInterface(&raw_storage);
    std::shared_ptr<data::DMatrixProxy> x{new data::DMatrixProxy{}};
    x->SetArrayData(raw.data());

    HostDeviceVector<float>* out_predt;
    learner->InplacePredict(x, PredictionType::kValue, std::numeric_limits<float>::quiet_NaN(),
                            &out_predt, 0, 2);
    auto h_out_predt = out_predt->HostVector();
    learner->InplacePredict(x, PredictionType::kValue, std::numeric_limits<float>::quiet_NaN(),
                            &out_predt, 0, 0);
    auto h_out_predt_full = out_predt->HostVector();

    ASSERT_TRUE(std::equal(h_out_predt.begin(), h_out_predt.end(), h_out_predt_full.begin()));
    // Out of range.
    ASSERT_THROW(learner->InplacePredict(x, PredictionType::kValue,
                                         std::numeric_limits<float>::quiet_NaN(), &out_predt, 0, 3),
                 dmlc::Error);
  }
}
}  // namespace xgboost
