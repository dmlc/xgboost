/*!
 * Copyright 2019-2020 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <dmlc/filesystem.h>
#include <xgboost/generic_parameters.h>

#include "xgboost/base.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/learner.h"
#include "../helpers.h"
#include "../../../src/gbm/gbtree.h"
#include "xgboost/predictor.h"

namespace xgboost {
TEST(GBTree, SelectTreeMethod) {
  size_t constexpr kCols = 10;

  GenericParameter generic_param;
  generic_param.UpdateAllowUnknown(Args{});
  LearnerModelParam mparam;
  mparam.base_score = 0.5;
  mparam.num_feature = kCols;
  mparam.num_output_group = 1;

  std::unique_ptr<GradientBooster> p_gbm {
    GradientBooster::Create("gbtree", &generic_param, &mparam)};
  auto& gbtree = dynamic_cast<gbm::GBTree&> (*p_gbm);

  // Test if `tree_method` can be set
  Args args {{"tree_method", "approx"}};
  gbtree.Configure({args.cbegin(), args.cend()});

  gbtree.Configure(args);
  auto const& tparam = gbtree.GetTrainParam();
  gbtree.Configure({{"tree_method", "approx"}});
  ASSERT_EQ(tparam.updater_seq, "grow_histmaker,prune");
  gbtree.Configure({{"tree_method", "exact"}});
  ASSERT_EQ(tparam.updater_seq, "grow_colmaker,prune");
  gbtree.Configure({{"tree_method", "hist"}});
  ASSERT_EQ(tparam.updater_seq, "grow_quantile_histmaker");
  gbtree.Configure({{"booster", "dart"}, {"tree_method", "hist"}});
  ASSERT_EQ(tparam.updater_seq, "grow_quantile_histmaker");

#ifdef XGBOOST_USE_CUDA
  generic_param.UpdateAllowUnknown(Args{{"gpu_id", "0"}});
  gbtree.Configure({{"tree_method", "gpu_hist"}});
  ASSERT_EQ(tparam.updater_seq, "grow_gpu_hist");
  gbtree.Configure({{"booster", "dart"}, {"tree_method", "gpu_hist"}});
  ASSERT_EQ(tparam.updater_seq, "grow_gpu_hist");
#endif  // XGBOOST_USE_CUDA
}

TEST(GBTree, WrongUpdater) {
  size_t constexpr kRows = 17;
  size_t constexpr kCols = 15;

  auto p_dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  p_dmat->Info().labels_.Resize(kRows);

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
  p_dmat->Info().labels_.Resize(kRows);

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
  data = HostDeviceVector<Entry>(data.HostVector(), 0);
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

  LearnerModelParam mparam;
  mparam.num_feature = kCols;
  mparam.num_output_group = 1;
  mparam.base_score = 0.5;

  GenericParameter gparam;
  gparam.Init(Args{});

  std::unique_ptr<GradientBooster> gbm {
    CreateTrainedGBM("gbtree", Args{}, kRows, kCols, &mparam, &gparam) };

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

  auto j_train_param = model["config"]["gbtree_train_param"];
  ASSERT_EQ(get<String>(j_train_param["num_parallel_tree"]), "1");
}

TEST(Dart, JsonIO) {
  size_t constexpr kRows = 16, kCols = 16;

  LearnerModelParam mparam;
  mparam.num_feature = kCols;
  mparam.base_score = 0.5;
  mparam.num_output_group = 1;

  GenericParameter gparam;
  gparam.Init(Args{});

  std::unique_ptr<GradientBooster> gbm {
    CreateTrainedGBM("dart", Args{}, kRows, kCols, &mparam, &gparam) };

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

TEST(Dart, Prediction) {
  size_t constexpr kRows = 16, kCols = 10;

  auto p_mat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  std::vector<bst_float> labels (kRows);
  for (size_t i = 0; i < kRows; ++i) {
    labels[i] = i % 2;
  }
  p_mat->Info().SetInfo("label", labels.data(), DataType::kFloat32, kRows);

  auto learner = std::unique_ptr<Learner>(Learner::Create({p_mat}));
  learner->SetParam("booster", "dart");
  learner->SetParam("rate_drop", "0.5");
  learner->Configure();

  for (size_t i = 0; i < 16; ++i) {
    learner->UpdateOneIter(i, p_mat);
  }

  HostDeviceVector<float> predts_training;
  learner->Predict(p_mat, false, &predts_training, 0, true);
  HostDeviceVector<float> predts_inference;
  learner->Predict(p_mat, false, &predts_inference, 0, false);

  auto& h_predts_training = predts_training.ConstHostVector();
  auto& h_predts_inference = predts_inference.ConstHostVector();
  ASSERT_EQ(h_predts_training.size(), h_predts_inference.size());
  for (size_t i = 0; i < predts_inference.Size(); ++i) {
    // Inference doesn't drop tree.
    ASSERT_GT(std::abs(h_predts_training[i] - h_predts_inference[i]), kRtEps);
  }
}

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
}  // namespace xgboost
