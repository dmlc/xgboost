#include <gtest/gtest.h>
#include <dmlc/filesystem.h>
#include <xgboost/generic_parameters.h>

#include "xgboost/base.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/learner.h"
#include "../helpers.h"
#include "../../../src/gbm/gbtree.h"

namespace xgboost {
TEST(GBTree, SelectTreeMethod) {
  size_t constexpr kCols = 10;

  GenericParameter generic_param;
  generic_param.UpdateAllowUnknown(Args{});
  LearnerModelParam mparam;
  mparam.base_score = 0.5;
  mparam.num_feature = kCols;
  mparam.num_output_group = 1;

  std::vector<std::shared_ptr<DMatrix> > caches;
  std::unique_ptr<GradientBooster> p_gbm {
    GradientBooster::Create("gbtree", &generic_param, &mparam, caches)};
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

#ifdef XGBOOST_USE_CUDA
TEST(GBTree, ChoosePredictor) {
  size_t constexpr kRows = 17;
  size_t constexpr kCols = 15;

  auto pp_dmat = CreateDMatrix(kRows, kCols, 0);
  std::shared_ptr<DMatrix> p_dmat {*pp_dmat};

  auto& data = (*(p_dmat->GetBatches<SparsePage>().begin())).data;
  p_dmat->Info().labels_.Resize(kRows);

  auto learner = std::unique_ptr<Learner>(Learner::Create({p_dmat}));
  learner->SetParams(Args{{"tree_method", "gpu_hist"}, {"gpu_id", "0"}});
  for (size_t i = 0; i < 4; ++i) {
    learner->UpdateOneIter(i, p_dmat.get());
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
    learner->UpdateOneIter(i, p_dmat.get());
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
    learner->UpdateOneIter(i, p_dmat.get());
  }
  // data is not pulled back into host
  ASSERT_FALSE(data.HostCanWrite());

  delete pp_dmat;
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
  ASSERT_EQ(get<Array>(gbtree_model["trees"]).size(), 1);
  ASSERT_EQ(get<Integer>(get<Object>(get<Array>(gbtree_model["trees"]).front()).at("id")), 0);
  ASSERT_EQ(get<Array>(gbtree_model["tree_info"]).size(), 1);

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
  ASSERT_NE(get<Array>(model["model"]["weight_drop"]).size(), 0);
}

TEST(Dart, Prediction) {
  size_t constexpr kRows = 16, kCols = 10;

  auto pp_dmat = CreateDMatrix(kRows, kCols, 0);
  auto& p_mat = *pp_dmat;

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
    learner->UpdateOneIter(i, p_mat.get());
  }

  HostDeviceVector<float> predts_training;
  learner->Predict(p_mat.get(), false, &predts_training, 0, true);
  HostDeviceVector<float> predts_inference;
  learner->Predict(p_mat.get(), false, &predts_inference, 0, false);

  auto& h_predts_training = predts_training.ConstHostVector();
  auto& h_predts_inference = predts_inference.ConstHostVector();
  ASSERT_EQ(h_predts_training.size(), h_predts_inference.size());
  for (size_t i = 0; i < predts_inference.Size(); ++i) {
    // Inference doesn't drop tree.
    ASSERT_GT(std::abs(h_predts_training[i] - h_predts_inference[i]), kRtEps);
  }

  delete pp_dmat;
}
}  // namespace xgboost
