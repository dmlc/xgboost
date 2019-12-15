#include <gtest/gtest.h>
#include <dmlc/filesystem.h>
#include <xgboost/generic_parameters.h>

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
  std::unique_ptr<GradientBooster> p_gbm{
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

// Some other parts of test are in `Tree.Json_IO'.
TEST(GBTree, Json_IO) {
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

TEST(Dart, Json_IO) {
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
}  // namespace xgboost
