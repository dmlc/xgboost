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
  generic_param.InitAllowUnknown(Args{});
  std::unique_ptr<GradientBooster> p_gbm{
    GradientBooster::Create("gbtree", &generic_param, {}, 0)};
  auto& gbtree = dynamic_cast<gbm::GBTree&> (*p_gbm);

  // Test if `tree_method` can be set
  std::string n_feat = std::to_string(kCols);
  Args args {{"tree_method", "approx"}, {"num_feature", n_feat}};
  gbtree.Configure({args.cbegin(), args.cend()});

  gbtree.Configure(args);
  auto const& tparam = gbtree.GetTrainParam();
  gbtree.Configure({{"tree_method", "approx"}, {"num_feature", n_feat}});
  ASSERT_EQ(tparam.updater_seq, "grow_histmaker,prune");
  gbtree.Configure({{"tree_method", "exact"}, {"num_feature", n_feat}});
  ASSERT_EQ(tparam.updater_seq, "grow_colmaker,prune");
  gbtree.Configure({{"tree_method", "hist"}, {"num_feature", n_feat}});
  ASSERT_EQ(tparam.updater_seq, "grow_quantile_histmaker");
  ASSERT_EQ(tparam.predictor, "cpu_predictor");
  gbtree.Configure({{"booster", "dart"}, {"tree_method", "hist"},
                    {"num_feature", n_feat}});
  ASSERT_EQ(tparam.updater_seq, "grow_quantile_histmaker");
  ASSERT_EQ(tparam.predictor, "cpu_predictor");

#ifdef XGBOOST_USE_CUDA
  generic_param.InitAllowUnknown(Args{{"gpu_id", "0"}});
  gbtree.Configure({{"tree_method", "gpu_hist"}, {"num_feature", n_feat}});
  ASSERT_EQ(tparam.updater_seq, "grow_gpu_hist");
  ASSERT_EQ(tparam.predictor, "gpu_predictor");
  gbtree.Configure({{"booster", "dart"}, {"tree_method", "gpu_hist"},
                    {"num_feature", n_feat}});
  ASSERT_EQ(tparam.updater_seq, "grow_gpu_hist");
  ASSERT_EQ(tparam.predictor, "gpu_predictor");
#endif
}

#ifdef XGBOOST_USE_CUDA
TEST(GBTree, ChoosePredictor) {
  size_t constexpr kNumRows = 17;
  size_t constexpr kCols = 15;
  auto pp_mat = CreateDMatrix(kNumRows, kCols, 0);
  auto& p_mat = *pp_mat;

  std::vector<bst_float> labels (kNumRows);
  for (size_t i = 0; i < kNumRows; ++i) {
    labels[i] = i % 2;
  }
  p_mat->Info().SetInfo("label", labels.data(), DataType::kFloat32, kNumRows);

  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {p_mat};
  std::string n_feat = std::to_string(kCols);
  Args args {{"tree_method", "approx"}, {"num_feature", n_feat}};
  GenericParameter generic_param;
  generic_param.InitAllowUnknown(Args{{"gpu_id", "0"}});

  auto& data = (*(p_mat->GetBatches<SparsePage>().begin())).data;

  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->SetParams(Args{{"tree_method", "gpu_hist"}});
  for (size_t i = 0; i < 4; ++i) {
    learner->UpdateOneIter(i, p_mat.get());
  }
  ASSERT_TRUE(data.HostCanWrite());
  dmlc::TemporaryDirectory tempdir;
  const std::string fname = tempdir.path + "/model_para.bst";

  {
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname.c_str(), "w"));
    learner->Save(fo.get());
  }

  // a new learner
  learner = std::unique_ptr<Learner>(Learner::Create(mat));
  {
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r"));
    learner->Load(fi.get());
  }
  learner->SetParams(Args{{"tree_method", "gpu_hist"}, {"gpu_id", "0"}});
  for (size_t i = 0; i < 4; ++i) {
    learner->UpdateOneIter(i, p_mat.get());
  }
  ASSERT_TRUE(data.HostCanWrite());

  // pull data into device.
  data = HostDeviceVector<Entry>(data.HostVector(), 0);
  data.DeviceSpan();
  ASSERT_FALSE(data.HostCanWrite());

  // another new learner
  learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->SetParams(Args{{"tree_method", "gpu_hist"}, {"gpu_id", "0"}});
  for (size_t i = 0; i < 4; ++i) {
    learner->UpdateOneIter(i, p_mat.get());
  }
  // data is not pulled back into host
  ASSERT_FALSE(data.HostCanWrite());
}
#endif

TEST(GBTree, PredictIncorrect) {
  size_t constexpr kRows = 10;
  size_t constexpr kCols = 10;
  GenericParameter generic_param;
  generic_param.InitAllowUnknown(Args{});
  std::unique_ptr<GradientBooster> p_gbm{
    GradientBooster::Create("gbtree", &generic_param, {}, 0)};
  auto& gbtree = dynamic_cast<gbm::GBTree&> (*p_gbm);
  std::string n_feat = std::to_string(kCols);
  gbtree.Configure(Args{{"tree_method", "exact"},
                        {"num_feature", n_feat}});

  // Create DMatrix with columns != num_feature.
  auto incorrect_test_mat = CreateDMatrix(kRows, kCols - 1, 0);
  HostDeviceVector<float> tmp;
  ASSERT_ANY_THROW(
      { gbtree.PredictBatch(incorrect_test_mat->get(), &tmp, 0); });
  ASSERT_ANY_THROW(
      { gbtree.PredictLeaf(incorrect_test_mat->get(), &tmp.HostVector(), 0); });
  ASSERT_ANY_THROW({
    gbtree.PredictContribution(incorrect_test_mat->get(), &tmp.HostVector(), 0,
                               false, 0, 0);
  });
  ASSERT_ANY_THROW({
    gbtree.PredictInteractionContributions(incorrect_test_mat->get(),
                                           &tmp.HostVector(), 0, false);
  });
  delete incorrect_test_mat;
}

}  // namespace xgboost
