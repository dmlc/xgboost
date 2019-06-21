#include <gtest/gtest.h>
#include <xgboost/generic_parameters.h>
#include "../helpers.h"
#include "../../../src/gbm/gbtree.h"

namespace xgboost {
using Arg = std::pair<std::string, std::string>;
TEST(GBTree, SelectTreeMethod) {
  size_t constexpr kRows = 10;
  size_t constexpr kCols = 10;
  auto mat_ptr = CreateDMatrix(kRows, kCols, 0);
  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {*mat_ptr};

  LearnerTrainParam learner_param;
  learner_param.InitAllowUnknown(std::vector<Arg>{Arg("n_gpus", "0")});
  std::unique_ptr<GradientBooster> p_gbm{
    GradientBooster::Create("gbtree", &learner_param, {}, 0)};
  auto& gbtree = dynamic_cast<gbm::GBTree&> (*p_gbm);

  // Test if `tree_method` can be set
  std::string n_feat = std::to_string(kCols);
  gbtree.Configure({Arg{"tree_method", "approx"}, Arg{"num_feature", n_feat}});
  auto const& tparam = gbtree.GetTrainParam();
  ASSERT_EQ(tparam.updater_seq, "grow_histmaker,prune");
  gbtree.Configure({Arg("tree_method", "exact"), Arg("num_feature", n_feat)});
  ASSERT_EQ(tparam.updater_seq, "grow_colmaker,prune");
  gbtree.Configure({Arg("tree_method", "hist"), Arg("num_feature", n_feat)});
  ASSERT_EQ(tparam.updater_seq, "grow_quantile_histmaker");
  ASSERT_EQ(tparam.predictor, "cpu_predictor");
  gbtree.Configure({Arg{"booster", "dart"}, Arg{"tree_method", "hist"},
                    Arg{"num_feature", n_feat}});
  ASSERT_EQ(tparam.updater_seq, "grow_quantile_histmaker");
#ifdef XGBOOST_USE_CUDA
  learner_param.InitAllowUnknown(std::vector<Arg>{Arg{"n_gpus", "1"}});
  gbtree.Configure({Arg("tree_method", "gpu_exact"),
                    Arg("num_feature", n_feat)});
  ASSERT_EQ(tparam.updater_seq, "grow_gpu,prune");
  ASSERT_EQ(tparam.predictor, "gpu_predictor");
  gbtree.Configure({Arg("tree_method", "gpu_hist"), Arg("num_feature", n_feat)});
  ASSERT_EQ(tparam.updater_seq, "grow_gpu_hist");
  ASSERT_EQ(tparam.predictor, "gpu_predictor");
  gbtree.Configure({Arg{"booster", "dart"}, Arg{"tree_method", "gpu_hist"},
                    Arg{"num_feature", n_feat}});
  ASSERT_EQ(tparam.updater_seq, "grow_gpu_hist");
#endif

  delete mat_ptr;
}

TEST(GBTree, PredictIncorrect) {
  size_t constexpr kRows = 10;
  size_t constexpr kCols = 10;
  LearnerTrainParam learner_param;
  learner_param.InitAllowUnknown(std::vector<Arg>{});
  std::unique_ptr<GradientBooster> p_gbm{
    GradientBooster::Create("gbtree", &learner_param, {}, 0)};
  auto& gbtree = dynamic_cast<gbm::GBTree&> (*p_gbm);
  std::string n_feat = std::to_string(kCols);
  gbtree.Configure({Arg("tree_method", "exact"), Arg("num_feature", n_feat)});
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
