#include <gtest/gtest.h>
#include <xgboost/generic_parameters.h>
#include "../helpers.h"
#include "../../../src/gbm/gbtree.h"

namespace xgboost {
TEST(GBTree, SelectTreeMethod) {
  using Arg = std::pair<std::string, std::string>;
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
}  // namespace xgboost
