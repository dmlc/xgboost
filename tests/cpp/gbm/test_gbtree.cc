#include <gtest/gtest.h>
#include <xgboost/generic_parameters.h>
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
}  // namespace xgboost
