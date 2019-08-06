#include <gtest/gtest.h>
#include <xgboost/generic_parameters.h>
#include "../helpers.h"
#include "../../../src/gbm/gbtree.h"

namespace xgboost {
TEST(GBTree, SelectTreeMethod) {
  using Arg = std::pair<std::string, std::string>;
  size_t constexpr kRows = 10;
  size_t constexpr kCols = 10;
  auto p_shared_ptr_dmat = CreateDMatrix(kRows, kCols, 0);
  auto p_dmat {(*p_shared_ptr_dmat).get()};

  GenericParameter generic_param;
  generic_param.InitAllowUnknown(std::vector<Arg>{Arg("n_gpus", "0")});
  std::unique_ptr<GradientBooster> p_gbm{
    GradientBooster::Create("gbtree", &generic_param, {}, 0)};
  auto& gbtree = dynamic_cast<gbm::GBTree&> (*p_gbm);

  // Test if `tree_method` can be set
  std::string n_feat = std::to_string(kCols);
  std::map<std::string, std::string> args {Arg{"tree_method", "approx"}, Arg{"num_feature", n_feat}};
  gbtree.Configure({args.cbegin(), args.cend()});

  gbtree.ConfigureWithKnownData(args, p_dmat);
  auto const& tparam = gbtree.GetTrainParam();
  gbtree.ConfigureWithKnownData({Arg{"tree_method", "approx"}, Arg{"num_feature", n_feat}}, p_dmat);
  ASSERT_EQ(tparam.updater_seq, "grow_histmaker,prune");
  gbtree.ConfigureWithKnownData({Arg("tree_method", "exact"), Arg("num_feature", n_feat)}, p_dmat);
  ASSERT_EQ(tparam.updater_seq, "grow_colmaker,prune");
  gbtree.ConfigureWithKnownData({Arg("tree_method", "hist"), Arg("num_feature", n_feat)}, p_dmat);
  ASSERT_EQ(tparam.updater_seq, "grow_quantile_histmaker");
  ASSERT_EQ(tparam.predictor, "cpu_predictor");
  gbtree.ConfigureWithKnownData({Arg{"booster", "dart"}, Arg{"tree_method", "hist"},
                                 Arg{"num_feature", n_feat}}, p_dmat);
  ASSERT_EQ(tparam.updater_seq, "grow_quantile_histmaker");
#ifdef XGBOOST_USE_CUDA
  generic_param.InitAllowUnknown(std::vector<Arg>{Arg{"n_gpus", "1"}});
  gbtree.ConfigureWithKnownData({Arg("tree_method", "gpu_hist"), Arg("num_feature", n_feat)},
                                p_dmat);
  ASSERT_EQ(tparam.updater_seq, "grow_gpu_hist");
  ASSERT_EQ(tparam.predictor, "gpu_predictor");
  gbtree.ConfigureWithKnownData({Arg{"booster", "dart"}, Arg{"tree_method", "gpu_hist"},
                                 Arg{"num_feature", n_feat}}, p_dmat);
  ASSERT_EQ(tparam.updater_seq, "grow_gpu_hist");
#endif

  delete p_shared_ptr_dmat;
}
}  // namespace xgboost
