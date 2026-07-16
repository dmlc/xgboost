/**
 * Copyright 2025, XGBoost contributors
 */
#include <gtest/gtest.h>

#include <cmath>  // for isnan

#include "../../../../src/tree/gpu_hist/evaluate_splits.cuh"
#include "../../../../src/tree/gpu_hist/multi_evaluate_splits.cuh"
#include "../../helpers.h"
#include "dummy_quantizer.cuh"  // for MakeDummyQuantizers

namespace xgboost::tree::cuda_impl {
class GpuMultiHistEvaluatorBasicTest : public ::testing::Test {
 public:
  Context ctx{MakeCUDACtx(0)};
  bst_target_t n_targets = 2;
  bst_bin_t n_bins_per_feat_tar = 4;

  dh::device_vector<GradientPairInt64> parent_sum;
  dh::device_vector<GradientPairInt64> histogram;
  MultiEvaluateSplitInputs input;
  dh::device_vector<GradientQuantiser> quantizers;
  MultiEvaluateSplitSharedInputs shared_inputs;

  dh::device_vector<bst_feature_t> feature_segments;
  dh::device_vector<bst_feature_t> feature_set;
  dh::device_vector<float> feature_values{.0f, .1f, .2f, .3f};

  void SetUp() override {
    input.nidx = 0;
    input.depth = 0;

    parent_sum.resize(n_targets);
    parent_sum[0] = GradientPairInt64{56, 40};
    parent_sum[1] = GradientPairInt64{96, 128};

    histogram.resize(n_bins_per_feat_tar * n_targets);
    // first target, dense,                    // 0/0, 56/40
    histogram[0] = GradientPairInt64{8, 4};    // 8/4, 48/36
    histogram[1] = GradientPairInt64{12, 8};   // 20/12, 36/28
    histogram[2] = GradientPairInt64{16, 12};  // 36/24, 20/16
    histogram[3] = GradientPairInt64{20, 16};  // 56/40, 0/0

    // second target, dense                    // 0/0,  96/128
    histogram[4] = GradientPairInt64{11, 13};  // 11/13, 85/115
    histogram[5] = GradientPairInt64{19, 29};  // 30/42, 66/86
    histogram[6] = GradientPairInt64{27, 45};  // 57/87, 39/41
    histogram[7] = GradientPairInt64{39, 41};  // 96/128, 0/0

    input.parent_sum = dh::ToSpan(parent_sum);
    input.histogram = dh::ToSpan(histogram);

    quantizers = MakeDummyQuantizers(2);

    shared_inputs.roundings = dh::ToSpan(quantizers);

    feature_segments.resize(2);
    feature_segments[0] = 0;
    feature_segments[1] = static_cast<bst_feature_t>(n_bins_per_feat_tar);
    shared_inputs.feature_segments = dh::ToSpan(feature_segments);

    feature_set.resize(1, 0);
    input.feature_set = dh::ToSpan(feature_set);

    shared_inputs.feature_values = dh::ToSpan(feature_values).data();

    shared_inputs.n_total_bins_per_tar = n_bins_per_feat_tar;
    shared_inputs.max_active_feature = 1;

    TrainParam param;
    param.Init(Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}, {"learning_rate", "1"}});
    shared_inputs.param = GPUTrainingParam{param};
  }

  void TestEmptyHess() {
    // Turn all Hessian values into 0.
    thrust::transform(histogram.begin(), histogram.end(), histogram.begin(),
                      [] XGBOOST_DEVICE(GradientPairInt64 const& bin) {
                        return GradientPairInt64{bin.GetQuantisedGrad(), 0};
                      });
    MultiHistEvaluator evaluator;
    auto candidate = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs);
    TrainParam param;
    param.Init(Args{});
    ASSERT_FALSE(candidate.IsValid(param, 100));
  }
};

namespace {
template <typename T, typename V = std::remove_cv_t<T>>
void AssertDeviceVecEq(common::Span<T> span, std::vector<V> const& exp) {
  std::vector<V> h_vec(span.size());
  dh::CopyDeviceSpanToVector(&h_vec, span);
  AssertVecEq(h_vec, exp);
}
}  // namespace

TEST_F(GpuMultiHistEvaluatorBasicTest, Root) {
  using OnePass = MultiEvaluateSplitSharedInputs;

  std::vector<GradientPairInt64> exp_left_sum{{36, 24}, {57, 87}};
  std::vector<GradientPairInt64> exp_right_sum{{20, 16}, {39, 41}};
  std::vector<float> exp_base_weight{-1.4, -0.75};
  std::vector<float> exp_left_weight{-1.5, -0.655172};
  std::vector<float> exp_right_weight{-1.25, -0.951219};

  for (auto one_pass : {OnePass::kNone, OnePass::kForward, OnePass::kBackward}) {
    auto shared = this->shared_inputs;
    shared.one_pass = one_pass;
    MultiHistEvaluator evaluator;
    auto candidate = evaluator.EvaluateSingleSplit(&ctx, input, shared);
    ASSERT_NEAR(candidate.split.loss_chg, 3.04239, 1e-5);

    std::vector<float> base, left, right;
    evaluator.CopyNodeWeightsToHost(candidate.nidx, candidate.base_weight.size(), &base, &left,
                                    &right);
    AssertVecEq(base, exp_base_weight);
    AssertVecEq(left, exp_left_weight);
    AssertVecEq(right, exp_right_weight);

    std::stringstream ss;
    ss << candidate;
    auto str = ss.str();
    if (one_pass != OnePass::kBackward) {
      ASSERT_NE(str.find("left_sum"), std::string::npos);
      ASSERT_EQ(str.find("right_sum"), std::string::npos);
      AssertDeviceVecEq(candidate.split.child_sum, exp_left_sum);
    } else {
      ASSERT_EQ(str.find("left_sum"), std::string::npos);
      ASSERT_NE(str.find("right_sum"), std::string::npos);
      AssertDeviceVecEq(candidate.split.child_sum, exp_right_sum);
    }
  }
}

TEST_F(GpuMultiHistEvaluatorBasicTest, EmptyHess) { this->TestEmptyHess(); }

TEST_F(GpuMultiHistEvaluatorBasicTest, CategoricalOneHot) {
  // Reuse the dense histogram from the fixture, but treat the single feature as
  // categorical with category ids {0, 1, 2, 3} as the cut values. Since the histogram is
  // dense (feature_sum == parent_sum), there are no missing values, so the one-hot split
  // separates the chosen category from the rest. The best partition is category 3 (bin
  // 3), which is the same partition the numerical `Root` test finds, hence the identical
  // loss_chg / child weights.
  dh::device_vector<float> cat_values{0.0f, 1.0f, 2.0f, 3.0f};
  dh::device_vector<FeatureType> ft(1, FeatureType::kCategorical);

  auto shared = this->shared_inputs;
  shared.feature_values = dh::ToSpan(cat_values).data();
  shared.feature_types = dh::ToSpan(ft);
  shared.cat_storage_size = common::CatBitField::ComputeStorageSize(4);
  TrainParam param;
  param.Init(Args{{"min_child_weight", "0"},
                  {"reg_lambda", "0"},
                  {"learning_rate", "1"},
                  {"max_cat_to_onehot", "100"}});
  shared.param = GPUTrainingParam{param};

  MultiHistEvaluator evaluator;
  evaluator.Reset(&ctx, shared.feature_segments, shared.feature_types, param);
  auto candidate = evaluator.EvaluateSingleSplit(&ctx, input, shared);

  ASSERT_TRUE(candidate.split.is_cat);
  ASSERT_NEAR(candidate.split.loss_chg, 3.04239, 1e-5);
  // The matching category goes right with the missing values; the chosen child sum stored
  // is the non-missing "other categories" (left child), so dir is kRightDir.
  ASSERT_EQ(candidate.split.dir, kRightDir);
  ASSERT_EQ(static_cast<bst_cat_t>(candidate.split.fvalue), 3);
  auto h_cats = evaluator.GetHostNodeCats(candidate.nidx);
  common::KCatBitField cats{h_cats};
  ASSERT_TRUE(cats.Check(3));

  // child_sum is the "other categories" = parent - hist[bin 3].
  std::vector<GradientPairInt64> exp_child_sum{{36, 24}, {57, 87}};
  AssertDeviceVecEq(candidate.split.child_sum, exp_child_sum);

  std::vector<float> base, left, right;
  evaluator.CopyNodeWeightsToHost(candidate.nidx, candidate.base_weight.size(), &base, &left,
                                  &right);
  // Left child = other categories {36,24},{57,87}; right child = category 3 {20,16},{39,41}.
  std::vector<float> exp_base_weight{-1.4, -0.75};
  std::vector<float> exp_left_weight{-1.5, -0.655172};
  std::vector<float> exp_right_weight{-1.25, -0.951219};
  AssertVecEq(base, exp_base_weight);
  AssertVecEq(left, exp_left_weight);
  AssertVecEq(right, exp_right_weight);
}

TEST_F(GpuMultiHistEvaluatorBasicTest, CategoricalPartition) {
  // The optimal split groups categories {2, 3} together. No one-hot split can represent
  // this 2-vs-2 partition.
  parent_sum[0] = GradientPairInt64{-9, 4};
  parent_sum[1] = GradientPairInt64{-7, 4};

  // target 0
  histogram[0] = GradientPairInt64{-4, 1};
  histogram[1] = GradientPairInt64{-3, 1};
  histogram[2] = GradientPairInt64{-1, 1};
  histogram[3] = GradientPairInt64{-1, 1};
  // target 1
  histogram[4] = GradientPairInt64{-3, 1};
  histogram[5] = GradientPairInt64{-2, 1};
  histogram[6] = GradientPairInt64{-1, 1};
  histogram[7] = GradientPairInt64{-1, 1};

  dh::device_vector<float> cat_values{0.0f, 1.0f, 2.0f, 3.0f};
  dh::device_vector<FeatureType> ft(1, FeatureType::kCategorical);

  auto shared = this->shared_inputs;
  shared.feature_values = dh::ToSpan(cat_values).data();
  shared.feature_types = dh::ToSpan(ft);
  shared.cat_storage_size = common::CatBitField::ComputeStorageSize(4);
  TrainParam param;
  param.Init(Args{{"min_child_weight", "0"},
                  {"reg_lambda", "0"},
                  {"learning_rate", "1"},
                  {"max_cat_to_onehot", "1"}});
  shared.param = GPUTrainingParam{param};

  MultiHistEvaluator evaluator;
  evaluator.Reset(&ctx, shared.feature_segments, shared.feature_types, param);
  auto candidate = evaluator.EvaluateSingleSplit(&ctx, input, shared);

  ASSERT_TRUE(candidate.split.is_cat);
  ASSERT_NEAR(candidate.split.loss_chg, 8.5, 1e-5);
  ASSERT_EQ(candidate.split.dir, kLeftDir);
  ASSERT_EQ(candidate.split.findex, 0);
  ASSERT_TRUE(std::isnan(candidate.split.fvalue));
  ASSERT_EQ(candidate.split.thresh, 1);
  auto h_cats = evaluator.GetHostNodeCats(candidate.nidx);
  common::KCatBitField cats{h_cats};
  ASSERT_FALSE(cats.Check(0));
  ASSERT_FALSE(cats.Check(1));
  ASSERT_TRUE(cats.Check(2));
  ASSERT_TRUE(cats.Check(3));

  // Categories {2, 3} form the right child. `child_sum` stores the non-missing child,
  // which is the right child when missing values default left.
  std::vector<GradientPairInt64> exp_child_sum{{-2, 2}, {-2, 2}};
  AssertDeviceVecEq(candidate.split.child_sum, exp_child_sum);

  std::vector<float> base, left, right;
  evaluator.CopyNodeWeightsToHost(candidate.nidx, candidate.base_weight.size(), &base, &left,
                                  &right);
  AssertVecEq(base, std::vector<float>{2.25f, 1.75f});
  AssertVecEq(left, std::vector<float>{3.5f, 2.5f});
  AssertVecEq(right, std::vector<float>{1.0f, 1.0f});
  ASSERT_EQ(candidate.left_sum, 4.0);
  ASSERT_EQ(candidate.right_sum, 4.0);

  // max_cat_threshold=1 does not enumerate a partition. The resulting root-only tree
  // must still retain its base weight and Hessian.
  TrainParam no_split_param;
  no_split_param.Init(Args{{"min_child_weight", "0"},
                           {"reg_lambda", "0"},
                           {"learning_rate", "1"},
                           {"max_cat_to_onehot", "1"},
                           {"max_cat_threshold", "1"}});
  shared.param = GPUTrainingParam{no_split_param};

  MultiHistEvaluator no_split_evaluator;
  no_split_evaluator.Reset(&ctx, shared.feature_segments, shared.feature_types, no_split_param);
  auto no_split = no_split_evaluator.EvaluateSingleSplit(&ctx, input, shared);

  ASSERT_TRUE(no_split.split.child_sum.empty());
  ASSERT_FALSE(no_split.IsValid(no_split_param, 100));
  AssertDeviceVecEq(no_split.base_weight, std::vector<float>{2.25f, 1.75f});
  ASSERT_EQ(no_split.left_sum, 8.0);
  ASSERT_EQ(no_split.right_sum, 0.0);
}
}  // namespace xgboost::tree::cuda_impl
