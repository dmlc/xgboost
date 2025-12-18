/**
 * Copyright 2025, XGBoost contributors
 */
#include <gtest/gtest.h>

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
  dh::device_vector<float> feature_values{.0f, .1f, .2f, .3f};
  dh::device_vector<float> min_values{-1.0f};

  void SetUp() override {
    parent_sum.resize(n_targets);
    parent_sum[0] = GradientPairInt64{56, 40};
    parent_sum[1] = GradientPairInt64{96, 128};

    histogram.resize(n_bins_per_feat_tar * n_targets);
    // first target, dense,                    // 0/0, 56/40
    histogram[0] = GradientPairInt64{8, 4};    // 8/4, 48/36
    histogram[2] = GradientPairInt64{12, 8};   // 20/12, 36/28
    histogram[4] = GradientPairInt64{16, 12};  // 36/24, 20/16
    histogram[6] = GradientPairInt64{20, 16};  // 56/40, 0/0

    // second target, dense                    // 0/0,  96/128
    histogram[1] = GradientPairInt64{11, 13};  // 11/13, 85/115
    histogram[3] = GradientPairInt64{19, 29};  // 30/42, 66/86
    histogram[5] = GradientPairInt64{27, 45};  // 57/87, 39/41
    histogram[7] = GradientPairInt64{39, 41};  // 96/128, 0/0

    input.parent_sum = dh::ToSpan(parent_sum);
    input.histogram = dh::ToSpan(histogram);

    quantizers = MakeDummyQuantizers(2);

    shared_inputs.roundings = dh::ToSpan(quantizers);

    feature_segments.resize(2);
    feature_segments[0] = 0;
    feature_segments[1] = static_cast<bst_feature_t>(n_bins_per_feat_tar);
    shared_inputs.feature_segments = dh::ToSpan(feature_segments);

    shared_inputs.feature_values = dh::ToSpan(feature_values).data();
    shared_inputs.min_values = dh::ToSpan(min_values).data();

    shared_inputs.n_bins_per_feat_tar = n_bins_per_feat_tar;
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
    ASSERT_TRUE(candidate.base_weight.empty());
    ASSERT_TRUE(candidate.left_weight.empty());
    ASSERT_TRUE(candidate.right_weight.empty());
    ASSERT_TRUE(candidate.split.child_sum.empty());
  }
};

namespace {
template <typename T, typename V = std::remove_cv_t<T>>
void CheckSpan(common::Span<T> span, std::vector<V> const& exp) {
  std::vector<V> h_vec(span.size());
  dh::CopyDeviceSpanToVector(&h_vec, span);
  ASSERT_EQ(h_vec.size(), exp.size());
  for (std::size_t i = 0; i < h_vec.size(); ++i) {
    if constexpr (std::is_floating_point_v<V>) {
      ASSERT_NEAR(h_vec[i], exp[i], 1e-5);
    } else {
      ASSERT_EQ(h_vec[i], exp[i]);
    }
  }
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
    CheckSpan(candidate.left_weight, exp_left_weight);
    CheckSpan(candidate.right_weight, exp_right_weight);
    CheckSpan(candidate.base_weight, exp_base_weight);

    std::stringstream ss;
    ss << candidate;
    auto str = ss.str();
    if (one_pass != OnePass::kBackward) {
      ASSERT_NE(str.find("left_sum"), std::string::npos);
      ASSERT_EQ(str.find("right_sum"), std::string::npos);
      CheckSpan(candidate.split.child_sum, exp_left_sum);
    } else {
      ASSERT_EQ(str.find("left_sum"), std::string::npos);
      ASSERT_NE(str.find("right_sum"), std::string::npos);
      CheckSpan(candidate.split.child_sum, exp_right_sum);
    }
  }
}

TEST_F(GpuMultiHistEvaluatorBasicTest, EmptyHess) { this->TestEmptyHess(); }

TEST(EvalScan, Basic) {
  bst_target_t n_targets = 2;
  bst_bin_t n_bins_per_feat_tar = 64;

  dh::device_vector<GradientPairInt64> parent_sum;
  dh::device_vector<GradientPairInt64> histogram;
  MultiEvaluateSplitInputs input;
  dh::device_vector<GradientQuantiser> quantizers;
  MultiEvaluateSplitSharedInputs shared_inputs;

  dh::device_vector<bst_feature_t> feature_segments;
  dh::device_vector<float> feature_values(n_bins_per_feat_tar, 0);
  dh::device_vector<float> min_values{-1.0f};

  // setup
  parent_sum.resize(n_targets);
  parent_sum[0] = GradientPairInt64{56, 40};
  parent_sum[1] = GradientPairInt64{96, 128};

  histogram.resize(n_bins_per_feat_tar * n_targets);
  thrust::fill(histogram.begin(), histogram.end(), GradientPairInt64{1, 1});

  input.parent_sum = dh::ToSpan(parent_sum);
  input.histogram = dh::ToSpan(histogram);

  quantizers = MakeDummyQuantizers(2);

  shared_inputs.roundings = dh::ToSpan(quantizers);

  feature_segments.resize(2);
  feature_segments[0] = 0;
  feature_segments[1] = static_cast<bst_feature_t>(n_bins_per_feat_tar);
  shared_inputs.feature_segments = dh::ToSpan(feature_segments);

  shared_inputs.feature_values = dh::ToSpan(feature_values).data();
  shared_inputs.min_values = dh::ToSpan(min_values).data();

  shared_inputs.n_bins_per_feat_tar = n_bins_per_feat_tar;
  TrainParam param;
  param.Init(Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}, {"learning_rate", "0.3"}});
  shared_inputs.param = GPUTrainingParam{param};
  shared_inputs.one_pass = MultiEvaluateSplitSharedInputs::kForward;

  auto ctx = MakeCUDACtx(0);
  MultiHistEvaluator evaluator;
  auto candidate = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs);
}
}  // namespace xgboost::tree::cuda_impl
