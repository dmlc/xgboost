/*!
 * Copyright 2020-2022 by XGBoost contributors
 */
#include <gtest/gtest.h>
#include "../../../../src/tree/gpu_hist/evaluate_splits.cuh"
#include "../../helpers.h"
#include "../../histogram_helpers.h"
#include "../test_evaluate_splits.h"  // TestPartitionBasedSplit

namespace xgboost {
namespace tree {
namespace {
auto ZeroParam() {
  auto args = Args{{"min_child_weight", "0"},
                   {"lambda", "0"}};
  TrainParam tparam;
  tparam.UpdateAllowUnknown(args);
  return tparam;
}
}  // anonymous namespace

void TestEvaluateSingleSplit(bool is_categorical) {
  GradientPairPrecise parent_sum(0.0, 1.0);
  TrainParam tparam = ZeroParam();
  GPUTrainingParam param{tparam};

  common::HistogramCuts cuts;
  cuts.cut_values_.HostVector() = std::vector<float>{1.0, 2.0, 11.0, 12.0};
  cuts.cut_ptrs_.HostVector() = std::vector<uint32_t>{0, 2, 4};
  cuts.min_vals_.HostVector() =  std::vector<float>{0.0, 0.0};
  cuts.cut_ptrs_.SetDevice(0);
  cuts.cut_values_.SetDevice(0);
  cuts.min_vals_.SetDevice(0);
  thrust::device_vector<bst_feature_t> feature_set =
      std::vector<bst_feature_t>{0, 1};

  // Setup gradients so that second feature gets higher gain
  thrust::device_vector<GradientPairPrecise> feature_histogram =
      std::vector<GradientPairPrecise>{
          {-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}};

  thrust::device_vector<int> monotonic_constraints(feature_set.size(), 0);
  dh::device_vector<FeatureType> feature_types(feature_set.size(),
                                               FeatureType::kCategorical);
  common::Span<FeatureType> d_feature_types;
  if (is_categorical) {
    auto max_cat = *std::max_element(cuts.cut_values_.HostVector().begin(),
                                     cuts.cut_values_.HostVector().end());
    cuts.SetCategorical(true, max_cat);
    d_feature_types = dh::ToSpan(feature_types);
  }

  EvaluateSplitInputs input{1,0,
                                          parent_sum,
                                          dh::ToSpan(feature_set),
                                          dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{
      param,
      d_feature_types,
      cuts.cut_ptrs_.ConstDeviceSpan(),
      cuts.cut_values_.ConstDeviceSpan(),
      cuts.min_vals_.ConstDeviceSpan(),
  };

  GPUHistEvaluator evaluator{
      tparam, static_cast<bst_feature_t>(feature_set.size()), 0};
  evaluator.Reset(cuts, dh::ToSpan(feature_types), feature_set.size(), tparam, 0);
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(input, shared_inputs).split;

  EXPECT_EQ(result.findex, 1);
  EXPECT_EQ(result.fvalue, 11.0);
  EXPECT_FLOAT_EQ(result.left_sum.GetGrad() + result.right_sum.GetGrad(),
                  parent_sum.GetGrad());
  EXPECT_FLOAT_EQ(result.left_sum.GetHess() + result.right_sum.GetHess(),
                  parent_sum.GetHess());
}

TEST(GpuHist, EvaluateSingleSplit) {
  TestEvaluateSingleSplit(false);
}

TEST(GpuHist, EvaluateCategoricalSplit) {
  TestEvaluateSingleSplit(true);
}

TEST(GpuHist, EvaluateSingleSplitMissing) {
  GradientPairPrecise parent_sum(1.0, 1.5);
  TrainParam tparam = ZeroParam();
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set =
      std::vector<bst_feature_t>{0};
  thrust::device_vector<uint32_t> feature_segments =
      std::vector<bst_row_t>{0, 2};
  thrust::device_vector<float> feature_values = std::vector<float>{1.0, 2.0};
  thrust::device_vector<float> feature_min_values = std::vector<float>{0.0};
  thrust::device_vector<GradientPairPrecise> feature_histogram =
      std::vector<GradientPairPrecise>{{-0.5, 0.5}, {0.5, 0.5}};
  thrust::device_vector<int> monotonic_constraints(feature_set.size(), 0);
  EvaluateSplitInputs input{1,0,
                                          parent_sum,
                                          dh::ToSpan(feature_set),
                                          dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{
      param,
      {},
      dh::ToSpan(feature_segments),
      dh::ToSpan(feature_values),
      dh::ToSpan(feature_min_values),
  };

  GPUHistEvaluator evaluator(tparam, feature_set.size(), 0);
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(input, shared_inputs).split;

  EXPECT_EQ(result.findex, 0);
  EXPECT_EQ(result.fvalue, 1.0);
  EXPECT_EQ(result.dir, kRightDir);
  EXPECT_EQ(result.left_sum, GradientPairPrecise(-0.5, 0.5));
  EXPECT_EQ(result.right_sum, GradientPairPrecise(1.5, 1.0));
}

TEST(GpuHist, EvaluateSingleSplitEmpty) {
  TrainParam tparam = ZeroParam();
  GPUHistEvaluator evaluator(tparam, 1, 0);
  DeviceSplitCandidate result =
      evaluator.EvaluateSingleSplit(EvaluateSplitInputs{}, EvaluateSplitSharedInputs{}).split;
  EXPECT_EQ(result.findex, -1);
  EXPECT_LT(result.loss_chg, 0.0f);
}

// Feature 0 has a better split, but the algorithm must select feature 1
TEST(GpuHist, EvaluateSingleSplitFeatureSampling) {
  GradientPairPrecise parent_sum(0.0, 1.0);
  TrainParam tparam = ZeroParam();
  tparam.UpdateAllowUnknown(Args{});
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set =
      std::vector<bst_feature_t>{1};
  thrust::device_vector<uint32_t> feature_segments =
      std::vector<bst_row_t>{0, 2, 4};
  thrust::device_vector<float> feature_values =
      std::vector<float>{1.0, 2.0, 11.0, 12.0};
  thrust::device_vector<float> feature_min_values =
      std::vector<float>{0.0, 10.0};
  thrust::device_vector<GradientPairPrecise> feature_histogram =
      std::vector<GradientPairPrecise>{
          {-10.0, 0.5}, {10.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}};
  thrust::device_vector<int> monotonic_constraints(2, 0);
  EvaluateSplitInputs input{1,0,
                                          parent_sum,
                                          dh::ToSpan(feature_set),
                                          dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{
      param,
                                          {},
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
  };

  GPUHistEvaluator evaluator(tparam, feature_min_values.size(), 0);
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(input, shared_inputs).split;

  EXPECT_EQ(result.findex, 1);
  EXPECT_EQ(result.fvalue, 11.0);
  EXPECT_EQ(result.left_sum, GradientPairPrecise(-0.5, 0.5));
  EXPECT_EQ(result.right_sum, GradientPairPrecise(0.5, 0.5));
}

// Features 0 and 1 have identical gain, the algorithm must select 0
TEST(GpuHist, EvaluateSingleSplitBreakTies) {
  GradientPairPrecise parent_sum(0.0, 1.0);
  TrainParam tparam = ZeroParam();
  tparam.UpdateAllowUnknown(Args{});
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set =
      std::vector<bst_feature_t>{0, 1};
  thrust::device_vector<uint32_t> feature_segments =
      std::vector<bst_row_t>{0, 2, 4};
  thrust::device_vector<float> feature_values =
      std::vector<float>{1.0, 2.0, 11.0, 12.0};
  thrust::device_vector<float> feature_min_values =
      std::vector<float>{0.0, 10.0};
  thrust::device_vector<GradientPairPrecise> feature_histogram =
      std::vector<GradientPairPrecise>{
          {-0.5, 0.5}, {0.5, 0.5}, {-0.5, 0.5}, {0.5, 0.5}};
  thrust::device_vector<int> monotonic_constraints(2, 0);
  EvaluateSplitInputs input{1,0,
                                          parent_sum,
                                          dh::ToSpan(feature_set),
                                          dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{
      param,
                                          {},
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
  };

  GPUHistEvaluator evaluator(tparam, feature_min_values.size(), 0);
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(input,shared_inputs).split;

  EXPECT_EQ(result.findex, 0);
  EXPECT_EQ(result.fvalue, 1.0);
}

TEST(GpuHist, EvaluateSplits) {
  thrust::device_vector<DeviceSplitCandidate> out_splits(2);
  GradientPairPrecise parent_sum(0.0, 1.0);
  TrainParam tparam = ZeroParam();
  tparam.UpdateAllowUnknown(Args{});
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set =
      std::vector<bst_feature_t>{0, 1};
  thrust::device_vector<uint32_t> feature_segments =
      std::vector<bst_row_t>{0, 2, 4};
  thrust::device_vector<float> feature_values =
      std::vector<float>{1.0, 2.0, 11.0, 12.0};
  thrust::device_vector<float> feature_min_values =
      std::vector<float>{0.0, 0.0};
  thrust::device_vector<GradientPairPrecise> feature_histogram_left =
      std::vector<GradientPairPrecise>{
          {-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}};
  thrust::device_vector<GradientPairPrecise> feature_histogram_right =
      std::vector<GradientPairPrecise>{
          {-1.0, 0.5}, {1.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}};
  thrust::device_vector<int> monotonic_constraints(feature_set.size(), 0);
  EvaluateSplitInputs input_left{
      1,0,
      parent_sum,
      dh::ToSpan(feature_set),
      dh::ToSpan(feature_histogram_left)};
  EvaluateSplitInputs input_right{
      2,0,
      parent_sum,
      dh::ToSpan(feature_set),
      dh::ToSpan(feature_histogram_right)};
  EvaluateSplitSharedInputs shared_inputs{
      param,
                                          {},
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
  };

  GPUHistEvaluator evaluator{
      tparam, static_cast<bst_feature_t>(feature_min_values.size()), 0};
  dh::device_vector<EvaluateSplitInputs> inputs = std::vector<EvaluateSplitInputs>{input_left,input_right};
  evaluator.LaunchEvaluateSplits(input_left.feature_set.size(),dh::ToSpan(inputs),shared_inputs, evaluator.GetEvaluator(),
                           dh::ToSpan(out_splits));

  DeviceSplitCandidate result_left = out_splits[0];
  EXPECT_EQ(result_left.findex, 1);
  EXPECT_EQ(result_left.fvalue, 11.0);

  DeviceSplitCandidate result_right = out_splits[1];
  EXPECT_EQ(result_right.findex, 0);
  EXPECT_EQ(result_right.fvalue, 1.0);
}

TEST_F(TestPartitionBasedSplit, GpuHist) {
  dh::device_vector<FeatureType> ft{std::vector<FeatureType>{FeatureType::kCategorical}};
  GPUHistEvaluator evaluator{param_,
                                                  static_cast<bst_feature_t>(info_.num_col_), 0};

  cuts_.cut_ptrs_.SetDevice(0);
  cuts_.cut_values_.SetDevice(0);
  cuts_.min_vals_.SetDevice(0);

  evaluator.Reset(cuts_, dh::ToSpan(ft), info_.num_col_, param_, 0);

  dh::device_vector<GradientPairPrecise> d_hist(hist_[0].size());
  auto node_hist = hist_[0];
  dh::safe_cuda(cudaMemcpy(d_hist.data().get(), node_hist.data(), node_hist.size_bytes(),
                           cudaMemcpyHostToDevice));
  dh::device_vector<bst_feature_t> feature_set{std::vector<bst_feature_t>{0}};

  EvaluateSplitInputs input{0, 0, total_gpair_, dh::ToSpan(feature_set), dh::ToSpan(d_hist)};
  EvaluateSplitSharedInputs shared_inputs{
      GPUTrainingParam{param_},          dh::ToSpan(ft),
      cuts_.cut_ptrs_.ConstDeviceSpan(), cuts_.cut_values_.ConstDeviceSpan(),
      cuts_.min_vals_.ConstDeviceSpan(),
  };
  auto split = evaluator.EvaluateSingleSplit(input, shared_inputs).split;
  ASSERT_NEAR(split.loss_chg, best_score_, 1e-16);
}

}  // namespace tree
}  // namespace xgboost
