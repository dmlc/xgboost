#include <gtest/gtest.h>
#include "../../../../src/tree/gpu_hist/evaluate_splits.cuh"
#include "../../helpers.h"
#include "../../histogram_helpers.h"

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
  thrust::device_vector<DeviceSplitCandidate> out_splits(1);
  GradientPair parent_sum(0.0, 1.0);
  TrainParam tparam = ZeroParam();
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set =
      std::vector<bst_feature_t>{0, 1};
  thrust::device_vector<uint32_t> feature_segments =
      std::vector<bst_row_t>{0, 2, 4};
  thrust::device_vector<float> feature_values =
      std::vector<float>{1.0, 2.0, 11.0, 12.0};
  thrust::device_vector<float> feature_min_values =
      std::vector<float>{0.0, 0.0};
  // Setup gradients so that second feature gets higher gain
  thrust::device_vector<GradientPair> feature_histogram =
      std::vector<GradientPair>{
          {-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}};

  thrust::device_vector<int> monotonic_constraints(feature_set.size(), 0);
  dh::device_vector<FeatureType> feature_types(feature_set.size(),
                                               FeatureType::kCategorical);
  common::Span<FeatureType> d_feature_types;
  if (is_categorical) {
    d_feature_types = dh::ToSpan(feature_types);
  }
  EvaluateSplitInputs<GradientPair> input{1,
                                          parent_sum,
                                          param,
                                          dh::ToSpan(feature_set),
                                          d_feature_types,
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
                                          dh::ToSpan(feature_histogram)};
  TreeEvaluator tree_evaluator(tparam, feature_min_values.size(), 0);
  auto evaluator = tree_evaluator.GetEvaluator<GPUTrainingParam>();
  EvaluateSingleSplit(dh::ToSpan(out_splits), evaluator, input);

  DeviceSplitCandidate result = out_splits[0];
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
  thrust::device_vector<DeviceSplitCandidate> out_splits(1);
  GradientPair parent_sum(1.0, 1.5);
  TrainParam tparam = ZeroParam();
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set =
      std::vector<bst_feature_t>{0};
  thrust::device_vector<uint32_t> feature_segments =
      std::vector<bst_row_t>{0, 2};
  thrust::device_vector<float> feature_values = std::vector<float>{1.0, 2.0};
  thrust::device_vector<float> feature_min_values = std::vector<float>{0.0};
  thrust::device_vector<GradientPair> feature_histogram =
      std::vector<GradientPair>{{-0.5, 0.5}, {0.5, 0.5}};
  thrust::device_vector<int> monotonic_constraints(feature_set.size(), 0);
  EvaluateSplitInputs<GradientPair> input{1,
                                          parent_sum,
                                          param,
                                          dh::ToSpan(feature_set),
                                          {},
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
                                          dh::ToSpan(feature_histogram)};

  TreeEvaluator tree_evaluator(tparam, feature_set.size(), 0);
  auto evaluator = tree_evaluator.GetEvaluator<GPUTrainingParam>();
  EvaluateSingleSplit(dh::ToSpan(out_splits), evaluator, input);

  DeviceSplitCandidate result = out_splits[0];
  EXPECT_EQ(result.findex, 0);
  EXPECT_EQ(result.fvalue, 1.0);
  EXPECT_EQ(result.dir, kRightDir);
  EXPECT_EQ(result.left_sum, GradientPair(-0.5, 0.5));
  EXPECT_EQ(result.right_sum, GradientPair(1.5, 1.0));
}

TEST(GpuHist, EvaluateSingleSplitEmpty) {
  DeviceSplitCandidate nonzeroed;
  nonzeroed.findex = 1;
  nonzeroed.loss_chg = 1.0;

  thrust::device_vector<DeviceSplitCandidate> out_split(1);
  out_split[0] = nonzeroed;

  TrainParam tparam = ZeroParam();
  TreeEvaluator tree_evaluator(tparam, 1, 0);
  auto evaluator = tree_evaluator.GetEvaluator<GPUTrainingParam>();
  EvaluateSingleSplit(dh::ToSpan(out_split), evaluator,
                      EvaluateSplitInputs<GradientPair>{});

  DeviceSplitCandidate result = out_split[0];
  EXPECT_EQ(result.findex, -1);
  EXPECT_LT(result.loss_chg, 0.0f);
}

// Feature 0 has a better split, but the algorithm must select feature 1
TEST(GpuHist, EvaluateSingleSplitFeatureSampling) {
  thrust::device_vector<DeviceSplitCandidate> out_splits(1);
  GradientPair parent_sum(0.0, 1.0);
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
  thrust::device_vector<GradientPair> feature_histogram =
      std::vector<GradientPair>{
          {-10.0, 0.5}, {10.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}};
  thrust::device_vector<int> monotonic_constraints(2, 0);
  EvaluateSplitInputs<GradientPair> input{1,
                                          parent_sum,
                                          param,
                                          dh::ToSpan(feature_set),
                                          {},
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
                                          dh::ToSpan(feature_histogram)};

  TreeEvaluator tree_evaluator(tparam, feature_min_values.size(), 0);
  auto evaluator = tree_evaluator.GetEvaluator<GPUTrainingParam>();
  EvaluateSingleSplit(dh::ToSpan(out_splits), evaluator, input);

  DeviceSplitCandidate result = out_splits[0];
  EXPECT_EQ(result.findex, 1);
  EXPECT_EQ(result.fvalue, 11.0);
  EXPECT_EQ(result.left_sum, GradientPair(-0.5, 0.5));
  EXPECT_EQ(result.right_sum, GradientPair(0.5, 0.5));
}

// Features 0 and 1 have identical gain, the algorithm must select 0
TEST(GpuHist, EvaluateSingleSplitBreakTies) {
  thrust::device_vector<DeviceSplitCandidate> out_splits(1);
  GradientPair parent_sum(0.0, 1.0);
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
  thrust::device_vector<GradientPair> feature_histogram =
      std::vector<GradientPair>{
          {-0.5, 0.5}, {0.5, 0.5}, {-0.5, 0.5}, {0.5, 0.5}};
  thrust::device_vector<int> monotonic_constraints(2, 0);
  EvaluateSplitInputs<GradientPair> input{1,
                                          parent_sum,
                                          param,
                                          dh::ToSpan(feature_set),
                                          {},
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
                                          dh::ToSpan(feature_histogram)};

  TreeEvaluator tree_evaluator(tparam, feature_min_values.size(), 0);
  auto evaluator = tree_evaluator.GetEvaluator<GPUTrainingParam>();
  EvaluateSingleSplit(dh::ToSpan(out_splits), evaluator, input);

  DeviceSplitCandidate result = out_splits[0];
  EXPECT_EQ(result.findex, 0);
  EXPECT_EQ(result.fvalue, 1.0);
}

TEST(GpuHist, EvaluateSplits) {
  thrust::device_vector<DeviceSplitCandidate> out_splits(2);
  GradientPair parent_sum(0.0, 1.0);
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
  thrust::device_vector<GradientPair> feature_histogram_left =
      std::vector<GradientPair>{
          {-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}};
  thrust::device_vector<GradientPair> feature_histogram_right =
      std::vector<GradientPair>{
          {-1.0, 0.5}, {1.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}};
  thrust::device_vector<int> monotonic_constraints(feature_set.size(), 0);
  EvaluateSplitInputs<GradientPair> input_left{
      1,
      parent_sum,
      param,
      dh::ToSpan(feature_set),
      {},
      dh::ToSpan(feature_segments),
      dh::ToSpan(feature_values),
      dh::ToSpan(feature_min_values),
      dh::ToSpan(feature_histogram_left)};
  EvaluateSplitInputs<GradientPair> input_right{
      2,
      parent_sum,
      param,
      dh::ToSpan(feature_set),
      {},
      dh::ToSpan(feature_segments),
      dh::ToSpan(feature_values),
      dh::ToSpan(feature_min_values),
      dh::ToSpan(feature_histogram_right)};

  TreeEvaluator tree_evaluator(tparam, feature_min_values.size(), 0);
  auto evaluator = tree_evaluator.GetEvaluator<GPUTrainingParam>();
  EvaluateSplits(dh::ToSpan(out_splits), evaluator, input_left, input_right);

  DeviceSplitCandidate result_left = out_splits[0];
  EXPECT_EQ(result_left.findex, 1);
  EXPECT_EQ(result_left.fvalue, 11.0);

  DeviceSplitCandidate result_right = out_splits[1];
  EXPECT_EQ(result_right.findex, 0);
  EXPECT_EQ(result_right.fvalue, 1.0);
}
}  // namespace tree
}  // namespace xgboost
