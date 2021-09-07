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

TEST(GpuHist, Foobar) {
  thrust::device_vector<DeviceSplitCandidate> out_splits(1);
  GradientPair parent_sum(6.4f, 12.8f);
  TrainParam tparam;
  std::vector<std::pair<std::string, std::string>> args{
      {"max_depth", "1"},
      {"max_leaves", "0"},

      // Disable all other parameters.
      {"colsample_bynode", "1"},
      {"colsample_bylevel", "1"},
      {"colsample_bytree", "1"},
      {"min_child_weight", "0.01"},
      {"reg_alpha", "0"},
      {"reg_lambda", "0"},
      {"max_delta_step", "0"}};
  tparam.Init(args);
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set =
      std::vector<bst_feature_t>{0, 1, 2, 3, 4, 5, 6, 7};
  thrust::device_vector<uint32_t> feature_segments =
      std::vector<bst_row_t>{0, 3, 6, 9, 12, 15, 18, 21, 24};
  thrust::device_vector<float> feature_values =
      std::vector<float>{0.30f, 0.67f, 1.64f,
                         0.32f, 0.77f, 1.95f,
                         0.29f, 0.70f, 1.80f,
                         0.32f, 0.75f, 1.85f,
                         0.18f, 0.59f, 1.69f,
                         0.25f, 0.74f, 2.00f,
                         0.26f, 0.74f, 1.98f,
                         0.26f, 0.71f, 1.83f};
  thrust::device_vector<float> feature_min_values =
      std::vector<float>{0.1f, 0.2f, 0.3f, 0.1f, 0.2f, 0.3f, 0.2f, 0.2f};
  // Setup gradients so that second feature gets higher gain
  thrust::device_vector<GradientPair> feature_histogram =
      std::vector<GradientPair>{
          {0.8314f, 0.7147f}, {1.7989f, 3.7312f}, {3.3846f, 3.4598f},
          {2.9277f, 3.5886f}, {1.8429f, 2.4152f}, {1.2443f, 1.9019f},
          {1.6380f, 2.9174f}, {1.5657f, 2.5107f}, {2.8111f, 2.4776f},
          {2.1322f, 3.0651f}, {3.2927f, 3.8540f}, {0.5899f, 0.9866f},
          {1.5185f, 1.6263f}, {2.0686f, 3.1844f}, {2.4278f, 3.0950f},
          {1.5105f, 2.1403f}, {2.6922f, 4.2217f}, {1.8122f, 1.5437f},
          {0.0000f, 0.0000f}, {4.3245f, 5.7955f}, {1.6903f, 2.1103f},
          {2.4012f, 4.4754f}, {3.6136f, 3.4303f}, {0.0000f, 0.0000f}};

  common::Span<FeatureType> d_feature_types;
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
  LOG(CONSOLE) << "findex = " << result.findex << ", fvalue = " << result.fvalue;
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
