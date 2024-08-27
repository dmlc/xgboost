/**
 * Copyright 2020-2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/host_vector.h>

#include "../../../../src/tree/gpu_hist/evaluate_splits.cuh"
#include "../../collective/test_worker.h"  // for BaseMGPUTest
#include "../../helpers.h"
#include "../test_evaluate_splits.h"  // TestPartitionBasedSplit

namespace xgboost::tree {
namespace {
auto ZeroParam() {
  auto args = Args{{"min_child_weight", "0"}, {"lambda", "0"}};
  TrainParam tparam;
  tparam.UpdateAllowUnknown(args);
  return tparam;
}

GradientQuantiser DummyRoundingFactor(Context const* ctx) {
  thrust::device_vector<GradientPair> gpair(1);
  gpair[0] = {1000.f, 1000.f};  // Tests should not exceed sum of 1000
  return {ctx, dh::ToSpan(gpair), MetaInfo()};
}
}  // anonymous namespace

thrust::device_vector<GradientPairInt64> ConvertToInteger(Context const* ctx,
                                                          std::vector<GradientPairPrecise> x) {
  auto r = DummyRoundingFactor(ctx);
  std::vector<GradientPairInt64> y(x.size());
  for (std::size_t i = 0; i < x.size(); i++) {
    y[i] = r.ToFixedPoint(GradientPair(x[i]));
  }
  return y;
}

TEST_F(TestCategoricalSplitWithMissing, GPUHistEvaluator) {
  auto ctx = MakeCUDACtx(0);
  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0};
  GPUTrainingParam param{param_};
  cuts_.cut_ptrs_.SetDevice(ctx.Device());
  cuts_.cut_values_.SetDevice(ctx.Device());
  cuts_.min_vals_.SetDevice(ctx.Device());
  thrust::device_vector<GradientPairInt64> feature_histogram{
      ConvertToInteger(&ctx, feature_histogram_)};

  dh::device_vector<FeatureType> feature_types(feature_set.size(), FeatureType::kCategorical);
  auto d_feature_types = dh::ToSpan(feature_types);
  auto quantiser = DummyRoundingFactor(&ctx);
  EvaluateSplitInputs input{1, 0, quantiser.ToFixedPoint(parent_sum_), dh::ToSpan(feature_set),
                            dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          d_feature_types,
                                          cuts_.cut_ptrs_.ConstDeviceSpan(),
                                          cuts_.cut_values_.ConstDeviceSpan(),
                                          cuts_.min_vals_.ConstDeviceSpan(),
                                          false};

  GPUHistEvaluator evaluator{param_, static_cast<bst_feature_t>(feature_set.size()), ctx.Device()};

  evaluator.Reset(&ctx, cuts_, dh::ToSpan(feature_types), feature_set.size(), param_, false);
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;

  ASSERT_EQ(result.thresh, 1);
  this->CheckResult(result.loss_chg, result.findex, result.fvalue, result.is_cat,
                    result.dir == kLeftDir, quantiser.ToFloatingPoint(result.left_sum),
                    quantiser.ToFloatingPoint(result.right_sum));
}

TEST(GpuHist, PartitionBasic) {
  auto ctx = MakeCUDACtx(0);
  TrainParam tparam = ZeroParam();
  tparam.max_cat_to_onehot = 0;
  GPUTrainingParam param{tparam};

  common::HistogramCuts cuts;
  cuts.cut_values_.HostVector() = std::vector<float>{0.0, 1.0, 2.0};
  cuts.cut_ptrs_.HostVector() = std::vector<uint32_t>{0, 3};
  cuts.min_vals_.HostVector() = std::vector<float>{0.0};
  cuts.cut_ptrs_.SetDevice(ctx.Device());
  cuts.cut_values_.SetDevice(ctx.Device());
  cuts.min_vals_.SetDevice(ctx.Device());
  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0};

  thrust::device_vector<int> monotonic_constraints(feature_set.size(), 0);
  dh::device_vector<FeatureType> feature_types(feature_set.size(), FeatureType::kCategorical);
  common::Span<FeatureType> d_feature_types;
  auto max_cat =
      *std::max_element(cuts.cut_values_.HostVector().begin(), cuts.cut_values_.HostVector().end());
  cuts.SetCategorical(true, max_cat);
  d_feature_types = dh::ToSpan(feature_types);
  auto quantiser = DummyRoundingFactor(&ctx);
  EvaluateSplitSharedInputs shared_inputs{
      param,
      quantiser,
      d_feature_types,
      cuts.cut_ptrs_.ConstDeviceSpan(),
      cuts.cut_values_.ConstDeviceSpan(),
      cuts.min_vals_.ConstDeviceSpan(),
      false,
  };

  GPUHistEvaluator evaluator{tparam, static_cast<bst_feature_t>(feature_set.size()), ctx.Device()};
  evaluator.Reset(&ctx, cuts, dh::ToSpan(feature_types), feature_set.size(), tparam, false);

  {
    // -1.0s go right
    // -3.0s go left
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-5.0, 3.0});
    auto feature_histogram = ConvertToInteger(&ctx, {{-1.0, 1.0}, {-1.0, 1.0}, {-3.0, 1.0}});
    EvaluateSplitInputs input{0, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    auto cats = std::bitset<32>(evaluator.GetHostNodeCats(input.nidx)[0]);
    EXPECT_EQ(result.dir, kLeftDir);
    EXPECT_EQ(cats, std::bitset<32>("11000000000000000000000000000000"));
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }

  {
    // -1.0s go right
    // -3.0s go left
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-7.0, 3.0});
    auto feature_histogram = ConvertToInteger(&ctx, {{-1.0, 1.0}, {-3.0, 1.0}, {-3.0, 1.0}});
    EvaluateSplitInputs input{1, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    auto cats = std::bitset<32>(evaluator.GetHostNodeCats(input.nidx)[0]);
    EXPECT_EQ(result.dir, kLeftDir);
    EXPECT_EQ(cats, std::bitset<32>("10000000000000000000000000000000"));
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }
  {
    // All -1.0, gain from splitting should be 0.0
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-3.0, 3.0});
    auto feature_histogram = ConvertToInteger(&ctx, {{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}});
    EvaluateSplitInputs input{2, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    EXPECT_EQ(result.dir, kLeftDir);
    EXPECT_FLOAT_EQ(result.loss_chg, 0.0f);
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }
  // With 3.0/3.0 missing values
  // Forward, first 2 categories are selected, while the last one go to left along with missing
  // value
  {
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{0.0, 6.0});
    auto feature_histogram = ConvertToInteger(&ctx, {{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}});
    EvaluateSplitInputs input{3, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    auto cats = std::bitset<32>(evaluator.GetHostNodeCats(input.nidx)[0]);
    EXPECT_EQ(cats, std::bitset<32>("11000000000000000000000000000000"));
    EXPECT_EQ(result.dir, kLeftDir);
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }
  {
    // -1.0s go right
    // -3.0s go left
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-5.0, 3.0});
    auto feature_histogram = ConvertToInteger(&ctx, {{-1.0, 1.0}, {-3.0, 1.0}, {-1.0, 1.0}});
    EvaluateSplitInputs input{4, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    auto cats = std::bitset<32>(evaluator.GetHostNodeCats(input.nidx)[0]);
    EXPECT_EQ(result.dir, kLeftDir);
    EXPECT_EQ(cats, std::bitset<32>("10100000000000000000000000000000"));
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }
  {
    // -1.0s go right
    // -3.0s go left
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-5.0, 3.0});
    auto feature_histogram = ConvertToInteger(&ctx, {{-3.0, 1.0}, {-1.0, 1.0}, {-3.0, 1.0}});
    EvaluateSplitInputs input{5, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    auto cats = std::bitset<32>(evaluator.GetHostNodeCats(input.nidx)[0]);
    EXPECT_EQ(cats, std::bitset<32>("01000000000000000000000000000000"));
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }
}

TEST(GpuHist, PartitionTwoFeatures) {
  auto ctx = MakeCUDACtx(0);
  TrainParam tparam = ZeroParam();
  tparam.max_cat_to_onehot = 0;
  GPUTrainingParam param{tparam};

  common::HistogramCuts cuts;
  cuts.cut_values_.HostVector() = std::vector<float>{0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
  cuts.cut_ptrs_.HostVector() = std::vector<uint32_t>{0, 3, 6};
  cuts.min_vals_.HostVector() = std::vector<float>{0.0, 0.0};
  cuts.cut_ptrs_.SetDevice(ctx.Device());
  cuts.cut_values_.SetDevice(ctx.Device());
  cuts.min_vals_.SetDevice(ctx.Device());
  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0, 1};

  thrust::device_vector<int> monotonic_constraints(feature_set.size(), 0);
  dh::device_vector<FeatureType> feature_types(feature_set.size(), FeatureType::kCategorical);
  common::Span<FeatureType> d_feature_types(dh::ToSpan(feature_types));
  auto max_cat =
      *std::max_element(cuts.cut_values_.HostVector().begin(), cuts.cut_values_.HostVector().end());
  cuts.SetCategorical(true, max_cat);

  auto quantiser = DummyRoundingFactor(&ctx);
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          d_feature_types,
                                          cuts.cut_ptrs_.ConstDeviceSpan(),
                                          cuts.cut_values_.ConstDeviceSpan(),
                                          cuts.min_vals_.ConstDeviceSpan(),
                                          false};

  GPUHistEvaluator evaluator{tparam, static_cast<bst_feature_t>(feature_set.size()), ctx.Device()};
  evaluator.Reset(&ctx, cuts, dh::ToSpan(feature_types), feature_set.size(), tparam, false);

  {
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-6.0, 3.0});
    auto feature_histogram = ConvertToInteger(
        &ctx, {{-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-4.0, 1.0}});
    EvaluateSplitInputs input{0, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    auto cats = std::bitset<32>(evaluator.GetHostNodeCats(input.nidx)[0]);
    EXPECT_EQ(result.findex, 1);
    EXPECT_EQ(cats, std::bitset<32>("11000000000000000000000000000000"));
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }

  {
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-6.0, 3.0});
    auto feature_histogram = ConvertToInteger(
        &ctx, {{-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-1.0, 1.0}, {-2.5, 1.0}, {-2.5, 1.0}});
    EvaluateSplitInputs input{1, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    auto cats = std::bitset<32>(evaluator.GetHostNodeCats(input.nidx)[0]);
    EXPECT_EQ(result.findex, 1);
    EXPECT_EQ(cats, std::bitset<32>("10000000000000000000000000000000"));
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }
}

TEST(GpuHist, PartitionTwoNodes) {
  auto ctx = MakeCUDACtx(0);
  TrainParam tparam = ZeroParam();
  tparam.max_cat_to_onehot = 0;
  GPUTrainingParam param{tparam};

  common::HistogramCuts cuts;
  cuts.cut_values_.HostVector() = std::vector<float>{0.0, 1.0, 2.0};
  cuts.cut_ptrs_.HostVector() = std::vector<uint32_t>{0, 3};
  cuts.min_vals_.HostVector() = std::vector<float>{0.0};
  cuts.cut_ptrs_.SetDevice(ctx.Device());
  cuts.cut_values_.SetDevice(ctx.Device());
  cuts.min_vals_.SetDevice(ctx.Device());
  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0};

  thrust::device_vector<int> monotonic_constraints(feature_set.size(), 0);
  dh::device_vector<FeatureType> feature_types(feature_set.size(), FeatureType::kCategorical);
  common::Span<FeatureType> d_feature_types(dh::ToSpan(feature_types));
  auto max_cat =
      *std::max_element(cuts.cut_values_.HostVector().begin(), cuts.cut_values_.HostVector().end());
  cuts.SetCategorical(true, max_cat);

  auto quantiser = DummyRoundingFactor(&ctx);
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          d_feature_types,
                                          cuts.cut_ptrs_.ConstDeviceSpan(),
                                          cuts.cut_values_.ConstDeviceSpan(),
                                          cuts.min_vals_.ConstDeviceSpan(),
                                          false};

  GPUHistEvaluator evaluator{tparam, static_cast<bst_feature_t>(feature_set.size()), ctx.Device()};
  evaluator.Reset(&ctx, cuts, dh::ToSpan(feature_types), feature_set.size(), tparam, false);

  {
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-6.0, 3.0});
    auto feature_histogram_a = ConvertToInteger(
        &ctx, {{-1.0, 1.0}, {-2.5, 1.0}, {-2.5, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-4.0, 1.0}});
    thrust::device_vector<EvaluateSplitInputs> inputs(2);
    inputs[0] = EvaluateSplitInputs{0, 0, parent_sum, dh::ToSpan(feature_set),
                                    dh::ToSpan(feature_histogram_a)};
    auto feature_histogram_b = ConvertToInteger(&ctx, {{-1.0, 1.0}, {-1.0, 1.0}, {-4.0, 1.0}});
    inputs[1] = EvaluateSplitInputs{1, 0, parent_sum, dh::ToSpan(feature_set),
                                    dh::ToSpan(feature_histogram_b)};
    thrust::device_vector<GPUExpandEntry> results(2);
    evaluator.EvaluateSplits(&ctx, {0, 1}, 1, dh::ToSpan(inputs), shared_inputs,
                             dh::ToSpan(results));
    EXPECT_EQ(std::bitset<32>(evaluator.GetHostNodeCats(0)[0]),
              std::bitset<32>("10000000000000000000000000000000"));
    EXPECT_EQ(std::bitset<32>(evaluator.GetHostNodeCats(1)[0]),
              std::bitset<32>("11000000000000000000000000000000"));
  }
}

void TestEvaluateSingleSplit(bool is_categorical) {
  auto ctx = MakeCUDACtx(0);
  auto quantiser = DummyRoundingFactor(&ctx);
  auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{0.0, 1.0});
  TrainParam tparam = ZeroParam();
  GPUTrainingParam param{tparam};

  common::HistogramCuts cuts{
      MakeCutsForTest({1.0, 2.0, 11.0, 12.0}, {0, 2, 4}, {0.0, 0.0}, ctx.Device())};
  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0, 1};

  // Setup gradients so that second feature gets higher gain
  auto feature_histogram =
      ConvertToInteger(&ctx, {{-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}});

  dh::device_vector<FeatureType> feature_types(feature_set.size(), FeatureType::kCategorical);
  common::Span<FeatureType> d_feature_types;
  if (is_categorical) {
    auto max_cat = *std::max_element(cuts.cut_values_.HostVector().begin(),
                                     cuts.cut_values_.HostVector().end());
    cuts.SetCategorical(true, max_cat);
    d_feature_types = dh::ToSpan(feature_types);
  }

  EvaluateSplitInputs input{1, 0, parent_sum, dh::ToSpan(feature_set),
                            dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          d_feature_types,
                                          cuts.cut_ptrs_.ConstDeviceSpan(),
                                          cuts.cut_values_.ConstDeviceSpan(),
                                          cuts.min_vals_.ConstDeviceSpan(),
                                          false};

  GPUHistEvaluator evaluator{tparam, static_cast<bst_feature_t>(feature_set.size()), ctx.Device()};
  evaluator.Reset(&ctx, cuts, dh::ToSpan(feature_types), feature_set.size(), tparam, false);
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;

  EXPECT_EQ(result.findex, 1);
  if (is_categorical) {
    ASSERT_TRUE(std::isnan(result.fvalue));
  } else {
    EXPECT_EQ(result.fvalue, 11.0);
  }
  EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
}

TEST(GpuHist, EvaluateSingleSplit) { TestEvaluateSingleSplit(false); }

TEST(GpuHist, EvaluateSingleCategoricalSplit) { TestEvaluateSingleSplit(true); }

TEST(GpuHist, EvaluateSingleSplitMissing) {
  auto ctx = MakeCUDACtx(0);
  auto quantiser = DummyRoundingFactor(&ctx);
  auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{1.0, 1.5});
  TrainParam tparam = ZeroParam();
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0};
  thrust::device_vector<uint32_t> feature_segments = std::vector<bst_idx_t>{0, 2};
  thrust::device_vector<float> feature_values = std::vector<float>{1.0, 2.0};
  thrust::device_vector<float> feature_min_values = std::vector<float>{0.0};
  auto feature_histogram = ConvertToInteger(&ctx, {{-0.5, 0.5}, {0.5, 0.5}});
  EvaluateSplitInputs input{1, 0, parent_sum, dh::ToSpan(feature_set),
                            dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          {},
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
                                          false};

  GPUHistEvaluator evaluator(tparam, feature_set.size(), FstCU());
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;

  EXPECT_EQ(result.findex, 0);
  EXPECT_EQ(result.fvalue, 1.0);
  EXPECT_EQ(result.dir, kRightDir);
  EXPECT_EQ(result.left_sum, quantiser.ToFixedPoint(GradientPairPrecise(-0.5, 0.5)));
  EXPECT_EQ(result.right_sum, quantiser.ToFixedPoint(GradientPairPrecise(1.5, 1.0)));
}

TEST(GpuHist, EvaluateSingleSplitEmpty) {
  auto ctx = MakeCUDACtx(0);
  TrainParam tparam = ZeroParam();
  GPUHistEvaluator evaluator(tparam, 1, FstCU());
  DeviceSplitCandidate result =
      evaluator
          .EvaluateSingleSplit(
              &ctx, EvaluateSplitInputs{},
              EvaluateSplitSharedInputs{
                  GPUTrainingParam(tparam), DummyRoundingFactor(&ctx), {}, {}, {}, {}, false})
          .split;
  EXPECT_EQ(result.findex, -1);
  EXPECT_LT(result.loss_chg, 0.0f);
}

// Feature 0 has a better split, but the algorithm must select feature 1
TEST(GpuHist, EvaluateSingleSplitFeatureSampling) {
  auto ctx = MakeCUDACtx(0);
  auto quantiser = DummyRoundingFactor(&ctx);
  auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{0.0, 1.0});
  TrainParam tparam = ZeroParam();
  tparam.UpdateAllowUnknown(Args{});
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{1};
  thrust::device_vector<uint32_t> feature_segments = std::vector<bst_idx_t>{0, 2, 4};
  thrust::device_vector<float> feature_values = std::vector<float>{1.0, 2.0, 11.0, 12.0};
  thrust::device_vector<float> feature_min_values = std::vector<float>{0.0, 10.0};
  auto feature_histogram =
      ConvertToInteger(&ctx, {{-10.0, 0.5}, {10.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}});
  EvaluateSplitInputs input{1, 0, parent_sum, dh::ToSpan(feature_set),
                            dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          {},
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
                                          false};

  GPUHistEvaluator evaluator(tparam, feature_min_values.size(), FstCU());
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;

  EXPECT_EQ(result.findex, 1);
  EXPECT_EQ(result.fvalue, 11.0);
  EXPECT_EQ(result.left_sum, quantiser.ToFixedPoint(GradientPairPrecise(-0.5, 0.5)));
  EXPECT_EQ(result.right_sum, quantiser.ToFixedPoint(GradientPairPrecise(0.5, 0.5)));
}

// Features 0 and 1 have identical gain, the algorithm must select 0
TEST(GpuHist, EvaluateSingleSplitBreakTies) {
  auto ctx = MakeCUDACtx(0);
  auto quantiser = DummyRoundingFactor(&ctx);
  auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{0.0, 1.0});
  TrainParam tparam = ZeroParam();
  tparam.UpdateAllowUnknown(Args{});
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0, 1};
  thrust::device_vector<uint32_t> feature_segments = std::vector<bst_idx_t>{0, 2, 4};
  thrust::device_vector<float> feature_values = std::vector<float>{1.0, 2.0, 11.0, 12.0};
  thrust::device_vector<float> feature_min_values = std::vector<float>{0.0, 10.0};
  auto feature_histogram =
      ConvertToInteger(&ctx, {{-0.5, 0.5}, {0.5, 0.5}, {-0.5, 0.5}, {0.5, 0.5}});
  EvaluateSplitInputs input{1, 0, parent_sum, dh::ToSpan(feature_set),
                            dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          {},
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
                                          false};

  GPUHistEvaluator evaluator(tparam, feature_min_values.size(), FstCU());
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;

  EXPECT_EQ(result.findex, 0);
  EXPECT_EQ(result.fvalue, 1.0);
}

TEST(GpuHist, EvaluateSplits) {
  auto ctx = MakeCUDACtx(0);
  thrust::device_vector<DeviceSplitCandidate> out_splits(2);
  auto quantiser = DummyRoundingFactor(&ctx);
  auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{0.0, 1.0});
  TrainParam tparam = ZeroParam();
  tparam.UpdateAllowUnknown(Args{});
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0, 1};
  thrust::device_vector<uint32_t> feature_segments = std::vector<bst_idx_t>{0, 2, 4};
  thrust::device_vector<float> feature_values = std::vector<float>{1.0, 2.0, 11.0, 12.0};
  thrust::device_vector<float> feature_min_values = std::vector<float>{0.0, 0.0};
  auto feature_histogram_left =
      ConvertToInteger(&ctx, {{-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}});
  auto feature_histogram_right =
      ConvertToInteger(&ctx, {{-1.0, 0.5}, {1.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}});
  EvaluateSplitInputs input_left{1, 0, parent_sum, dh::ToSpan(feature_set),
                                 dh::ToSpan(feature_histogram_left)};
  EvaluateSplitInputs input_right{2, 0, parent_sum, dh::ToSpan(feature_set),
                                  dh::ToSpan(feature_histogram_right)};
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          {},
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
                                          false};

  GPUHistEvaluator evaluator{tparam, static_cast<bst_feature_t>(feature_min_values.size()),
                             FstCU()};
  dh::device_vector<EvaluateSplitInputs> inputs =
      std::vector<EvaluateSplitInputs>{input_left, input_right};
  evaluator.LaunchEvaluateSplits(input_left.feature_set.size(), dh::ToSpan(inputs), shared_inputs,
                                 evaluator.GetEvaluator(), dh::ToSpan(out_splits));

  DeviceSplitCandidate result_left = out_splits[0];
  EXPECT_EQ(result_left.findex, 1);
  EXPECT_EQ(result_left.fvalue, 11.0);

  DeviceSplitCandidate result_right = out_splits[1];
  EXPECT_EQ(result_right.findex, 0);
  EXPECT_EQ(result_right.fvalue, 1.0);
}

TEST_F(TestPartitionBasedSplit, GpuHist) {
  auto ctx = MakeCUDACtx(0);
  dh::device_vector<FeatureType> ft{std::vector<FeatureType>{FeatureType::kCategorical}};
  GPUHistEvaluator evaluator{param_, static_cast<bst_feature_t>(info_.num_col_), ctx.Device()};

  cuts_.cut_ptrs_.SetDevice(ctx.Device());
  cuts_.cut_values_.SetDevice(ctx.Device());
  cuts_.min_vals_.SetDevice(ctx.Device());

  evaluator.Reset(&ctx, cuts_, dh::ToSpan(ft), info_.num_col_, param_, false);

  // Convert the sample histogram to fixed point
  auto quantiser = DummyRoundingFactor(&ctx);
  thrust::host_vector<GradientPairInt64> h_hist;
  for (auto e : hist_[0]) {
    h_hist.push_back(quantiser.ToFixedPoint(e));
  }
  dh::device_vector<GradientPairInt64> d_hist = h_hist;
  dh::device_vector<bst_feature_t> feature_set{std::vector<bst_feature_t>{0}};

  EvaluateSplitInputs input{0, 0, quantiser.ToFixedPoint(total_gpair_), dh::ToSpan(feature_set),
                            dh::ToSpan(d_hist)};
  EvaluateSplitSharedInputs shared_inputs{GPUTrainingParam{param_},
                                          quantiser,
                                          dh::ToSpan(ft),
                                          cuts_.cut_ptrs_.ConstDeviceSpan(),
                                          cuts_.cut_values_.ConstDeviceSpan(),
                                          cuts_.min_vals_.ConstDeviceSpan(),
                                          false};
  auto split = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
  ASSERT_NEAR(split.loss_chg, best_score_, 1e-2);
}

class MGPUHistTest : public collective::BaseMGPUTest {};

namespace {
void VerifyColumnSplitEvaluateSingleSplit(bool is_categorical) {
  auto ctx = MakeCUDACtx(GPUIDX);
  auto rank = collective::GetRank();
  auto quantiser = DummyRoundingFactor(&ctx);
  auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{0.0, 1.0});
  TrainParam tparam = ZeroParam();
  GPUTrainingParam param{tparam};

  common::HistogramCuts cuts{
      rank == 0 ? MakeCutsForTest({1.0, 2.0}, {0, 2, 2}, {0.0, 0.0}, ctx.Device())
                : MakeCutsForTest({11.0, 12.0}, {0, 0, 2}, {0.0, 0.0}, ctx.Device())};
  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0, 1};

  // Setup gradients so that second feature gets higher gain
  auto feature_histogram = rank == 0 ? ConvertToInteger(&ctx, {{-0.5, 0.5}, {0.5, 0.5}})
                                     : ConvertToInteger(&ctx, {{-1.0, 0.5}, {1.0, 0.5}});

  dh::device_vector<FeatureType> feature_types(feature_set.size(), FeatureType::kCategorical);
  common::Span<FeatureType> d_feature_types;
  if (is_categorical) {
    auto max_cat = *std::max_element(cuts.cut_values_.HostVector().begin(),
                                     cuts.cut_values_.HostVector().end());
    cuts.SetCategorical(true, max_cat);
    d_feature_types = dh::ToSpan(feature_types);
  }

  EvaluateSplitInputs input{1, 0, parent_sum, dh::ToSpan(feature_set),
                            dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          d_feature_types,
                                          cuts.cut_ptrs_.ConstDeviceSpan(),
                                          cuts.cut_values_.ConstDeviceSpan(),
                                          cuts.min_vals_.ConstDeviceSpan(),
                                          false};

  GPUHistEvaluator evaluator{tparam, static_cast<bst_feature_t>(feature_set.size()), ctx.Device()};
  evaluator.Reset(&ctx, cuts, dh::ToSpan(feature_types), feature_set.size(), tparam, true);
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;

  EXPECT_EQ(result.findex, 1);
  if (is_categorical) {
    ASSERT_TRUE(std::isnan(result.fvalue));
  } else {
    EXPECT_EQ(result.fvalue, 11.0);
  }
  EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
}
}  // anonymous namespace

TEST_F(MGPUHistTest, ColumnSplitEvaluateSingleSplit) {
  if (curt::AllVisibleGPUs() > 1) {
    // We can't emulate multiple GPUs with NCCL.
    this->DoTest([] { VerifyColumnSplitEvaluateSingleSplit(false); }, false, true);
  }
  this->DoTest([] { VerifyColumnSplitEvaluateSingleSplit(false); }, true, true);
}

TEST_F(MGPUHistTest, ColumnSplitEvaluateSingleCategoricalSplit) {
  if (curt::AllVisibleGPUs() > 1) {
    // We can't emulate multiple GPUs with NCCL.
    this->DoTest([] { VerifyColumnSplitEvaluateSingleSplit(true); }, false, true);
  }
  this->DoTest([] { VerifyColumnSplitEvaluateSingleSplit(true); }, true, true);
}
}  // namespace xgboost::tree
