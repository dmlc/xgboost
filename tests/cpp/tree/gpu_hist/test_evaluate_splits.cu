/**
 * Copyright 2020-2026, XGBoost contributors
 */
#include <gtest/gtest.h>

#include <algorithm>  // for any_of, max_element
#include <bitset>     // for bitset
#include <cmath>      // for isnan
#include <cstdint>    // for uint32_t
#include <vector>     // for vector

#include "../../../../src/tree/gpu_hist/evaluate_splits.cuh"
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
  GradientQuantiserGroup group{ctx, linalg::MakeVec(ctx->Device(), dh::ToSpan(gpair))};
  return group[0];
}

struct Node {
  bst_node_t nidx;
  GradientPairPrecise parent_sum;
  std::vector<GradientPairPrecise> histogram;
};

struct Result {
  DeviceSplitCandidate split;
  GradientPairInt64 left_sum;
  GradientPairInt64 right_sum;
  std::bitset<32> cats;
};

// Evaluate single-target splits from host inputs.
class ScalarEvaluator {
  Context const* ctx_;
  dh::device_vector<GradientQuantiser> roundings_;
  dh::device_vector<std::uint32_t> cut_ptrs_;
  dh::device_vector<float> cut_values_;
  dh::device_vector<FeatureType> feature_types_;
  std::size_t cat_storage_size_{0};
  TrainParam param_;
  std::vector<dh::device_vector<GradientPairInt64>> storage_;
  dh::device_vector<bst_feature_t> feature_set_;
  dh::device_vector<EvaluateSplitInputs> inputs_;

 public:
  GradientQuantiser quantiser;
  GPUHistEvaluator evaluator;

  ScalarEvaluator(Context const* ctx, TrainParam const& param,
                  std::vector<std::uint32_t> const& cut_ptrs, std::vector<float> const& cut_values,
                  std::vector<FeatureType> const& feature_types = {})
      : ctx_{ctx},
        cut_ptrs_{cut_ptrs},
        cut_values_{cut_values},
        feature_types_{feature_types},
        param_{param},
        quantiser{DummyRoundingFactor(ctx)} {
    roundings_ = std::vector<GradientQuantiser>{quantiser};
    auto is_cat = [](FeatureType ft) { return ft == FeatureType::kCategorical; };
    if (std::any_of(feature_types.cbegin(), feature_types.cend(), is_cat)) {
      auto max_cat = *std::max_element(cut_values.cbegin(), cut_values.cend());
      cat_storage_size_ = common::CatBitField::ComputeStorageSize(common::AsCat(max_cat) + 1);
    }
    evaluator.Reset(ctx, dh::ToSpan(cut_ptrs_), dh::ToSpan(feature_types_), param, 1);
  }
  ScalarEvaluator(Context const* ctx, TrainParam const& param, common::HistogramCuts const& cuts,
                  std::vector<FeatureType> const& feature_types)
      : ScalarEvaluator{ctx, param, cuts.cut_ptrs_.ConstHostVector(),
                        cuts.cut_values_.ConstHostVector(), feature_types} {}

  [[nodiscard]] GradientPairInt64 ToFixed(GradientPairPrecise const& g) const {
    return quantiser.ToFixedPoint(GradientPair(g));
  }

  [[nodiscard]] EvaluateSplitSharedInputs Shared() {
    return {dh::ToSpan(roundings_),
            dh::ToSpan(cut_ptrs_),
            thrust::raw_pointer_cast(cut_values_.data()),
            dh::ToSpan(feature_types_),
            cat_storage_size_,
            static_cast<bst_bin_t>(cut_values_.size()),
            static_cast<bst_feature_t>(feature_set_.size()),
            EvalParam{param_}};
  }

  common::Span<EvaluateSplitInputs const> MakeInputs(std::vector<Node> const& nodes,
                                                     std::vector<bst_feature_t> const& fset) {
    feature_set_ = fset;
    storage_.clear();
    storage_.reserve(nodes.size());
    std::vector<EvaluateSplitInputs> h_inputs;
    for (auto const& node : nodes) {
      std::vector<GradientPairInt64> h_data{ToFixed(node.parent_sum)};
      for (auto const& g : node.histogram) {
        h_data.push_back(ToFixed(g));
      }
      auto d_data = dh::ToSpan(storage_.emplace_back(h_data));
      h_inputs.push_back(
          {node.nidx, 0, d_data.subspan(0, 1), dh::ToSpan(feature_set_), d_data.subspan(1)});
    }
    inputs_ = h_inputs;
    return dh::ToSpan(inputs_);
  }

  std::vector<Result> Evaluate(std::vector<Node> const& nodes,
                               std::vector<bst_feature_t> const& fset) {
    auto d_inputs = this->MakeInputs(nodes, fset);
    dh::device_vector<GPUExpandEntry> d_out(nodes.size());
    bst_node_t max_nidx = 0;
    for (auto const& node : nodes) {
      max_nidx = std::max(max_nidx, node.nidx);
    }
    evaluator.EvaluateSplits(ctx_, d_inputs, this->Shared(), max_nidx, dh::ToSpan(d_out));
    evaluator.CopyToHost(ctx_, 0, max_nidx);
    ctx_->CUDACtx()->Stream().Sync();
    std::vector<GPUExpandEntry> h_out(d_out.size());
    thrust::copy(d_out.cbegin(), d_out.cend(), h_out.begin());

    std::vector<Result> results;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      Result r;
      r.split = h_out[i].split;
      if (!r.split.child_sum.empty()) {
        GradientPairInt64 child;
        dh::safe_cuda(cudaMemcpy(&child, r.split.child_sum.data(), sizeof(child),
                                 cudaMemcpyDeviceToHost));
        auto parent = ToFixed(nodes[i].parent_sum);
        r.left_sum = r.split.dir == kRightDir ? child : parent - child;
        r.right_sum = parent - r.left_sum;
      }
      if (cat_storage_size_ != 0) {
        r.cats = std::bitset<32>(evaluator.GetHostNodeCats(nodes[i].nidx)[0]);
      }
      results.push_back(r);
    }
    return results;
  }

  Result EvaluateOne(Node const& node, std::vector<bst_feature_t> const& fset) {
    return this->Evaluate({node}, fset).front();
  }
};

constexpr auto kCat = FeatureType::kCategorical;
}  // anonymous namespace

TEST_F(TestCategoricalSplitWithMissing, GPUHistEvaluator) {
  auto ctx = MakeCUDACtx(0);
  ScalarEvaluator eval{&ctx, param_, cuts_, {kCat}};
  auto r = eval.EvaluateOne({1, parent_sum_, feature_histogram_}, {0});
  ASSERT_EQ(r.split.thresh, 1);
  this->CheckResult(r.split.loss_chg, r.split.findex, r.split.fvalue, r.split.is_cat,
                    r.split.dir == kLeftDir, eval.quantiser.ToFloatingPoint(r.left_sum),
                    eval.quantiser.ToFloatingPoint(r.right_sum));
}

TEST(GpuHist, PartitionBasic) {
  auto ctx = MakeCUDACtx(0);
  TrainParam tparam = ZeroParam();
  tparam.max_cat_to_onehot = 0;
  ScalarEvaluator eval{&ctx, tparam, {0, 3}, {0.0, 1.0, 2.0}, {kCat}};
  auto check = [&](bst_node_t nidx, GradientPairPrecise parent,
                   std::vector<GradientPairPrecise> hist) {
    auto r = eval.EvaluateOne({nidx, parent, hist}, {0});
    EXPECT_EQ(r.left_sum + r.right_sum, eval.ToFixed(parent));
    return r;
  };
  {
    // -1.0s go right
    // -3.0s go left
    auto r = check(0, {-5.0, 3.0}, {{-1.0, 1.0}, {-1.0, 1.0}, {-3.0, 1.0}});
    EXPECT_EQ(r.split.dir, kLeftDir);
    EXPECT_EQ(r.cats, std::bitset<32>("11000000000000000000000000000000"));
  }
  {
    auto r = check(1, {-7.0, 3.0}, {{-1.0, 1.0}, {-3.0, 1.0}, {-3.0, 1.0}});
    EXPECT_EQ(r.split.dir, kLeftDir);
    EXPECT_EQ(r.cats, std::bitset<32>("10000000000000000000000000000000"));
  }
  {
    // All -1.0, gain from splitting should be 0.0
    auto r = check(2, {-3.0, 3.0}, {{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}});
    EXPECT_EQ(r.split.dir, kLeftDir);
    EXPECT_FLOAT_EQ(r.split.loss_chg, 0.0f);
  }
  {
    // With 3.0/3.0 missing values. Forward, first 2 categories are selected, while the last
    // one go to left along with missing value
    auto r = check(3, {0.0, 6.0}, {{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}});
    EXPECT_EQ(r.cats, std::bitset<32>("11000000000000000000000000000000"));
    EXPECT_EQ(r.split.dir, kLeftDir);
  }
  {
    auto r = check(4, {-5.0, 3.0}, {{-1.0, 1.0}, {-3.0, 1.0}, {-1.0, 1.0}});
    EXPECT_EQ(r.split.dir, kLeftDir);
    EXPECT_EQ(r.cats, std::bitset<32>("10100000000000000000000000000000"));
  }
  {
    auto r = check(5, {-5.0, 3.0}, {{-3.0, 1.0}, {-1.0, 1.0}, {-3.0, 1.0}});
    EXPECT_EQ(r.cats, std::bitset<32>("01000000000000000000000000000000"));
  }
}

TEST(GpuHist, PartitionTwoFeatures) {
  auto ctx = MakeCUDACtx(0);
  TrainParam tparam = ZeroParam();
  tparam.max_cat_to_onehot = 0;
  ScalarEvaluator eval{&ctx, tparam, {0, 3, 6}, {0.0, 1.0, 2.0, 0.0, 1.0, 2.0}, {kCat, kCat}};
  GradientPairPrecise parent{-6.0, 3.0};
  {
    auto r = eval.EvaluateOne(
        {0, parent, {{-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-4.0, 1.0}}},
        {0, 1});
    EXPECT_EQ(r.split.findex, 1);
    EXPECT_EQ(r.cats, std::bitset<32>("11000000000000000000000000000000"));
    EXPECT_EQ(r.left_sum + r.right_sum, eval.ToFixed(parent));
  }
  {
    auto r = eval.EvaluateOne(
        {1, parent, {{-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-1.0, 1.0}, {-2.5, 1.0}, {-2.5, 1.0}}},
        {0, 1});
    EXPECT_EQ(r.split.findex, 1);
    EXPECT_EQ(r.cats, std::bitset<32>("10000000000000000000000000000000"));
    EXPECT_EQ(r.left_sum + r.right_sum, eval.ToFixed(parent));
  }
}

TEST(GpuHist, PartitionInterleavedFeatures) {
  auto ctx = MakeCUDACtx(0);
  TrainParam tparam = ZeroParam();
  tparam.max_cat_to_onehot = 3;
  // Features 0 and 4 are partition-based. Feature 1 is numerical, feature 2 is one-hot, and
  // feature 3 is empty.
  ScalarEvaluator eval{&ctx,
                       tparam,
                       {0, 3, 6, 8, 8, 11},
                       {0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0},
                       {kCat, FeatureType::kNumerical, kCat, kCat, kCat}};
  std::vector<bst_feature_t> fset{0, 1, 2, 3, 4};
  GradientPairPrecise parent{-6.0, 3.0};

  std::vector<GradientPairPrecise> best_split_feature_4{
      {-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0},
      {-3.0, 1.0}, {-3.0, 2.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-4.0, 1.0}};
  auto r = eval.EvaluateOne({0, parent, best_split_feature_4}, fset);
  EXPECT_EQ(r.split.findex, 4);
  EXPECT_EQ(r.cats, std::bitset<32>("11000000000000000000000000000000"));
  EXPECT_EQ(r.left_sum + r.right_sum, eval.ToFixed(parent));

  std::vector<GradientPairPrecise> best_split_feature_0{
      {-1.0, 1.0}, {-1.0, 1.0}, {-4.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0},
      {-3.0, 1.0}, {-3.0, 2.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}};
  auto results =
      eval.Evaluate({{1, parent, best_split_feature_4}, {2, parent, best_split_feature_0}}, fset);
  EXPECT_EQ(results[0].split.findex, 4);
  EXPECT_EQ(results[1].split.findex, 0);
  EXPECT_EQ(results[0].cats, std::bitset<32>("11000000000000000000000000000000"));
  EXPECT_EQ(results[1].cats, std::bitset<32>("11000000000000000000000000000000"));

  // Feature 0 is bins 0-2. Node 0 weights 4, 1, 1 sort as 1, 2, 0. Node 1 weights 1, 4, 1
  // sort as 0, 2, 1. Reading the other node's histogram fails that node. Feature 4 is bins
  // 8-10. Both nodes have weights 4, 1, 1 and sort as 9, 10, 8. Swapping a tie or skipping
  // the scatter leaves a different order.
  std::vector<GradientPairPrecise> tied_node_0(11, {-1.0, 1.0}), tied_node_1(11, {-1.0, 1.0});
  tied_node_0[0] = tied_node_0[8] = {-4.0, 1.0};
  tied_node_1[1] = tied_node_1[8] = {-4.0, 1.0};
  auto d_inputs = eval.MakeInputs({{0, parent, tied_node_0}, {1, parent, tied_node_1}}, fset);
  auto sorted = eval.evaluator.SortHistogram(&ctx, d_inputs, eval.Shared());
  std::vector<bst_bin_t> h_sorted(sorted.size());
  dh::CopyDeviceSpanToVector(&h_sorted, sorted);
  // The sorted indices are local to each node.
  std::size_t const total_bins = 11;
  EXPECT_EQ(h_sorted[0], 1);
  EXPECT_EQ(h_sorted[1], 2);
  EXPECT_EQ(h_sorted[2], 0);
  EXPECT_EQ(h_sorted[total_bins + 0], 0);
  EXPECT_EQ(h_sorted[total_bins + 1], 2);
  EXPECT_EQ(h_sorted[total_bins + 2], 1);
  for (std::size_t node = 0; node < 2; ++node) {
    auto base = node * total_bins;
    EXPECT_EQ(h_sorted[base + 8], 9);
    EXPECT_EQ(h_sorted[base + 9], 10);
    EXPECT_EQ(h_sorted[base + 10], 8);
  }
}

TEST(GpuHist, PartitionTwoNodes) {
  auto ctx = MakeCUDACtx(0);
  TrainParam tparam = ZeroParam();
  tparam.max_cat_to_onehot = 0;
  ScalarEvaluator eval{&ctx, tparam, {0, 3}, {0.0, 1.0, 2.0}, {kCat}};
  GradientPairPrecise parent{-6.0, 3.0};
  auto results = eval.Evaluate({{0, parent, {{-1.0, 1.0}, {-2.5, 1.0}, {-2.5, 1.0}}},
                                {1, parent, {{-1.0, 1.0}, {-1.0, 1.0}, {-4.0, 1.0}}}},
                               {0});
  EXPECT_EQ(results[0].cats, std::bitset<32>("10000000000000000000000000000000"));
  EXPECT_EQ(results[1].cats, std::bitset<32>("11000000000000000000000000000000"));
}

void TestEvaluateSingleSplit(bool is_categorical) {
  auto ctx = MakeCUDACtx(0);
  std::vector<FeatureType> ft;
  if (is_categorical) {
    ft = {kCat, kCat};
  }
  ScalarEvaluator eval{&ctx, ZeroParam(), {0, 2, 4}, {1.0, 2.0, 11.0, 12.0}, ft};
  GradientPairPrecise parent{0.0, 1.0};
  // Setup gradients so that second feature gets higher gain
  auto r = eval.EvaluateOne(
      {1, parent, {{-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}}}, {0, 1});

  EXPECT_EQ(r.split.findex, 1);
  if (is_categorical) {
    ASSERT_TRUE(std::isnan(r.split.fvalue));
  } else {
    EXPECT_EQ(r.split.fvalue, 11.0);
  }
  EXPECT_EQ(r.left_sum + r.right_sum, eval.ToFixed(parent));
}

TEST(GpuHist, EvaluateSingleSplit) { TestEvaluateSingleSplit(false); }

TEST(GpuHist, EvaluateSingleCategoricalSplit) { TestEvaluateSingleSplit(true); }

TEST(GpuHist, EvaluateSingleSplitMissing) {
  auto ctx = MakeCUDACtx(0);
  ScalarEvaluator eval{&ctx, ZeroParam(), {0, 2}, {1.0, 2.0}};
  auto r = eval.EvaluateOne({1, {1.0, 1.5}, {{-0.5, 0.5}, {0.5, 0.5}}}, {0});

  EXPECT_EQ(r.split.findex, 0);
  EXPECT_EQ(r.split.fvalue, 1.0);
  EXPECT_EQ(r.split.dir, kRightDir);
  EXPECT_EQ(r.left_sum, eval.ToFixed({-0.5, 0.5}));
  EXPECT_EQ(r.right_sum, eval.ToFixed({1.5, 1.0}));
}

TEST(GpuHist, EvaluateSingleSplitMissingOnly) {
  auto ctx = MakeCUDACtx(0);
  // The forward and backward endpoints give equal gain. CPU hist keeps the
  // forward cut, so the observed bin goes left and missing values go right.
  ScalarEvaluator eval{&ctx, ZeroParam(), {0, 1}, {2.0}};
  auto r = eval.EvaluateOne({0, {0.0, 2.0}, {{1.0, 1.0}}}, {0});
  EXPECT_EQ(r.split.fvalue, 2.0);
  EXPECT_EQ(r.split.dir, kRightDir);
  EXPECT_EQ(r.left_sum, eval.ToFixed({1.0, 1.0}));
  EXPECT_EQ(r.right_sum, eval.ToFixed({-1.0, 1.0}));
}

TEST(GpuHist, EvaluateSingleSplitEmpty) {
  auto ctx = MakeCUDACtx(0);
  ScalarEvaluator eval{&ctx, ZeroParam(), {0, 1}, {1.0}};
  auto r = eval.EvaluateOne({0, {0.0, 0.0}, {{0.0, 0.0}}}, {});
  EXPECT_EQ(r.split.findex, -1);
  EXPECT_LT(r.split.loss_chg, 0.0f);
}

// Feature 0 has a better split, but the algorithm must select feature 1
TEST(GpuHist, EvaluateSingleSplitFeatureSampling) {
  auto ctx = MakeCUDACtx(0);
  ScalarEvaluator eval{&ctx, ZeroParam(), {0, 2, 4}, {1.0, 2.0, 11.0, 12.0}};
  auto r = eval.EvaluateOne(
      {1, {0.0, 1.0}, {{-10.0, 0.5}, {10.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}}}, {1});

  EXPECT_EQ(r.split.findex, 1);
  EXPECT_EQ(r.split.fvalue, 11.0);
  EXPECT_EQ(r.left_sum, eval.ToFixed({-0.5, 0.5}));
  EXPECT_EQ(r.right_sum, eval.ToFixed({0.5, 0.5}));
}

// Features 0 and 1 have identical gain, the algorithm must select 0
TEST(GpuHist, EvaluateSingleSplitBreakTies) {
  auto ctx = MakeCUDACtx(0);
  ScalarEvaluator eval{&ctx, ZeroParam(), {0, 2, 4}, {1.0, 2.0, 11.0, 12.0}};
  auto r = eval.EvaluateOne(
      {1, {0.0, 1.0}, {{-0.5, 0.5}, {0.5, 0.5}, {-0.5, 0.5}, {0.5, 0.5}}}, {0, 1});

  EXPECT_EQ(r.split.findex, 0);
  EXPECT_EQ(r.split.fvalue, 1.0);
}

// Without missing values, only the forward pass is run and missing values go right.
TEST(GpuHist, EvaluateSingleSplitDense) {
  auto ctx = MakeCUDACtx(0);
  ScalarEvaluator eval{&ctx, ZeroParam(), {0, 3, 6}, {1.0, 2.0, 3.0, 11.0, 12.0, 13.0}};
  // The bins of every feature add up to the parent sum.
  auto r = eval.EvaluateOne(
      {1,
       {0.0, 1.5},
       {{-1.0, 0.5}, {0.25, 0.5}, {0.75, 0.5}, {-0.25, 0.5}, {-0.5, 0.5}, {0.75, 0.5}}},
      {0, 1});

  EXPECT_EQ(r.split.dir, kRightDir);
  EXPECT_EQ(r.split.findex, 0);
  EXPECT_EQ(r.split.fvalue, 1.0);
  EXPECT_FLOAT_EQ(r.split.loss_chg, 3.0f);
  EXPECT_EQ(r.left_sum, eval.ToFixed({-1.0, 0.5}));
  EXPECT_EQ(r.right_sum, eval.ToFixed({1.0, 1.0}));
}

TEST(GpuHist, EvaluateSplits) {
  auto ctx = MakeCUDACtx(0);
  ScalarEvaluator eval{&ctx, ZeroParam(), {0, 2, 4}, {1.0, 2.0, 11.0, 12.0}};
  GradientPairPrecise parent{0.0, 1.0};
  auto results =
      eval.Evaluate({{1, parent, {{-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}}},
                     {2, parent, {{-1.0, 0.5}, {1.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}}}},
                    {0, 1});

  EXPECT_EQ(results[0].split.findex, 1);
  EXPECT_EQ(results[0].split.fvalue, 11.0);
  EXPECT_EQ(results[1].split.findex, 0);
  EXPECT_EQ(results[1].split.fvalue, 1.0);
}

TEST_F(TestPartitionBasedSplit, GpuHist) {
  auto ctx = MakeCUDACtx(0);
  ScalarEvaluator eval{&ctx, param_, cuts_, {kCat}};
  auto node_hist = hist_[0];
  std::vector<GradientPairPrecise> hist(node_hist.cbegin(), node_hist.cend());
  auto r = eval.EvaluateOne({0, total_gpair_, hist}, {0});
  ASSERT_NEAR(r.split.loss_chg, best_score_, 1e-2);
}
}  // namespace xgboost::tree
