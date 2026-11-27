/**
 * Copyright 2020-2026, XGBoost contributors
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>  // for any_of, max_element
#include <bitset>     // for bitset
#include <cmath>      // for isnan
#include <cstdint>    // for uint32_t
#include <string>     // for to_string
#include <vector>     // for vector

#include "../../../../src/tree/driver.h"  // for IsValidExpandEntry
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

// A node with per-target parent sums and a target-major histogram. A single-target node is
// replicated across all targets by the evaluator.
struct Node {
  bst_node_t nidx;
  std::vector<GradientPairPrecise> parent_sum;  // [n_targets]
  std::vector<GradientPairPrecise> histogram;   // [n_targets, n_bins]

  Node(bst_node_t nidx, GradientPairPrecise parent, std::vector<GradientPairPrecise> hist)
      : nidx{nidx}, parent_sum{parent}, histogram{std::move(hist)} {}
  Node(bst_node_t nidx, std::vector<GradientPairPrecise> parent,
       std::vector<GradientPairPrecise> hist)
      : nidx{nidx}, parent_sum{std::move(parent)}, histogram{std::move(hist)} {}
};

struct Result : public GPUExpandEntry {
  std::vector<GradientPairInt64> left_sum;
  std::vector<GradientPairInt64> right_sum;
  std::vector<float> base_weight;
  std::vector<float> left_weight;
  std::vector<float> right_weight;
  std::bitset<32> cats;
};

// Evaluate splits from host inputs.
class EvalHelper {
  Context const* ctx_;
  bst_target_t n_targets_;
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

  EvalHelper(Context const* ctx, TrainParam const& param,
             std::vector<std::uint32_t> const& cut_ptrs, std::vector<float> const& cut_values,
             std::vector<FeatureType> const& feature_types = {}, bst_target_t n_targets = 1)
      : ctx_{ctx},
        n_targets_{n_targets},
        cut_ptrs_{cut_ptrs},
        cut_values_{cut_values},
        feature_types_{feature_types},
        param_{param},
        quantiser{DummyRoundingFactor(ctx)} {
    roundings_ = std::vector<GradientQuantiser>(n_targets, quantiser);
    auto is_cat = [](FeatureType ft) {
      return ft == FeatureType::kCategorical;
    };
    if (std::any_of(feature_types.cbegin(), feature_types.cend(), is_cat)) {
      auto max_cat = *std::max_element(cut_values.cbegin(), cut_values.cend());
      cat_storage_size_ = common::CatBitField::ComputeStorageSize(common::AsCat(max_cat) + 1);
    }
    evaluator.Reset(ctx, dh::ToSpan(cut_ptrs_), dh::ToSpan(feature_types_), param, n_targets);
  }
  EvalHelper(Context const* ctx, TrainParam const& param, common::HistogramCuts const& cuts,
             std::vector<FeatureType> const& feature_types)
      : EvalHelper{ctx, param, cuts.cut_ptrs_.ConstHostVector(), cuts.cut_values_.ConstHostVector(),
                   feature_types} {}

  // Quantize the sums of all targets. A single sum is replicated across all targets.
  [[nodiscard]] std::vector<GradientPairInt64> ToFixed(
      std::vector<GradientPairPrecise> const& sums) const {
    CHECK(sums.size() == 1 || sums.size() == n_targets_);
    std::vector<GradientPairInt64> out;
    for (bst_target_t t = 0; t < n_targets_; ++t) {
      out.push_back(quantiser.ToFixedPoint(GradientPair(sums[sums.size() == 1 ? 0 : t])));
    }
    return out;
  }
  [[nodiscard]] std::vector<GradientPairInt64> ToFixed(GradientPairPrecise const& g) const {
    return this->ToFixed(std::vector{g});
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
      auto replicate = node.parent_sum.size() == 1 ? n_targets_ : 1;
      CHECK_EQ(node.parent_sum.size() * replicate, n_targets_);
      CHECK_EQ(node.histogram.size() * replicate, cut_values_.size() * n_targets_);
      auto h_data = this->ToFixed(node.parent_sum);
      for (bst_target_t t = 0; t < replicate; ++t) {
        for (auto const& g : node.histogram) {
          h_data.push_back(quantiser.ToFixedPoint(GradientPair(g)));
        }
      }
      auto d_data = dh::ToSpan(storage_.emplace_back(h_data));
      h_inputs.push_back({node.nidx, 0, d_data.subspan(0, n_targets_), dh::ToSpan(feature_set_),
                          d_data.subspan(n_targets_)});
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

    auto weights = evaluator.GetHostNodeWeights();
    auto to_vec = [](common::Span<float const> s) {
      return std::vector<float>(s.cbegin(), s.cend());
    };
    std::vector<Result> results;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      Result r;
      static_cast<GPUExpandEntry&>(r) = h_out[i];
      auto nidx = nodes[i].nidx;
      r.base_weight = to_vec(weights.Base(nidx));
      r.left_weight = to_vec(weights.Left(nidx));
      r.right_weight = to_vec(weights.Right(nidx));
      if (!r.split.child_sum.empty()) {
        std::vector<GradientPairInt64> child(r.split.child_sum.size());
        dh::CopyDeviceSpanToVector(&child, r.split.child_sum);
        auto parent = this->ToFixed(nodes[i].parent_sum);
        for (bst_target_t t = 0; t < n_targets_; ++t) {
          auto left = r.split.dir == kRightDir ? child[t] : parent[t] - child[t];
          r.left_sum.push_back(left);
          r.right_sum.push_back(parent[t] - left);
        }
      }
      if (cat_storage_size_ != 0) {
        r.cats = std::bitset<32>(evaluator.GetHostNodeCats(nidx)[0]);
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
  EvalHelper eval{&ctx, param_, cuts_, {kCat}};
  auto r = eval.EvaluateOne({1, parent_sum_, feature_histogram_}, {0});
  ASSERT_EQ(r.split.thresh, 1);
  this->CheckResult(r.split.loss_chg, r.split.findex, r.split.fvalue, r.split.is_cat,
                    r.split.dir == kLeftDir, eval.quantiser.ToFloatingPoint(r.left_sum[0]),
                    eval.quantiser.ToFloatingPoint(r.right_sum[0]));
}

TEST(GpuHist, PartitionBasic) {
  auto ctx = MakeCUDACtx(0);
  TrainParam tparam = ZeroParam();
  tparam.max_cat_to_onehot = 0;
  EvalHelper eval{&ctx, tparam, {0, 3}, {0.0, 1.0, 2.0}, {kCat}};
  auto check = [&](bst_node_t nidx, GradientPairPrecise parent,
                   std::vector<GradientPairPrecise> hist) {
    return eval.EvaluateOne({nidx, parent, hist}, {0});
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

TEST(GpuHist, PartitionInterleavedFeatures) {
  auto ctx = MakeCUDACtx(0);
  TrainParam tparam = ZeroParam();
  tparam.max_cat_to_onehot = 3;
  // Features 0 and 4 are partition-based. Feature 1 is numerical, feature 2 is one-hot, and
  // feature 3 is empty.
  EvalHelper eval{&ctx,
                  tparam,
                  {0, 3, 6, 8, 8, 11},
                  {0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0},
                  {kCat, FeatureType::kNumerical, kCat, kCat, kCat}};
  std::vector<bst_feature_t> fset{0, 1, 2, 3, 4};
  GradientPairPrecise parent{-6.0, 3.0};

  std::vector<GradientPairPrecise> best_split_feature_4{
      {-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0},
      {-3.0, 1.0}, {-3.0, 2.0}, {0.0, 1.0},  {-3.0, 1.0}, {-3.0, 1.0}};
  std::vector<GradientPairPrecise> best_split_feature_0{
      {-1.0, 1.0}, {-1.0, 1.0}, {-4.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0},
      {-3.0, 1.0}, {-3.0, 2.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}};
  auto results =
      eval.Evaluate({{1, parent, best_split_feature_4}, {2, parent, best_split_feature_0}}, fset);
  EXPECT_EQ(results[0].split.findex, 4);
  EXPECT_EQ(results[1].split.findex, 0);
  EXPECT_EQ(results[0].cats, std::bitset<32>("10000000000000000000000000000000"));
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

// Numerical and one-hot splits don't depend on the number of targets. Each single-target node
// is replicated across all targets, which must select the same split as a single target and
// scale the gain by the number of targets.
class GpuHistEvaluatorTargets : public ::testing::TestWithParam<bst_target_t> {
 protected:
  Context ctx_{MakeCUDACtx(0)};
  bst_target_t n_targets_{GetParam()};

  EvalHelper MakeEval(std::vector<std::uint32_t> const& cut_ptrs,
                      std::vector<float> const& cut_values,
                      std::vector<FeatureType> const& feature_types = {}) {
    return EvalHelper{&ctx_, ZeroParam(), cut_ptrs, cut_values, feature_types, n_targets_};
  }
  [[nodiscard]] std::vector<float> Replicate(float v) const {
    return std::vector<float>(n_targets_, v);
  }
};

TEST_P(GpuHistEvaluatorTargets, OneHotDense) {
  auto eval = this->MakeEval({0, 2, 4}, {1.0, 2.0, 11.0, 12.0}, {kCat, kCat});
  auto r =
      eval.EvaluateOne({1, {0.0, 1.0}, {{-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}}}, {0, 1});

  EXPECT_EQ(r.split.findex, 1);
  EXPECT_TRUE(std::isnan(r.split.fvalue));
  EXPECT_EQ(r.split.dir, kLeftDir);
}

TEST_P(GpuHistEvaluatorTargets, EvaluateSingleSplitMissing) {
  auto eval = this->MakeEval({0, 2}, {1.0, 2.0});
  auto r = eval.EvaluateOne({1, {1.0, 1.5}, {{-0.5, 0.5}, {0.5, 0.5}}}, {0});

  EXPECT_EQ(r.split.findex, 0);
  EXPECT_EQ(r.split.fvalue, 1.0);
  EXPECT_EQ(r.split.dir, kRightDir);
  EXPECT_EQ(r.left_sum, eval.ToFixed({-0.5, 0.5}));
  EXPECT_EQ(r.right_sum, eval.ToFixed({1.5, 1.0}));
}

TEST_P(GpuHistEvaluatorTargets, EvaluateSingleSplitEmpty) {
  auto eval = this->MakeEval({0, 1}, {1.0});
  auto r = eval.EvaluateOne({0, {0.0, 0.0}, {{0.0, 0.0}}}, {});
  EXPECT_EQ(r.split.findex, -1);
  EXPECT_LT(r.split.loss_chg, 0.0f);
}

TEST_P(GpuHistEvaluatorTargets, OneHotMissingRight) {
  auto eval = this->MakeEval({0, 3}, {0.0, 1.0, 2.0}, {kCat});
  // The missing row and category 0 both have gradient +1.
  auto r = eval.EvaluateOne({0, {0, 4}, {{1, 1}, {-1, 1}, {-1, 1}}}, {0});
  EXPECT_EQ(r.split.thresh, 0);
  EXPECT_EQ(r.split.dir, kRightDir);
  EXPECT_EQ(r.left_sum, eval.ToFixed({-2, 2}));
  EXPECT_EQ(r.right_sum, eval.ToFixed({2, 2}));
}

// Feature 0 has a better split, but the algorithm must select feature 1
TEST_P(GpuHistEvaluatorTargets, EvaluateSingleSplitFeatureSampling) {
  auto eval = this->MakeEval({0, 2, 4}, {1.0, 2.0, 11.0, 12.0});
  auto r =
      eval.EvaluateOne({1, {0.0, 1.0}, {{-10.0, 0.5}, {10.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}}}, {1});

  EXPECT_EQ(r.split.findex, 1);
  EXPECT_EQ(r.split.fvalue, 11.0);
  EXPECT_EQ(r.left_sum, eval.ToFixed({-0.5, 0.5}));
  EXPECT_EQ(r.right_sum, eval.ToFixed({0.5, 0.5}));
}

// Features 0 and 1 have identical gain, the algorithm must select 0
TEST_P(GpuHistEvaluatorTargets, EvaluateSingleSplitBreakTies) {
  auto eval = this->MakeEval({0, 2, 4}, {1.0, 2.0, 11.0, 12.0});
  auto r =
      eval.EvaluateOne({1, {0.0, 1.0}, {{-0.5, 0.5}, {0.5, 0.5}, {-0.5, 0.5}, {0.5, 0.5}}}, {0, 1});

  EXPECT_EQ(r.split.findex, 0);
  EXPECT_EQ(r.split.fvalue, 1.0);
}

// Without missing values, only the forward pass is run and missing values go right.
TEST_P(GpuHistEvaluatorTargets, EvaluateSingleSplitDense) {
  auto eval = this->MakeEval({0, 3, 6}, {1.0, 2.0, 3.0, 11.0, 12.0, 13.0});
  // The bins of every feature add up to the parent sum.
  auto r = eval.EvaluateOne(
      {1,
       {0.0, 1.5},
       {{-1.0, 0.5}, {0.25, 0.5}, {0.75, 0.5}, {-0.25, 0.5}, {-0.5, 0.5}, {0.75, 0.5}}},
      {0, 1});

  EXPECT_EQ(r.split.dir, kRightDir);
  EXPECT_EQ(r.split.findex, 0);
  EXPECT_EQ(r.split.fvalue, 1.0);
  EXPECT_FLOAT_EQ(r.split.loss_chg, 3.0f * n_targets_);
  EXPECT_EQ(r.left_sum, eval.ToFixed({-1.0, 0.5}));
  EXPECT_EQ(r.right_sum, eval.ToFixed({1.0, 1.0}));
  // weight = -G / H
  EXPECT_EQ(r.base_weight, this->Replicate(0.0f));
  EXPECT_EQ(r.left_weight, this->Replicate(2.0f));
  EXPECT_EQ(r.right_weight, this->Replicate(-1.0f));
}

TEST_P(GpuHistEvaluatorTargets, EvaluateSplits) {
  auto eval = this->MakeEval({0, 2, 4}, {1.0, 2.0, 11.0, 12.0});
  GradientPairPrecise parent{0.0, 1.0};
  auto results = eval.Evaluate({{1, parent, {{-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}}},
                                {2, parent, {{-1.0, 0.5}, {1.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}}}},
                               {0, 1});

  EXPECT_EQ(results[0].split.findex, 1);
  EXPECT_EQ(results[0].split.fvalue, 11.0);
  EXPECT_EQ(results[1].split.findex, 0);
  EXPECT_EQ(results[1].split.fvalue, 1.0);
}

INSTANTIATE_TEST_SUITE_P(GpuHist, GpuHistEvaluatorTargets, ::testing::Values(1, 3),
                         [](auto const& info) { return std::to_string(info.param); });

namespace {
TrainParam MultiTargetParam(Args args = {}) {
  TrainParam param;
  param.UpdateAllowUnknown(
      Args{{"min_child_weight", "0"}, {"reg_lambda", "0"}, {"learning_rate", "1"}});
  param.UpdateAllowUnknown(args);
  return param;
}

Node MultiTargetDenseNode() {
  // Two targets for a single feature with 4 bins. The comments are the left/right sums for
  // each split point.
  return {0,
          std::vector<GradientPairPrecise>{{56, 40}, {96, 128}},
          {
              // First target, 0/0, 56/40
              {8, 4},    // 8/4, 48/36
              {12, 8},   // 20/12, 36/28
              {16, 12},  // 36/24, 20/16
              {20, 16},  // 56/40, 0/0
              // Second target, 0/0, 96/128
              {11, 13},  // 11/13, 85/115
              {19, 29},  // 30/42, 66/86
              {27, 45},  // 57/87, 39/41
              {39, 41},  // 96/128, 0/0
          }};
}
}  // anonymous namespace

TEST(GpuHist, MultiTargetDense) {
  auto ctx = MakeCUDACtx(0);
  // The best one-hot split isolates category 3, which is the same partition as the best
  // numerical split.
  for (bool is_cat : {false, true}) {
    auto param = MultiTargetParam(Args{{"max_cat_to_onehot", "100"}});
    std::vector<FeatureType> ft;
    std::vector<float> cut_values{.0f, .1f, .2f, .3f};
    if (is_cat) {
      ft = {kCat};
      cut_values = {0.0f, 1.0f, 2.0f, 3.0f};
    }
    EvalHelper eval{&ctx, param, {0, 4}, cut_values, ft, 2};
    auto r = eval.EvaluateOne(MultiTargetDenseNode(), {0});
    ASSERT_EQ(r.split.is_cat, is_cat);
    ASSERT_NEAR(r.split.loss_chg, 3.04239, 1e-5);
    // No missing value, only the forward pass is performed.
    ASSERT_EQ(r.split.dir, is_cat ? kLeftDir : kRightDir);
    if (is_cat) {
      ASSERT_TRUE(std::isnan(r.split.fvalue));
      ASSERT_EQ(r.split.thresh, 3);
      ASSERT_EQ(r.cats, std::bitset<32>("00010000000000000000000000000000"));
    } else {
      ASSERT_EQ(r.split.fvalue, .2f);
    }
    ASSERT_EQ(r.left_sum, eval.ToFixed({{36, 24}, {57, 87}}));
    ASSERT_EQ(r.right_sum, eval.ToFixed({{20, 16}, {39, 41}}));
    ASSERT_EQ(r.base_weight, (std::vector<float>{-1.4, -0.75}));
    ASSERT_THAT(r.left_weight, ::testing::Pointwise(::testing::FloatNear(1e-5), {-1.5, -0.655172}));
    ASSERT_THAT(r.right_weight,
                ::testing::Pointwise(::testing::FloatNear(1e-5), {-1.25, -0.951219}));
  }

  auto param = MultiTargetParam(Args{{"reg_alpha", "0.1"}, {"max_delta_step", "1.45"}});
  EvalHelper eval{&ctx, param, {0, 4}, {.0f, .1f, .2f, .3f}, {}, 2};
  auto r = eval.EvaluateOne(MultiTargetDenseNode(), {0});
  ASSERT_NEAR(r.split.loss_chg, 2.55177, 1e-5);
}

TEST(GpuHist, MultiTargetPartition) {
  auto ctx = MakeCUDACtx(0);
  // The optimal split groups categories {2, 3} together. No one-hot split can represent this
  // 2-vs-2 partition.
  Node node{0,
            std::vector<GradientPairPrecise>{{-9, 4}, {-7, 4}},
            {
                {-4, 1},
                {-3, 1},
                {-1, 1},
                {-1, 1},  // target 0
                {-3, 1},
                {-2, 1},
                {-1, 1},
                {-1, 1},  // target 1
            }};
  std::vector<float> cut_values{0.0f, 1.0f, 2.0f, 3.0f};
  auto param = MultiTargetParam(Args{{"max_cat_to_onehot", "1"}});
  EvalHelper eval{&ctx, param, {0, 4}, cut_values, {kCat}, 2};
  auto r = eval.EvaluateOne(node, {0});

  ASSERT_TRUE(r.split.is_cat);
  ASSERT_NEAR(r.split.loss_chg, 8.5, 1e-5);
  ASSERT_EQ(r.split.dir, kLeftDir);
  ASSERT_EQ(r.split.findex, 0);
  ASSERT_TRUE(std::isnan(r.split.fvalue));
  ASSERT_EQ(r.split.thresh, 1);
  // Categories {2, 3} form the right child.
  ASSERT_EQ(r.cats, std::bitset<32>("00110000000000000000000000000000"));
  ASSERT_EQ(r.right_sum, eval.ToFixed({{-2, 2}, {-2, 2}}));
  ASSERT_EQ(r.base_weight, (std::vector<float>{2.25f, 1.75f}));
  ASSERT_EQ(r.left_weight, (std::vector<float>{3.5f, 2.5f}));
  ASSERT_EQ(r.right_weight, (std::vector<float>{1.0f, 1.0f}));
  ASSERT_EQ(r.left_hess, 4.0);
  ASSERT_EQ(r.right_hess, 4.0);

  // max_cat_threshold=1 does not enumerate a partition. The resulting root-only tree must
  // still retain its base weight and Hessian.
  auto no_split_param =
      MultiTargetParam(Args{{"max_cat_to_onehot", "1"}, {"max_cat_threshold", "1"}});
  EvalHelper no_split_eval{&ctx, no_split_param, {0, 4}, cut_values, {kCat}, 2};
  auto no_split = no_split_eval.EvaluateOne(node, {0});
  ASSERT_TRUE(no_split.split.child_sum.empty());
  ASSERT_FALSE(IsValidExpandEntry(no_split, no_split_param, 100));
  ASSERT_EQ(no_split.base_weight, (std::vector<float>{2.25f, 1.75f}));
  ASSERT_EQ(no_split.left_hess + no_split.right_hess, 8.0);
}

TEST_F(TestPartitionBasedSplit, GpuHist) {
  auto ctx = MakeCUDACtx(0);
  EvalHelper eval{&ctx, param_, cuts_, {kCat}};
  auto node_hist = hist_[0];
  std::vector<GradientPairPrecise> hist(node_hist.cbegin(), node_hist.cend());
  auto r = eval.EvaluateOne({0, total_gpair_, hist}, {0});
  ASSERT_NEAR(r.split.loss_chg, best_score_, 1e-2);
}
}  // namespace xgboost::tree
