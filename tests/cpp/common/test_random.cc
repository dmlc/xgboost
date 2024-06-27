/**
 * Copyright 2018-2023, XGBoost Contributors
 */
#include "../../../src/common/random.h"
#include "../helpers.h"
#include "gtest/gtest.h"
#include "xgboost/context.h"  // for Context

namespace xgboost::common {
namespace {
void TestBasic(Context const* ctx) {
  int n = 128;
  ColumnSampler cs{1u};
  std::vector<float> feature_weights;

  // No node sampling
  cs.Init(ctx, n, feature_weights, 1.0f, 0.5f, 0.5f);
  auto set0 = cs.GetFeatureSet(0);
  ASSERT_EQ(set0->Size(), 32);

  auto set1 = cs.GetFeatureSet(0);

  ASSERT_EQ(set0->HostVector(), set1->HostVector());

  auto set2 = cs.GetFeatureSet(1);
  ASSERT_NE(set1->HostVector(), set2->HostVector());
  ASSERT_EQ(set2->Size(), 32);

  // Node sampling
  cs.Init(ctx, n, feature_weights, 0.5f, 1.0f, 0.5f);
  auto set3 = cs.GetFeatureSet(0);
  ASSERT_EQ(set3->Size(), 32);

  auto set4 = cs.GetFeatureSet(0);

  ASSERT_NE(set3->HostVector(), set4->HostVector());
  ASSERT_EQ(set4->Size(), 32);

  // No level or node sampling, should be the same at different depth
  cs.Init(ctx, n, feature_weights, 1.0f, 1.0f, 0.5f);
  ASSERT_EQ(cs.GetFeatureSet(0)->HostVector(), cs.GetFeatureSet(1)->HostVector());

  cs.Init(ctx, n, feature_weights, 1.0f, 1.0f, 1.0f);
  auto set5 = cs.GetFeatureSet(0);
  ASSERT_EQ(set5->Size(), n);
  cs.Init(ctx, n, feature_weights, 1.0f, 1.0f, 1.0f);
  auto set6 = cs.GetFeatureSet(0);
  ASSERT_EQ(set5->HostVector(), set6->HostVector());

  // Should always be a minimum of one feature
  cs.Init(ctx, n, feature_weights, 1e-16f, 1e-16f, 1e-16f);
  ASSERT_EQ(cs.GetFeatureSet(0)->Size(), 1);
}
}  // namespace

TEST(ColumnSampler, Test) {
  Context ctx;
  TestBasic(&ctx);
}

#if defined(XGBOOST_USE_CUDA)
TEST(ColumnSampler, GPUTest) {
  auto ctx = MakeCUDACtx(0);
  TestBasic(&ctx);
}
#endif  // defined(XGBOOST_USE_CUDA)

// Test if different threads using the same seed produce the same result
TEST(ColumnSampler, ThreadSynchronisation) {
  Context ctx;
  // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
#if defined(__linux__)
  std::int64_t const n_threads = std::thread::hardware_concurrency() * 128;
#else
  std::int64_t const n_threads = std::thread::hardware_concurrency();
#endif
  // NOLINTEND(clang-analyzer-deadcode.DeadStores)
  int n = 128;
  size_t iterations = 10;
  size_t levels = 5;
  std::vector<bst_feature_t> reference_result;
  std::vector<float> feature_weights;
  bool success = true; // Cannot use google test asserts in multithreaded region
#pragma omp parallel num_threads(n_threads)
  {
    for (auto j = 0ull; j < iterations; j++) {
      ColumnSampler cs(j);
      cs.Init(&ctx, n, feature_weights, 0.5f, 0.5f, 0.5f);
      for (auto level = 0ull; level < levels; level++) {
        auto result = cs.GetFeatureSet(level)->ConstHostVector();
#pragma omp single
        { reference_result = result; }
        if (result != reference_result) {
          success = false;
        }
#pragma omp barrier
      }
    }
  }
  ASSERT_TRUE(success);
}

namespace {
void TestWeightedSampling(Context const* ctx) {
  auto test_basic = [ctx](int first) {
    std::vector<float> feature_weights(2);
    feature_weights[0] = std::abs(first - 1.0f);
    feature_weights[1] = first - 0.0f;
    ColumnSampler cs{0};
    cs.Init(ctx, 2, feature_weights, 1.0, 1.0, 0.5);
    auto feature_sets = cs.GetFeatureSet(0);
    auto const& h_feat_set = feature_sets->HostVector();
    ASSERT_EQ(h_feat_set.size(), 1);
    ASSERT_EQ(h_feat_set[0], first - 0);
  };

  test_basic(0);
  test_basic(1);

  size_t constexpr kCols = 64;
  std::vector<float> feature_weights(kCols);
  SimpleLCG rng;
  SimpleRealUniformDistribution<float> dist(.0f, 12.0f);
  std::generate(feature_weights.begin(), feature_weights.end(), [&]() { return dist(&rng); });
  ColumnSampler cs{0};
  cs.Init(ctx, kCols, feature_weights, 0.5f, 1.0f, 1.0f);
  std::vector<bst_feature_t> features(kCols);
  std::iota(features.begin(), features.end(), 0);
  std::vector<float> freq(kCols, 0);
  for (size_t i = 0; i < 1024; ++i) {
    auto fset = cs.GetFeatureSet(0);
    ASSERT_EQ(kCols * 0.5, fset->Size());
    auto const& h_fset = fset->HostVector();
    for (auto f : h_fset) {
      freq[f] += 1.0f;
    }
  }

  auto norm = std::accumulate(freq.cbegin(), freq.cend(), .0f);
  for (auto& f : freq) {
    f /= norm;
  }
  norm = std::accumulate(feature_weights.cbegin(), feature_weights.cend(), .0f);
  for (auto& f : feature_weights) {
    f /= norm;
  }

  for (size_t i = 0; i < feature_weights.size(); ++i) {
    EXPECT_NEAR(freq[i], feature_weights[i], 1e-2);
  }
}
}  // namespace

TEST(ColumnSampler, WeightedSampling) {
  Context ctx;
  TestWeightedSampling(&ctx);
}

#if defined(XGBOOST_USE_CUDA)
TEST(ColumnSampler, GPUWeightedSampling) {
  auto ctx = MakeCUDACtx(0);
  TestWeightedSampling(&ctx);
}
#endif  // defined(XGBOOST_USE_CUDA)

namespace {
void TestWeightedMultiSampling(Context const* ctx) {
  size_t constexpr kCols = 32;
  std::vector<float> feature_weights(kCols, 0);
  for (size_t i = 0; i < feature_weights.size(); ++i) {
    feature_weights[i] = i;
  }
  ColumnSampler cs{0};
  float bytree{0.5}, bylevel{0.5}, bynode{0.5};
  cs.Init(ctx, feature_weights.size(), feature_weights, bytree, bylevel, bynode);
  auto feature_set = cs.GetFeatureSet(0);
  size_t n_sampled = kCols * bytree * bylevel * bynode;
  ASSERT_EQ(feature_set->Size(), n_sampled);
  feature_set = cs.GetFeatureSet(1);
  ASSERT_EQ(feature_set->Size(), n_sampled);
}
}  // namespace

TEST(ColumnSampler, WeightedMultiSampling) {
  Context ctx;
  TestWeightedMultiSampling(&ctx);
}

#if defined(XGBOOST_USE_CUDA)
TEST(ColumnSampler, GPUWeightedMultiSampling) {
  auto ctx = MakeCUDACtx(0);
  TestWeightedMultiSampling(&ctx);
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::common
