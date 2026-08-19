/**
 * Copyright 2018-2026, XGBoost Contributors
 */
#include <algorithm>  // for shuffle
#include <cstddef>    // for size_t
#include <numeric>    // for iota
#include <sstream>    // for stringstream
#include <vector>     // for vector

#include "../../../src/common/random.h"
#include "../helpers.h"
#include "gtest/gtest.h"
#include "xgboost/context.h"  // for Context, SerializableRandomEngine
#include "xgboost/json.h"     // for Json, Object, String, get, IsA

namespace xgboost::common {
namespace {
void TestBasic(Context const* ctx) {
  int n = 128;
  ColumnSampler cs;
  HostDeviceVector<float> feature_weights;

  // No node sampling
  cs.Init(ctx, n, feature_weights, 1.0f, 0.5f, 0.5f);
  auto set0 = cs.GetFeatureSet(ctx, 0);
  ASSERT_EQ(set0->Size(), 32);

  auto set1 = cs.GetFeatureSet(ctx, 0);

  ASSERT_EQ(set0->HostVector(), set1->HostVector());

  auto set2 = cs.GetFeatureSet(ctx, 1);
  ASSERT_NE(set1->HostVector(), set2->HostVector());
  ASSERT_EQ(set2->Size(), 32);

  // Node sampling
  cs.Init(ctx, n, feature_weights, 0.5f, 1.0f, 0.5f);
  auto set3 = cs.GetFeatureSet(ctx, 0);
  ASSERT_EQ(set3->Size(), 32);

  auto set4 = cs.GetFeatureSet(ctx, 0);

  ASSERT_NE(set3->HostVector(), set4->HostVector());
  ASSERT_EQ(set4->Size(), 32);

  // No level or node sampling, should be the same at different depth
  cs.Init(ctx, n, feature_weights, 1.0f, 1.0f, 0.5f);
  ASSERT_EQ(cs.GetFeatureSet(ctx, 0)->HostVector(), cs.GetFeatureSet(ctx, 1)->HostVector());

  cs.Init(ctx, n, feature_weights, 1.0f, 1.0f, 1.0f);
  auto set5 = cs.GetFeatureSet(ctx, 0);
  ASSERT_EQ(set5->Size(), n);
  cs.Init(ctx, n, feature_weights, 1.0f, 1.0f, 1.0f);
  auto set6 = cs.GetFeatureSet(ctx, 0);
  ASSERT_EQ(set5->HostVector(), set6->HostVector());

  // Should always be a minimum of one feature
  cs.Init(ctx, n, feature_weights, 1e-16f, 1e-16f, 1e-16f);
  ASSERT_EQ(cs.GetFeatureSet(ctx, 0)->Size(), 1);
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

// Test if different threads using the same seed produce the same result.
// Each thread gets its own Context (since ctx->Rng() is not thread-safe) with the same
// seed. All threads should produce identical column samples.
TEST(ColumnSampler, ThreadSynchronisation) {
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
  HostDeviceVector<float> feature_weights;
  bool success = true;
#pragma omp parallel num_threads(n_threads)
  {
    for (auto j = 0ull; j < iterations; j++) {
      Context ctx;
      ctx.Init({{"seed", std::to_string(j)}});
      ColumnSampler cs;
      cs.Init(&ctx, n, feature_weights, 0.5f, 0.5f, 0.5f);
      for (auto level = 0ull; level < levels; level++) {
        auto result = cs.GetFeatureSet(&ctx, level)->ConstHostVector();
#pragma omp single
        {
          reference_result = result;
        }
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
    HostDeviceVector<float> feature_weights(2);
    feature_weights.HostVector()[0] = std::abs(first - 1.0f);
    feature_weights.HostVector()[1] = first - 0.0f;
    ColumnSampler cs;
    cs.Init(ctx, 2, feature_weights, 1.0, 1.0, 0.5);
    auto feature_sets = cs.GetFeatureSet(ctx, 0);
    auto const& h_feat_set = feature_sets->HostVector();
    ASSERT_EQ(h_feat_set.size(), 1);
    ASSERT_EQ(h_feat_set[0], first - 0);
  };

  test_basic(0);
  test_basic(1);

  size_t constexpr kCols = 64;
  HostDeviceVector<float> feature_weights(kCols);
  SimpleLCG rng;
  SimpleRealUniformDistribution<float> dist(.0f, 12.0f);
  std::generate(feature_weights.HostVector().begin(), feature_weights.HostVector().end(),
                [&]() { return dist(&rng); });
  ColumnSampler cs;
  cs.Init(ctx, kCols, feature_weights, 0.5f, 1.0f, 1.0f);
  std::vector<bst_feature_t> features(kCols);
  std::iota(features.begin(), features.end(), 0);
  std::vector<float> freq(kCols, 0);
  for (size_t i = 0; i < 1024; ++i) {
    auto fset = cs.GetFeatureSet(ctx, 0);
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
  auto& h_feature_weights = feature_weights.HostVector();
  norm = std::accumulate(h_feature_weights.cbegin(), h_feature_weights.cend(), .0f);
  for (auto& f : h_feature_weights) {
    f /= norm;
  }

  for (size_t i = 0; i < h_feature_weights.size(); ++i) {
    EXPECT_NEAR(freq[i], h_feature_weights[i], 1e-2);
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
  HostDeviceVector<float> feature_weights(kCols, 0);
  auto& h_feature_weights = feature_weights.HostVector();
  for (size_t i = 0; i < h_feature_weights.size(); ++i) {
    h_feature_weights[i] = i;
  }
  ColumnSampler cs;
  float bytree{0.5}, bylevel{0.5}, bynode{0.5};
  cs.Init(ctx, h_feature_weights.size(), feature_weights, bytree, bylevel, bynode);
  auto feature_set = cs.GetFeatureSet(ctx, 0);
  size_t n_sampled = kCols * bytree * bylevel * bynode;
  ASSERT_EQ(feature_set->Size(), n_sampled);
  feature_set = cs.GetFeatureSet(ctx, 1);
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

namespace {
// The state must survive a round trip, and must do so through plain integers only. The
// textual form of `std::mt19937` differs between standard library implementations, so a
// snapshot holding it could not be moved between platforms. See
// https://github.com/dmlc/xgboost/issues/12459 .
TEST(RngState, RoundTrip) {
  SerializableRandomEngine rng;
  rng.seed(1994);
  for (std::size_t i = 0; i < 17; ++i) {
    static_cast<void>(rng());
  }

  Json out{Object{}};
  SaveRng(&out, rng);

  SerializableRandomEngine loaded;
  ASSERT_TRUE(LoadRng(out, &loaded));
  ASSERT_EQ(loaded.Seed(), rng.Seed());
  ASSERT_EQ(loaded.NumAdvanced(), rng.NumAdvanced());
  for (std::size_t i = 0; i < 32; ++i) {
    ASSERT_EQ(loaded(), rng());
  }
}

TEST(RngState, IsPlatformIndependent) {
  SerializableRandomEngine rng;
  rng.seed(3);
  static_cast<void>(rng());

  Json out{Object{}};
  SaveRng(&out, rng);

  // The seed and the draw count, spelled out by us. Nothing here can vary with the
  // standard library that wrote it.
  ASSERT_EQ(get<String const>(out["rng_state"]), "3 1");
}

TEST(RngState, RejectsLegacyState) {
  // 3.3.0 and earlier wrote the text produced by `operator<<` for `std::mt19937`. It must
  // be refused rather than misread, and refusing it must not throw.
  SerializableRandomEngine rng;
  rng.seed(7);

  std::stringstream ss;
  ss << std::hex << RandomEngine{1994};
  Json legacy{Object{}};
  legacy["rng_state"] = String{ss.str()};
  ASSERT_FALSE(LoadRng(legacy, &rng));
  // Left untouched for the caller to re-seed.
  ASSERT_EQ(rng.Seed(), 7u);
  ASSERT_EQ(rng.NumAdvanced(), 0ul);

  // Neither a missing state nor a malformed one is an error.
  Json empty{Object{}};
  ASSERT_FALSE(LoadRng(empty, &rng));
  for (auto const& malformed : {"", " ", "12", "1 2 3", "-1 2", "1 x"}) {
    Json bad{Object{}};
    bad["rng_state"] = String{malformed};
    ASSERT_FALSE(LoadRng(bad, &rng)) << "accepted: '" << malformed << "'";
  }
  ASSERT_EQ(rng.Seed(), 7u);
}

// Advancing by hand and restoring must agree, since restoring replays the draws with
// `discard`.
TEST(RngState, RestoreMatchesReplay) {
  SerializableRandomEngine reference;
  reference.seed(11);
  for (std::size_t i = 0; i < 1000; ++i) {
    static_cast<void>(reference());
  }

  SerializableRandomEngine restored;
  restored.Restore(11, 1000);
  ASSERT_EQ(restored.NumAdvanced(), reference.NumAdvanced());
  for (std::size_t i = 0; i < 8; ++i) {
    ASSERT_EQ(restored(), reference());
  }
}

// The wrapper must stay a drop-in for `std::mt19937` so that call sites can keep handing
// it to standard distributions and algorithms.
TEST(RngState, MatchesStdEngine) {
  SerializableRandomEngine wrapped{1994};
  RandomEngine plain{1994};

  ASSERT_EQ(SerializableRandomEngine::min(), RandomEngine::min());
  ASSERT_EQ(SerializableRandomEngine::max(), RandomEngine::max());
  for (std::size_t i = 0; i < 64; ++i) {
    ASSERT_EQ(wrapped(), plain());
  }
  ASSERT_EQ(wrapped.NumAdvanced(), 64ul);

  std::vector<int> a(32), b(32);
  std::iota(a.begin(), a.end(), 0);
  std::iota(b.begin(), b.end(), 0);
  wrapped.seed(7);
  plain.seed(7);
  std::shuffle(a.begin(), a.end(), wrapped);
  std::shuffle(b.begin(), b.end(), plain);
  ASSERT_EQ(a, b);
}
}  // namespace
}  // namespace xgboost::common
