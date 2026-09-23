/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>
#include <xgboost/gradient.h>
#include <xgboost/objective.h>

#include <cmath>    // for exp
#include <cstdint>  // for uint32_t
#include <cstring>  // for memcpy
#include <memory>   // for unique_ptr
#include <string>   // for to_string
#include <vector>   // for vector

#include "../../../src/common/exact_multinomial/packed_stats.h"
#include "../helpers.h"

namespace xgboost {
namespace {
struct MulticlassFixture {
  char const* name;
  std::int32_t n_classes;
  std::vector<float> preds;
  std::vector<float> labels;
  std::vector<float> weights;
};

/**
 * @brief Inputs covering the numerical corners the objective already has to handle:
 *        weighted and unweighted rows, saturating logits, the binary case, a zero weight
 *        and a class whose probability underflows.
 */
std::vector<MulticlassFixture> Fixtures() {
  return {
      {"k3_weighted", 3, {1.0f, 0.0f, 2.0f, 2.0f, 0.0f, 1.0f}, {1.0f, 0.0f}, {0.5f, 2.0f}},
      {"k3_unweighted", 3, {1.0f, 0.0f, 2.0f, 2.0f, 0.0f, 1.0f}, {1.0f, 0.0f}, {}},
      {"k3_extreme", 3, {100.0f, -100.0f, 0.0f, -50.0f, 50.0f, 0.0f}, {0.0f, 2.0f}, {}},
      {"k2_basic", 2, {0.5f, -0.5f, -2.0f, 3.0f}, {0.0f, 1.0f}, {}},
      {"k7_spread",
       7,
       {0.0f, 1.0f, -1.0f, 5.0f, -12.0f, 2.5f, -0.25f, 3.0f, 3.0f, -8.0f, 0.5f, 1.25f, -2.0f, 0.0f},
       {6.0f, 0.0f},
       {1.5f, 0.0f}},
  };
}

/**
 * @brief Raw bits of the gradient pairs produced by `GetGradient` before the exact Hessian
 *        work started, as (grad, hess) pairs in row major order.
 *
 * These lock the existing multinomial gradient against any floating point drift caused by
 * sharing arithmetic with the exact producer. They were captured from the unmodified
 * objective; a mismatch means the scalar path changed and must be investigated rather than
 * re-baselined. Note the negative zero and the denormal below: a tolerance based check
 * would not notice either.
 */
std::vector<std::uint32_t> GoldenBits(std::string const& name) {
  if (name == "k3_weighted") {
    return {0x3dfa9a1au, 0x3dfa9a1au, 0xbee8f3c2u, 0x3ee8f3c2u, 0x3eaa4d3bu, 0x3eaa4d3bu,
            0xbf2b658au, 0x3f2b658au, 0x3e3861f3u, 0x3e3861f3u, 0x3efa9a1au, 0x3efa9a1au};
  }
  if (name == "k3_unweighted") {
    return {0x3e7a9a1au, 0x3e7a9a1au, 0xbf68f3c2u, 0x3f68f3c2u, 0x3f2a4d3bu, 0x3f2a4d3bu,
            0xbeab658au, 0x3eab658au, 0x3db861f3u, 0x3db861f3u, 0x3e7a9a1au, 0x3e7a9a1au};
  }
  if (name == "k3_extreme") {
    return {0x00000000u, 0x24e69595u, 0x00000000u, 0x24e69595u, 0x0000001bu, 0x24e69595u,
            0x0000001bu, 0x24e69595u, 0x3f800000u, 0x3f800000u, 0xbf800000u, 0x3f800000u};
  }
  if (name == "k2_basic") {
    return {0xbe89b2b0u, 0x3e89b2b0u, 0x3e89b2b1u, 0x3e89b2b1u,
            0x3bdb4fb4u, 0x3bdb4fb4u, 0xbbdb4f80u, 0x3bdb4f80u};
  }
  if (name == "k7_spread") {
    return {0x3c1487e3u, 0x3c1487e3u, 0x3cc9dfd2u, 0x3cc9dfd2u, 0x3b5a90d3u, 0x3b5a90d3u,
            0x3fac37dau, 0x3fac37dau, 0x336f3be0u, 0x336f3be0u, 0x3de22f38u, 0x3de22f38u,
            0xbfbf18a6u, 0x3fbf18a6u, 0x80000000u, 0x24e69595u, 0x00000000u, 0x24e69595u,
            0x00000000u, 0x24e69595u, 0x00000000u, 0x24e69595u, 0x00000000u, 0x24e69595u,
            0x00000000u, 0x24e69595u, 0x00000000u, 0x24e69595u};
  }
  return {};
}

MetaInfo MakeInfo(std::vector<float> const& labels, std::vector<float> const& weights) {
  MetaInfo info;
  info.num_row_ = labels.size();
  info.labels = linalg::Tensor<float, 2>{labels.cbegin(),
                                         labels.cend(),
                                         {labels.size(), static_cast<std::size_t>(1)},
                                         DeviceOrd::CPU()};
  info.weights_.HostVector() = weights;
  return info;
}

std::unique_ptr<ObjFunction> MakeMulticlassObj(Context const* ctx, std::int32_t n_classes,
                                               char const* name = "multi:softprob") {
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create(name, ctx)};
  obj->Configure({{"num_class", std::to_string(n_classes)}});
  return obj;
}

std::uint32_t Bits(float v) {
  std::uint32_t out;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}

/** @brief Logits whose float softmax reproduces @p probability. */
std::vector<float> LogitsFor(std::vector<double> const& probability) {
  std::vector<float> out;
  out.reserve(probability.size());
  for (auto p : probability) {
    out.push_back(static_cast<float>(std::log(p)));
  }
  return out;
}

/** @brief Softmax in double, used as the higher precision reference. */
std::vector<double> ReferenceProbability(std::vector<float> const& logits) {
  auto wmax = *std::max_element(logits.cbegin(), logits.cend());
  double wsum = 0.0;
  std::vector<double> out(logits.size());
  for (std::size_t k = 0; k < logits.size(); ++k) {
    out[k] = std::exp(static_cast<double>(logits[k]) - static_cast<double>(wmax));
    wsum += out[k];
  }
  for (auto& p : out) {
    p /= wsum;
  }
  return out;
}
}  // anonymous namespace

/**
 * Existing behaviour must not move. The exact producer shares its scalar arithmetic with
 * `GetGradient`, so this guards that sharing against any rounding drift.
 */
TEST(MulticlassExactObj, ExistingGradientBitsUnchanged) {
  Context ctx;
  for (auto const& fixture : Fixtures()) {
    auto obj = MakeMulticlassObj(&ctx, fixture.n_classes);
    auto info = MakeInfo(fixture.labels, fixture.weights);
    HostDeviceVector<float> preds{fixture.preds};
    linalg::Matrix<GradientPair> gpair;
    obj->GetGradient(preds, info, 0, &gpair);

    auto const& values = gpair.Data()->HostVector();
    auto golden = GoldenBits(fixture.name);
    ASSERT_EQ(values.size() * 2, golden.size()) << fixture.name;
    for (std::size_t i = 0; i < values.size(); ++i) {
      ASSERT_EQ(Bits(values[i].GetGrad()), golden[i * 2]) << fixture.name << " grad at " << i;
      ASSERT_EQ(Bits(values[i].GetHess()), golden[i * 2 + 1]) << fixture.name << " hess at " << i;
    }
  }
}

/**
 * The hard invariant: the combined producer's gradient is the scalar path's gradient, bit
 * for bit. Compared as raw bits so that a signed zero or a denormal cannot slip through.
 */
TEST(MulticlassExactObj, CombinedProducerGradientIsBitIdentical) {
  Context ctx;
  for (auto const& fixture : Fixtures()) {
    auto obj = MakeMulticlassObj(&ctx, fixture.n_classes);
    auto info = MakeInfo(fixture.labels, fixture.weights);
    HostDeviceVector<float> preds{fixture.preds};

    linalg::Matrix<GradientPair> scalar_gpair;
    obj->GetGradient(preds, info, 0, &scalar_gpair);

    GradientContainer container;
    obj->GetGradientAndExactHessian(preds, info, 0, &container.gpair, &container.exact_hessian);

    auto const& lhs = scalar_gpair.Data()->HostVector();
    auto const& rhs = container.gpair.Data()->HostVector();
    ASSERT_EQ(lhs.size(), rhs.size()) << fixture.name;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
      ASSERT_EQ(Bits(lhs[i].GetGrad()), Bits(rhs[i].GetGrad())) << fixture.name << " grad " << i;
      ASSERT_EQ(Bits(lhs[i].GetHess()), Bits(rhs[i].GetHess())) << fixture.name << " hess " << i;
    }

    // The sidecar is filled in the same call.
    ASSERT_TRUE(container.HasExactHessian()) << fixture.name;
    ASSERT_EQ(container.exact_hessian.n_free, fixture.n_classes - 1) << fixture.name;
    ASSERT_EQ(container.exact_hessian.NumRows(), info.num_row_) << fixture.name;
  }
}

/**
 * p = [0.1, 0.3, 0.6] with class 2 as the reference gives
 *   H = [[0.09, -0.03], [-0.03, 0.21]]
 * Asserted through the packed row as well, so a transposed or mis-indexed write fails.
 */
TEST(MulticlassExactObj, KnownHessian) {
  Context ctx;
  auto obj = MakeMulticlassObj(&ctx, 3);
  auto logits = LogitsFor({0.1, 0.3, 0.6});
  auto info = MakeInfo({1.0f}, {});
  HostDeviceVector<float> preds{logits};

  GradientContainer container;
  obj->GetGradientAndExactHessian(preds, info, 0, &container.gpair, &container.exact_hessian);

  ASSERT_EQ(container.exact_hessian.n_free, 2);
  ASSERT_EQ(container.exact_hessian.RowSize(), 3);
  auto row = container.exact_hessian.HostRow(0);

  // Packed lower triangle order: H00, H10, H11.
  EXPECT_NEAR(row[0], 0.09f, 1e-6f);
  EXPECT_NEAR(row[1], -0.03f, 1e-6f);
  EXPECT_NEAR(row[2], 0.21f, 1e-6f);

  auto hessian = common::PackedHessianAtRow(row, container.exact_hessian.n_free, 0);
  EXPECT_NEAR(hessian.Get(0, 0), 0.09f, 1e-6f);
  EXPECT_NEAR(hessian.Get(0, 1), -0.03f, 1e-6f);
  EXPECT_NEAR(hessian.Get(1, 0), -0.03f, 1e-6f);
  EXPECT_NEAR(hessian.Get(1, 1), 0.21f, 1e-6f);

  // The reference class is excluded: p2 * (1 - p2) = 0.24 must appear nowhere, which is
  // what a parameterisation using the *first* class as the reference would have produced.
  for (auto v : row) {
    EXPECT_GT(std::fabs(v - 0.24f), 1e-3f);
  }

  // The gradient of the free classes is carried by gpair, not duplicated in the sidecar.
  auto gpair = container.gpair.HostView();
  EXPECT_NEAR(gpair(0, 0).GetGrad(), 0.1f, 1e-6f);   // p0 - y0
  EXPECT_NEAR(gpair(0, 1).GetGrad(), -0.7f, 1e-6f);  // p1 - y1
}

/** The Hessian scales linearly with the sample weight. */
TEST(MulticlassExactObj, WeightedHessian) {
  Context ctx;
  auto obj = MakeMulticlassObj(&ctx, 3);
  auto logits = LogitsFor({0.1, 0.3, 0.6});
  HostDeviceVector<float> preds{logits};
  float constexpr kWeight = 2.5f;

  GradientContainer unweighted;
  auto plain_info = MakeInfo({1.0f}, {});
  obj->GetGradientAndExactHessian(preds, plain_info, 0, &unweighted.gpair,
                                  &unweighted.exact_hessian);

  GradientContainer weighted;
  auto weighted_info = MakeInfo({1.0f}, {kWeight});
  obj->GetGradientAndExactHessian(preds, weighted_info, 0, &weighted.gpair,
                                  &weighted.exact_hessian);

  auto plain = unweighted.exact_hessian.HostRow(0);
  auto scaled = weighted.exact_hessian.HostRow(0);
  ASSERT_EQ(plain.size(), scaled.size());
  for (std::size_t k = 0; k < plain.size(); ++k) {
    EXPECT_NEAR(scaled[k], plain[k] * kWeight, 1e-6f) << "entry " << k;
  }
  EXPECT_NEAR(scaled[0], 0.09f * kWeight, 1e-6f);
  EXPECT_NEAR(scaled[2], 0.21f * kWeight, 1e-6f);
}

/**
 * The multinomial Hessian depends on the probabilities alone. Changing the label must not
 * perturb a single bit of it.
 */
TEST(MulticlassExactObj, HessianIsLabelIndependent) {
  Context ctx;
  auto obj = MakeMulticlassObj(&ctx, 3);
  auto logits = LogitsFor({0.1, 0.3, 0.6});
  HostDeviceVector<float> preds{logits};

  std::vector<std::uint32_t> reference;
  for (float label : {0.0f, 1.0f, 2.0f}) {
    GradientContainer container;
    auto info = MakeInfo({label}, {1.5f});
    obj->GetGradientAndExactHessian(preds, info, 0, &container.gpair, &container.exact_hessian);

    std::vector<std::uint32_t> bits;
    for (auto v : container.exact_hessian.HostRow(0)) {
      bits.push_back(Bits(v));
    }
    if (reference.empty()) {
      reference = bits;
      ASSERT_EQ(reference.size(), 3);
    } else {
      ASSERT_EQ(bits, reference) << "label " << label << " changed the Hessian";
    }
  }
}

/** The exact coordinates have dimension K - 1 for every K. */
TEST(MulticlassExactObj, FreeDimension) {
  Context ctx;
  for (std::int32_t n_classes : {2, 3, 5, 7}) {
    auto obj = MakeMulticlassObj(&ctx, n_classes);
    std::size_t constexpr kRows = 4;
    std::vector<float> logits(kRows * n_classes);
    for (std::size_t i = 0; i < logits.size(); ++i) {
      logits[i] = static_cast<float>(i % 5) * 0.5f - 1.0f;
    }
    HostDeviceVector<float> preds{logits};
    auto info = MakeInfo({0.0f, 1.0f, 0.0f, 1.0f}, {});

    GradientContainer container;
    obj->GetGradientAndExactHessian(preds, info, 0, &container.gpair, &container.exact_hessian);

    auto n_free = static_cast<bst_target_t>(n_classes - 1);
    ASSERT_EQ(container.exact_hessian.n_free, n_free) << "K=" << n_classes;
    ASSERT_EQ(container.exact_hessian.RowSize(), common::PackedHessianSize(n_free))
        << "K=" << n_classes;
    // (K-1) * K / 2 entries per row.
    ASSERT_EQ(container.exact_hessian.RowSize(),
              static_cast<std::size_t>(n_classes - 1) * n_classes / 2)
        << "K=" << n_classes;
    ASSERT_EQ(container.exact_hessian.NumRows(), kRows) << "K=" << n_classes;
    // The gradient keeps all K columns; only the exact coordinates drop the reference.
    ASSERT_EQ(container.gpair.Shape(1), static_cast<std::size_t>(n_classes)) << "K=" << n_classes;
  }
}

/**
 * The stored float Hessian against a double precision reference. The tolerance covers the
 * float probability calculation the objective performs, which the transport deliberately
 * matches; it is not an assertion that float equals double.
 */
TEST(MulticlassExactObj, HigherPrecisionReference) {
  Context ctx;
  std::int32_t constexpr kNumClasses = 7;
  auto obj = MakeMulticlassObj(&ctx, kNumClasses);
  std::vector<float> logits{0.0f, 1.0f, -1.0f, 5.0f, -9.0f, 2.5f, -0.25f};
  std::vector<float> weights{1.75f};
  HostDeviceVector<float> preds{logits};
  auto info = MakeInfo({3.0f}, weights);

  GradientContainer container;
  obj->GetGradientAndExactHessian(preds, info, 0, &container.gpair, &container.exact_hessian);

  auto probability = ReferenceProbability(logits);
  auto n_free = static_cast<std::size_t>(kNumClasses - 1);
  auto row = container.exact_hessian.HostRow(0);
  auto hessian = common::PackedHessianAtRow(row, n_free, 0);

  for (std::size_t i = 0; i < n_free; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      auto expected = static_cast<double>(weights[0]) * probability[i] *
                      ((i == j ? 1.0 : 0.0) - probability[j]);
      EXPECT_NEAR(static_cast<double>(hessian.Get(i, j)), expected, 1e-6)
          << "H(" << i << ", " << j << ")";
      // Symmetry is structural, but assert it so a future layout change cannot break it.
      EXPECT_EQ(Bits(hessian.Get(i, j)), Bits(hessian.Get(j, i)));
    }
  }
}

/**
 * Only the objectives that implement the exact Hessian advertise it, and the capability is
 * reported through ObjInfo rather than through the device the objective happens to run on.
 */
TEST(MulticlassExactObj, Capability) {
  Context ctx;
  ASSERT_TRUE(ctx.IsCPU());
  for (auto name : {"multi:softprob", "multi:softmax"}) {
    auto obj = MakeMulticlassObj(&ctx, 3, name);
    EXPECT_TRUE(obj->Task().exact_hess) << name;
    // The capability flag must not be entangled with the other task metadata.
    EXPECT_FALSE(obj->Task().const_hess) << name;
    EXPECT_EQ(obj->Task().task, ObjInfo::kClassification) << name;
  }
  for (auto name : {"reg:squarederror", "binary:logistic", "reg:absoluteerror"}) {
    std::unique_ptr<ObjFunction> obj{ObjFunction::Create(name, &ctx)};
    obj->Configure({});
    EXPECT_FALSE(obj->Task().exact_hess) << name;

    // The default entry point is an explicit failure, never a silent no-op.
    HostDeviceVector<float> preds{std::vector<float>{0.1f, 0.2f}};
    auto info = MakeInfo({0.0f, 1.0f}, {});
    GradientContainer container;
    EXPECT_THROW(obj->GetGradientAndExactHessian(preds, info, 0, &container.gpair,
                                                 &container.exact_hessian),
                 dmlc::Error)
        << name;
    EXPECT_FALSE(container.HasExactHessian()) << name;
  }
}

/**
 * The stale-sidecar hazard, at objective level.
 *
 * GetGradient is handed the gradient matrix, not the container, so it cannot and must not
 * invalidate the sidecar itself. This test pins that fact down: after an exact round, a
 * scalar round leaves the old sidecar in place, which is exactly why the owner of the
 * container clears it when a new gradient computation starts.
 */
TEST(MulticlassExactObj, SidecarGoesStaleWithoutClear) {
  Context ctx;
  auto obj = MakeMulticlassObj(&ctx, 3);
  GradientContainer container;

  // Round N: exact path, 2 rows.
  auto round_n = MakeInfo({1.0f, 0.0f}, {});
  HostDeviceVector<float> preds_n{std::vector<float>{1.0f, 0.0f, 2.0f, 2.0f, 0.0f, 1.0f}};
  obj->GetGradientAndExactHessian(preds_n, round_n, 0, &container.gpair, &container.exact_hessian);
  ASSERT_TRUE(container.HasExactHessian());
  ASSERT_EQ(container.exact_hessian.NumRows(), 2);

  // Round N+1: scalar path with a DIFFERENT row count. The gradient is replaced...
  auto round_n1 = MakeInfo({2.0f, 1.0f, 0.0f}, {});
  HostDeviceVector<float> preds_n1{
      std::vector<float>{0.5f, 0.25f, 0.0f, 1.0f, 0.0f, 2.0f, 2.0f, 1.0f, 0.0f}};
  obj->GetGradient(preds_n1, round_n1, 1, &container.gpair);
  ASSERT_EQ(container.gpair.Shape(0), 3);

  // ...but the objective could not touch the sidecar, so it is still round N's, and its row
  // count no longer matches the gradient. Consuming it here would read the wrong rows.
  EXPECT_TRUE(container.HasExactHessian());
  EXPECT_EQ(container.exact_hessian.NumRows(), 2);
  EXPECT_NE(container.exact_hessian.NumRows(), container.gpair.Shape(0));

  // The container owner resolves it. This is what LearnerImpl::GetGradient does.
  container.ClearExactHessian();
  EXPECT_FALSE(container.HasExactHessian());
}

/** Ordinary training never populates the sidecar. */
TEST(MulticlassExactObj, SidecarUnusedInNormalMode) {
  Context ctx;
  auto obj = MakeMulticlassObj(&ctx, 3);
  auto info = MakeInfo({1.0f, 0.0f}, {});
  HostDeviceVector<float> preds{std::vector<float>{1.0f, 0.0f, 2.0f, 2.0f, 0.0f, 1.0f}};

  GradientContainer container;
  ASSERT_FALSE(container.HasExactHessian());

  obj->GetGradient(preds, info, 0, &container.gpair);
  ASSERT_FALSE(container.gpair.Empty());
  EXPECT_FALSE(container.HasExactHessian());
  EXPECT_TRUE(container.exact_hessian.Empty());
  EXPECT_EQ(container.exact_hessian.n_free, 0);
  EXPECT_EQ(container.exact_hessian.NumRows(), 0);
}
}  // namespace xgboost
