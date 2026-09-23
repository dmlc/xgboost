/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/context.h>
#include <xgboost/gradient.h>

#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "../../../../src/common/exact_multinomial/packed_stats.h"
#include "../../../../src/tree/hist/sampler.h"
#include "../../../../src/tree/param.h"

namespace xgboost::tree {
namespace {
TrainParam MakeParam(double subsample, char const* method = "uniform") {
  TrainParam param;
  param.UpdateAllowUnknown(
      Args{{"subsample", std::to_string(subsample)}, {"sampling_method", method}});
  return param;
}

/** @brief True when every entry of the row is exactly zero. */
bool RowIsZero(linalg::MatrixView<GradientPair const> gpair, std::size_t row) {
  for (std::size_t t = 0; t < gpair.Shape(1); ++t) {
    if (gpair(row, t).GetGrad() != 0.0f || gpair(row, t).GetHess() != 0.0f) {
      return false;
    }
  }
  return true;
}

bool RowIsZero(common::Span<float const> row) {
  for (auto v : row) {
    if (v != 0.0f) {
      return false;
    }
  }
  return true;
}
}  // anonymous namespace

/**
 * The gradient and the exact Hessian must survive sampling together. If they disagreed, a
 * histogram bin would mix one row's gradient with another row's curvature, which no later
 * test would catch because both would still look individually well formed.
 */
TEST(ExactSampling, GradientAndHessianSelectTheSameRows) {
  for (double subsample : {0.3, 0.5, 0.8}) {
    Context ctx;
    std::size_t constexpr kRows = 512;
    bst_target_t constexpr kNumClasses = 4;
    bst_target_t constexpr kNumFree = kNumClasses - 1;

    // Every row starts non-zero so that "zero" unambiguously means "dropped".
    linalg::Matrix<GradientPair> gpair;
    gpair.Reshape(kRows, kNumClasses);
    auto h_gpair = gpair.HostView();
    for (std::size_t r = 0; r < kRows; ++r) {
      for (bst_target_t t = 0; t < kNumClasses; ++t) {
        h_gpair(r, t) = GradientPair{static_cast<float>(r + t + 1), static_cast<float>(t + 1)};
      }
    }

    ExactHessian hessian;
    hessian.Reshape(kRows, kNumFree);
    auto values = hessian.HostValues();
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<float>(i % 17) + 1.0f;
    }

    auto param = MakeParam(subsample);
    cpu_impl::Sampler sampler{param};
    sampler.Sample(&ctx, h_gpair);
    ASSERT_TRUE(sampler.IsSampling()) << "subsample=" << subsample;
    sampler.ApplySampling(&ctx, &hessian);

    std::size_t dropped = 0;
    for (std::size_t r = 0; r < kRows; ++r) {
      auto gradient_dropped = RowIsZero(
          linalg::MatrixView<GradientPair const>{h_gpair}, r);
      auto hessian_dropped = RowIsZero(hessian.HostRow(r));
      ASSERT_EQ(gradient_dropped, hessian_dropped)
          << "row " << r << " disagrees at subsample=" << subsample;
      dropped += gradient_dropped ? 1 : 0;
    }

    // The test is only meaningful if sampling actually removed a substantial share of rows.
    EXPECT_GT(dropped, kRows / 10) << "subsample=" << subsample;
    EXPECT_LT(dropped, kRows) << "subsample=" << subsample;
    auto kept_fraction = 1.0 - static_cast<double>(dropped) / kRows;
    EXPECT_NEAR(kept_fraction, subsample, 0.08) << "subsample=" << subsample;
  }
}

/** With no subsampling, neither the gradient nor the Hessian is touched. */
TEST(ExactSampling, NoSubsampleLeavesBothIntact) {
  Context ctx;
  std::size_t constexpr kRows = 64;
  bst_target_t constexpr kNumFree = 2;

  linalg::Matrix<GradientPair> gpair;
  gpair.Reshape(kRows, 3);
  auto h_gpair = gpair.HostView();
  for (std::size_t r = 0; r < kRows; ++r) {
    for (bst_target_t t = 0; t < 3; ++t) {
      h_gpair(r, t) = GradientPair{1.0f, 1.0f};
    }
  }
  ExactHessian hessian;
  hessian.Reshape(kRows, kNumFree);
  auto values = hessian.HostValues();
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = 2.0f;
  }

  auto param = MakeParam(1.0);
  cpu_impl::Sampler sampler{param};
  sampler.Sample(&ctx, h_gpair);
  EXPECT_FALSE(sampler.IsSampling());
  sampler.ApplySampling(&ctx, &hessian);

  for (std::size_t r = 0; r < kRows; ++r) {
    EXPECT_FALSE(RowIsZero(hessian.HostRow(r))) << "row " << r;
  }
}

/** An empty sidecar is a no-op, not a crash. */
TEST(ExactSampling, EmptySidecarIsSafe) {
  Context ctx;
  linalg::Matrix<GradientPair> gpair;
  gpair.Reshape(32, 3);
  auto h_gpair = gpair.HostView();
  for (std::size_t r = 0; r < 32; ++r) {
    for (bst_target_t t = 0; t < 3; ++t) {
      h_gpair(r, t) = GradientPair{1.0f, 1.0f};
    }
  }
  auto param = MakeParam(0.5);
  cpu_impl::Sampler sampler{param};
  sampler.Sample(&ctx, h_gpair);

  ExactHessian empty;
  EXPECT_NO_THROW(sampler.ApplySampling(&ctx, &empty));
  EXPECT_TRUE(empty.Empty());
}

/** Gradient-based sampling is rejected rather than silently mismatching the Hessian. */
TEST(ExactSampling, RejectsGradientBasedSampling) {
  Context ctx;
  std::size_t constexpr kRows = 128;
  linalg::Matrix<GradientPair> gpair;
  gpair.Reshape(kRows, 3);
  auto h_gpair = gpair.HostView();
  for (std::size_t r = 0; r < kRows; ++r) {
    for (bst_target_t t = 0; t < 3; ++t) {
      h_gpair(r, t) = GradientPair{static_cast<float>(r + 1), 1.0f};
    }
  }
  ExactHessian hessian;
  hessian.Reshape(kRows, 2);

  auto param = MakeParam(0.5, "gradient_based");
  cpu_impl::Sampler sampler{param};
  sampler.Sample(&ctx, h_gpair);
  ASSERT_TRUE(sampler.IsSampling());
  EXPECT_THROW(sampler.ApplySampling(&ctx, &hessian), dmlc::Error);
}
}  // namespace xgboost::tree
