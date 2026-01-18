/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../../src/tree/gpu_hist/gradient_based_sampler.cuh"
#include "../../../../src/tree/param.h"  // TrainParam
#include "../../helpers.h"
#include "dummy_quantizer.cuh"

namespace xgboost::tree {
void VerifySampling(float subsample, int sampling_method, bool check_sum = true) {
  auto ctx = MakeCUDACtx(0);

  constexpr size_t kRows = 4096;
  bst_idx_t sample_rows = kRows * subsample;
  bst_target_t n_targets = 1;

  auto q = MakeDummyQuantizer();
  dh::caching_device_vector<GradientQuantiser> roundings{q};

  auto gpair = GenerateRandomGradients(kRows);
  gpair.SetDevice(ctx.Device());
  auto d_gpair = linalg::MakeTensorView(&ctx, gpair.ConstDeviceSpan(), kRows, n_targets);
  linalg::Matrix<GradientPairInt64> gpair_i64;
  CalcQuantizedGpairs(&ctx, d_gpair, dh::ToSpan(roundings), &gpair_i64);

  GradientPairPrecise sum_gpair{};
  for (const auto& gp : gpair_i64.HostView()) {
    sum_gpair += q.ToFloatingPoint(gp);
  }

  GradientBasedSampler sampler{kRows, subsample, sampling_method};
  sampler.Sample(&ctx, gpair_i64.View(ctx.Device()).Slice(linalg::All(), 0), q);

  GradientPairPrecise sum_sampled_gpair{};
  for (auto const& gp : gpair_i64.HostView()) {
    sum_sampled_gpair += q.ToFloatingPoint(gp);
  }
  if (check_sum) {
    ASSERT_NEAR(sum_gpair.GetGrad(), sum_sampled_gpair.GetGrad(), 0.03f * kRows);
    ASSERT_NEAR(sum_gpair.GetHess(), sum_sampled_gpair.GetHess(), 0.03f * kRows);
  } else {
    ASSERT_NEAR(sum_gpair.GetGrad() / kRows, sum_sampled_gpair.GetGrad() / sample_rows, 0.03f);
    ASSERT_NEAR(sum_gpair.GetHess() / kRows, sum_sampled_gpair.GetHess() / sample_rows, 0.03f);
  }
}

TEST(GradientBasedSampler, NoSampling) {
  constexpr float kSubsample = 1.0f;
  constexpr int kSamplingMethod = TrainParam::kUniform;
  VerifySampling(kSubsample, kSamplingMethod);
}

TEST(GradientBasedSampler, UniformSampling) {
  constexpr float kSubsample = 0.5;
  constexpr int kSamplingMethod = TrainParam::kUniform;
  constexpr bool kCheckSum = false;
  VerifySampling(kSubsample, kSamplingMethod, kCheckSum);
}

TEST(GradientBasedSampler, GradientBasedSampling) {
  constexpr float kSubsample = 0.8;
  constexpr int kSamplingMethod = TrainParam::kGradientBased;
  VerifySampling(kSubsample, kSamplingMethod);
}
}  // namespace xgboost::tree
