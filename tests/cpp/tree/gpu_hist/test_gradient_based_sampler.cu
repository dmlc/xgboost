/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../../src/tree/gpu_hist/gradient_based_sampler.cuh"
#include "../../../../src/tree/param.h"  // TrainParam
#include "../../helpers.h"
#include "../test_sampler.h"  // VerifyApplySamplingMask
#include "dummy_quantizer.cuh"

namespace xgboost::tree {
void VerifySampling(float subsample, int sampling_method, bst_target_t n_targets = 1,
                    bool check_sum = true) {
  auto ctx = MakeCUDACtx(0);

  constexpr size_t kRows = 4096;
  bst_idx_t sample_rows = kRows * subsample;

  auto [gpair_i64, quantizer] = GenerateGradientsFixedPoint(&ctx, kRows, n_targets);

  // Copy quantizers to host for summing
  std::vector<GradientQuantiser> h_quantizers(n_targets, MakeDummyQuantizer());
  dh::safe_cuda(cudaMemcpy(h_quantizers.data(), quantizer.Quantizers().data(),
                           n_targets * sizeof(GradientQuantiser), cudaMemcpyDeviceToHost));

  auto h_gpair = gpair_i64.HostView();
  auto sum_gradients = [&]() {
    std::vector<GradientPairPrecise> sum(n_targets);
    for (std::size_t i = 0; i < kRows; ++i) {
      for (bst_target_t t = 0; t < n_targets; ++t) {
        sum[t] += h_quantizers[t].ToFloatingPoint(h_gpair(i, t));
      }
    }
    return sum;
  };

  auto sum_gpair = sum_gradients();

  GradientBasedSampler sampler{kRows, subsample, sampling_method};
  sampler.Sample(&ctx, gpair_i64.View(ctx.Device()), quantizer.Quantizers());

  // Refresh host view after device modification
  h_gpair = gpair_i64.HostView();
  auto sum_sampled_gpair = sum_gradients();

  // Verify gradient sums per target
  for (bst_target_t t = 0; t < n_targets; ++t) {
    if (check_sum) {
      // Gradient-based sampling preserves the sum approximately
      ASSERT_NEAR(sum_gpair[t].GetGrad(), sum_sampled_gpair[t].GetGrad(), 0.03f * kRows);
      ASSERT_NEAR(sum_gpair[t].GetHess(), sum_sampled_gpair[t].GetHess(), 0.03f * kRows);
    } else {
      // Uniform sampling preserves the mean approximately
      auto mean_grad = sum_gpair[t].GetGrad() / kRows;
      auto mean_hess = sum_gpair[t].GetHess() / kRows;
      auto sampled_mean_grad = sum_sampled_gpair[t].GetGrad() / sample_rows;
      auto sampled_mean_hess = sum_sampled_gpair[t].GetHess() / sample_rows;
      ASSERT_NEAR(mean_grad, sampled_mean_grad, mean_grad * 0.1);
      ASSERT_NEAR(mean_hess, sampled_mean_hess, mean_hess * 0.1);
    }
  }

  // For multi-target, verify that rows are either fully sampled or fully zeroed
  h_gpair = gpair_i64.HostView();
  std::size_t sampled_count = 0;
  for (std::size_t i = 0; i < kRows; ++i) {
    bool first_is_zero =
        (h_gpair(i, 0).GetQuantisedGrad() == 0 && h_gpair(i, 0).GetQuantisedHess() == 0);
    for (bst_target_t t = 1; t < n_targets; ++t) {
      bool is_zero =
          (h_gpair(i, t).GetQuantisedGrad() == 0 && h_gpair(i, t).GetQuantisedHess() == 0);
      ASSERT_EQ(first_is_zero, is_zero);
    }
    if (!first_is_zero) {
      ++sampled_count;
    }
  }

  // Verify approximately the right fraction of rows are sampled
  if (subsample < 1.0f) {
    double sampled_fraction = static_cast<double>(sampled_count) / kRows;
    ASSERT_NEAR(sampled_fraction, subsample, 0.05);
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
  // Uniform sampling preserves the mean, not the sum (check_sum = false)
  constexpr bool kCheckSum = false;
  // Single target
  VerifySampling(kSubsample, kSamplingMethod, 1, kCheckSum);
  // Multi-target
  VerifySampling(kSubsample, kSamplingMethod, 3, kCheckSum);
}

TEST(GradientBasedSampler, GradientBasedSampling) {
  constexpr float kSubsample = 0.8;
  constexpr int kSamplingMethod = TrainParam::kGradientBased;
  VerifySampling(kSubsample, kSamplingMethod, 1);
  VerifySampling(kSubsample, kSamplingMethod, 3);
}

TEST(GradientBasedSampler, ApplySamplingMask) {
  auto ctx = MakeCUDACtx(0);

  bst_idx_t n_samples = 1024;
  bst_target_t n_split_targets = 2;
  bst_target_t n_value_targets = 4;  // More targets than split gradient
  constexpr float kSubsample = 0.5f;

  // Generate and sample the split gradient
  auto [split_gpair, quantizer] = GenerateGradientsFixedPoint(&ctx, n_samples, n_split_targets);
  GradientBasedSampler sampler{n_samples, kSubsample, TrainParam::kUniform};
  sampler.Sample(&ctx, split_gpair.View(ctx.Device()), quantizer.Quantizers());

  // Generate value gradient (more targets than split)
  auto value_gpair = GenerateRandomGradients(&ctx, n_samples, n_value_targets);
  // Apply the sampling mask
  cuda_impl::ApplySamplingMask(&ctx, split_gpair, &value_gpair.gpair);

  // Verify using the shared test helper
  VerifyApplySamplingMask(split_gpair.HostView(), value_gpair.gpair.HostView(), kSubsample);
}
}  // namespace xgboost::tree
