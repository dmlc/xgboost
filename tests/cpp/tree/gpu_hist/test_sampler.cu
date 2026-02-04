/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <algorithm>  // for sort
#include <limits>     // for numeric_limits
#include <numeric>    // for partial_sum
#include <vector>     // for vector

#include "../../../../src/tree/gpu_hist/sampler.cuh"
#include "../../../../src/tree/hist/sampler.h"  // for cpu_impl::CalculateThreshold
#include "../../../../src/tree/param.h"         // TrainParam
#include "../../helpers.h"
#include "../test_sampler.h"  // VerifyApplySamplingMask
#include "dummy_quantizer.cuh"

namespace xgboost::tree::cuda_impl {
void CalcFloatGrad(linalg::MatrixView<GradientPairInt64> in_gpair,
                   common::Span<GradientQuantiser const> roundings,
                   linalg::Matrix<GradientPair>* p_out_gpair) {
  auto& out_gpair = *p_out_gpair;
  out_gpair.Reshape(in_gpair.Shape());
  auto h_out_gpair = out_gpair.HostView();
  for (std::size_t i = 0; i < in_gpair.Shape(0); ++i) {
    for (std::size_t j = 0; j < in_gpair.Shape(1); ++j) {
      auto g64 = roundings[j].ToFloatingPoint(in_gpair(i, j));
      h_out_gpair(i, j) = GradientPair(g64.GetGrad(), g64.GetHess());
    }
  }
}

void VerifySampling(float subsample, int sampling_method, bst_target_t n_targets = 1,
                    bool check_sum = true) {
  auto ctx = MakeCUDACtx(0);

  constexpr size_t kRows = 4096;
  auto [gpair_i64, quantizer] = GenerateGradientsFixedPoint(&ctx, kRows, n_targets);

  // Copy quantizers to host for summing
  std::vector<GradientQuantiser> h_quantizers(n_targets, MakeDummyQuantizer());
  dh::safe_cuda(cudaMemcpy(h_quantizers.data(), quantizer.Quantizers().data(),
                           n_targets * sizeof(GradientQuantiser), cudaMemcpyDeviceToHost));

  auto h_gpair_i64 = gpair_i64.HostView();
  auto sum_gradients = [&]() {
    std::vector<GradientPairPrecise> sum(n_targets);
    for (std::size_t i = 0; i < kRows; ++i) {
      for (bst_target_t t = 0; t < n_targets; ++t) {
        sum[t] += h_quantizers[t].ToFloatingPoint(h_gpair_i64(i, t));
      }
    }
    return sum;
  };

  auto sum_gpair = sum_gradients();

  Sampler sampler{kRows, subsample, sampling_method};
  sampler.Sample(&ctx, gpair_i64.View(ctx.Device()), quantizer.Quantizers());

  // Refresh host view after device modification
  h_gpair_i64 = gpair_i64.HostView();
  linalg::Matrix<GradientPair> gpair;
  CalcFloatGrad(h_gpair_i64, common::Span{h_quantizers}, &gpair);
  auto h_gpair = gpair.HostView();
  auto sum_sampled_gpair = sum_gradients();
  CheckSampling(subsample, n_targets, check_sum, sum_sampled_gpair, sum_gpair, h_gpair);
}

TEST(GpuSampler, NoSampling) {
  constexpr float kSubsample = 1.0f;
  constexpr int kSamplingMethod = TrainParam::kUniform;
  VerifySampling(kSubsample, kSamplingMethod);
}

TEST(GpuSampler, UniformSampling) {
  constexpr float kSubsample = 0.5;
  constexpr int kSamplingMethod = TrainParam::kUniform;
  // Uniform sampling preserves the mean, not the sum (check_sum = false)
  constexpr bool kCheckSum = false;
  VerifySampling(kSubsample, kSamplingMethod, 1, kCheckSum);
  VerifySampling(kSubsample, kSamplingMethod, 3, kCheckSum);
}

TEST(GpuSampler, GradientBasedSampling) {
  constexpr float kSubsample = 0.8;
  constexpr int kSamplingMethod = TrainParam::kGradientBased;
  VerifySampling(kSubsample, kSamplingMethod, 1);
  VerifySampling(kSubsample, kSamplingMethod, 3);
}

TEST(GpuSampler, ApplySampling) {
  auto ctx = MakeCUDACtx(0);

  bst_idx_t n_samples = 1024;
  bst_target_t n_split_targets = 2, n_value_targets = 4;
  constexpr float kSubsample = 0.5f;
  constexpr int kSamplingMethod = TrainParam::kGradientBased;

  // Generate and sample the split gradient
  auto [split_gpair, quantizer] = GenerateGradientsFixedPoint(&ctx, n_samples, n_split_targets);
  Sampler sampler{n_samples, kSubsample, kSamplingMethod};
  sampler.Sample(&ctx, split_gpair.View(ctx.Device()), quantizer.Quantizers());
  auto d_roundings = quantizer.Quantizers();
  std::vector<GradientQuantiser> h_roundings(d_roundings.size(), MakeDummyQuantizer());
  thrust::copy(dh::tcbegin(d_roundings), dh::tcend(d_roundings), h_roundings.begin());

  // Generate value gradient (more targets than split)
  auto value_gpair = GenerateRandomGradients(&ctx, n_samples, n_value_targets);
  auto h_value_before = value_gpair.gpair.HostView();
  linalg::Matrix<GradientPair> sampled;
  CalcFloatGrad(split_gpair.HostView(), dh::ToSpan(h_roundings), &sampled);

  sampler.ApplySampling(&ctx, split_gpair, &value_gpair.gpair);
  CheckSamplingMask(sampled.HostView(), value_gpair.gpair.HostView(), kSubsample);

  auto h_value_after = value_gpair.gpair.HostView();
  std::vector<float> thresholds;
  auto reg_abs_grad = ::xgboost::tree::cpu_impl::CalcRegAbsGrad(&ctx, h_value_before, &thresholds);

  dh::device_vector<float> d_sorted(thresholds);
  dh::device_vector<float> d_csum(n_samples);
  auto threshold_index =
      cuda_impl::CalculateThresholdIndex(&ctx, dh::ToSpan(d_sorted), dh::ToSpan(d_csum), n_samples,
                                         static_cast<bst_idx_t>(n_samples * kSubsample));
  float threshold = d_sorted[threshold_index];

  auto h_sampled_split = sampled.HostView();
  CheckValueReweight(h_sampled_split, h_value_before, h_value_after, reg_abs_grad, threshold);
}
}  // namespace xgboost::tree::cuda_impl

namespace xgboost::tree {
// Test consistency between CPU and GPU threshold calculations
TEST(CalculateThreshold, CpuGpuConsistency) {
  auto ctx = MakeCUDACtx(0);

  // Test with various gradient distributions
  std::vector<std::vector<float>> test_cases = {
      {0.5f, 5.0f, 1.0f, 2.0f, 2.0f},                                // Basic
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},  // All equal
      {0.1f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f},                         // Varied
  };

  std::vector<float> subsample_rates = {0.3f, 0.5f, 0.8f};

  for (auto const& rag : test_cases) {
    for (float subsample : subsample_rates) {
      bst_idx_t n = rag.size();
      bst_idx_t sample_rows = static_cast<bst_idx_t>(n * subsample);

      // CPU calculation
      std::vector<float> cpu_sorted = rag;
      std::sort(cpu_sorted.begin(), cpu_sorted.end());
      cpu_sorted.push_back(std::numeric_limits<float>::max());
      std::vector<float> cpu_csum(n);
      std::partial_sum(cpu_sorted.begin(), cpu_sorted.end() - 1, cpu_csum.begin());
      float cpu_threshold = cpu_impl::CalculateThreshold(common::Span{cpu_sorted},
                                                         common::Span{cpu_csum}, n, sample_rows);

      // GPU calculation
      std::vector<float> gpu_sorted = rag;
      std::sort(gpu_sorted.begin(), gpu_sorted.end());
      gpu_sorted.push_back(std::numeric_limits<float>::max());
      dh::device_vector<float> d_sorted(gpu_sorted);
      dh::device_vector<float> d_csum(n);
      std::size_t threshold_index = cuda_impl::CalculateThresholdIndex(
          &ctx, dh::ToSpan(d_sorted), dh::ToSpan(d_csum), n, sample_rows);
      float gpu_threshold = d_sorted[threshold_index];

      // Both should produce similar expected sample counts
      auto calc_expected = [&](float threshold) {
        float expected = 0.0f;
        for (bst_idx_t i = 0; i < n; ++i) {
          expected += std::min(SamplingProbability(threshold, cpu_sorted[i]), 1.0f);
        }
        return expected;
      };

      float cpu_expected = calc_expected(cpu_threshold);
      float gpu_expected = calc_expected(gpu_threshold);
      // Both should be close to target sample_rows
      EXPECT_NEAR(cpu_expected, sample_rows, 0.1f);
      EXPECT_NEAR(gpu_expected, sample_rows, 0.1f);
      EXPECT_NEAR(cpu_expected, gpu_expected, 0.1f);
    }
  }
}
}  // namespace xgboost::tree
