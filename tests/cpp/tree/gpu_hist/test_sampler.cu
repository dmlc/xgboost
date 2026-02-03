/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../../src/tree/gpu_hist/sampler.cuh"
#include "../../../../src/tree/param.h"  // TrainParam
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

  GradientBasedSampler sampler{kRows, subsample, sampling_method};
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

  // Generate and sample the split gradient
  auto [split_gpair, quantizer] = GenerateGradientsFixedPoint(&ctx, n_samples, n_split_targets);
  GradientBasedSampler sampler{n_samples, kSubsample, TrainParam::kUniform};
  sampler.Sample(&ctx, split_gpair.View(ctx.Device()), quantizer.Quantizers());
  auto d_roundings = quantizer.Quantizers();
  std::vector<GradientQuantiser> h_roundings(d_roundings.size(), MakeDummyQuantizer());
  thrust::copy(dh::tcbegin(d_roundings), dh::tcend(d_roundings), h_roundings.begin());

  // Generate value gradient (more targets than split)
  auto value_gpair = GenerateRandomGradients(&ctx, n_samples, n_value_targets);
  linalg::Matrix<GradientPair> sampled;
  CalcFloatGrad(split_gpair.HostView(), dh::ToSpan(h_roundings), &sampled);

  cuda_impl::ApplySampling(&ctx, split_gpair, &value_gpair.gpair);
  CheckSamplingMask(sampled.HostView(), value_gpair.gpair.HostView(), kSubsample);
}
}  // namespace xgboost::tree::cuda_impl
