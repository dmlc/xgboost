/**
 * Copyright 2023-2026, XGBoost Contributors
 */
#include "../test_sampler.h"  // VerifyApplySamplingMask

#include <gtest/gtest.h>

#include <cmath>    // std::exp
#include <cstddef>  // std::size_t
#include <string>   // std::to_string
#include <vector>   // std::vector

#include "../../../../src/tree/hist/sampler.h"  // SampleGradient
#include "../../../../src/tree/param.h"         // TrainParam
#include "../../helpers.h"                      // GenerateRandomGradients
#include "xgboost/base.h"                       // GradientPair,bst_target_t
#include "xgboost/context.h"                    // Context
#include "xgboost/data.h"                       // MetaInfo
#include "xgboost/linalg.h"                     // Matrix,Constants

namespace xgboost::tree::cpu_impl {
void VerifySampling(float subsample, int sampling_method, bst_target_t n_targets = 1,
                    bool check_sum = true) {
  Context ctx;

  constexpr std::size_t kRows = 4096;
  // Generate random gradients
  auto gpair_container = GenerateRandomGradients(&ctx, kRows, n_targets);
  auto h_gpair = gpair_container.gpair.HostView();

  auto sum_gradients = [&]() {
    std::vector<GradientPairPrecise> sum(n_targets);
    for (std::size_t i = 0; i < kRows; ++i) {
      for (bst_target_t t = 0; t < n_targets; ++t) {
        sum[t] += GradientPairPrecise{h_gpair(i, t).GetGrad(), h_gpair(i, t).GetHess()};
      }
    }
    return sum;
  };

  auto sum_gpair = sum_gradients();

  TrainParam param;
  param.UpdateAllowUnknown(Args{
      {"subsample", std::to_string(subsample)},
      {"sampling_method", sampling_method == TrainParam::kUniform ? "uniform" : "gradient_based"}});
  SampleGradient(&ctx, param, h_gpair);

  auto sum_sampled_gpair = sum_gradients();
  CheckSampling(subsample, n_targets, check_sum, sum_sampled_gpair, sum_gpair, h_gpair);
}

TEST(CpuSampler, NoSampling) {
  constexpr float kSubsample = 1.0f;
  constexpr int kSamplingMethod = TrainParam::kUniform;
  VerifySampling(kSubsample, kSamplingMethod);
}

TEST(CpuSampler, UniformSampling) {
  constexpr float kSubsample = 0.5;
  constexpr int kSamplingMethod = TrainParam::kUniform;
  // Uniform sampling preserves the mean, not the sum (check_sum = false)
  constexpr bool kCheckSum = false;
  // Single target
  VerifySampling(kSubsample, kSamplingMethod, 1, kCheckSum);
  // Multi-target
  VerifySampling(kSubsample, kSamplingMethod, 3, kCheckSum);
}

TEST(CpuSampler, GradientBasedSampling) {
  constexpr float kSubsample = 0.8;
  constexpr int kSamplingMethod = TrainParam::kGradientBased;
  VerifySampling(kSubsample, kSamplingMethod, 1);
  VerifySampling(kSubsample, kSamplingMethod, 3);
}

TEST(CpuSampler, ApplySampling) {
  Context ctx;

  std::size_t n_samples = 1024;
  std::size_t n_split_targets = 2, n_value_targets = 4;
  constexpr float kSubsample = 0.5f;

  TrainParam param;
  param.UpdateAllowUnknown(Args{{"subsample", std::to_string(kSubsample)}});

  // Generate and sample the split gradient
  std::size_t split_shape[2] = {n_samples, n_split_targets};
  linalg::Matrix<GradientPair> split_gpair{split_shape, ctx.Device()};
  *split_gpair.Data() = GenerateRandomGradients(n_samples * n_split_targets, 0.0f, 1.0f);
  SampleGradient(&ctx, param, split_gpair.HostView());

  // Generate value gradient (more targets than split)
  std::size_t value_shape[2] = {n_samples, n_value_targets};
  linalg::Matrix<GradientPair> value_gpair{value_shape, ctx.Device()};
  *value_gpair.Data() = GenerateRandomGradients(n_samples * n_value_targets, 0.0f, 1.0f);

  cpu_impl::ApplySamplingMask(&ctx, split_gpair, &value_gpair);
  CheckSamplingMask(split_gpair.HostView(), value_gpair.HostView(), kSubsample);
}
}  // namespace xgboost::tree::cpu_impl
