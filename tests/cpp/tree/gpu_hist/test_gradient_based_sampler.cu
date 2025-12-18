/**
 * Copyright 2020-2025, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../../src/data/ellpack_page.cuh"
#include "../../../../src/tree/gpu_hist/gradient_based_sampler.cuh"
#include "../../../../src/tree/param.h"
#include "../../../../src/tree/param.h"  // TrainParam
#include "../../helpers.h"

namespace xgboost::tree {
void VerifySampling(size_t page_size, float subsample, int sampling_method, bool check_sum = true) {
  constexpr size_t kRows = 4096;
  constexpr size_t kCols = 1;
  bst_idx_t sample_rows = kRows * subsample;

  auto dmat = RandomDataGenerator{kRows, kCols, 0.0f}.GenerateDMatrix(true);
  auto gpair = GenerateRandomGradients(kRows);
  GradientPair sum_gpair{};
  for (const auto& gp : gpair.ConstHostVector()) {
    sum_gpair += gp;
  }
  auto ctx = MakeCUDACtx(0);
  gpair.SetDevice(ctx.Device());

  auto param = BatchParam{256, tree::TrainParam::DftSparseThreshold()};
  auto page = (*dmat->GetBatches<EllpackPage>(&ctx, param).begin()).Impl();
  if (page_size != 0) {
    EXPECT_NE(page->n_rows, kRows);
  }

  GradientBasedSampler sampler(&ctx, kRows, param, subsample, sampling_method);
  auto sample = sampler.Sample(&ctx, gpair.DeviceSpan(), dmat.get());

  EXPECT_EQ(sample.p_fmat->Info().num_row_, kRows);
  EXPECT_EQ(sample.gpair.size(), kRows);

  GradientPair sum_sampled_gpair{};
  std::vector<GradientPair> sampled_gpair_h(sample.gpair.size());
  dh::CopyDeviceSpanToVector(&sampled_gpair_h, sample.gpair);
  for (const auto& gp : sampled_gpair_h) {
    sum_sampled_gpair += gp;
  }
  if (check_sum) {
    EXPECT_NEAR(sum_gpair.GetGrad(), sum_sampled_gpair.GetGrad(), 0.03f * kRows);
    EXPECT_NEAR(sum_gpair.GetHess(), sum_sampled_gpair.GetHess(), 0.03f * kRows);
  } else {
    EXPECT_NEAR(sum_gpair.GetGrad() / kRows, sum_sampled_gpair.GetGrad() / sample_rows, 0.03f);
    EXPECT_NEAR(sum_gpair.GetHess() / kRows, sum_sampled_gpair.GetHess() / sample_rows, 0.03f);
  }
}

TEST(GradientBasedSampler, NoSampling) {
  constexpr size_t kPageSize = 0;
  constexpr float kSubsample = 1.0f;
  constexpr int kSamplingMethod = TrainParam::kUniform;
  VerifySampling(kPageSize, kSubsample, kSamplingMethod);
}

TEST(GradientBasedSampler, UniformSampling) {
  constexpr size_t kPageSize = 0;
  constexpr float kSubsample = 0.5;
  constexpr int kSamplingMethod = TrainParam::kUniform;
  constexpr bool kCheckSum = false;
  VerifySampling(kPageSize, kSubsample, kSamplingMethod, kCheckSum);
}

TEST(GradientBasedSampler, GradientBasedSampling) {
  constexpr size_t kPageSize = 0;
  constexpr float kSubsample = 0.8;
  constexpr int kSamplingMethod = TrainParam::kGradientBased;
  VerifySampling(kPageSize, kSubsample, kSamplingMethod);
}
}  // namespace xgboost::tree
