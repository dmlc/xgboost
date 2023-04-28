/**
 * Copyright 2020-2023, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../../src/data/ellpack_page.cuh"
#include "../../../../src/tree/gpu_hist/gradient_based_sampler.cuh"
#include "../../../../src/tree/param.h"
#include "../../../../src/tree/param.h"  // TrainParam
#include "../../filesystem.h"            // dmlc::TemporaryDirectory
#include "../../helpers.h"

namespace xgboost {
namespace tree {

void VerifySampling(size_t page_size,
                    float subsample,
                    int sampling_method,
                    bool fixed_size_sampling = true,
                    bool check_sum = true) {
  constexpr size_t kRows = 4096;
  constexpr size_t kCols = 1;
  size_t sample_rows = kRows * subsample;

  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrix(
      kRows, kCols, kRows / (page_size == 0 ? kRows : page_size), tmpdir.path + "/cache"));
  auto gpair = GenerateRandomGradients(kRows);
  GradientPair sum_gpair{};
  for (const auto& gp : gpair.ConstHostVector()) {
    sum_gpair += gp;
  }
  gpair.SetDevice(0);

  Context ctx{MakeCUDACtx(0)};
  auto param = BatchParam{256, tree::TrainParam::DftSparseThreshold()};
  auto page = (*dmat->GetBatches<EllpackPage>(&ctx, param).begin()).Impl();
  if (page_size != 0) {
    EXPECT_NE(page->n_rows, kRows);
  }

  GradientBasedSampler sampler(&ctx, page, kRows, param, subsample, sampling_method);
  auto sample = sampler.Sample(&ctx, gpair.DeviceSpan(), dmat.get());

  if (fixed_size_sampling) {
    EXPECT_EQ(sample.sample_rows, kRows);
    EXPECT_EQ(sample.page->n_rows, kRows);
    EXPECT_EQ(sample.gpair.size(), kRows);
  } else {
    EXPECT_NEAR(sample.sample_rows, sample_rows, kRows * 0.03);
    EXPECT_NEAR(sample.page->n_rows, sample_rows, kRows * 0.03f);
    EXPECT_NEAR(sample.gpair.size(), sample_rows, kRows * 0.03f);
  }

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

// In external mode, when not sampling, we concatenate the pages together.
TEST(GradientBasedSampler, NoSamplingExternalMemory) {
  constexpr size_t kRows = 2048;
  constexpr size_t kCols = 1;
  constexpr float kSubsample = 1.0f;
  constexpr size_t kPageSize = 1024;

  // Create a DMatrix with multiple batches.
  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix> dmat(
      CreateSparsePageDMatrix(kRows, kCols, kRows / kPageSize, tmpdir.path + "/cache"));
  auto gpair = GenerateRandomGradients(kRows);
  gpair.SetDevice(0);

  Context ctx{MakeCUDACtx(0)};
  auto param = BatchParam{256, tree::TrainParam::DftSparseThreshold()};
  auto page = (*dmat->GetBatches<EllpackPage>(&ctx, param).begin()).Impl();
  EXPECT_NE(page->n_rows, kRows);

  GradientBasedSampler sampler(&ctx, page, kRows, param, kSubsample, TrainParam::kUniform);
  auto sample = sampler.Sample(&ctx, gpair.DeviceSpan(), dmat.get());
  auto sampled_page = sample.page;
  EXPECT_EQ(sample.sample_rows, kRows);
  EXPECT_EQ(sample.gpair.size(), gpair.Size());
  EXPECT_EQ(sample.gpair.data(), gpair.DevicePointer());
  EXPECT_EQ(sampled_page->n_rows, kRows);

  std::vector<common::CompressedByteT> buffer(sampled_page->gidx_buffer.HostVector());
  common::CompressedIterator<common::CompressedByteT>
      ci(buffer.data(), sampled_page->NumSymbols());

  size_t offset = 0;
  for (auto& batch : dmat->GetBatches<EllpackPage>(&ctx, param)) {
    auto page = batch.Impl();
    std::vector<common::CompressedByteT> page_buffer(page->gidx_buffer.HostVector());
    common::CompressedIterator<common::CompressedByteT>
        page_ci(page_buffer.data(), page->NumSymbols());
    size_t num_elements = page->n_rows * page->row_stride;
    for (size_t i = 0; i < num_elements; i++) {
      EXPECT_EQ(ci[i + offset], page_ci[i]);
    }
    offset += num_elements;
  }
}

TEST(GradientBasedSampler, UniformSampling) {
  constexpr size_t kPageSize = 0;
  constexpr float kSubsample = 0.5;
  constexpr int kSamplingMethod = TrainParam::kUniform;
  constexpr bool kFixedSizeSampling = true;
  constexpr bool kCheckSum = false;
  VerifySampling(kPageSize, kSubsample, kSamplingMethod, kFixedSizeSampling, kCheckSum);
}

TEST(GradientBasedSampler, UniformSamplingExternalMemory) {
  constexpr size_t kPageSize = 1024;
  constexpr float kSubsample = 0.5;
  constexpr int kSamplingMethod = TrainParam::kUniform;
  constexpr bool kFixedSizeSampling = false;
  constexpr bool kCheckSum = false;
  VerifySampling(kPageSize, kSubsample, kSamplingMethod, kFixedSizeSampling, kCheckSum);
}

TEST(GradientBasedSampler, GradientBasedSampling) {
  constexpr size_t kPageSize = 0;
  constexpr float kSubsample = 0.8;
  constexpr int kSamplingMethod = TrainParam::kGradientBased;
  VerifySampling(kPageSize, kSubsample, kSamplingMethod);
}

TEST(GradientBasedSampler, GradientBasedSamplingExternalMemory) {
  constexpr size_t kPageSize = 1024;
  constexpr float kSubsample = 0.8;
  constexpr int kSamplingMethod = TrainParam::kGradientBased;
  constexpr bool kFixedSizeSampling = false;
  VerifySampling(kPageSize, kSubsample, kSamplingMethod, kFixedSizeSampling);
}

};  // namespace tree
};  // namespace xgboost
