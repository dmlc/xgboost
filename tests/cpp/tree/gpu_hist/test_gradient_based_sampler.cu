#include <gtest/gtest.h>

#include "../../../../src/data/ellpack_page.cuh"
#include "../../../../src/tree/gpu_hist/gradient_based_sampler.cuh"
#include "../../helpers.h"

namespace xgboost {
namespace tree {

void VerifySampling(size_t page_size, float subsample, int sampling_method) {
  constexpr size_t kRows = 4096;
  constexpr size_t kCols = 1;
  size_t sample_rows = kRows * subsample;

  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix> dmat(
      CreateSparsePageDMatrixWithRC(kRows, kCols, page_size, true, tmpdir));
  auto gpair = GenerateRandomGradients(kRows);
  GradientPair sum_gpair{};
  for (const auto& gp : gpair.ConstHostVector()) {
    sum_gpair += gp;
  }
  gpair.SetDevice(0);

  BatchParam param{0, 256, 0, page_size};
  auto page = (*dmat->GetBatches<EllpackPage>(param).begin()).Impl();
  if (page_size != 0) {
    EXPECT_NE(page->matrix.n_rows, kRows);
  }

  GradientBasedSampler sampler(page, kRows, param, subsample, sampling_method);
  auto sample = sampler.Sample(gpair.DeviceSpan(), dmat.get());
  EXPECT_EQ(sample.sample_rows, sample_rows);
  EXPECT_EQ(sample.page->matrix.n_rows, sample_rows);
  EXPECT_EQ(sample.gpair.size(), sample_rows);

  GradientPair sum_sampled_gpair{};
  std::vector<GradientPair> sampled_gpair_h(sample.gpair.size());
  dh::CopyDeviceSpanToVector(&sampled_gpair_h, sample.gpair);
  for (const auto& gp : sampled_gpair_h) {
    sum_sampled_gpair += gp;
  }
  EXPECT_NEAR(sum_gpair.GetGrad(), sum_sampled_gpair.GetGrad(), 0.01f * kRows);
  EXPECT_NEAR(sum_gpair.GetHess(), sum_sampled_gpair.GetHess(), 0.01f * kRows);
}

TEST(GradientBasedSampler, NoSampling) {
  constexpr size_t kPageSize = 0;
  constexpr float kSubsample = 1.0f;
  constexpr int kSamplingMethod = TrainParam::kUniform;
  VerifySampling(kPageSize, kSubsample, kSamplingMethod);
}

// In external mode, when not sampling, we concatenate the pages together.
TEST(GradientBasedSampler, NoSampling_ExternalMemory) {
  constexpr size_t kRows = 2048;
  constexpr size_t kCols = 1;
  constexpr float kSubsample = 1.0f;
  constexpr size_t kPageSize = 1024;

  // Create a DMatrix with multiple batches.
  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix>
      dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true, tmpdir));
  auto gpair = GenerateRandomGradients(kRows);
  gpair.SetDevice(0);

  BatchParam param{0, 256, 0, kPageSize};
  auto page = (*dmat->GetBatches<EllpackPage>(param).begin()).Impl();
  EXPECT_NE(page->matrix.n_rows, kRows);

  GradientBasedSampler sampler(page, kRows, param, kSubsample, TrainParam::kUniform);
  auto sample = sampler.Sample(gpair.DeviceSpan(), dmat.get());
  auto sampled_page = sample.page;
  EXPECT_EQ(sample.sample_rows, kRows);
  EXPECT_EQ(sample.gpair.size(), gpair.Size());
  EXPECT_EQ(sample.gpair.data(), gpair.DevicePointer());
  EXPECT_EQ(sampled_page->matrix.n_rows, kRows);

  std::vector<common::CompressedByteT> buffer(sampled_page->gidx_buffer.size());
  dh::CopyDeviceSpanToVector(&buffer, sampled_page->gidx_buffer);
  common::CompressedIterator<common::CompressedByteT>
      ci(buffer.data(), sampled_page->matrix.info.NumSymbols());

  size_t offset = 0;
  for (auto& batch : dmat->GetBatches<EllpackPage>(param)) {
    auto page = batch.Impl();
    std::vector<common::CompressedByteT> page_buffer(page->gidx_buffer.size());
    dh::CopyDeviceSpanToVector(&page_buffer, page->gidx_buffer);
    common::CompressedIterator<common::CompressedByteT>
        page_ci(page_buffer.data(), page->matrix.info.NumSymbols());
    size_t num_elements = page->matrix.n_rows * page->matrix.info.row_stride;
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
  VerifySampling(kPageSize, kSubsample, kSamplingMethod);
}

TEST(GradientBasedSampler, UniformSampling_ExternalMemory) {
  constexpr size_t kPageSize = 1024;
  constexpr float kSubsample = 0.5;
  constexpr int kSamplingMethod = TrainParam::kUniform;
  VerifySampling(kPageSize, kSubsample, kSamplingMethod);
}

TEST(GradientBasedSampler, GradientBasedSampling) {
  constexpr size_t kPageSize = 0;
  constexpr float kSubsample = 0.5;
  constexpr int kSamplingMethod = TrainParam::kGradientBased;
  VerifySampling(kPageSize, kSubsample, kSamplingMethod);
}

TEST(GradientBasedSampler, GradientBasedSampling_ExternalMemory) {
  constexpr size_t kPageSize = 1024;
  constexpr float kSubsample = 0.5;
  constexpr int kSamplingMethod = TrainParam::kGradientBased;
  VerifySampling(kPageSize, kSubsample, kSamplingMethod);
}

};  // namespace tree
};  // namespace xgboost
