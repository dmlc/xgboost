#include <gtest/gtest.h>

#include "../../../../src/data/ellpack_page.cuh"
#include "../../../../src/tree/gpu_hist/gradient_based_sampler.cuh"
#include "../../helpers.h"

namespace xgboost {
namespace tree {

TEST(GradientBasedSampler, NoSampling) {
  constexpr size_t kRows = 1024;
  constexpr size_t kCols = 4;
  constexpr float kSubsample = 1.0;
  constexpr size_t kPageSize = 0;

  // Create a DMatrix with a single batche.
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true));
  auto gpair = GenerateRandomGradients(kRows);
  gpair.SetDevice(0);

  BatchParam param{0, 256, 0, kPageSize};
  auto page = (*dmat->GetBatches<EllpackPage>(param).begin()).Impl();

  GradientBasedSampler sampler(page, kRows, param, kSubsample, TrainParam::kUniform);
  auto sample = sampler.Sample(gpair.DeviceSpan(), dmat.get());
  EXPECT_EQ(sample.sample_rows, kRows);
  EXPECT_EQ(sample.page, page);
  EXPECT_EQ(sample.gpair.size(), gpair.Size());
  EXPECT_EQ(sample.gpair.data(), gpair.DevicePointer());
}

TEST(GradientBasedSampler, NoSampling_ExternalMemory) {
  constexpr size_t kRows = 1024;
  constexpr size_t kCols = 4;
  constexpr float kSubsample = 1.0;
  constexpr size_t kPageSize = 1024;

  // Create a DMatrix with multiple batches.
  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix>
      dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true, tmpdir));
  auto gpair = GenerateRandomGradients(kRows);
  gpair.SetDevice(0);

  BatchParam param{0, 256, 0, kPageSize};
  auto page = (*dmat->GetBatches<EllpackPage>(param).begin()).Impl();

  GradientBasedSampler sampler(page, kRows, param, kSubsample, TrainParam::kUniform);
  auto sample = sampler.Sample(gpair.DeviceSpan(), dmat.get());
  auto sampled_page = sample.page;
  auto sampled_gpair = sample.gpair;
  EXPECT_EQ(sample.sample_rows, kRows);
  EXPECT_EQ(sampled_gpair.size(), kRows);
  EXPECT_EQ(sampled_page->matrix.n_rows, kRows);

  auto gpair_h = gpair.ConstHostVector();
  std::vector<GradientPair> sampled_gpair_h(sampled_gpair.size());
  dh::CopyDeviceSpanToVector(&sampled_gpair_h, sampled_gpair);
  EXPECT_EQ(gpair_h, sampled_gpair_h);

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
  constexpr size_t kRows = 2048;
  constexpr size_t kCols = 16;
  constexpr float kSubsample = 0.5;
  constexpr size_t kSampleRows = kRows * kSubsample;
  constexpr size_t kPageSize = 0;

  // Create a DMatrix with a single batche.
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true));
  auto gpair = GenerateRandomGradients(kRows);
  float sum_gradients = 0;
  for (auto gp : gpair.ConstHostVector()) {
    sum_gradients += gp.GetGrad();
  }
  gpair.SetDevice(0);

  BatchParam param{0, 256, 0, kPageSize};
  auto page = (*dmat->GetBatches<EllpackPage>(param).begin()).Impl();

  GradientBasedSampler sampler(page, kRows, param, kSubsample, TrainParam::kUniform);
  auto sample = sampler.Sample(gpair.DeviceSpan(), dmat.get());
  auto sampled_gpair = sample.gpair;
  EXPECT_EQ(sample.sample_rows, kRows);
  EXPECT_EQ(sample.page, page);
  EXPECT_EQ(sample.gpair.size(), kRows);

  float sum_sampled_gradients = 0;
  std::vector<GradientPair> sampled_gpair_h(sampled_gpair.size());
  dh::CopyDeviceSpanToVector(&sampled_gpair_h, sampled_gpair);
  for (auto gp : sampled_gpair_h) {
    sum_sampled_gradients += gp.GetGrad();
  }
  EXPECT_NEAR(sum_gradients / kRows, sum_sampled_gradients / kSampleRows, 0.02);
}

TEST(GradientBasedSampler, UniformSampling_ExternalMemory) {
  constexpr size_t kRows = 2048;
  constexpr size_t kCols = 16;
  constexpr float kSubsample = 0.5;
  constexpr size_t kSampleRows = kRows * kSubsample;
  constexpr size_t kPageSize = 1024;

  // Create a DMatrix with multiple batches.
  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix>
      dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true, tmpdir));
  auto gpair = GenerateRandomGradients(kRows);
  float sum_gradients = 0;
  for (auto gp : gpair.ConstHostVector()) {
    sum_gradients += gp.GetGrad();
  }
  gpair.SetDevice(0);

  BatchParam param{0, 256, 0, kPageSize};
  auto page = (*dmat->GetBatches<EllpackPage>(param).begin()).Impl();

  GradientBasedSampler sampler(page, kRows, param, kSubsample, TrainParam::kUniform);
  auto sample = sampler.Sample(gpair.DeviceSpan(), dmat.get());
  auto sampled_page = sample.page;
  auto sampled_gpair = sample.gpair;
  EXPECT_EQ(sample.sample_rows, kSampleRows);
  EXPECT_EQ(sampled_page->matrix.n_rows, kSampleRows);
  EXPECT_EQ(sampled_gpair.size(), kSampleRows);

  float sum_sampled_gradients = 0;
  std::vector<GradientPair> sampled_gpair_h(sampled_gpair.size());
  dh::CopyDeviceSpanToVector(&sampled_gpair_h, sampled_gpair);
  for (auto gp : sampled_gpair_h) {
    sum_sampled_gradients += gp.GetGrad();
  }
  EXPECT_NEAR(sum_gradients / kRows, sum_sampled_gradients / kSampleRows, 0.02);
}

TEST(GradientBasedSampler, GradientBasedSampling) {
  constexpr size_t kRows = 2048;
  constexpr size_t kCols = 16;
  constexpr float kSubsample = 0.5;
  constexpr size_t kSampleRows = kRows * kSubsample;
  constexpr size_t kPageSize = 0;

  // Create a DMatrix with a single batche.
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true));
  auto gpair = GenerateRandomGradients(kRows);
  float sum_gradients = 0;
  for (auto gp : gpair.ConstHostVector()) {
    sum_gradients += gp.GetGrad();
  }
  gpair.SetDevice(0);

  BatchParam param{0, 256, 0, kPageSize};
  auto page = (*dmat->GetBatches<EllpackPage>(param).begin()).Impl();

  GradientBasedSampler sampler(page, kRows, param, kSubsample, TrainParam::kGradientBased);
  auto sample = sampler.Sample(gpair.DeviceSpan(), dmat.get());
  auto sampled_gpair = sample.gpair;
  EXPECT_EQ(sample.sample_rows, kRows);
  EXPECT_EQ(sample.page, page);
  EXPECT_EQ(sampled_gpair.size(), kRows);

  float sum_sampled_gradients = 0;
  std::vector<GradientPair> sampled_gpair_h(sampled_gpair.size());
  dh::CopyDeviceSpanToVector(&sampled_gpair_h, sampled_gpair);
  for (auto gp : sampled_gpair_h) {
    sum_sampled_gradients += gp.GetGrad();
  }
  // TODO(rongou): gradient pairs need to be rescaled to get accurate statistics.
  EXPECT_NEAR(sum_gradients / kRows, sum_sampled_gradients / kSampleRows, 0.15);
}

TEST(GradientBasedSampler, GradientBasedSampling_ExternalMemory) {
  constexpr size_t kRows = 2048;
  constexpr size_t kCols = 16;
  constexpr float kSubsample = 0.5;
  constexpr size_t kSampleRows = kRows * kSubsample;
  constexpr size_t kPageSize = 1024;

  // Create a DMatrix with multiple batches.
  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix>
      dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true, tmpdir));
  auto gpair = GenerateRandomGradients(kRows);
  float sum_gradients = 0;
  for (auto gp : gpair.ConstHostVector()) {
    sum_gradients += gp.GetGrad();
  }
  gpair.SetDevice(0);

  BatchParam param{0, 256, 0, kPageSize};
  auto page = (*dmat->GetBatches<EllpackPage>(param).begin()).Impl();

  GradientBasedSampler sampler(page, kRows, param, kSubsample, TrainParam::kGradientBased);
  auto sample = sampler.Sample(gpair.DeviceSpan(), dmat.get());
  auto sampled_page = sample.page;
  auto sampled_gpair = sample.gpair;
  EXPECT_EQ(sample.sample_rows, kSampleRows);
  EXPECT_EQ(sampled_page->matrix.n_rows, kSampleRows);
  EXPECT_EQ(sampled_gpair.size(), kSampleRows);

  float sum_sampled_gradients = 0;
  std::vector<GradientPair> sampled_gpair_h(sampled_gpair.size());
  dh::CopyDeviceSpanToVector(&sampled_gpair_h, sampled_gpair);
  for (auto gp : sampled_gpair_h) {
    sum_sampled_gradients += gp.GetGrad();
  }
  // TODO(rongou): gradient pairs need to be rescaled to get accurate statistics.
  EXPECT_NEAR(sum_gradients / kRows, sum_sampled_gradients / kSampleRows, 0.15);
}

};  // namespace tree
};  // namespace xgboost
