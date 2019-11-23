#include <gtest/gtest.h>

#include "../../../../src/data/ellpack_page.cuh"
#include "../../../../src/tree/gpu_hist/gradient_based_sampler.cuh"
#include "../../helpers.h"

namespace xgboost {
namespace tree {

TEST(GradientBasedSampler, Sample) {
  constexpr size_t kRows = 2048;
  constexpr size_t kCols = 16;
  constexpr size_t kSampleRows = 512;
  constexpr size_t kPageSize = 1024;

  // Create a DMatrix with multiple batches.
  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix>
      dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true, tmpdir));
  auto gpair = GenerateRandomGradients(kRows);
  gpair.SetDevice(0);

  GradientBasedSampler sampler;
  BatchParam param{0, 256, 0, kPageSize};
  auto sample = sampler.Sample(&gpair, dmat.get(), param, kSampleRows);
  auto page = sample.page;
  auto sampled_gpair = sample.gpair;
  EXPECT_NEAR(sampled_gpair.Size(), kSampleRows, 12);
  EXPECT_NEAR(page->matrix.n_rows, kSampleRows, 12);
  EXPECT_EQ(page->matrix.n_rows, sampled_gpair.Size());

  float sum_gradients = 0;
  for (auto gp : gpair.ConstHostVector()) {
    sum_gradients += gp.GetGrad();
  }
  float sum_sampled_gradients = 0;
  for (auto gp : sampled_gpair.ConstHostVector()) {
    sum_sampled_gradients += gp.GetGrad();
  }
  EXPECT_FLOAT_EQ(sum_gradients, sum_sampled_gradients);
}
};  // namespace tree
};  // namespace xgboost
