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

  GradientBasedSampler sampler;
  BatchParam param{0, 256, 0, kPageSize};
  auto sample = sampler.Sample(&gpair, dmat.get(), param, kSampleRows);
  auto page = sample.page;
  auto scaled_gpair = sample.gpair;
  EXPECT_NEAR(page->matrix.n_rows, kSampleRows, 5);
  EXPECT_EQ(page->matrix.n_rows, scaled_gpair->Size());

  float gradients = 0;
  for (auto gp : gpair.ConstHostVector()) {
    gradients += gp.GetGrad();
  }
  float scaled_gradients = 0;
  for (auto gp : scaled_gpair->ConstHostVector()) {
    scaled_gradients += gp.GetGrad();
  }
  EXPECT_FLOAT_EQ(gradients, scaled_gradients);
}
};  // namespace tree
};  // namespace xgboost
