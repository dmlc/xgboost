#include <gtest/gtest.h>

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
  sampler.Sample(&gpair, dmat.get(), kSampleRows);
}
};  // namespace tree
};  // namespace xgboost
