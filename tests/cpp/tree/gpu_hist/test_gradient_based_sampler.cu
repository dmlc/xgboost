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

  BatchParam param{0, 256, 0, kPageSize};
  auto page = (*dmat->GetBatches<EllpackPage>(param).begin()).Impl();

  GradientBasedSampler sampler(param, page->matrix.info, kRows, kSampleRows);
  auto sample = sampler.Sample(&gpair, dmat.get());
  page = sample.page;
  auto sampled_gpair = sample.gpair;
  EXPECT_EQ(sampled_gpair.size(), kSampleRows);
  EXPECT_EQ(page->matrix.n_rows, kSampleRows);
  EXPECT_EQ(page->matrix.n_rows, sampled_gpair.size());

  float sum_gradients = 0;
  for (auto gp : gpair.ConstHostVector()) {
    sum_gradients += gp.GetGrad();
  }

  float sum_sampled_gradients = 0;
  std::vector<GradientPair> sampled_gpair_h(sampled_gpair.size());
  thrust::copy(sampled_gpair.begin(), sampled_gpair.end(), sampled_gpair_h.begin());
  for (auto gp : sampled_gpair_h) {
    sum_sampled_gradients += gp.GetGrad();
  }
  EXPECT_FLOAT_EQ(sum_gradients, sum_sampled_gradients);
}

};  // namespace tree
};  // namespace xgboost
