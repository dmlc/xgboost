/**
 * Copyright 2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>  // for BatchParam

#include "../helpers.h"  // for RandomDataGenerator

namespace xgboost::data {
class ExtMemQuantileDMatrixGpu : public ::testing::TestWithParam<float> {
 public:
  void Run(float sparsity) {
    bst_idx_t n_samples = 256, n_features = 16, n_batches = 4;
    bst_bin_t max_bin = 64;
    bst_target_t n_targets = 3;
    auto ctx = MakeCUDACtx(0);
    auto p_fmat = RandomDataGenerator{n_samples, n_features, sparsity}
                      .Bins(max_bin)
                      .Batches(n_batches)
                      .Targets(n_targets)
                      .Device(ctx.Device())
                      .GenerateExtMemQuantileDMatrix("temp", true);
  }
};

TEST_P(ExtMemQuantileDMatrixGpu, Basic) { this->Run(this->GetParam()); }

INSTANTIATE_TEST_SUITE_P(ExtMemQuantileDMatrix, ExtMemQuantileDMatrixGpu, ::testing::ValuesIn([] {
                           std::vector<float> sparsities{0.0f, 0.2f, 0.4f, 0.8f};
                           return sparsities;
                         }()));
}  // namespace xgboost::data
