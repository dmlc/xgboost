/**
 * Copyright 2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>  // for BatchParam

#include <algorithm>  // for equal

#include "../../../src/common/column_matrix.h"  // for ColumnMatrix
#include "../../../src/data/gradient_index.h"   // for GHistIndexMatrix
#include "../../../src/tree/param.h"            // for TrainParam
#include "../helpers.h"                         // for RandomDataGenerator

namespace xgboost::data {
namespace {
class ExtMemQuantileDMatrixCpu : public ::testing::TestWithParam<float> {
 public:
  void Run(float sparsity) {
    bst_idx_t n_samples = 256, n_features = 16, n_batches = 4;
    bst_bin_t max_bin = 64;
    bst_target_t n_targets = 3;
    auto p_fmat = RandomDataGenerator{n_samples, n_features, sparsity}
                      .Bins(max_bin)
                      .Batches(n_batches)
                      .Targets(n_targets)
                      .GenerateExtMemQuantileDMatrix("temp", true);
    ASSERT_FALSE(p_fmat->SingleColBlock());

    BatchParam p{max_bin, tree::TrainParam::DftSparseThreshold()};
    Context ctx;

    // Loop over the batches and count the number of pages
    bst_idx_t batch_cnt = 0;
    bst_idx_t base_cnt = 0;
    bst_idx_t row_cnt = 0;
    for (auto const& page : p_fmat->GetBatches<GHistIndexMatrix>(&ctx, p)) {
      ASSERT_EQ(page.base_rowid, base_cnt);
      ++batch_cnt;
      base_cnt += n_samples / n_batches;
      row_cnt += page.Size();
      ASSERT_EQ((sparsity == 0.0f), page.IsDense());
    }
    ASSERT_EQ(n_batches, batch_cnt);
    ASSERT_EQ(p_fmat->Info().num_row_, n_samples);
    EXPECT_EQ(p_fmat->Info().num_row_, row_cnt);
    ASSERT_EQ(p_fmat->Info().num_col_, n_features);
    if (sparsity == 0.0f) {
      ASSERT_EQ(p_fmat->Info().num_nonzero_, n_samples * n_features);
    } else {
      ASSERT_LT(p_fmat->Info().num_nonzero_, n_samples * n_features);
      ASSERT_GT(p_fmat->Info().num_nonzero_, 0);
    }
    ASSERT_EQ(p_fmat->Info().labels.Shape(0), n_samples);
    ASSERT_EQ(p_fmat->Info().labels.Shape(1), n_targets);

    // Compare against the sparse page DMatrix
    auto p_sparse = RandomDataGenerator{n_samples, n_features, sparsity}
                        .Bins(max_bin)
                        .Batches(n_batches)
                        .Targets(n_targets)
                        .GenerateSparsePageDMatrix("temp", true);
    auto it = p_fmat->GetBatches<GHistIndexMatrix>(&ctx, p).begin();
    for (auto const& page : p_sparse->GetBatches<GHistIndexMatrix>(&ctx, p)) {
      auto orig = it.Page();
      // Check the CSR matrix
      auto orig_cuts = it.Page()->Cuts();
      auto sparse_cuts = page.Cuts();
      ASSERT_EQ(orig_cuts.Values(), sparse_cuts.Values());
      ASSERT_EQ(orig_cuts.MinValues(), sparse_cuts.MinValues());
      ASSERT_EQ(orig_cuts.Ptrs(), sparse_cuts.Ptrs());

      auto orig_ptr = orig->data.data();
      auto sparse_ptr = page.data.data();
      ASSERT_EQ(orig->data.size(), page.data.size());

      auto equal = std::equal(orig_ptr, orig_ptr + orig->data.size(), sparse_ptr);
      ASSERT_TRUE(equal);

      // Check the column matrix
      common::ColumnMatrix const& orig_columns = orig->Transpose();
      common::ColumnMatrix const& sparse_columns = page.Transpose();

      std::string str_orig, str_sparse;
      common::AlignedMemWriteStream fo_orig{&str_orig}, fo_sparse{&str_sparse};
      auto n_bytes_orig = orig_columns.Write(&fo_orig);
      auto n_bytes_sparse = sparse_columns.Write(&fo_sparse);
      ASSERT_EQ(n_bytes_orig, n_bytes_sparse);
      ASSERT_EQ(str_orig, str_sparse);

      ++it;
    }

    // Check meta info
    auto h_y_sparse = p_sparse->Info().labels.HostView();
    auto h_y = p_fmat->Info().labels.HostView();
    for (std::size_t i = 0, m = h_y_sparse.Shape(0); i < m; ++i) {
      for (std::size_t j = 0, n = h_y_sparse.Shape(1); j < n; ++j) {
        ASSERT_EQ(h_y(i, j), h_y_sparse(i, j));
      }
    }
  }
};
}  // anonymous namespace

TEST_P(ExtMemQuantileDMatrixCpu, Basic) { this->Run(this->GetParam()); }

INSTANTIATE_TEST_SUITE_P(ExtMemQuantileDMatrix, ExtMemQuantileDMatrixCpu, ::testing::ValuesIn([] {
                           std::vector<float> sparsities{
                               0.0f, tree::TrainParam::DftSparseThreshold(), 0.4f, 0.8f};
                           return sparsities;
                         }()));
}  // namespace xgboost::data
