/**
 * Copyright 2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>  // for BatchParam

#include <algorithm>  // for equal

#include "../../../src/data/gradient_index.h"  // for GHistIndexMatrix
#include "../../../src/tree/param.h"           // for TrainParam
#include "../helpers.h"                        // for RandomDataGenerator

namespace xgboost::data {
TEST(ExtMemQuantileDMatrix, Basic) {
  bst_idx_t n_samples = 256, n_features = 16, n_batches = 4;
  bst_bin_t max_bin = 64;
  auto p_fmat = RandomDataGenerator{n_samples, n_features, 0.0f}
                    .Bins(max_bin)
                    .Batches(n_batches)
                    .GenerateExtMemQuantileDMatrix("temp", false);
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
    ASSERT_TRUE(page.IsDense());
  }
  ASSERT_EQ(n_batches, batch_cnt);
  ASSERT_EQ(p_fmat->Info().num_row_, n_samples);
  EXPECT_EQ(p_fmat->Info().num_row_, row_cnt);
  ASSERT_EQ(p_fmat->Info().num_col_, n_features);
  ASSERT_EQ(p_fmat->Info().num_nonzero_, n_samples * n_features);

  // Compare against the sparse page DMatrix
  auto p_sparse = RandomDataGenerator{n_samples, n_features, 0.0f}
                      .Bins(max_bin)
                      .Batches(n_batches)
                      .GenerateSparsePageDMatrix("temp", false);
  auto it = p_fmat->GetBatches<GHistIndexMatrix>(&ctx, p).begin();
  for (auto const& page : p_sparse->GetBatches<GHistIndexMatrix>(&ctx, p)) {
    auto orig = it.Page();

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
    ++it;
  }
}
}  // namespace xgboost::data
