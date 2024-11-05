/**
 * Copyright 2024, XGBoost Contributors
 */
#include <xgboost/base.h>
#include <xgboost/context.h>

#include "../../../src/tree/param.h"  // for TrainParam
#include "../helpers.h"               // for RandomDataGenerator

namespace xgboost::data {
template <typename Page, typename Equal, typename NoMissing>
void TestExtMemQdmBasic(Context const* ctx, bool on_host, float sparsity, Equal&& check_equal,
                        NoMissing&& no_missing) {
  bst_idx_t n_samples = 256, n_features = 16, n_batches = 4;
  bst_bin_t max_bin = 64;
  bst_target_t n_targets = 3;
  BatchParam p{max_bin, tree::TrainParam::DftSparseThreshold()};

  auto p_fmat = RandomDataGenerator{n_samples, n_features, sparsity}
                    .Bins(max_bin)
                    .Batches(n_batches)
                    .Targets(n_targets)
                    .Device(ctx->Device())
                    .OnHost(on_host)
                    .GenerateExtMemQuantileDMatrix("temp", true);
  ASSERT_FALSE(p_fmat->SingleColBlock());

  // Loop over the batches and count the number of pages
  bst_idx_t batch_cnt = 0, base_cnt = 0, row_cnt = 0;
  for (auto const& page : p_fmat->GetBatches<Page>(ctx, p)) {
    ASSERT_EQ(page.BaseRowId(), base_cnt);
    ++batch_cnt;
    base_cnt += n_samples / n_batches;
    row_cnt += page.Size();
    ASSERT_EQ((sparsity == 0.0f), no_missing(page));
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
                      .Device(ctx->Device())
                      .OnHost(on_host)
                      .GenerateSparsePageDMatrix("temp", true);
  auto it = p_fmat->GetBatches<Page>(ctx, p).begin();
  for (auto const& page : p_sparse->GetBatches<Page>(ctx, p)) {
    auto orig = it.Page();
    check_equal(ctx, *orig, page);
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
}  // namespace xgboost::data
