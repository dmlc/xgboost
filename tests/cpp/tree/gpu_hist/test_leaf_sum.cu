/**
 * Copyright 2025, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/linalg.h>  // for Constant

#include <cstdint>  // for int64_t
#include <vector>   // for vector

#include "../../../../src/common/device_vector.cuh"
#include "../../../../src/tree/gpu_hist/leaf_sum.cuh"
#include "../../../../src/tree/gpu_hist/row_partitioner.cuh"  // for LeafInfo, RowBuffers
#include "../../helpers.h"
#include "dummy_quantizer.cuh"  // for MakeDummyQuantizers

namespace xgboost::tree::cuda_impl {
TEST(LeafGradSum, Basic) {
  auto ctx = MakeCUDACtx(0);

  bst_target_t n_targets = 2;
  bst_idx_t n_samples = 6;
  bst_idx_t n_leaves = 2;

  // Create leaf information
  std::vector<LeafInfo> h_leaves(n_leaves);
  h_leaves[0].nidx = 1;
  h_leaves[0].node.segment = Segment{0, 2};
  h_leaves[1].nidx = 2;
  h_leaves[1].node.segment = Segment{2, 6};
  h_leaves[1].node.buffer = 1;

  // The gradient of row i is i.
  auto gpairs = linalg::Constant(&ctx, GradientPair{}, n_samples, n_targets);
  auto h_gpairs = gpairs.HostView();
  for (bst_idx_t i = 0; i < n_samples; ++i) {
    for (bst_target_t t = 0; t < n_targets; ++t) {
      h_gpairs(i, t) = GradientPair{static_cast<float>(i), static_cast<float>(i)};
    }
  }

  // Each leaf reads its own buffer, the rest of a buffer is stale.
  std::vector<RowIndexT> h_ridx0{0, 1, 5, 5, 5, 5};
  std::vector<RowIndexT> h_ridx1{5, 5, 2, 3, 4, 5};
  dh::device_vector<RowIndexT> ridx0(h_ridx0), ridx1(h_ridx1);

  auto quantizers = MakeDummyQuantizers(n_targets);
  auto out_sum = linalg::Constant(&ctx, GradientPairInt64{}, n_leaves, n_targets);

  RowBuffers<RowIndexT const> ridx{{dh::ToSpan(ridx0), dh::ToSpan(ridx1)}};
  LeafGradSum(&ctx, h_leaves, dh::ToSpan(quantizers), ridx, gpairs.View(ctx.Device()),
              out_sum.View(ctx.Device()));

  auto h_out = out_sum.HostView();
  std::int64_t expected[] = {0 + 1, 2 + 3 + 4 + 5};
  for (bst_idx_t i = 0; i < n_leaves; ++i) {
    for (bst_target_t t = 0; t < n_targets; ++t) {
      ASSERT_EQ(h_out(i, t).GetQuantisedGrad(), expected[i]);
      ASSERT_EQ(h_out(i, t).GetQuantisedHess(), expected[i]);
    }
  }
}
}  // namespace xgboost::tree::cuda_impl
