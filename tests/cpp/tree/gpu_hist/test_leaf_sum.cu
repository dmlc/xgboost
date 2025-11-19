/**
 * Copyright 2025, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/sequence.h>  // for sequence
#include <xgboost/linalg.h>   // for Constant

#include <vector>  // for vector

#include "../../../../src/common/device_vector.cuh"
#include "../../../../src/tree/gpu_hist/leaf_sum.cuh"
#include "../../../../src/tree/gpu_hist/row_partitioner.cuh"  // for LeafInfo
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
  h_leaves[0].node.segment = Segment{0, 3};
  h_leaves[1].nidx = 2;
  h_leaves[1].node.segment = Segment{3, 6};

  auto gpairs = linalg::Constant(&ctx, GradientPair{1.0f, 1.0f}, n_samples, n_targets);

  dh::device_vector<RowIndexT> sorted_ridx(n_samples);
  thrust::sequence(ctx.CUDACtx()->CTP(), sorted_ridx.begin(), sorted_ridx.end(), 0);

  auto quantizers = MakeDummyQuantizers(n_targets);
  auto out_sum = linalg::Constant(&ctx, GradientPairInt64{}, n_leaves, n_targets);

  LeafGradSum(&ctx, h_leaves, dh::ToSpan(quantizers), dh::ToSpan(sorted_ridx),
              gpairs.View(ctx.Device()), out_sum.View(ctx.Device()));

  for (auto v : out_sum.HostView()) {
    ASSERT_EQ(v.GetQuantisedGrad(), 3);
    ASSERT_EQ(v.GetQuantisedHess(), 3);
  }
}
}  // namespace xgboost::tree::cuda_impl
