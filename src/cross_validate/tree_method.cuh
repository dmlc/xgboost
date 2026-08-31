/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/cuda_context.cuh"  // for CUDAContext
#include "../common/device_helpers.cuh"
#include "../tree/tree_view.h"  // for MultiTargetTreeView
#include "../tree/updater_gpu_hist.cuh"
#include "xgboost/context.h"  // for Context
#include "xgboost/span.h"     // for Span

namespace xgboost::cv {
inline void DebugCheckValid(Context const* ctx, bst_idx_t n_expected,
                            common::Span<bst_node_t const> d_position) {
  // Every training row of the unit, and only those, must have received a position.
  auto n_valid = thrust::count_if(
      ctx->CUDACtx()->CTP(), dh::tcbegin(d_position), dh::tcend(d_position),
      [] XGBOOST_DEVICE(bst_node_t nidx) { return nidx != RegTree::kInvalidNodeId; });
  CHECK_EQ(static_cast<bst_idx_t>(n_valid), n_expected);
}

template <template <typename> typename GoLeftOp, typename Acc>
void RouteHeldOut(Context const* ctx, common::Span<bst_idx_t const> ridxs,
                  tree::MultiTargetTreeView tree, GoLeftOp<Acc> go_left,
                  common::Span<bst_node_t> in_out_position) {
  dh::LaunchN(ridxs.size(), ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t i) {
    auto ridx = ridxs[i];
    auto nidx = in_out_position[ridx];
    if (tree.IsLeaf(nidx)) {
      return;
    }
    bool is_left = go_left(ridx, tree::cuda_impl::MultiTargetHistMaker::NodeSplitData{nidx});
    in_out_position[ridx] = is_left ? tree.LeftChild(nidx) : tree.RightChild(nidx);
  });
}
}  // namespace xgboost::cv
