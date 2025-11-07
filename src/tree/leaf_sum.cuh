/**
 * Copyright 2025, XGBoost contributors
 */
#pragma once

#include <vector>  // for vector

#include "gpu_hist/quantiser.cuh"        // for GradientQuantiser
#include "gpu_hist/row_partitioner.cuh"  // for RowIndexT, LeafInfo
#include "updater_gpu_common.cuh"        // for GPUTrainingParam
#include "xgboost/context.h"             // for Context
#include "xgboost/linalg.h"              // for MatrixView
#include "xgboost/span.h"                // for Span

namespace xgboost::tree::cuda_impl {
// shape(out_sum) == (n_leaves, n_targets)
void LeafGradSum(Context const* ctx, std::vector<LeafInfo> const& h_leaves,
                 common::Span<GradientQuantiser const> roundings,
                 common::Span<RowIndexT const> sorted_ridx,
                 linalg::MatrixView<GradientPair const> grad,
                 linalg::MatrixView<GradientPairInt64> out_sum);

// shape(grad_sum) == (n_leaves, n_targets)
// shape(out_weights) == (n_leaves, n_targets)
void LeafWeight(Context const* ctx, GPUTrainingParam const& param,
                common::Span<GradientQuantiser const> roundings,
                linalg::MatrixView<GradientPairInt64 const> grad_sum,
                linalg::MatrixView<float> out_weights);
}  // namespace xgboost::tree::cuda_impl
