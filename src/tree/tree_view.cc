/**
 * Copyright 2025, XGBoost Contributors
 */
#include "tree_view.h"

#include "xgboost/context.h"             // for Context
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for MakeTensorView
#include "xgboost/span.h"                // for Span

namespace xgboost::tree {
namespace {
template <typename T>
auto DispatchPtr(Context const* ctx, HostDeviceVector<T> const& vec) {
  if (ctx->IsCPU()) {
    return vec.ConstHostPointer();
  }
  return vec.ConstDevicePointer();
}
}  // namespace

ScalarTreeView::ScalarTreeView(Context const* ctx, RegTree const* tree)
    : nodes{tree->GetNodes(ctx).data()},
      stats{tree->GetStats().data()},
      cats{tree->GetCategoriesMatrix()},
      n{tree->NumNodes()} {
  CHECK(!tree->IsMultiTarget());
}

MultiTargetTreeView::MultiTargetTreeView(Context const* ctx, RegTree const* tree)
    : left{DispatchPtr(ctx, tree->GetMultiTargetTree()->left_)},
      right{DispatchPtr(ctx, tree->GetMultiTargetTree()->right_)},
      parent{DispatchPtr(ctx, tree->GetMultiTargetTree()->parent_)},
      split_index{DispatchPtr(ctx, tree->GetMultiTargetTree()->split_index_)},
      default_left{DispatchPtr(ctx, tree->GetMultiTargetTree()->default_left_)},
      split_conds{DispatchPtr(ctx, tree->GetMultiTargetTree()->split_conds_)},
      cats{tree->GetCategoriesMatrix()},
      n{tree->NumNodes()},
      weights{[&]() {
        auto const* mt_tree = tree->GetMultiTargetTree();
        auto n_targets = mt_tree->NumTargets();
        auto n_leaves = mt_tree->weights_.Size() / mt_tree->NumTargets();
        CHECK_GE(n_leaves, 1);
        common::Span<float const> weights;
        if (ctx->IsCPU()) {
          weights = tree->GetMultiTargetTree()->weights_.ConstHostSpan();
        } else {
          tree->GetMultiTargetTree()->weights_.SetDevice(ctx->Device());
          weights = tree->GetMultiTargetTree()->weights_.ConstDeviceSpan();
        }
        return linalg::MakeTensorView(ctx, weights, n_leaves, n_targets);
      }()} {}

MultiTargetTreeView::MultiTargetTreeView(RegTree const* tree)
    : left{tree->GetMultiTargetTree()->left_.ConstHostPointer()},
      right{tree->GetMultiTargetTree()->right_.ConstHostPointer()},
      parent{tree->GetMultiTargetTree()->parent_.ConstHostPointer()},
      split_index{tree->GetMultiTargetTree()->split_index_.ConstHostPointer()},
      default_left{tree->GetMultiTargetTree()->default_left_.ConstHostPointer()},
      split_conds{tree->GetMultiTargetTree()->split_conds_.ConstHostPointer()},
      cats{tree->GetCategoriesMatrix()},
      n{tree->NumNodes()},
      weights{[&]() {
        auto const* mt_tree = tree->GetMultiTargetTree();
        auto n_targets = mt_tree->NumTargets();
        auto n_leaves = mt_tree->weights_.Size() / mt_tree->NumTargets();
        CHECK_GE(n_leaves, 1);
        common::Span<float const> weights;
        weights = tree->GetMultiTargetTree()->weights_.ConstHostSpan();
        return linalg::MakeTensorView(DeviceOrd::CPU(), weights, n_leaves, n_targets);
      }()} {}
}  // namespace xgboost::tree
