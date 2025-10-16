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
  vec.SetDevice(ctx->Device());
  return vec.ConstDevicePointer();
}

auto DispatchWeight(DeviceOrd device, RegTree const* tree) {
  auto const* mt_tree = tree->GetMultiTargetTree();
  auto n_targets = mt_tree->NumTargets();
  auto n_leaves = mt_tree->NumLeaves();
  CHECK_GE(n_leaves, 1);
  common::Span<float const> weights = tree->GetMultiTargetTree()->Weights(device);
  return linalg::MakeTensorView(device, weights, n_leaves, n_targets);
}
}  // namespace

ScalarTreeView::ScalarTreeView(Context const* ctx, RegTree const* tree)
    : CategoriesMixIn{tree->GetCategoriesMatrix(ctx->Device())},
      nodes{tree->GetNodes(ctx->Device()).data()},
      stats{tree->GetStats(ctx->Device()).data()},
      n{tree->NumNodes()} {
  CHECK(!tree->IsMultiTarget());
}

MultiTargetTreeView::MultiTargetTreeView(Context const* ctx, RegTree const* tree)
    : CategoriesMixIn{tree->GetCategoriesMatrix(ctx->Device())},
      left{DispatchPtr(ctx, tree->GetMultiTargetTree()->left_)},
      right{DispatchPtr(ctx, tree->GetMultiTargetTree()->right_)},
      parent{DispatchPtr(ctx, tree->GetMultiTargetTree()->parent_)},
      split_index{DispatchPtr(ctx, tree->GetMultiTargetTree()->split_index_)},
      default_left{DispatchPtr(ctx, tree->GetMultiTargetTree()->default_left_)},
      split_conds{DispatchPtr(ctx, tree->GetMultiTargetTree()->split_conds_)},
      n{tree->NumNodes()},
      weights{DispatchWeight(ctx->Device(), tree)} {
  CHECK(tree->IsMultiTarget());
}

MultiTargetTreeView::MultiTargetTreeView(RegTree const* tree)
    : CategoriesMixIn{tree->GetCategoriesMatrix(DeviceOrd::CPU())},
      left{tree->GetMultiTargetTree()->left_.ConstHostPointer()},
      right{tree->GetMultiTargetTree()->right_.ConstHostPointer()},
      parent{tree->GetMultiTargetTree()->parent_.ConstHostPointer()},
      split_index{tree->GetMultiTargetTree()->split_index_.ConstHostPointer()},
      default_left{tree->GetMultiTargetTree()->default_left_.ConstHostPointer()},
      split_conds{tree->GetMultiTargetTree()->split_conds_.ConstHostPointer()},
      n{tree->NumNodes()},
      weights{DispatchWeight(DeviceOrd::CPU(), tree)} {
  CHECK(tree->IsMultiTarget());
}
}  // namespace xgboost::tree
