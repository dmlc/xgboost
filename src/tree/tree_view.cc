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
auto DispatchPtr(DeviceOrd device, HostDeviceVector<T> const& vec) {
  if (device.IsCPU()) {
    return vec.ConstHostPointer();
  }
  vec.SetDevice(device);
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

ScalarTreeView::ScalarTreeView(DeviceOrd device, RegTree const* tree)
    : CategoriesMixIn{tree->GetCategoriesMatrix(device)},
      nodes{tree->GetNodes(device).data()},
      stats{tree->GetStats(device).data()},
      n{tree->NumNodes()} {
  CHECK(!tree->IsMultiTarget());
}

MultiTargetTreeView::MultiTargetTreeView(DeviceOrd device, RegTree const* tree)
    : CategoriesMixIn{tree->GetCategoriesMatrix(device)},
      left{DispatchPtr(device, tree->GetMultiTargetTree()->left_)},
      right{DispatchPtr(device, tree->GetMultiTargetTree()->right_)},
      parent{DispatchPtr(device, tree->GetMultiTargetTree()->parent_)},
      split_index{DispatchPtr(device, tree->GetMultiTargetTree()->split_index_)},
      default_left{DispatchPtr(device, tree->GetMultiTargetTree()->default_left_)},
      split_conds{DispatchPtr(device, tree->GetMultiTargetTree()->split_conds_)},
      n{tree->NumNodes()},
      weights{DispatchWeight(device, tree)} {
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
