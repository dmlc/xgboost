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
  common::Span<float const> weights = tree->GetMultiTargetTree()->LeafWeights(device);
  if (n_leaves > 0) {
    CHECK(!weights.empty());
  }
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
      leaf_weights{DispatchWeight(device, tree)} {}

MultiTargetTreeView::MultiTargetTreeView(RegTree const* tree)
    : MultiTargetTreeView{DeviceOrd::CPU(), tree} {}
}  // namespace xgboost::tree
