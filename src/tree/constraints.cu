/**
 * Copyright 2019-2026, XGBoost contributors
 */
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <algorithm>  // for max
#include <limits>     // for numeric_limits
#include <set>
#include <string>

#include "../common/cuda_context.cuh"  // for CUDAContext
#include "../common/device_helpers.cuh"
#include "constraints.cuh"
#include "constraints.h"
#include "param.h"
#include "xgboost/logging.h"
#include "xgboost/span.h"
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost {

size_t FeatureInteractionConstraintDevice::Features() const { return h_sets_ptr_.size() - 1; }

void FeatureInteractionConstraintDevice::Configure(tree::TrainParam const& param,
                                                   int32_t const n_features) {
  has_constraint_ = true;
  if (param.interaction_constraints.length() == 0) {
    has_constraint_ = false;
    return;
  }
  // --- Parse interaction constraints
  // Interaction constraints parsed from string parameter.  After
  // parsing, this looks like {{0, 1, 2}, {2, 3 ,4}}.
  std::vector<std::vector<bst_feature_t>> h_feature_constraints;
  try {
    ParseInteractionConstraint(param.interaction_constraints, &h_feature_constraints);
  } catch (dmlc::Error const& e) {
    LOG(FATAL) << "Failed to parse feature interaction constraint:\n"
               << param.interaction_constraints << "\n"
               << "With error:\n"
               << e.what();
  }
  n_sets_ = h_feature_constraints.size();

  size_t const n_feat_storage = LBitField64::ComputeStorageSize(n_features);
  if (n_feat_storage == 0 && n_features != 0) {
    LOG(FATAL) << "Wrong storage size, n_features: " << n_features;
  }

  // --- Initialize allowed features attached to nodes.
  int32_t n_nodes{param.MaxNodes()};
  node_constraints_.resize(n_nodes);
  node_constraints_storage_.resize(n_nodes);
  for (auto& n : node_constraints_storage_) {
    n.resize(LBitField64::ComputeStorageSize(n_features));
  }
  for (size_t i = 0; i < node_constraints_storage_.size(); ++i) {
    auto span = dh::ToSpan(node_constraints_storage_[i]);
    node_constraints_[i] = LBitField64(span);
  }
  s_node_constraints_ =
      common::Span<LBitField64>(node_constraints_.data(), node_constraints_.size());

  // Represent constraints as CSR format, flatten is the value vector,
  // ptr is row_ptr vector in CSR.
  std::vector<uint32_t> h_feature_constraints_flatten;
  for (auto const& constraints : h_feature_constraints) {
    for (auto c : constraints) {
      h_feature_constraints_flatten.emplace_back(c);
    }
  }
  std::vector<size_t> h_feature_constraints_ptr;
  size_t n_features_in_constraints = 0;
  h_feature_constraints_ptr.emplace_back(n_features_in_constraints);
  for (auto const& v : h_feature_constraints) {
    n_features_in_constraints += v.size();
    h_feature_constraints_ptr.emplace_back(n_features_in_constraints);
  }
  // Copy the CSR to device.
  d_fconstraints_.resize(h_feature_constraints_flatten.size());
  thrust::copy(h_feature_constraints_flatten.cbegin(), h_feature_constraints_flatten.cend(),
               d_fconstraints_.begin());
  d_fconstraints_ptr_.resize(h_feature_constraints_ptr.size());
  thrust::copy(h_feature_constraints_ptr.cbegin(), h_feature_constraints_ptr.cend(),
               d_fconstraints_ptr_.begin());

  // --- Compute interaction sets attached to each feature.
  // Use a set to eliminate duplicated entries.
  CHECK_LE(n_sets_, std::numeric_limits<GroupId>::max());
  std::vector<std::set<GroupId>> h_features_set(n_features);
  GroupId cid = 0;
  for (auto const& constraints : h_feature_constraints) {
    for (auto const& feat : constraints) {
      h_features_set.at(feat).insert(cid);
    }
    cid++;
  }
  // Compute device sets.
  std::vector<GroupId> h_sets;
  h_sets_ptr_.assign(1, 0);
  for (auto const& feature : h_features_set) {
    for (auto constraint_id : feature) {
      h_sets.emplace_back(constraint_id);
    }
    // empty set is well defined here.
    h_sets_ptr_.push_back(h_sets.size());
  }
  d_sets_ = h_sets;
}

FeatureInteractionConstraintDevice::FeatureInteractionConstraintDevice(
    tree::TrainParam const& param, int32_t const n_features)
    : has_constraint_{true}, n_sets_{0} {
  this->Configure(param, n_features);
}

void FeatureInteractionConstraintDevice::Reset(Context const* ctx) {
  node_groups_.clear();
  for (auto& node : node_constraints_storage_) {
    thrust::fill(ctx->CUDACtx()->CTP(), node.begin(), node.end(), 0);
  }
}

common::Span<bst_feature_t> FeatureInteractionConstraintDevice::PrepareBatch(
    std::size_t n_sampled) {
  if (!has_constraint_) {
    return {};
  }
  // Discard the previous results before growing the buffer.
  result_buffer_.clear();
  result_buffer_.resize(n_sampled);
  return dh::ToSpan(result_buffer_);
}

common::Span<bst_feature_t const> FeatureInteractionConstraintDevice::Query(
    Context const* ctx, common::Span<bst_feature_t const> feature_list, bst_node_t nidx,
    common::Span<bst_feature_t> out) const {
  if (!has_constraint_ || nidx == 0 || feature_list.empty()) {
    return feature_list;
  }
  CHECK_GE(out.size(), feature_list.size());
  auto allowed = s_node_constraints_[nidx];
  auto end = thrust::copy_if(ctx->CUDACtx()->CTP(), dh::tcbegin(feature_list),
                             dh::tcend(feature_list), dh::tbegin(out),
                             [=] __device__(bst_feature_t fidx) { return allowed.Check(fidx); });
  return out.first(end - dh::tbegin(out));
}

// One block per group containing the split feature. Children inherit only groups that
// also contained every earlier split feature on this path.
__global__ void InteractionConstraintSplitKernel(
    bst_feature_t split_feature, bool is_root,
    common::Span<FeatureInteractionConstraintDevice::GroupId const> groups,
    common::Span<bst_feature_t const> members, common::Span<std::size_t const> group_ptr,
    LBitField64 parent_groups, LBitField64 left_groups, LBitField64 right_groups,
    LBitField64 left_features, LBitField64 right_features) {
  auto group_idx_in_set = blockIdx.x;  // ID for IC groups that a feature belongs to
  if (group_idx_in_set == 0 && threadIdx.x == 0) {
    // A feature outside all groups can still be used repeatedly.
    left_features.Set(split_feature);
    right_features.Set(split_feature);
  }
  if (group_idx_in_set >= groups.size()) {
    return;
  }
  auto ic_group_idx = groups[group_idx_in_set];  // The actual IC group ID.
  if (!is_root && !parent_groups.Check(ic_group_idx)) {
    return;
  }
  if (threadIdx.x == 0) {
    // The current group becomes valid for the left and right child
    left_groups.Set(ic_group_idx);
    right_groups.Set(ic_group_idx);
  }
  auto end = group_ptr[ic_group_idx + 1];
  for (auto i = group_ptr[ic_group_idx] + threadIdx.x; i < end; i += blockDim.x) {
    // Enable each feature in the group.
    left_features.Set(members[i]);
    right_features.Set(members[i]);
  }
}

void FeatureInteractionConstraintDevice::Split(Context const* ctx, bst_node_t nidx,
                                               bst_feature_t feature_id, bst_node_t left_idx,
                                               bst_node_t right_idx) {
  if (!has_constraint_) {
    return;
  }
  detail::CheckFeatureInteractionSplit(nidx, left_idx, right_idx);
  CHECK_LT(right_idx, s_node_constraints_.size());
  CHECK_NE(s_node_constraints_.size(), 0);

  auto n_words = LBitField64::ComputeStorageSize(n_sets_);
  auto required = (static_cast<std::size_t>(std::max(left_idx, right_idx)) + 1) * n_words;
  if (node_groups_.size() < required) {
    node_groups_.resize(std::max(required, node_groups_.size() * 2), 0, ctx->CUDACtx()->Stream());
  }
  auto group_mask = [&](bst_node_t nidx) {
    return LBitField64{dh::ToSpan(node_groups_).subspan(nidx * n_words, n_words)};
  };
  auto begin = h_sets_ptr_.at(feature_id);
  auto s_sets = dh::ToSpan(this->d_sets_);
  auto groups = s_sets.subspan(begin, h_sets_ptr_.at(feature_id + 1) - begin);
  // A single block handles the split feature itself when it belongs to no group.
  CHECK_LE(groups.size(), std::numeric_limits<GroupId>::max());
  auto n_blocks = static_cast<std::uint32_t>(std::max<std::size_t>(1, groups.size()));
  dh::LaunchKernel{n_blocks, 256u, 0, ctx->CUDACtx()->Stream()} (
      InteractionConstraintSplitKernel, feature_id, nidx == RegTree::kRoot, groups,
      dh::ToSpan(d_fconstraints_), dh::ToSpan(d_fconstraints_ptr_), group_mask(nidx), group_mask(left_idx),
      group_mask(right_idx), s_node_constraints_[left_idx], s_node_constraints_[right_idx]);
}

}  // namespace xgboost
