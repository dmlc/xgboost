/**
 * Copyright 2019-2026, XGBoost contributors
 */
#include <algorithm>  // for clamp, max, min
#include <limits>     // for numeric_limits
#include <string>     // for string
#include <vector>     // for vector

#include "../common/cuda_context.cuh"  // for CUDAContext
#include "../common/device_helpers.cuh"
#include "constraints.cuh"
#include "constraints.h"
#include "param.h"
#include "xgboost/logging.h"
#include "xgboost/span.h"
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost {

FeatureInteractionConstraintDevice::State::State(Context const* ctx, std::string const& spec,
                                                 bst_feature_t n_features)
    : constraints{spec} {
  std::vector<std::vector<bst_feature_t>> sets;
  ParseInteractionConstraint(spec, &sets);
  CHECK_LE(sets.size(), std::numeric_limits<GroupId>::max());

  std::vector<bst_feature_t> h_members;
  std::vector<std::size_t> h_group_ptr{0};
  std::vector<std::vector<GroupId>> by_feature(n_features);
  for (GroupId gid = 0; gid < sets.size(); ++gid) {
    auto const& set = sets[gid];
    h_members.insert(h_members.end(), set.begin(), set.end());
    h_group_ptr.push_back(h_members.size());
    for (auto f : set) {
      auto& ids = by_feature.at(f);
      if (ids.empty() || ids.back() != gid) {
        ids.push_back(gid);
      }
    }
  }

  std::vector<GroupId> h_groups;
  feature_ptr.reserve(by_feature.size() + 1);
  feature_ptr.push_back(0);
  for (auto const& ids : by_feature) {
    h_groups.insert(h_groups.end(), ids.begin(), ids.end());
    feature_ptr.push_back(h_groups.size());
  }

  auto stream = ctx->CUDACtx()->Stream();
  dh::CopyTo(h_members, &members, stream);
  dh::CopyTo(h_group_ptr, &group_ptr, stream);
  dh::CopyTo(h_groups, &groups, stream);
  stream.Sync();
}

bst_feature_t FeatureInteractionConstraintDevice::Features() const {
  return state_ ? static_cast<bst_feature_t>(state_->feature_ptr.size() - 1) : 0;
}

void FeatureInteractionConstraintDevice::Reset(Context const* ctx, tree::TrainParam const& param,
                                               bst_feature_t n_features) {
  auto const& spec = param.interaction_constraints;
  if (spec.empty()) {
    state_.reset();
    return;
  }
  if (!state_ || state_->constraints != spec) {
    state_.emplace(ctx, spec, n_features);
  }

  state_->n_nodes = param.MaxNodes();
  auto n_words = LBitField64::ComputeStorageSize(this->Features());
  state_->node_constraints.clear();
  state_->node_constraints.resize(state_->n_nodes * n_words, 0, ctx->CUDACtx()->Stream());
  state_->node_groups.clear();
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
  if (!state_) {
    return;
  }
  auto& state = *state_;
  detail::CheckFeatureInteractionSplit(nidx, left_idx, right_idx);
  CHECK_GE(std::min({nidx, left_idx, right_idx}), 0);
  CHECK_LT(std::max({nidx, left_idx, right_idx}), state.n_nodes);

  auto n_words = LBitField64::ComputeStorageSize(state.group_ptr.size() - 1);
  auto required = (static_cast<std::size_t>(std::max(left_idx, right_idx)) + 1) * n_words;
  if (state.node_groups.size() < required) {
    auto max_words = static_cast<std::size_t>(state.n_nodes) * n_words;
    auto new_size = std::clamp(state.node_groups.size() * 2, required, max_words);
    state.node_groups.resize(new_size, 0, ctx->CUDACtx()->Stream());
  }
  detail::NodeBitFields group_masks{dh::ToSpan(state.node_groups), n_words};
  auto feature_masks = this->FeatureMasks();
  auto begin = state.feature_ptr.at(feature_id);
  auto s_sets = dh::ToSpan(state.groups);
  auto groups = s_sets.subspan(begin, state.feature_ptr.at(feature_id + 1) - begin);
  // A single block handles the split feature itself when it belongs to no group.
  CHECK_LE(groups.size(), std::numeric_limits<GroupId>::max());
  auto n_blocks = static_cast<std::uint32_t>(std::max<std::size_t>(1, groups.size()));
  dh::LaunchKernel{n_blocks, 256u, 0, ctx->CUDACtx()->Stream()}(  // NOLINT
      InteractionConstraintSplitKernel, feature_id, nidx == RegTree::kRoot, groups,
      dh::ToSpan(state.members), dh::ToSpan(state.group_ptr), group_masks[nidx],
      group_masks[left_idx], group_masks[right_idx], feature_masks[left_idx],
      feature_masks[right_idx]);
}
}  // namespace xgboost
