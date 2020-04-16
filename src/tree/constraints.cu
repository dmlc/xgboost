/*!
 * Copyright 2019 XGBoost contributors
 */
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <string>
#include <set>

#include "xgboost/logging.h"
#include "xgboost/span.h"
#include "constraints.cuh"
#include "param.h"
#include "../common/device_helpers.cuh"

namespace xgboost {

size_t FeatureInteractionConstraintDevice::Features() const {
  return d_sets_ptr_.size() - 1;
}

void FeatureInteractionConstraintDevice::Configure(
    tree::TrainParam const& param, int32_t const n_features) {
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
               << "With error:\n" << e.what();
  }
  n_sets_ = h_feature_constraints.size();

  size_t const n_feat_storage = LBitField64::ComputeStorageSize(n_features);
  if (n_feat_storage == 0 && n_features != 0) {
    LOG(FATAL) << "Wrong storage size, n_features: " << n_features;
  }

  // --- Initialize allowed features attached to nodes.
  int32_t n_nodes { param.MaxNodes() };
  node_constraints_.resize(n_nodes);
  node_constraints_storage_.resize(n_nodes);
  for (auto& n : node_constraints_storage_) {
    n.resize(LBitField64::ComputeStorageSize(n_features));
  }
  for (size_t i = 0; i < node_constraints_storage_.size(); ++i) {
    auto span = dh::ToSpan(node_constraints_storage_[i]);
    node_constraints_[i] = LBitField64(span);
  }
  s_node_constraints_ = common::Span<LBitField64>(node_constraints_.data(),
                                               node_constraints_.size());

  // Represent constraints as CSR format, flatten is the value vector,
  // ptr is row_ptr vector in CSR.
  std::vector<uint32_t> h_feature_constraints_flatten;
  for (auto const& constraints : h_feature_constraints) {
    for (uint32_t c : constraints) {
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
  s_fconstraints_ = dh::ToSpan(d_fconstraints_);
  d_fconstraints_ptr_.resize(h_feature_constraints_ptr.size());
  thrust::copy(h_feature_constraints_ptr.cbegin(), h_feature_constraints_ptr.cend(),
               d_fconstraints_ptr_.begin());
  s_fconstraints_ptr_ = dh::ToSpan(d_fconstraints_ptr_);

  // --- Compute interaction sets attached to each feature.
  // Use a set to eliminate duplicated entries.
  std::vector<std::set<int32_t> > h_features_set(n_features);
  int32_t cid = 0;
  for (auto const& constraints : h_feature_constraints) {
    for (auto const& feat : constraints) {
      h_features_set.at(feat).insert(cid);
    }
    cid++;
  }
  // Compute device sets.
  std::vector<int32_t> h_sets;
  int32_t ptr = 0;
  std::vector<int32_t> h_sets_ptr {ptr};
  for (auto const& feature : h_features_set) {
    for (auto constraint_id : feature) {
      h_sets.emplace_back(constraint_id);
    }
    // empty set is well defined here.
    ptr += feature.size();
    h_sets_ptr.emplace_back(ptr);
  }
  d_sets_ = h_sets;
  d_sets_ptr_ = h_sets_ptr;
  s_sets_ = dh::ToSpan(d_sets_);
  s_sets_ptr_ = dh::ToSpan(d_sets_ptr_);

  d_feature_buffer_storage_.resize(LBitField64::ComputeStorageSize(n_features));
  feature_buffer_ = LBitField64{dh::ToSpan(d_feature_buffer_storage_)};

  // --- Initialize result buffers.
  output_buffer_bits_storage_.resize(LBitField64::ComputeStorageSize(n_features));
  output_buffer_bits_ = LBitField64(dh::ToSpan(output_buffer_bits_storage_));
  input_buffer_bits_storage_.resize(LBitField64::ComputeStorageSize(n_features));
  input_buffer_bits_ = LBitField64(dh::ToSpan(input_buffer_bits_storage_));
  result_buffer_.resize(n_features);
  s_result_buffer_ = dh::ToSpan(result_buffer_);
}

FeatureInteractionConstraintDevice::FeatureInteractionConstraintDevice(
    tree::TrainParam const& param, int32_t const n_features) :
    has_constraint_{true}, n_sets_{0} {
  this->Configure(param, n_features);
}

void FeatureInteractionConstraintDevice::Reset() {
  for (auto& node : node_constraints_storage_) {
    thrust::fill(node.begin(), node.end(), 0);
  }
}

__global__ void ClearBuffersKernel(
    LBitField64 result_buffer_output, LBitField64 result_buffer_input) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < result_buffer_output.Size()) {
    result_buffer_output.Clear(tid);
  }
  if (tid < result_buffer_input.Size()) {
    result_buffer_input.Clear(tid);
  }
}

void FeatureInteractionConstraintDevice::ClearBuffers() {
  CHECK_EQ(output_buffer_bits_.Size(), input_buffer_bits_.Size());
  CHECK_LE(feature_buffer_.Size(), output_buffer_bits_.Size());
  uint32_t constexpr kBlockThreads = 256;
  auto const n_grids = static_cast<uint32_t>(
      common::DivRoundUp(input_buffer_bits_.Size(), kBlockThreads));
  dh::LaunchKernel {n_grids, kBlockThreads} (
      ClearBuffersKernel,
      output_buffer_bits_, input_buffer_bits_);
}

common::Span<bst_feature_t> FeatureInteractionConstraintDevice::QueryNode(int32_t node_id) {
  if (!has_constraint_) { return {}; }
  CHECK_LT(node_id, s_node_constraints_.size());

  ClearBuffers();

  thrust::counting_iterator<int32_t> begin(0);
  thrust::counting_iterator<int32_t> end(result_buffer_.size());
  auto p_result_buffer = result_buffer_.data();
  LBitField64 node_constraints = s_node_constraints_[node_id];

  thrust::device_ptr<bst_feature_t> const out_end = thrust::copy_if(
      thrust::device,
      begin, end,
      p_result_buffer,
      [=]__device__(int32_t pos) {
        bool res = node_constraints.Check(pos);
        return res;
      });
  size_t const n_available = std::distance(result_buffer_.data(), out_end);

  return {s_result_buffer_.data(), s_result_buffer_.data() + n_available};
}

__global__ void SetInputBufferKernel(common::Span<bst_feature_t> feature_list_input,
                                     LBitField64 result_buffer_input) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < feature_list_input.size()) {
    result_buffer_input.Set(feature_list_input[tid]);
  }
}

__global__ void QueryFeatureListKernel(LBitField64 node_constraints,
                                       LBitField64 result_buffer_input,
                                       LBitField64 result_buffer_output) {
  result_buffer_output |= node_constraints;
  result_buffer_output &= result_buffer_input;
}

common::Span<bst_feature_t> FeatureInteractionConstraintDevice::Query(
    common::Span<bst_feature_t> feature_list, int32_t nid) {
  if (!has_constraint_ || nid == 0) {
    return feature_list;
  }

  ClearBuffers();

  LBitField64 node_constraints = s_node_constraints_[nid];
  CHECK_EQ(input_buffer_bits_.Size(), output_buffer_bits_.Size());

  uint32_t constexpr kBlockThreads = 256;
  auto n_grids = static_cast<uint32_t>(
      common::DivRoundUp(output_buffer_bits_.Size(), kBlockThreads));
  dh::LaunchKernel {n_grids, kBlockThreads} (
      SetInputBufferKernel,
      feature_list, input_buffer_bits_);
  dh::LaunchKernel {n_grids, kBlockThreads} (
      QueryFeatureListKernel,
      node_constraints, input_buffer_bits_, output_buffer_bits_);

  thrust::counting_iterator<int32_t> begin(0);
  thrust::counting_iterator<int32_t> end(result_buffer_.size());

  LBitField64 local_result_buffer = output_buffer_bits_;

  thrust::device_ptr<bst_feature_t> const out_end = thrust::copy_if(
      thrust::device,
      begin, end,
      result_buffer_.data(),
      [=]__device__(int32_t pos) {
        bool res = local_result_buffer.Check(pos);
        return res;
      });
  size_t const n_available = std::distance(result_buffer_.data(), out_end);

  common::Span<bst_feature_t> result =
      {s_result_buffer_.data(), s_result_buffer_.data() + n_available};
  return result;
}

// Find interaction sets for each feature, then store all features in
// those sets in a buffer.
__global__ void RestoreFeatureListFromSetsKernel(
    LBitField64 feature_buffer,

    bst_feature_t fid,
    common::Span<bst_feature_t> feature_interactions,
    common::Span<size_t> feature_interactions_ptr,  // of size n interaction set + 1

    common::Span<bst_feature_t> interactions_list,
    common::Span<size_t> interactions_list_ptr) {
  auto const tid_x = threadIdx.x + blockIdx.x * blockDim.x;
  auto const tid_y = threadIdx.y + blockIdx.y * blockDim.y;
  // painful mapping: fid -> sets related to it -> features related to sets.
  auto const beg = interactions_list_ptr[fid];
  auto const end = interactions_list_ptr[fid+1];
  auto const n_sets = end - beg;
  if (tid_x < n_sets) {
    auto const set_id_pos = beg + tid_x;
    auto const set_id = interactions_list[set_id_pos];
    auto const set_beg = feature_interactions_ptr[set_id];
    auto const set_end = feature_interactions_ptr[set_id + 1];
    auto const feature_pos = set_beg + tid_y;
    if (feature_pos < set_end) {
      feature_buffer.Set(feature_interactions[feature_pos]);
    }
  }
}

__global__ void InteractionConstraintSplitKernel(LBitField64 feature,
                                                 int32_t feature_id,
                                                 LBitField64 node,
                                                 LBitField64 left,
                                                 LBitField64 right) {
  auto tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid > node.Size()) {
    return;
  }
  // enable constraints from feature
  node |= feature;
  // clear the buffer after use
  if (tid < feature.Size()) {
    feature.Clear(tid);
  }

  // enable constraints from parent
  left  |= node;
  right |= node;

  if (tid == feature_id) {
    // enable the split feature, set all of them at last instead of
    // setting it for parent to avoid race.
    node.Set(feature_id);
    left.Set(feature_id);
    right.Set(feature_id);
  }
}

void FeatureInteractionConstraintDevice::Split(
    bst_node_t node_id, bst_feature_t feature_id, bst_node_t left_id, bst_node_t right_id) {
  if (!has_constraint_) { return; }
  CHECK_NE(node_id, left_id)
      << " Split node: " << node_id << " and its left child: "
      << left_id << " cannot be the same.";
  CHECK_NE(node_id, right_id)
      << " Split node: " << node_id << " and its left child: "
      << right_id << " cannot be the same.";
  CHECK_LT(right_id, s_node_constraints_.size());
  CHECK_NE(s_node_constraints_.size(), 0);

  LBitField64 node = s_node_constraints_[node_id];
  LBitField64 left = s_node_constraints_[left_id];
  LBitField64 right = s_node_constraints_[right_id];

  dim3 const block3(16, 64, 1);
  dim3 const grid3(common::DivRoundUp(n_sets_, 16),
                   common::DivRoundUp(s_fconstraints_.size(), 64));
  dh::LaunchKernel {grid3, block3} (
      RestoreFeatureListFromSetsKernel,
      feature_buffer_, feature_id,
      s_fconstraints_, s_fconstraints_ptr_,
      s_sets_, s_sets_ptr_);

  uint32_t constexpr kBlockThreads = 256;
  auto n_grids = static_cast<uint32_t>(common::DivRoundUp(node.Size(), kBlockThreads));

  dh::LaunchKernel {n_grids, kBlockThreads} (
      InteractionConstraintSplitKernel,
      feature_buffer_,
      feature_id,
      node, left, right);
}

}  // namespace xgboost
