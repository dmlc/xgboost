/*!
 * Copyright 2019 XGBoost contributors
 */
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <xgboost/logging.h>

#include <algorithm>
#include <bitset>
#include <string>
#include <sstream>
#include <set>

#include "constraints.cuh"
#include "param.h"
#include "../common/span.h"
#include "../common/device_helpers.cuh"


namespace xgboost {

BitField::value_type constexpr BitField::kValueSize;
BitField::value_type constexpr BitField::kOne;

size_t FeatureInteractionConstraint::Features() const {
  return d_sets_ptr_.size() - 1;
}

void FeatureInteractionConstraint::Configure(
    tree::TrainParam const& param, int32_t const n_features) {
  has_constraint_ = true;
  if (param.interaction_constraints.length() == 0) {
    has_constraint_ = false;
    return;
  }
  // --- Parse interaction constraints
  std::istringstream iss(param.interaction_constraints);
  dmlc::JSONReader reader(&iss);
  // Interaction constraints parsed from string parameter.  After
  // parsing, this looks like {{0, 1, 2}, {2, 3 ,4}}.
  std::vector<std::vector<int32_t>> h_feature_constraints;
  try {
    reader.Read(&h_feature_constraints);
  } catch (dmlc::Error const& e) {
    LOG(FATAL) << "Failed to parse feature interaction constraint:\n"
               << param.interaction_constraints << "\n"
               << "With error:\n" << e.what();
  }
  n_sets_ = h_feature_constraints.size();

  size_t const n_feat_storage = BitField::ComputeStorageSize(n_features);
  if (n_feat_storage == 0 && n_features != 0) {
    LOG(FATAL) << "Wrong storage size, n_features: " << n_features;
  }

  // --- Initialize allowed features attached to nodes.
  if (param.max_depth == 0 && param.max_leaves == 0) {
    LOG(FATAL) << "Max leaves and max depth cannot both be unconstrained for gpu_hist.";
  }
  int32_t n_nodes {0};
  if (param.max_depth != 0) {
    n_nodes = std::pow(2, param.max_depth + 1);
  } else {
    n_nodes = param.max_leaves * 2 - 1;
  }
  CHECK_NE(n_nodes, 0);
  node_constraints_.resize(n_nodes);
  node_constraints_storage_.resize(n_nodes);
  for (auto& n : node_constraints_storage_) {
    n.resize(BitField::ComputeStorageSize(n_features));
  }
  for (size_t i = 0; i < node_constraints_storage_.size(); ++i) {
    auto span = dh::ToSpan(node_constraints_storage_[i]);
    node_constraints_[i] = BitField(span);
  }
  s_node_constraints_ = common::Span<BitField>(node_constraints_.data(),
                                               node_constraints_.size());

  // Represent constraints as CSR format, flatten is the value vector,
  // ptr is row_ptr vector in CSR.
  std::vector<int32_t> h_feature_constraints_flatten;
  for (auto const& constraints : h_feature_constraints) {
    for (int32_t c : constraints) {
      h_feature_constraints_flatten.emplace_back(c);
    }
  }
  std::vector<int32_t> h_feature_constraints_ptr;
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

  d_feature_buffer_storage_.resize(BitField::ComputeStorageSize(n_features));
  feature_buffer_ = dh::ToSpan(d_feature_buffer_storage_);

  // --- Initialize result buffers.
  output_buffer_bits_storage_.resize(BitField::ComputeStorageSize(n_features));
  output_buffer_bits_ = BitField(dh::ToSpan(output_buffer_bits_storage_));
  input_buffer_bits_storage_.resize(BitField::ComputeStorageSize(n_features));
  input_buffer_bits_ = BitField(dh::ToSpan(input_buffer_bits_storage_));
  result_buffer_.resize(n_features);
  s_result_buffer_ = dh::ToSpan(result_buffer_);
}

FeatureInteractionConstraint::FeatureInteractionConstraint(
    tree::TrainParam const& param, int32_t const n_features) :
    has_constraint_{true}, n_sets_{0} {
  this->Configure(param, n_features);
}

void FeatureInteractionConstraint::Reset() {
  for (auto& node : node_constraints_storage_) {
    thrust::fill(node.begin(), node.end(), 0);
  }
}

__global__ void ClearBuffersKernel(
    BitField result_buffer_output, BitField result_buffer_input) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < result_buffer_output.Size()) {
    result_buffer_output.Clear(tid);
  }
  if (tid < result_buffer_input.Size()) {
    result_buffer_input.Clear(tid);
  }
}

void FeatureInteractionConstraint::ClearBuffers() {
  CHECK_EQ(output_buffer_bits_.Size(), input_buffer_bits_.Size());
  CHECK_LE(feature_buffer_.Size(), output_buffer_bits_.Size());
  int constexpr kBlockThreads = 256;
  const int n_grids = static_cast<int>(
      common::DivRoundUp(input_buffer_bits_.Size(), kBlockThreads));
  ClearBuffersKernel<<<n_grids, kBlockThreads>>>(
      output_buffer_bits_, input_buffer_bits_);
}

common::Span<int32_t> FeatureInteractionConstraint::QueryNode(int32_t node_id) {
  if (!has_constraint_) { return {}; }
  CHECK_LT(node_id, s_node_constraints_.size());

  ClearBuffers();

  thrust::counting_iterator<int32_t> begin(0);
  thrust::counting_iterator<int32_t> end(result_buffer_.size());
  auto p_result_buffer = result_buffer_.data();
  BitField node_constraints = s_node_constraints_[node_id];

  thrust::device_ptr<int32_t> const out_end = thrust::copy_if(
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

__global__ void SetInputBufferKernel(common::Span<int32_t> feature_list_input,
                                     BitField result_buffer_input) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < feature_list_input.size()) {
    result_buffer_input.Set(feature_list_input[tid]);
  }
}

__global__ void QueryFeatureListKernel(BitField node_constraints,
                                       BitField result_buffer_input,
                                       BitField result_buffer_output) {
  result_buffer_output |= node_constraints;
  result_buffer_output &= result_buffer_input;
}

common::Span<int32_t> FeatureInteractionConstraint::Query(
    common::Span<int32_t> feature_list, int32_t nid) {
  if (!has_constraint_ || nid == 0) {
    return feature_list;
  }

  ClearBuffers();

  BitField node_constraints = s_node_constraints_[nid];
  CHECK_EQ(input_buffer_bits_.Size(), output_buffer_bits_.Size());

  int constexpr kBlockThreads = 256;
  const int n_grids = static_cast<int>(
      common::DivRoundUp(output_buffer_bits_.Size(), kBlockThreads));
  SetInputBufferKernel<<<n_grids, kBlockThreads>>>(feature_list, input_buffer_bits_);

  QueryFeatureListKernel<<<n_grids, kBlockThreads>>>(
      node_constraints, input_buffer_bits_, output_buffer_bits_);

  thrust::counting_iterator<int32_t> begin(0);
  thrust::counting_iterator<int32_t> end(result_buffer_.size());

  BitField local_result_buffer = output_buffer_bits_;

  thrust::device_ptr<int32_t> const out_end = thrust::copy_if(
      thrust::device,
      begin, end,
      result_buffer_.data(),
      [=]__device__(int32_t pos) {
        bool res = local_result_buffer.Check(pos);
        return res;
      });
  size_t const n_available = std::distance(result_buffer_.data(), out_end);

  common::Span<int32_t> result =
      {s_result_buffer_.data(), s_result_buffer_.data() + n_available};
  return result;
}

// Find interaction sets for each feature, then store all features in
// those sets in a buffer.
__global__ void RestoreFeatureListFromSetsKernel(
    BitField feature_buffer,

    int32_t fid,
    common::Span<int32_t> feature_interactions,
    common::Span<int32_t> feature_interactions_ptr,  // of size n interaction set + 1

    common::Span<int32_t> interactions_list,
    common::Span<int32_t> interactions_list_ptr) {
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

__global__ void InteractionConstraintSplitKernel(BitField feature,
                                                 int32_t feature_id,
                                                 BitField node,
                                                 BitField left,
                                                 BitField right) {
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

void FeatureInteractionConstraint::Split(
    int32_t node_id, int32_t feature_id, int32_t left_id, int32_t right_id) {
  if (!has_constraint_) { return; }
  CHECK_NE(node_id, left_id)
      << " Split node: " << node_id << " and its left child: "
      << left_id << " cannot be the same.";
  CHECK_NE(node_id, right_id)
      << " Split node: " << node_id << " and its left child: "
      << right_id << " cannot be the same.";
  CHECK_LT(right_id, s_node_constraints_.size());
  CHECK_NE(s_node_constraints_.size(), 0);

  BitField node = s_node_constraints_[node_id];
  BitField left = s_node_constraints_[left_id];
  BitField right = s_node_constraints_[right_id];

  dim3 const block3(16, 64, 1);
  dim3 const grid3(common::DivRoundUp(n_sets_, 16),
                   common::DivRoundUp(s_fconstraints_.size(), 64));
  RestoreFeatureListFromSetsKernel<<<grid3, block3>>>
      (feature_buffer_,
       feature_id,
       s_fconstraints_,
       s_fconstraints_ptr_,
       s_sets_,
       s_sets_ptr_);

  int constexpr kBlockThreads = 256;
  const int n_grids = static_cast<int>(common::DivRoundUp(node.Size(), kBlockThreads));
  InteractionConstraintSplitKernel<<<n_grids, kBlockThreads>>>
      (feature_buffer_,
       feature_id,
       node, left, right);
}

}  // namespace xgboost
