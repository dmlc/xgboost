/**
 * Copyright 2023-2026, XGBoost Contributors
 */
#include "xgboost/multi_target_tree_model.h"

#include <algorithm>    // for copy_n
#include <cstddef>      // for size_t
#include <cstdint>      // for int32_t, uint8_t
#include <limits>       // for numeric_limits
#include <string_view>  // for string_view
#include <utility>      // for move
#include <vector>       // for vector

#include "../common/cuda_rt_utils.h"  // for MemcpyAsync
#include "../common/linalg_op.h"      // for cbegin
#include "io_utils.h"                 // for I32ArrayT, FloatArrayT, GetElem, ...
#include "xgboost/base.h"             // for bst_node_t, bst_feature_t, bst_target_t
#include "xgboost/json.h"             // for Json, get, Object, Number, Integer, ...
#include "xgboost/logging.h"
#include "xgboost/tree_model.h"  // for TreeParam

namespace xgboost {
MultiTargetTree::MultiTargetTree(TreeParam const* param)
    : param_{param},
      left_(1ul, InvalidNodeId()),
      right_(1ul, InvalidNodeId()),
      parent_(1ul, InvalidNodeId()),
      split_index_(1ul, 0),
      default_left_(1ul, 0),
      split_conds_(1ul, DftBadValue()),
      gain_(1ul, 0.0f),
      sum_hess_(1ul, 0.0f) {
  CHECK_GT(param_->size_leaf_vector, 1);
}

MultiTargetTree::MultiTargetTree(MultiTargetTree const& that)
    : param_{that.param_},
      left_(that.left_.Size(), 0, that.left_.Device()),
      right_(that.right_.Size(), 0, that.right_.Device()),
      parent_(that.parent_.Size(), 0, that.parent_.Device()),
      split_index_(that.split_index_.Size(), 0, that.split_index_.Device()),
      default_left_(that.default_left_.Size(), 0, that.default_left_.Device()),
      split_conds_(that.split_conds_.Size(), 0.0f, that.split_conds_.Device()),
      weights_(that.weights_.Size(), 0.0f, that.weights_.Device()),
      leaf_weights_(that.leaf_weights_.Size(), 0.0f, that.leaf_weights_.Device()),
      gain_(that.gain_.Size(), 0.0f, that.gain_.Device()),
      sum_hess_(that.sum_hess_.Size(), 0.0f, that.sum_hess_.Device()) {
  this->left_.Copy(that.left_);
  this->right_.Copy(that.right_);
  this->parent_.Copy(that.parent_);
  this->split_index_.Copy(that.split_index_);
  this->default_left_.Copy(that.default_left_);
  this->split_conds_.Copy(that.split_conds_);
  this->weights_.Copy(that.weights_);
  this->leaf_weights_.Copy(that.leaf_weights_);
  this->gain_.Copy(that.gain_);
  this->sum_hess_.Copy(that.sum_hess_);
}

void MultiTargetTree::SetRoot(linalg::VectorView<float const> weight, float sum_hess) {
  CHECK(!weight.Empty());
  auto const next_nidx = RegTree::kRoot + 1;

  this->weights_.SetDevice(weight.Device());
  this->weights_.Resize(weight.Size(), DftBadValue());

  CHECK_LE(weight.Size(), this->NumTargets());
  CHECK_GE(weights_.Size(), next_nidx * weight.Size());

  if (weight.Device().IsCUDA()) {
    auto out_weight = weights_.DeviceSpan().subspan(RegTree::kRoot * weight.Size(), weight.Size());
    CHECK(weight.Contiguous());
    curt::MemcpyAsync(out_weight.data(), weight.Values().data(), out_weight.size_bytes(),
                      curt::DefaultStream());
  } else {
    auto out_weight = weights_.HostSpan().subspan(RegTree::kRoot * weight.Size(), weight.Size());
    for (std::size_t i = 0, n = weight.Size(); i < n; ++i) {
      out_weight[i] = weight(i);
    }
  }

  // Set root statistics
  sum_hess_.Resize(next_nidx, 0.0f);
  sum_hess_.HostVector()[RegTree::kRoot] = sum_hess;
  gain_.Resize(next_nidx, 0.0f);

  CHECK_EQ(this->param_->num_nodes, 1);
  CHECK_EQ(this->NumSplitTargets(), weight.Size());
}

void MultiTargetTree::Expand(bst_node_t nidx, bst_feature_t split_idx, float split_cond,
                             bool default_left, linalg::VectorView<float const> base_weight,
                             linalg::VectorView<float const> left_weight,
                             linalg::VectorView<float const> right_weight, float gain,
                             float sum_hess, float left_sum, float right_sum) {
  CHECK(this->IsLeaf(nidx));
  CHECK_GE(parent_.Size(), 1);
  CHECK_EQ(parent_.Size(), left_.Size());
  CHECK_EQ(left_.Size(), right_.Size());
  auto n_split_targets = this->NumSplitTargets();
  CHECK_EQ(base_weight.Size(), n_split_targets);

  std::size_t n = param_->num_nodes + 2;
  CHECK_LT(split_idx, this->param_->num_feature);
  left_.Resize(n, InvalidNodeId());
  right_.Resize(n, InvalidNodeId());
  parent_.Resize(n, InvalidNodeId());

  auto left_child = parent_.Size() - 2;
  auto right_child = parent_.Size() - 1;

  CHECK_NE(left_child, nidx);
  left_.HostVector()[nidx] = left_child;
  right_.HostVector()[nidx] = right_child;

  auto& h_parent = parent_.HostVector();
  if (nidx != 0) {
    CHECK_NE(h_parent[nidx], InvalidNodeId());
  }

  h_parent[left_child] = nidx;
  h_parent[right_child] = nidx;

  split_index_.Resize(n);
  split_index_.HostVector()[nidx] = split_idx;

  split_conds_.Resize(n, DftBadValue());
  split_conds_.HostVector()[nidx] = split_cond;

  default_left_.Resize(n);
  default_left_.HostVector()[nidx] = static_cast<std::uint8_t>(default_left);

  // Set weights
  weights_.Resize(n * base_weight.Size());
  auto p_weight = this->NodeWeight(nidx, n_split_targets);
  CHECK_GE(p_weight.Size(), base_weight.Size());
  auto l_weight = this->NodeWeight(left_child, n_split_targets);
  CHECK_GE(l_weight.Size(), left_weight.Size());
  auto r_weight = this->NodeWeight(right_child, n_split_targets);
  CHECK_GE(r_weight.Size(), right_weight.Size());

  CHECK_EQ(base_weight.Size(), left_weight.Size());
  CHECK_EQ(base_weight.Size(), right_weight.Size());

  for (std::size_t i = 0, n = base_weight.Size(); i < n; ++i) {
    p_weight(i) = base_weight(i);
    l_weight(i) = left_weight(i);
    r_weight(i) = right_weight(i);
  }

  gain_.Resize(n, 0.0f);
  gain_.HostVector()[nidx] = gain;

  sum_hess_.Resize(n, 0.0f);
  auto& h_hess = sum_hess_.HostVector();
  h_hess[nidx] = sum_hess;
  h_hess[left_child] = left_sum;
  h_hess[right_child] = right_sum;
}

void MultiTargetTree::SetLeaves(std::vector<bst_node_t> leaves, common::Span<float const> weights) {
  auto is_partial_tree = this->NumLeaves() == 0;
  CHECK(is_partial_tree || leaves.size() == this->NumLeaves());
  auto n_targets = this->NumTargets();
  std::int32_t nidx_in_set = 0;
  auto n_leaves = leaves.size();
  this->leaf_weights_.Resize(n_leaves * n_targets);
  auto h_weights = this->leaf_weights_.HostSpan();
  // Reuse the right child as the leaf weight mapping.
  auto h_leaf_mapping = this->right_.HostSpan();

  for (auto nidx : leaves) {
    CHECK(this->IsLeaf(nidx));
    auto w_in = weights.subspan(nidx_in_set * n_targets, n_targets);
    auto w_out = h_weights.subspan(nidx_in_set * n_targets, n_targets);
    std::copy(w_in.cbegin(), w_in.cend(), w_out.begin());
    if (is_partial_tree) {
      CHECK_EQ(h_leaf_mapping[nidx], InvalidNodeId());
    }
    h_leaf_mapping[nidx] = nidx_in_set;
    nidx_in_set++;
  }
}

void MultiTargetTree::SetLeaves() {
  CHECK_EQ(this->NumLeaves(), 0);
  auto n_targets = this->NumTargets();
  CHECK_EQ(n_targets, this->NumSplitTargets());
  auto n_nodes = this->param_->num_nodes;
  // Reuse the right child as the leaf weight mapping.
  auto h_leaf_mapping = this->right_.HostSpan();

  bst_node_t nidx_in_set = 0;
  auto& h_weights = this->leaf_weights_.HostVector();
  CHECK(h_weights.empty());
  for (bst_node_t nidx = 0; nidx < n_nodes; ++nidx) {
    if (!IsLeaf(nidx)) {
      continue;
    }
    auto w_in = this->NodeWeight(nidx);
    h_weights.resize((nidx_in_set + 1) * n_targets);
    auto w_out = common::Span{h_weights}.subspan(nidx_in_set * n_targets, n_targets);
    std::copy(linalg::cbegin(w_in), linalg::cend(w_in), w_out.begin());
    CHECK_EQ(h_leaf_mapping[nidx], InvalidNodeId());
    h_leaf_mapping[nidx] = nidx_in_set;
    nidx_in_set++;
  }
}

template <bool typed, bool feature_is_64>
void LoadModelImpl(Json const& in, HostDeviceVector<float>* p_weights,
                   HostDeviceVector<float>* p_leaf_weights, HostDeviceVector<bst_node_t>* p_lefts,
                   HostDeviceVector<bst_node_t>* p_rights, HostDeviceVector<bst_node_t>* p_parents,
                   HostDeviceVector<float>* p_conds, HostDeviceVector<bst_feature_t>* p_fidx,
                   HostDeviceVector<std::uint8_t>* p_dft_left, HostDeviceVector<float>* p_gain,
                   HostDeviceVector<float>* p_sum_hess) {
  namespace tf = tree_field;

  auto get_float = [&](std::string_view name, HostDeviceVector<float>* p_out) {
    auto& values = get<FloatArrayT<typed>>(get<Object const>(in).find(name)->second);
    auto& out = *p_out;
    out.Resize(values.size());
    auto& h_out = out.HostVector();
    for (std::size_t i = 0; i < values.size(); ++i) {
      h_out[i] = GetElem<Number>(values, i);
    }
  };
  get_float(tf::kBaseWeight, p_weights);
  get_float(tf::kLeafWeight, p_leaf_weights);
  get_float(tf::kSplitCond, p_conds);

  auto get_nidx = [&](std::string_view name, HostDeviceVector<bst_node_t>* p_nidx) {
    auto& nidx = get<I32ArrayT<typed>>(get<Object const>(in).find(name)->second);
    auto& out_nidx = p_nidx->HostVector();
    out_nidx.resize(nidx.size());
    for (std::size_t i = 0; i < nidx.size(); ++i) {
      out_nidx[i] = GetElem<Integer>(nidx, i);
    }
  };
  get_nidx(tf::kLeft, p_lefts);
  get_nidx(tf::kRight, p_rights);
  get_nidx(tf::kParent, p_parents);

  auto const& splits = get<IndexArrayT<typed, feature_is_64> const>(in[tf::kSplitIdx]);
  p_fidx->Resize(splits.size());
  auto& out_fidx = p_fidx->HostVector();
  for (std::size_t i = 0; i < splits.size(); ++i) {
    out_fidx[i] = GetElem<Integer>(splits, i);
  }

  auto const& dft_left = get<U8ArrayT<typed> const>(in[tf::kDftLeft]);
  p_dft_left->Resize(dft_left.size());
  auto& out_dft_l = p_dft_left->HostVector();
  for (std::size_t i = 0; i < dft_left.size(); ++i) {
    out_dft_l[i] = GetElem<Boolean>(dft_left, i);
  }

  // Load statistics
  get_float(tf::kLossChg, p_gain);
  get_float(tf::kSumHess, p_sum_hess);
}

void MultiTargetTree::LoadModel(Json const& in) {
  namespace tf = tree_field;
  bool typed = IsA<F32Array>(in[tf::kBaseWeight]);
  bool feature_is_64 = IsA<I64Array>(in[tf::kSplitIdx]);

  if (typed && feature_is_64) {
    LoadModelImpl<true, true>(in, &weights_, &leaf_weights_, &left_, &right_, &parent_,
                              &split_conds_, &split_index_, &default_left_, &gain_, &sum_hess_);
  } else if (typed && !feature_is_64) {
    LoadModelImpl<true, false>(in, &weights_, &leaf_weights_, &left_, &right_, &parent_,
                               &split_conds_, &split_index_, &default_left_, &gain_, &sum_hess_);
  } else if (!typed && feature_is_64) {
    LoadModelImpl<false, true>(in, &weights_, &leaf_weights_, &left_, &right_, &parent_,
                               &split_conds_, &split_index_, &default_left_, &gain_, &sum_hess_);
  } else {
    LoadModelImpl<false, false>(in, &weights_, &leaf_weights_, &left_, &right_, &parent_,
                                &split_conds_, &split_index_, &default_left_, &gain_, &sum_hess_);
  }
}

void MultiTargetTree::SaveModel(Json* p_out) const {
  CHECK(p_out);
  auto& out = *p_out;

  auto n_nodes = param_->num_nodes;

  // nodes
  I32Array lefts(n_nodes);
  I32Array rights(n_nodes);
  I32Array parents(n_nodes);
  F32Array conds(n_nodes);
  U8Array default_left(n_nodes);
  F32Array weights(this->weights_.Size());

  auto n_leaves = this->NumLeaves();
  CHECK_GE(n_leaves, 1);
  F32Array leaf_weights(n_leaves * this->NumTargets());

  auto const& h_left = this->left_.ConstHostVector();
  auto const& h_right = this->right_.ConstHostVector();
  auto const& h_parent = this->parent_.ConstHostVector();
  auto const& h_split_index = this->split_index_.ConstHostVector();
  auto const& h_split_conds = this->split_conds_.ConstHostVector();
  auto const& h_default_left = this->default_left_.ConstHostVector();
  auto save_tree = [&](auto* p_indices_array) {
    auto& indices_array = *p_indices_array;
    for (bst_node_t nidx = 0; nidx < n_nodes; ++nidx) {
      CHECK_LT(nidx, left_.Size());
      lefts.Set(nidx, h_left[nidx]);
      CHECK_LT(nidx, right_.Size());
      rights.Set(nidx, h_right[nidx]);
      CHECK_LT(nidx, parent_.Size());
      parents.Set(nidx, h_parent[nidx]);
      CHECK_LT(nidx, split_index_.Size());
      indices_array.Set(nidx, h_split_index[nidx]);
      conds.Set(nidx, h_split_conds[nidx]);
      default_left.Set(nidx, h_default_left[nidx]);

      // Save internal weights
      auto in_weight = this->NodeWeight(nidx);
      auto weight_out = common::Span<float>(weights.GetArray())
                            .subspan(nidx * this->NumSplitTargets(), this->NumSplitTargets());
      CHECK_EQ(in_weight.Size(), weight_out.size());
      std::copy_n(in_weight.Values().data(), in_weight.Size(), weight_out.data());

      // Save leaf weights
      if (IsLeaf(nidx)) {
        auto in_weight = this->LeafValue(nidx);
        auto leaf_idx = this->LeafIdx(nidx);
        auto weight_out = common::Span<float>(leaf_weights.GetArray())
                              .subspan(leaf_idx * this->NumTargets(), this->NumTargets());
        CHECK_EQ(in_weight.Size(), weight_out.size());
        std::copy_n(in_weight.Values().data(), in_weight.Size(), weight_out.data());
      }
    }
  };

  namespace tf = tree_field;

  if (this->param_->num_feature >
      static_cast<bst_feature_t>(std::numeric_limits<std::int32_t>::max())) {
    I64Array indices_64(n_nodes);
    save_tree(&indices_64);
    out[tf::kSplitIdx] = std::move(indices_64);
  } else {
    I32Array indices_32(n_nodes);
    save_tree(&indices_32);
    out[tf::kSplitIdx] = std::move(indices_32);
  }

  out[tf::kBaseWeight] = std::move(weights);
  out[tf::kLeafWeight] = std::move(leaf_weights);
  out[tf::kLeft] = std::move(lefts);
  out[tf::kRight] = std::move(rights);
  out[tf::kParent] = std::move(parents);

  out[tf::kSplitCond] = std::move(conds);
  out[tf::kDftLeft] = std::move(default_left);

  // Save statistics (gain and sum_hess)
  F32Array gains(n_nodes);
  F32Array sum_hess(n_nodes);
  auto const& h_gain = this->gain_.ConstHostVector();
  auto const& h_sum_hess = this->sum_hess_.ConstHostVector();
  for (bst_node_t nidx = 0; nidx < n_nodes; ++nidx) {
    gains.Set(nidx, h_gain[nidx]);
    sum_hess.Set(nidx, h_sum_hess[nidx]);
  }
  out[tf::kLossChg] = std::move(gains);
  out[tf::kSumHess] = std::move(sum_hess);
}

[[nodiscard]] bst_target_t MultiTargetTree::NumTargets() const { return param_->size_leaf_vector; }
[[nodiscard]] bst_target_t MultiTargetTree::NumSplitTargets() const {
  auto n_targets = this->weights_.Size() / this->left_.Size();
  CHECK_NE(n_targets, 0);
  return n_targets;
}
[[nodiscard]] std::size_t MultiTargetTree::Size() const { return parent_.Size(); }

[[nodiscard]] MultiTargetTree* MultiTargetTree::Copy(TreeParam const* param) const {
  auto ptr = new MultiTargetTree{*this};
  ptr->param_ = param;
  return ptr;
}

[[nodiscard]] std::size_t MultiTargetTree::MemCostBytes() const {
  std::size_t n_bytes = 0;
  n_bytes += left_.SizeBytes();
  n_bytes += right_.SizeBytes();
  n_bytes += parent_.SizeBytes();
  n_bytes += split_index_.SizeBytes();
  n_bytes += default_left_.SizeBytes();
  n_bytes += split_conds_.SizeBytes();
  n_bytes += weights_.SizeBytes();
  n_bytes += leaf_weights_.SizeBytes();
  n_bytes += gain_.SizeBytes();
  n_bytes += sum_hess_.SizeBytes();
  return n_bytes;
}
}  // namespace xgboost
