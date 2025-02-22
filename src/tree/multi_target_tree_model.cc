/**
 * Copyright 2023-2025, XGBoost Contributors
 */
#include "xgboost/multi_target_tree_model.h"

#include <algorithm>    // for copy_n
#include <cstddef>      // for size_t
#include <cstdint>      // for int32_t, uint8_t
#include <limits>       // for numeric_limits
#include <string_view>  // for string_view
#include <utility>      // for move
#include <vector>       // for vector

#include "io_utils.h"      // for I32ArrayT, FloatArrayT, GetElem, ...
#include "xgboost/base.h"  // for bst_node_t, bst_feature_t, bst_target_t
#include "xgboost/json.h"  // for Json, get, Object, Number, Integer, ...
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
      split_conds_(1ul, std::numeric_limits<float>::quiet_NaN()),
      weights_(param->size_leaf_vector, std::numeric_limits<float>::quiet_NaN()) {
  CHECK_GT(param_->size_leaf_vector, 1);
}

MultiTargetTree::MultiTargetTree(MultiTargetTree const& that)
    : param_{that.param_},
      left_(that.left_.Size(), 0, that.left_.Device()),
      right_(that.right_.Size(), 0, that.right_.Device()),
      parent_(that.parent_.Size(), 0, that.parent_.Device()),
      split_index_(that.split_index_.Size(), 0, that.split_index_.Device()),
      default_left_(that.default_left_.Size(), 0, that.default_left_.Device()),
      split_conds_(that.split_conds_.Size(), 0, that.split_conds_.Device()),
      weights_(that.weights_.Size(), 0, that.weights_.Device()) {
  this->left_.Copy(that.left_);
  this->right_.Copy(that.right_);
  this->parent_.Copy(that.parent_);
  this->split_index_.Copy(that.split_index_);
  this->default_left_.Copy(that.default_left_);
  this->split_conds_.Copy(that.split_conds_);
  this->weights_.Copy(that.weights_);
}

template <bool typed, bool feature_is_64>
void LoadModelImpl(Json const& in, HostDeviceVector<float>* p_weights,
                   HostDeviceVector<bst_node_t>* p_lefts, HostDeviceVector<bst_node_t>* p_rights,
                   HostDeviceVector<bst_node_t>* p_parents, HostDeviceVector<float>* p_conds,
                   HostDeviceVector<bst_feature_t>* p_fidx,
                   HostDeviceVector<std::uint8_t>* p_dft_left) {
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
}

void MultiTargetTree::LoadModel(Json const& in) {
  namespace tf = tree_field;
  bool typed = IsA<F32Array>(in[tf::kBaseWeight]);
  bool feature_is_64 = IsA<I64Array>(in[tf::kSplitIdx]);

  if (typed && feature_is_64) {
    LoadModelImpl<true, true>(in, &weights_, &left_, &right_, &parent_, &split_conds_,
                              &split_index_, &default_left_);
  } else if (typed && !feature_is_64) {
    LoadModelImpl<true, false>(in, &weights_, &left_, &right_, &parent_, &split_conds_,
                               &split_index_, &default_left_);
  } else if (!typed && feature_is_64) {
    LoadModelImpl<false, true>(in, &weights_, &left_, &right_, &parent_, &split_conds_,
                               &split_index_, &default_left_);
  } else {
    LoadModelImpl<false, false>(in, &weights_, &left_, &right_, &parent_, &split_conds_,
                                &split_index_, &default_left_);
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
  F32Array weights(n_nodes * this->NumTarget());

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

      auto in_weight = this->NodeWeight(nidx);
      auto weight_out = common::Span<float>(weights.GetArray())
                            .subspan(nidx * this->NumTarget(), this->NumTarget());
      CHECK_EQ(in_weight.Size(), weight_out.size());
      std::copy_n(in_weight.Values().data(), in_weight.Size(), weight_out.data());
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
  out[tf::kLeft] = std::move(lefts);
  out[tf::kRight] = std::move(rights);
  out[tf::kParent] = std::move(parents);

  out[tf::kSplitCond] = std::move(conds);
  out[tf::kDftLeft] = std::move(default_left);
}

void MultiTargetTree::SetLeaf(bst_node_t nidx, linalg::VectorView<float const> weight) {
  CHECK(this->IsLeaf(nidx)) << "Collapsing a split node to leaf " << MTNotImplemented();
  auto const next_nidx = nidx + 1;
  CHECK_EQ(weight.Size(), this->NumTarget());
  CHECK_GE(weights_.Size(), next_nidx * weight.Size());
  auto out_weight = weights_.HostSpan().subspan(nidx * weight.Size(), weight.Size());
  for (std::size_t i = 0; i < weight.Size(); ++i) {
    out_weight[i] = weight(i);
  }
}

void MultiTargetTree::Expand(bst_node_t nidx, bst_feature_t split_idx, float split_cond,
                             bool default_left, linalg::VectorView<float const> base_weight,
                             linalg::VectorView<float const> left_weight,
                             linalg::VectorView<float const> right_weight) {
  CHECK(this->IsLeaf(nidx));
  CHECK_GE(parent_.Size(), 1);
  CHECK_EQ(parent_.Size(), left_.Size());
  CHECK_EQ(left_.Size(), right_.Size());

  std::size_t n = param_->num_nodes + 2;
  CHECK_LT(split_idx, this->param_->num_feature);
  left_.Resize(n, InvalidNodeId());
  right_.Resize(n, InvalidNodeId());
  parent_.Resize(n, InvalidNodeId());

  auto left_child = parent_.Size() - 2;
  auto right_child = parent_.Size() - 1;

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

  split_conds_.Resize(n, std::numeric_limits<float>::quiet_NaN());
  split_conds_.HostVector()[nidx] = split_cond;

  default_left_.Resize(n);
  default_left_.HostVector()[nidx] = static_cast<std::uint8_t>(default_left);

  weights_.Resize(n * this->NumTarget());
  auto p_weight = this->NodeWeight(nidx);
  CHECK_EQ(p_weight.Size(), base_weight.Size());
  auto l_weight = this->NodeWeight(left_child);
  CHECK_EQ(l_weight.Size(), left_weight.Size());
  auto r_weight = this->NodeWeight(right_child);
  CHECK_EQ(r_weight.Size(), right_weight.Size());

  for (std::size_t i = 0; i < base_weight.Size(); ++i) {
    p_weight(i) = base_weight(i);
    l_weight(i) = left_weight(i);
    r_weight(i) = right_weight(i);
  }
}

bst_target_t MultiTargetTree::NumTarget() const { return param_->size_leaf_vector; }
std::size_t MultiTargetTree::Size() const { return parent_.Size(); }
}  // namespace xgboost
