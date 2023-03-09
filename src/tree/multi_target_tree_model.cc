/**
 * Copyright 2023 by XGBoost Contributors
 */
#include "xgboost/multi_target_tree_model.h"

#include <algorithm>             // for copy_n
#include <cstddef>               // for size_t
#include <cstdint>               // for int32_t, uint8_t
#include <limits>                // for numeric_limits
#include <string_view>           // for string_view
#include <utility>               // for move
#include <vector>                // for vector

#include "io_utils.h"            // for I32ArrayT, FloatArrayT, GetElem, ...
#include "xgboost/base.h"        // for bst_node_t, bst_feature_t, bst_target_t
#include "xgboost/json.h"        // for Json, get, Object, Number, Integer, ...
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

template <bool typed, bool feature_is_64>
void LoadModelImpl(Json const& in, std::vector<float>* p_weights, std::vector<bst_node_t>* p_lefts,
                   std::vector<bst_node_t>* p_rights, std::vector<bst_node_t>* p_parents,
                   std::vector<float>* p_conds, std::vector<bst_feature_t>* p_fidx,
                   std::vector<std::uint8_t>* p_dft_left) {
  namespace tf = tree_field;

  auto get_float = [&](std::string_view name, std::vector<float>* p_out) {
    auto& values = get<FloatArrayT<typed>>(get<Object const>(in).find(name)->second);
    auto& out = *p_out;
    out.resize(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
      out[i] = GetElem<Number>(values, i);
    }
  };
  get_float(tf::kBaseWeight, p_weights);
  get_float(tf::kSplitCond, p_conds);

  auto get_nidx = [&](std::string_view name, std::vector<bst_node_t>* p_nidx) {
    auto& nidx = get<I32ArrayT<typed>>(get<Object const>(in).find(name)->second);
    auto& out_nidx = *p_nidx;
    out_nidx.resize(nidx.size());
    for (std::size_t i = 0; i < nidx.size(); ++i) {
      out_nidx[i] = GetElem<Integer>(nidx, i);
    }
  };
  get_nidx(tf::kLeft, p_lefts);
  get_nidx(tf::kRight, p_rights);
  get_nidx(tf::kParent, p_parents);

  auto const& splits = get<IndexArrayT<typed, feature_is_64> const>(in[tf::kSplitIdx]);
  p_fidx->resize(splits.size());
  auto& out_fidx = *p_fidx;
  for (std::size_t i = 0; i < splits.size(); ++i) {
    out_fidx[i] = GetElem<Integer>(splits, i);
  }

  auto const& dft_left = get<U8ArrayT<typed> const>(in[tf::kDftLeft]);
  auto& out_dft_l = *p_dft_left;
  out_dft_l.resize(dft_left.size());
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

  auto save_tree = [&](auto* p_indices_array) {
    auto& indices_array = *p_indices_array;
    for (bst_node_t nidx = 0; nidx < n_nodes; ++nidx) {
      CHECK_LT(nidx, left_.size());
      lefts.Set(nidx, left_[nidx]);
      CHECK_LT(nidx, right_.size());
      rights.Set(nidx, right_[nidx]);
      CHECK_LT(nidx, parent_.size());
      parents.Set(nidx, parent_[nidx]);
      CHECK_LT(nidx, split_index_.size());
      indices_array.Set(nidx, split_index_[nidx]);
      conds.Set(nidx, split_conds_[nidx]);
      default_left.Set(nidx, default_left_[nidx]);

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
  CHECK_GE(weights_.size(), next_nidx * weight.Size());
  auto out_weight = common::Span<float>(weights_).subspan(nidx * weight.Size(), weight.Size());
  for (std::size_t i = 0; i < weight.Size(); ++i) {
    out_weight[i] = weight(i);
  }
}

void MultiTargetTree::Expand(bst_node_t nidx, bst_feature_t split_idx, float split_cond,
                             bool default_left, linalg::VectorView<float const> base_weight,
                             linalg::VectorView<float const> left_weight,
                             linalg::VectorView<float const> right_weight) {
  CHECK(this->IsLeaf(nidx));
  CHECK_GE(parent_.size(), 1);
  CHECK_EQ(parent_.size(), left_.size());
  CHECK_EQ(left_.size(), right_.size());

  std::size_t n = param_->num_nodes + 2;
  CHECK_LT(split_idx, this->param_->num_feature);
  left_.resize(n, InvalidNodeId());
  right_.resize(n, InvalidNodeId());
  parent_.resize(n, InvalidNodeId());

  auto left_child = parent_.size() - 2;
  auto right_child = parent_.size() - 1;

  left_[nidx] = left_child;
  right_[nidx] = right_child;

  if (nidx != 0) {
    CHECK_NE(parent_[nidx], InvalidNodeId());
  }

  parent_[left_child] = nidx;
  parent_[right_child] = nidx;

  split_index_.resize(n);
  split_index_[nidx] = split_idx;

  split_conds_.resize(n);
  split_conds_[nidx] = split_cond;
  default_left_.resize(n);
  default_left_[nidx] = static_cast<std::uint8_t>(default_left);

  weights_.resize(n * this->NumTarget());
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
std::size_t MultiTargetTree::Size() const { return parent_.size(); }
}  // namespace xgboost
