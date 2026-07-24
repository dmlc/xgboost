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
namespace tree::cuda_impl {
void CopyBatch(Context const* ctx, common::Span<void*> dsts, common::Span<void const*> srcs,
               common::Span<std::size_t const> sizes);
void ApplyLearningRate(Context const* ctx, common::Span<float> weights, float eta);
}  // namespace tree::cuda_impl

namespace {
template <typename T>
struct CopyBatchItem {
  std::size_t dst_offset;
  common::Span<T const> src;

  CopyBatchItem(std::size_t dst_offset, common::Span<T const> src)
      : dst_offset{dst_offset}, src{src} {}
};

template <typename T>
void CopyBatch(Context const* ctx, std::size_t size, std::vector<CopyBatchItem<T>> const& copies,
               HostDeviceVector<T>* out) {
  out->SetDevice(ctx->Device());
#if defined(XGBOOST_USE_CUDA)
  if (ctx->IsCUDA()) {
    (void)out->DeviceSpan();
    out->Resize(size);
    auto dst = out->DeviceSpan();
    std::vector<void*> dsts(copies.size());
    std::vector<void const*> srcs(copies.size());
    std::vector<std::size_t> sizes(copies.size());
    for (std::size_t i = 0; i < copies.size(); ++i) {
      dsts[i] = dst.data() + copies[i].dst_offset;
      srcs[i] = copies[i].src.data();
      sizes[i] = copies[i].src.size_bytes();
    }
    tree::cuda_impl::CopyBatch(ctx, dsts, srcs, sizes);
    return;
  }
#endif  // defined(XGBOOST_USE_CUDA)

  out->Resize(size);
  auto dst = out->HostSpan();
  for (auto const& copy : copies) {
    std::copy(copy.src.cbegin(), copy.src.cend(), dst.begin() + copy.dst_offset);
  }
}

void ApplyLearningRate(Context const* ctx, std::size_t offset, std::size_t size, float eta,
                       HostDeviceVector<float>* values) {
  values->SetDevice(ctx->Device());
#if defined(XGBOOST_USE_CUDA)
  if (ctx->IsCUDA()) {
    tree::cuda_impl::ApplyLearningRate(ctx, values->DeviceSpan().subspan(offset, size), eta);
    return;
  }
#endif  // defined(XGBOOST_USE_CUDA)

  auto out = values->HostSpan().subspan(offset, size);
  std::transform(out.cbegin(), out.cend(), out.begin(),
                 [eta](float weight) { return weight * eta; });
}
}  // namespace

namespace tree {
void CopyCategoryStorage(Context const* ctx, std::size_t offset, ExpandBatch const& batch,
                         HostDeviceVector<CatWordT>* out) {
  std::vector<CopyBatchItem<CatWordT>> copies;
  for (auto cats : batch.cat_bits) {
    if (!cats.empty()) {
      copies.emplace_back(offset, cats);
      offset += cats.size();
    }
  }
  CopyBatch(ctx, offset, copies, out);
}
}  // namespace tree

MultiTargetTree::MultiTargetTree(TreeParam const* param)
    : param_{param},
      left_(1ul, InvalidNodeId()),
      right_(1ul, InvalidNodeId()),
      parent_(1ul, InvalidNodeId()),
      split_index_(1ul, 0),
      default_left_(1ul, 0),
      split_conds_(1ul, DftBadValue()),
      loss_chg_(1ul, 0.0f),
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
      loss_chg_(that.loss_chg_.Size(), 0.0f, that.loss_chg_.Device()),
      sum_hess_(that.sum_hess_.Size(), 0.0f, that.sum_hess_.Device()) {
  this->left_.Copy(that.left_);
  this->right_.Copy(that.right_);
  this->parent_.Copy(that.parent_);
  this->split_index_.Copy(that.split_index_);
  this->default_left_.Copy(that.default_left_);
  this->split_conds_.Copy(that.split_conds_);
  this->weights_.Copy(that.weights_);
  this->leaf_weights_.Copy(that.leaf_weights_);
  this->loss_chg_.Copy(that.loss_chg_);
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
  loss_chg_.Resize(next_nidx, 0.0f);

  CHECK_EQ(this->param_->num_nodes, 1);
  CHECK_EQ(this->NumSplitTargets(), weight.Size());
}

void MultiTargetTree::Expand(Context const* ctx, tree::ExpandBatch const& batch) {
  auto const batch_size = batch.Size();
  auto const n_split_targets = this->NumSplitTargets();
  auto const old_n_nodes = this->Size();
  auto const n_nodes = old_n_nodes + batch_size * 2;
  left_.Resize(n_nodes, InvalidNodeId());
  right_.Resize(n_nodes, InvalidNodeId());
  parent_.Resize(n_nodes, InvalidNodeId());
  split_index_.Resize(n_nodes);
  split_conds_.Resize(n_nodes, DftBadValue());
  default_left_.Resize(n_nodes);

  auto h_left = left_.HostSpan();
  auto h_right = right_.HostSpan();
  auto h_parent = parent_.HostSpan();
  auto h_split_index = split_index_.HostSpan();
  auto h_split_conds = split_conds_.HostSpan();
  auto h_default_left = default_left_.HostSpan();
  for (std::size_t i = 0; i < batch_size; ++i) {
    auto const nidx = batch.nidxs[i];
    h_left[nidx] = static_cast<bst_node_t>(old_n_nodes + i * 2);
    h_right[nidx] = h_left[nidx] + 1;
    h_parent[h_left[nidx]] = nidx;
    h_parent[h_right[nidx]] = nidx;
    h_split_index[nidx] = batch.fidxs[i];
    h_split_conds[nidx] = batch.cat_bits[i].empty() ? batch.conds[i] : DftBadValue();
    h_default_left[nidx] = batch.dft_lefts[i];
  }

  std::vector<CopyBatchItem<float>> weight_copies;
  for (std::size_t i = 0; i < batch_size; ++i) {
    auto const nidx = batch.nidxs[i];
    weight_copies.emplace_back(nidx * n_split_targets, batch.base_weight_batch[i]);
    weight_copies.emplace_back(h_left[nidx] * n_split_targets, batch.left_weight_batch[i]);
    weight_copies.emplace_back(h_right[nidx] * n_split_targets, batch.right_weight_batch[i]);
  }
  CopyBatch(ctx, n_nodes * n_split_targets, weight_copies, &weights_);
  auto const n_child_weights = batch_size * 2 * n_split_targets;
  ApplyLearningRate(ctx, old_n_nodes * n_split_targets, n_child_weights, batch.eta, &weights_);

  loss_chg_.Resize(n_nodes, 0.0f);
  sum_hess_.Resize(n_nodes, 0.0f);
  auto h_loss_chg = loss_chg_.HostSpan();
  auto h_sum_hess = sum_hess_.HostSpan();
  for (std::size_t i = 0; i < batch_size; ++i) {
    auto const nidx = batch.nidxs[i];
    h_loss_chg[nidx] = batch.loss_chgs[i];
    h_sum_hess[nidx] = batch.left_sums[i] + batch.right_sums[i];
    h_sum_hess[h_left[nidx]] = batch.left_sums[i];
    h_sum_hess[h_right[nidx]] = batch.right_sums[i];
  }
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
                              &split_conds_, &split_index_, &default_left_, &loss_chg_, &sum_hess_);
  } else if (typed && !feature_is_64) {
    LoadModelImpl<true, false>(in, &weights_, &leaf_weights_, &left_, &right_, &parent_,
                               &split_conds_, &split_index_, &default_left_, &loss_chg_,
                               &sum_hess_);
  } else if (!typed && feature_is_64) {
    LoadModelImpl<false, true>(in, &weights_, &leaf_weights_, &left_, &right_, &parent_,
                               &split_conds_, &split_index_, &default_left_, &loss_chg_,
                               &sum_hess_);
  } else {
    LoadModelImpl<false, false>(in, &weights_, &leaf_weights_, &left_, &right_, &parent_,
                                &split_conds_, &split_index_, &default_left_, &loss_chg_,
                                &sum_hess_);
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
  F32Array loss_chg(n_nodes);
  F32Array sum_hess(n_nodes);

  auto n_leaves = this->NumLeaves();
  CHECK_GE(n_leaves, 1);
  F32Array leaf_weights(n_leaves * this->NumTargets());

  auto const& h_left = this->left_.ConstHostVector();
  auto const& h_right = this->right_.ConstHostVector();
  auto const& h_parent = this->parent_.ConstHostVector();
  auto const& h_split_index = this->split_index_.ConstHostVector();
  auto const& h_split_conds = this->split_conds_.ConstHostVector();
  auto const& h_default_left = this->default_left_.ConstHostVector();
  auto const& h_loss_chg = this->loss_chg_.ConstHostVector();
  auto const& h_sum_hess = this->sum_hess_.ConstHostVector();

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
      loss_chg.Set(nidx, h_loss_chg[nidx]);
      sum_hess.Set(nidx, h_sum_hess[nidx]);

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
  out[tf::kLossChg] = std::move(loss_chg);
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
  n_bytes += loss_chg_.SizeBytes();
  n_bytes += sum_hess_.SizeBytes();
  return n_bytes;
}
}  // namespace xgboost
