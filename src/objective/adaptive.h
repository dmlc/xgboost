/**
 * Copyright 2022-2026, XGBoost Contributors
 */
#pragma once

#include <algorithm>
#include <cstdint>  // std::int32_t
#include <limits>
#include <vector>  // std::vector

#include "../collective/aggregator.h"
#include "xgboost/base.h"                // for bst_node_t
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // MetaInfo
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/tree_model.h"          // RegTree

namespace xgboost::obj {
namespace detail {
inline void FillMissingLeaf(std::vector<bst_node_t> const& maybe_missing,
                            std::vector<bst_node_t>* p_nidx, std::vector<size_t>* p_nptr) {
  auto& h_node_idx = *p_nidx;
  auto& h_node_ptr = *p_nptr;

  for (auto leaf : maybe_missing) {
    if (std::binary_search(h_node_idx.cbegin(), h_node_idx.cend(), leaf)) {
      continue;
    }
    auto it = std::upper_bound(h_node_idx.cbegin(), h_node_idx.cend(), leaf);
    auto pos = it - h_node_idx.cbegin();
    h_node_idx.insert(h_node_idx.cbegin() + pos, leaf);
    h_node_ptr.insert(h_node_ptr.cbegin() + pos, h_node_ptr[pos]);
  }
}

inline void UpdateLeafValues(Context const* ctx, std::vector<float>* p_quantiles,
                             std::vector<bst_node_t> const& nidx, MetaInfo const& info,
                             float learning_rate, RegTree* p_tree) {
  auto& tree = *p_tree;
  auto& quantiles = *p_quantiles;
  auto const& h_node_idx = nidx;
  auto n_targets = tree.IsMultiTarget() ? tree.NumTargets() : 1;

  bst_idx_t n_leaf = collective::GlobalMax(ctx, info, static_cast<bst_idx_t>(h_node_idx.size()));
  auto n_values = n_leaf * n_targets;
  CHECK(quantiles.empty() || quantiles.size() == n_values);
  if (quantiles.empty()) {
    quantiles.resize(n_values, std::numeric_limits<float>::quiet_NaN());
  }

  // number of workers that have valid quantiles
  std::vector<int32_t> n_valids(quantiles.size());
  std::transform(quantiles.cbegin(), quantiles.cend(), n_valids.begin(),
                 [](float q) { return static_cast<int32_t>(!std::isnan(q)); });
  auto rc = collective::GlobalSum(ctx, info, linalg::MakeVec(n_valids.data(), n_valids.size()));
  collective::SafeColl(rc);

  // convert to 0 for all reduce
  std::replace_if(quantiles.begin(), quantiles.end(), [](float q) { return std::isnan(q); }, 0.f);
  // use the mean value
  rc = collective::GlobalSum(ctx, info, linalg::MakeVec(quantiles.data(), quantiles.size()));
  collective::SafeColl(rc);

  for (size_t i = 0; i < n_leaf; ++i) {
    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto idx = i * n_targets + t;
      if (n_valids[idx] > 0) {
        quantiles[idx] = quantiles[idx] / static_cast<float>(n_valids[idx]) * learning_rate;
      } else {
        // Use original leaf value if no worker can provide the quantile.
        if (tree.IsMultiTarget()) {
          quantiles[idx] = tree.GetMultiTargetTree()->LeafValue(h_node_idx[i])(t);
        } else {
          quantiles[idx] = tree[h_node_idx[i]].LeafValue() * learning_rate;
        }
      }
    }
  }

  if (tree.IsMultiTarget()) {
    tree.SetLeaves(h_node_idx, common::Span{quantiles});
  } else {
    for (size_t i = 0; i < nidx.size(); ++i) {
      auto nidx = h_node_idx[i];
      auto q = quantiles[i];
      CHECK(tree[nidx].IsLeaf());
      tree[nidx].SetLeaf(q);
    }
  }
}

inline std::size_t IdxY(MetaInfo const& info, bst_group_t group_idx) {
  std::size_t y_idx{0};
  if (info.labels.Shape(1) > 1) {
    y_idx = group_idx;
  }
  CHECK_LE(y_idx, info.labels.Shape(1));
  return y_idx;
}
}  // namespace detail

namespace cpu_impl {
void UpdateTreeLeaf(Context const* ctx, std::vector<bst_node_t> const& position,
                    bst_target_t group_idx, MetaInfo const& info, float learning_rate,
                    HostDeviceVector<float> const& predt, std::vector<float> const& alphas,
                    RegTree* p_tree);
}

namespace cuda_impl {
void UpdateTreeLeaf(Context const* ctx, common::Span<bst_node_t const> position,
                    bst_target_t group_idx, MetaInfo const& info, float learning_rate,
                    HostDeviceVector<float> const& predt, std::vector<float> const& alphas,
                    RegTree* p_tree);
}

inline void UpdateTreeLeaf(Context const* ctx, HostDeviceVector<bst_node_t> const& position,
                           bst_target_t group_idx, MetaInfo const& info, float learning_rate,
                           HostDeviceVector<float> const& predt, std::vector<float> const& alphas,
                           RegTree* p_tree) {
  if (ctx->IsCUDA()) {
    position.SetDevice(ctx->Device());
    cuda_impl::UpdateTreeLeaf(ctx, position.ConstDeviceSpan(), group_idx, info, learning_rate,
                              predt, alphas, p_tree);
  } else {
    cpu_impl::UpdateTreeLeaf(ctx, position.ConstHostVector(), group_idx, info, learning_rate, predt,
                             alphas, p_tree);
  }
}
}  // namespace xgboost::obj
