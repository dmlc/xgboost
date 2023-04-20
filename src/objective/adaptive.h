/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#pragma once

#include <algorithm>
#include <cstdint>  // std::int32_t
#include <limits>
#include <vector>   // std::vector

#include "../collective/aggregator.h"
#include "../collective/communicator-inl.h"
#include "../common/common.h"
#include "xgboost/base.h"                // bst_node_t
#include "xgboost/context.h"             // Context
#include "xgboost/data.h"                // MetaInfo
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/tree_model.h"          // RegTree

namespace xgboost {
namespace obj {
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

inline void UpdateLeafValues(std::vector<float>* p_quantiles, std::vector<bst_node_t> const& nidx,
                             MetaInfo const& info, float learning_rate, RegTree* p_tree) {
  auto& tree = *p_tree;
  auto& quantiles = *p_quantiles;
  auto const& h_node_idx = nidx;

  size_t n_leaf = collective::GlobalMax(info, h_node_idx.size());
  CHECK(quantiles.empty() || quantiles.size() == n_leaf);
  if (quantiles.empty()) {
    quantiles.resize(n_leaf, std::numeric_limits<float>::quiet_NaN());
  }

  // number of workers that have valid quantiles
  std::vector<int32_t> n_valids(quantiles.size());
  std::transform(quantiles.cbegin(), quantiles.cend(), n_valids.begin(),
                 [](float q) { return static_cast<int32_t>(!std::isnan(q)); });
  collective::GlobalSum(info, &n_valids);
  // convert to 0 for all reduce
  std::replace_if(
      quantiles.begin(), quantiles.end(), [](float q) { return std::isnan(q); }, 0.f);
  // use the mean value
  collective::GlobalSum(info, &quantiles);
  for (size_t i = 0; i < n_leaf; ++i) {
    if (n_valids[i] > 0) {
      quantiles[i] /= static_cast<float>(n_valids[i]);
    } else {
      // Use original leaf value if no worker can provide the quantile.
      quantiles[i] = tree[h_node_idx[i]].LeafValue();
    }
  }

  for (size_t i = 0; i < nidx.size(); ++i) {
    auto nidx = h_node_idx[i];
    auto q = quantiles[i];
    CHECK(tree[nidx].IsLeaf());
    tree[nidx].SetLeaf(q * learning_rate);
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

void UpdateTreeLeafDevice(Context const* ctx, common::Span<bst_node_t const> position,
                          std::int32_t group_idx, MetaInfo const& info, float learning_rate,
                          HostDeviceVector<float> const& predt, float alpha, RegTree* p_tree);

void UpdateTreeLeafHost(Context const* ctx, std::vector<bst_node_t> const& position,
                        std::int32_t group_idx, MetaInfo const& info, float learning_rate,
                        HostDeviceVector<float> const& predt, float alpha, RegTree* p_tree);
}  // namespace detail

inline void UpdateTreeLeaf(Context const* ctx, HostDeviceVector<bst_node_t> const& position,
                           std::int32_t group_idx, MetaInfo const& info, float learning_rate,
                           HostDeviceVector<float> const& predt, float alpha, RegTree* p_tree) {
  if (ctx->IsCPU()) {
    detail::UpdateTreeLeafHost(ctx, position.ConstHostVector(), group_idx, info, learning_rate,
                               predt, alpha, p_tree);
  } else {
    position.SetDevice(ctx->gpu_id);
    detail::UpdateTreeLeafDevice(ctx, position.ConstDeviceSpan(), group_idx, info, learning_rate,
                                 predt, alpha, p_tree);
  }
}
}  // namespace obj
}  // namespace xgboost
