/*!
 * Copyright 2022 by XGBoost Contributors
 */
#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "../collective/communicator-inl.h"
#include "../common/common.h"
#include "xgboost/generic_parameters.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/tree_model.h"

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

inline void UpdateLeafValues(std::vector<float>* p_quantiles, std::vector<bst_node_t> const nidx,
                             RegTree* p_tree) {
  auto& tree = *p_tree;
  auto& quantiles = *p_quantiles;
  auto const& h_node_idx = nidx;

  size_t n_leaf{h_node_idx.size()};
  collective::Allreduce<collective::Operation::kMax>(&n_leaf, 1);
  CHECK(quantiles.empty() || quantiles.size() == n_leaf);
  if (quantiles.empty()) {
    quantiles.resize(n_leaf, std::numeric_limits<float>::quiet_NaN());
  }

  // number of workers that have valid quantiles
  std::vector<int32_t> n_valids(quantiles.size());
  std::transform(quantiles.cbegin(), quantiles.cend(), n_valids.begin(),
                 [](float q) { return static_cast<int32_t>(!std::isnan(q)); });
  collective::Allreduce<collective::Operation::kSum>(n_valids.data(), n_valids.size());
  // convert to 0 for all reduce
  std::replace_if(
      quantiles.begin(), quantiles.end(), [](float q) { return std::isnan(q); }, 0.f);
  // use the mean value
  collective::Allreduce<collective::Operation::kSum>(quantiles.data(), quantiles.size());
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
    tree[nidx].SetLeaf(q);
  }
}

void UpdateTreeLeafDevice(Context const* ctx, common::Span<bst_node_t const> position,
                          MetaInfo const& info, HostDeviceVector<float> const& predt, float alpha,
                          RegTree* p_tree);

void UpdateTreeLeafHost(Context const* ctx, std::vector<bst_node_t> const& position,
                        MetaInfo const& info, HostDeviceVector<float> const& predt, float alpha,
                        RegTree* p_tree);
}  // namespace detail
}  // namespace obj
}  // namespace xgboost
