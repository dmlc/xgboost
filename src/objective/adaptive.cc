/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include "adaptive.h"

#include <limits>
#include <vector>

#include "../common/common.h"
#include "../common/numeric.h"
#include "../common/stats.h"
#include "../common/threading_utils.h"
#include "xgboost/tree_model.h"

namespace xgboost {
namespace obj {
namespace detail {
void EncodeTreeLeafHost(RegTree const& tree, std::vector<bst_node_t> const& position,
                        std::vector<size_t>* p_nptr, std::vector<bst_node_t>* p_nidx,
                        std::vector<size_t>* p_ridx) {
  auto& nptr = *p_nptr;
  auto& nidx = *p_nidx;
  auto& ridx = *p_ridx;
  ridx = common::ArgSort<size_t>(position);
  std::vector<bst_node_t> sorted_pos(position);
  // permutation
  for (size_t i = 0; i < position.size(); ++i) {
    sorted_pos[i] = position[ridx[i]];
  }
  // find the first non-sampled row
  size_t begin_pos =
      std::distance(sorted_pos.cbegin(), std::find_if(sorted_pos.cbegin(), sorted_pos.cend(),
                                                      [](bst_node_t nidx) { return nidx >= 0; }));
  CHECK_LE(begin_pos, sorted_pos.size());

  std::vector<bst_node_t> leaf;
  tree.WalkTree([&](bst_node_t nidx) {
    if (tree[nidx].IsLeaf()) {
      leaf.push_back(nidx);
    }
    return true;
  });

  if (begin_pos == sorted_pos.size()) {
    nidx = leaf;
    return;
  }

  auto beg_it = sorted_pos.begin() + begin_pos;
  common::RunLengthEncode(beg_it, sorted_pos.end(), &nptr);
  CHECK_GT(nptr.size(), 0);
  // skip the sampled rows in indptr
  std::transform(nptr.begin(), nptr.end(), nptr.begin(),
                 [begin_pos](size_t ptr) { return ptr + begin_pos; });

  size_t n_leaf = nptr.size() - 1;
  auto n_unique = std::unique(beg_it, sorted_pos.end()) - beg_it;
  CHECK_EQ(n_unique, n_leaf);
  nidx.resize(n_leaf);
  std::copy(beg_it, beg_it + n_unique, nidx.begin());

  if (n_leaf != leaf.size()) {
    FillMissingLeaf(leaf, &nidx, &nptr);
  }
}

void UpdateTreeLeafHost(Context const* ctx, std::vector<bst_node_t> const& position,
                        MetaInfo const& info, HostDeviceVector<float> const& predt, float alpha,
                        RegTree* p_tree) {
  auto& tree = *p_tree;

  std::vector<bst_node_t> nidx;
  std::vector<size_t> nptr;
  std::vector<size_t> ridx;
  EncodeTreeLeafHost(*p_tree, position, &nptr, &nidx, &ridx);
  size_t n_leaf = nidx.size();
  if (nptr.empty()) {
    std::vector<float> quantiles;
    UpdateLeafValues(&quantiles, nidx, p_tree);
    return;
  }

  CHECK(!position.empty());
  std::vector<float> quantiles(n_leaf, 0);
  std::vector<int32_t> n_valids(n_leaf, 0);

  auto const& h_node_idx = nidx;
  auto const& h_node_ptr = nptr;
  CHECK_LE(h_node_ptr.back(), info.num_row_);
  // loop over each leaf
  common::ParallelFor(quantiles.size(), ctx->Threads(), [&](size_t k) {
    auto nidx = h_node_idx[k];
    CHECK(tree[nidx].IsLeaf());
    CHECK_LT(k + 1, h_node_ptr.size());
    size_t n = h_node_ptr[k + 1] - h_node_ptr[k];
    auto h_row_set = common::Span<size_t const>{ridx}.subspan(h_node_ptr[k], n);
    // multi-target not yet supported.
    auto h_labels = info.labels.HostView().Slice(linalg::All(), 0);
    auto const& h_predt = predt.ConstHostVector();
    auto h_weights = linalg::MakeVec(&info.weights_);

    auto iter = common::MakeIndexTransformIter([&](size_t i) -> float {
      auto row_idx = h_row_set[i];
      return h_labels(row_idx) - h_predt[row_idx];
    });
    auto w_it = common::MakeIndexTransformIter([&](size_t i) -> float {
      auto row_idx = h_row_set[i];
      return h_weights(row_idx);
    });

    float q{0};
    if (info.weights_.Empty()) {
      q = common::Quantile(alpha, iter, iter + h_row_set.size());
    } else {
      q = common::WeightedQuantile(alpha, iter, iter + h_row_set.size(), w_it);
    }
    if (std::isnan(q)) {
      CHECK(h_row_set.empty());
    }
    quantiles.at(k) = q;
  });

  UpdateLeafValues(&quantiles, nidx, p_tree);
}
}  // namespace detail
}  // namespace obj
}  // namespace xgboost
