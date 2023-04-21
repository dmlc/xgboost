/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#include "adaptive.h"

#include <algorithm>                       // std::transform,std::find_if,std::copy,std::unique
#include <cmath>                           // std::isnan
#include <cstddef>                         // std::size_t
#include <iterator>                        // std::distance
#include <vector>                          // std::vector

#include "../common/algorithm.h"           // ArgSort
#include "../common/common.h"              // AssertGPUSupport
#include "../common/numeric.h"             // RunLengthEncode
#include "../common/stats.h"               // Quantile,WeightedQuantile
#include "../common/threading_utils.h"     // ParallelFor
#include "../common/transform_iterator.h"  // MakeIndexTransformIter
#include "xgboost/base.h"                  // bst_node_t
#include "xgboost/context.h"               // Context
#include "xgboost/data.h"                  // MetaInfo
#include "xgboost/host_device_vector.h"    // HostDeviceVector
#include "xgboost/linalg.h"                // MakeTensorView
#include "xgboost/span.h"                  // Span
#include "xgboost/tree_model.h"            // RegTree

namespace xgboost::obj::detail {
void EncodeTreeLeafHost(Context const* ctx, RegTree const& tree,
                        std::vector<bst_node_t> const& position, std::vector<size_t>* p_nptr,
                        std::vector<bst_node_t>* p_nidx, std::vector<size_t>* p_ridx) {
  auto& nptr = *p_nptr;
  auto& nidx = *p_nidx;
  auto& ridx = *p_ridx;
  ridx = common::ArgSort<size_t>(ctx, position.cbegin(), position.cend());
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
                        std::int32_t group_idx, MetaInfo const& info, float learning_rate,
                        HostDeviceVector<float> const& predt, float alpha, RegTree* p_tree) {
  auto& tree = *p_tree;

  std::vector<bst_node_t> nidx;
  std::vector<size_t> nptr;
  std::vector<size_t> ridx;
  EncodeTreeLeafHost(ctx, *p_tree, position, &nptr, &nidx, &ridx);
  size_t n_leaf = nidx.size();
  if (nptr.empty()) {
    std::vector<float> quantiles;
    UpdateLeafValues(&quantiles, nidx, info, learning_rate, p_tree);
    return;
  }

  CHECK(!position.empty());
  std::vector<float> quantiles(n_leaf, 0);
  std::vector<int32_t> n_valids(n_leaf, 0);

  auto const& h_node_idx = nidx;
  auto const& h_node_ptr = nptr;
  CHECK_LE(h_node_ptr.back(), info.num_row_);
  auto h_predt = linalg::MakeTensorView(ctx, predt.ConstHostSpan(), info.num_row_,
                                        predt.Size() / info.num_row_);

  collective::ApplyWithLabels(
      info, static_cast<void*>(quantiles.data()), quantiles.size() * sizeof(float), [&] {
        // loop over each leaf
        common::ParallelFor(quantiles.size(), ctx->Threads(), [&](size_t k) {
          auto nidx = h_node_idx[k];
          CHECK(tree[nidx].IsLeaf());
          CHECK_LT(k + 1, h_node_ptr.size());
          size_t n = h_node_ptr[k + 1] - h_node_ptr[k];
          auto h_row_set = common::Span<size_t const>{ridx}.subspan(h_node_ptr[k], n);

          auto h_labels = info.labels.HostView().Slice(linalg::All(), IdxY(info, group_idx));
          auto h_weights = linalg::MakeVec(&info.weights_);

          auto iter = common::MakeIndexTransformIter([&](size_t i) -> float {
            auto row_idx = h_row_set[i];
            return h_labels(row_idx) - h_predt(row_idx, group_idx);
          });
          auto w_it = common::MakeIndexTransformIter([&](size_t i) -> float {
            auto row_idx = h_row_set[i];
            return h_weights(row_idx);
          });

          float q{0};
          if (info.weights_.Empty()) {
            q = common::Quantile(ctx, alpha, iter, iter + h_row_set.size());
          } else {
            q = common::WeightedQuantile(ctx, alpha, iter, iter + h_row_set.size(), w_it);
          }
          if (std::isnan(q)) {
            CHECK(h_row_set.empty());
          }
          quantiles.at(k) = q;
        });
      });

  UpdateLeafValues(&quantiles, nidx, info, learning_rate, p_tree);
}

#if !defined(XGBOOST_USE_CUDA)
void UpdateTreeLeafDevice(Context const*, common::Span<bst_node_t const>, std::int32_t,
                          MetaInfo const&, float, HostDeviceVector<float> const&, float, RegTree*) {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::obj::detail
