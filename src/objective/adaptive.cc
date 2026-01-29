/**
 * Copyright 2022-2026, XGBoost Contributors
 */
#include "adaptive.h"

#include <algorithm>  // for transform,find_if,copy,unique,max
#include <cmath>      // std::isnan
#include <cstddef>    // std::size_t
#include <iterator>   // std::distance
#include <vector>     // std::vector

#include "../common/algorithm.h"           // ArgSort
#include "../common/linalg_op.h"           // for VecScaMul
#include "../common/numeric.h"             // RunLengthEncode
#include "../common/stats.h"               // Quantile,WeightedQuantile
#include "../common/threading_utils.h"     // ParallelFor
#include "../common/transform_iterator.h"  // MakeIndexTransformIter
#include "../tree/sample_position.h"       // for SamplePosition
#include "../tree/tree_view.h"             // for WalkTree
#include "xgboost/base.h"                  // bst_node_t
#include "xgboost/context.h"               // Context
#include "xgboost/data.h"                  // MetaInfo
#include "xgboost/host_device_vector.h"    // HostDeviceVector
#include "xgboost/linalg.h"                // MakeTensorView
#include "xgboost/span.h"                  // Span
#include "xgboost/tree_model.h"            // RegTree

#if !defined(XGBOOST_USE_CUDA)
#include "../common/common.h"  // AssertGPUSupport
#endif                         // !defined(XGBOOST_USE_CUDA)

namespace xgboost::obj {
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
  size_t begin_pos = std::distance(
      sorted_pos.cbegin(),
      std::find_if(sorted_pos.cbegin(), sorted_pos.cend(),
                   [](bst_node_t nidx) { return tree::SamplePosition::IsValid(nidx); }));
  CHECK_LE(begin_pos, sorted_pos.size());

  std::vector<bst_node_t> leaf;
  tree::WalkTree(tree, [&](auto const& tree, bst_node_t nidx) {
    if (tree.IsLeaf(nidx)) {
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
    detail::FillMissingLeaf(leaf, &nidx, &nptr);
  }
}

namespace cpu_impl {
namespace {
[[nodiscard]] std::int32_t AllocThreads(Context const* ctx, std::vector<size_t> const& h_node_ptr) {
  // A heuristic to use parallel sort. If we use multiple threads here, the sorting is
  // performed using a single thread as openmp cannot allocate new threads inside a
  // parallel region.
  std::int32_t n_threads;
  if constexpr (kHasParallelStableSort) {
    CHECK_GE(h_node_ptr.size(), 1);
    auto it = common::MakeIndexTransformIter(
        [&](std::size_t i) { return h_node_ptr[i + 1] - h_node_ptr[i]; });
    n_threads = std::any_of(it, it + h_node_ptr.size() - 1,
                            [](auto n) {
                              constexpr std::size_t kNeedParallelSort = 1ul << 19;
                              return n > kNeedParallelSort;
                            })
                    ? 1
                    : ctx->Threads();
  } else {
    n_threads = ctx->Threads();
  }
  return n_threads;
}
}  // namespace

void UpdateTreeLeaf(Context const* ctx, std::vector<bst_node_t> const& position,
                    bst_target_t group_idx, MetaInfo const& info, float learning_rate,
                    HostDeviceVector<float> const& predt, std::vector<float> const& alphas,
                    RegTree* p_tree) {
  std::vector<bst_node_t> nidx;
  std::vector<size_t> nptr;
  std::vector<size_t> ridx;
  EncodeTreeLeafHost(ctx, *p_tree, position, &nptr, &nidx, &ridx);
  std::size_t n_leaves = nidx.size();
  std::size_t n_alphas = alphas.size();
  if (nptr.empty()) {
    std::vector<float> quantiles;
    detail::UpdateLeafValues(ctx, &quantiles, nidx, info, learning_rate, p_tree);
    return;
  }

  CHECK(!position.empty());
  std::vector<float> quantiles(n_leaves * n_alphas, 0.0f);
  std::vector<bst_node_t> n_valids(n_leaves, 0);

  auto h_quantiles = linalg::MakeTensorView(ctx, common::Span{quantiles}, n_leaves, n_alphas);
  CHECK_LE(nptr.back(), info.num_row_);
  auto h_predt = linalg::MakeTensorView(ctx, predt.ConstHostSpan(), info.num_row_,
                                        predt.Size() / info.num_row_);
  if (p_tree->IsMultiTarget()) {
    CHECK_EQ(h_predt.Shape(1), alphas.size());
  }
  std::int32_t n_threads = AllocThreads(ctx, nptr);

  collective::ApplyWithLabels(
      ctx, info, static_cast<void*>(quantiles.data()), quantiles.size() * sizeof(float), [&] {
        // Loop over each leaf
        common::ParallelFor(n_leaves, n_threads, [&](auto k) {
          CHECK_LT(k + 1, nptr.size());
          size_t n = nptr[k + 1] - nptr[k];
          auto h_row_set = common::Span<size_t const>{ridx}.subspan(nptr[k], n);

          linalg::MatrixView<float const> h_labels = info.labels.HostView();
          auto h_weights = linalg::MakeVec(&info.weights_);
          // Loop over each target (quantile).
          for (std::size_t alpha_idx = 0; alpha_idx < n_alphas; ++alpha_idx) {
            // If it's vector-leaf, group_idx is 0, alpha_idx is used. Otherwise,
            // alpha_idx is 0, the group idx is used.
            auto predt_idx = std::max(alpha_idx, static_cast<std::size_t>(group_idx));
            // label is a single column for quantile regression, but it's a matrix for MAE.
            auto y_idx = std::max(alpha_idx, static_cast<std::size_t>(group_idx));
            y_idx = std::min(y_idx, h_labels.Shape(1) - 1);
            auto iter = common::MakeIndexTransformIter([&](std::size_t i) -> float {
              auto row_idx = h_row_set[i];
              return h_labels(row_idx, y_idx) - h_predt(row_idx, predt_idx);
            });
            auto w_it = common::MakeIndexTransformIter([&](std::size_t i) -> float {
              auto row_idx = h_row_set[i];
              return h_weights(row_idx);
            });
            auto alpha = alphas[alpha_idx];

            float q{0};
            if (info.weights_.Empty()) {
              q = common::Quantile(ctx, alpha, iter, iter + h_row_set.size());
            } else {
              q = common::WeightedQuantile(ctx, alpha, iter, iter + h_row_set.size(), w_it);
            }
            if (std::isnan(q)) {
              CHECK(h_row_set.empty());
            }
            h_quantiles(k, alpha_idx) = q;
          }
        });
      });

  if (p_tree->IsMultiTarget()) {
    linalg::VecScaMul(ctx, linalg::MakeVec(ctx->Device(), common::Span{quantiles}), learning_rate);
    p_tree->SetLeaves(nidx, common::Span{quantiles});
  } else {
    detail::UpdateLeafValues(ctx, &quantiles, nidx, info, learning_rate, p_tree);
  }
}
}  // namespace cpu_impl

namespace cuda_impl {
#if !defined(XGBOOST_USE_CUDA)
void UpdateTreeLeaf(Context const*, common::Span<bst_node_t const>, bst_target_t, MetaInfo const&,
                    float, HostDeviceVector<float> const&, std::vector<float> const&, RegTree*) {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace cuda_impl
}  // namespace xgboost::obj
