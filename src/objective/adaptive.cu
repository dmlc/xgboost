/**
 * Copyright 2022-2026, XGBoost Contributors
 */
#include <thrust/sort.h>

#include <algorithm>
#include <cub/cub.cuh>  // NOLINT
#include <vector>

#include "../collective/aggregator.h"
#include "../common/cuda_context.cuh"  // CUDAContext
#include "../common/cuda_stream.h"     // for Event, Stream
#include "../common/device_helpers.cuh"
#include "../common/linalg_op.h"  // for VecScaMul
#include "../common/stats.cuh"
#include "../tree/sample_position.h"  // for SamplePosition
#include "../tree/tree_view.h"        // for WalkTree
#include "adaptive.h"
#include "xgboost/context.h"

namespace xgboost::obj {
void EncodeTreeLeafDevice(Context const* ctx, common::Span<bst_node_t const> position,
                          dh::device_vector<size_t>* p_ridx, HostDeviceVector<size_t>* p_nptr,
                          HostDeviceVector<bst_node_t>* p_nidx, RegTree const& tree) {
  // copy position to buffer
  dh::safe_cuda(cudaSetDevice(ctx->Ordinal()));
  auto cuctx = ctx->CUDACtx();
  size_t n_samples = position.size();
  dh::device_vector<bst_node_t> sorted_position(position.size());
  dh::safe_cuda(cudaMemcpyAsync(sorted_position.data().get(), position.data(),
                                position.size_bytes(), cudaMemcpyDeviceToDevice, cuctx->Stream()));

  p_ridx->resize(position.size());
  dh::Iota(dh::ToSpan(*p_ridx), cuctx->Stream());
  // sort row index according to node index
  thrust::stable_sort_by_key(cuctx->TP(), sorted_position.begin(),
                             sorted_position.begin() + n_samples, p_ridx->begin());
  // Find the first one that's not sampled (nidx not been negated).
  size_t beg_pos = thrust::find_if(cuctx->CTP(), sorted_position.cbegin(), sorted_position.cend(),
                                   [] XGBOOST_DEVICE(bst_node_t nidx) {
                                     return tree::SamplePosition::IsValid(nidx);
                                   }) -
                   sorted_position.cbegin();
  if (beg_pos == sorted_position.size()) {
    auto& leaf = p_nidx->HostVector();
    tree::WalkTree(tree, [&](auto const& tree, bst_node_t nidx) {
      if (tree.IsLeaf(nidx)) {
        leaf.push_back(nidx);
      }
      return true;
    });
    return;
  }

  size_t n_leaf = tree.GetNumLeaves();
  size_t max_n_unique = n_leaf;

  dh::caching_device_vector<size_t> counts_out(max_n_unique + 1, 0);
  auto d_counts_out = dh::ToSpan(counts_out).subspan(0, max_n_unique);
  auto d_num_runs_out = dh::ToSpan(counts_out).subspan(max_n_unique, 1);
  dh::caching_device_vector<bst_node_t> unique_out(max_n_unique, 0);
  auto d_unique_out = dh::ToSpan(unique_out);

  size_t nbytes{0};
  auto begin_it = sorted_position.begin() + beg_pos;
  dh::safe_cuda(cub::DeviceRunLengthEncode::Encode(
      nullptr, nbytes, begin_it, unique_out.data().get(), counts_out.data().get(),
      d_num_runs_out.data(), n_samples - beg_pos, ctx->CUDACtx()->Stream()));
  dh::TemporaryArray<char> temp(nbytes);
  dh::safe_cuda(cub::DeviceRunLengthEncode::Encode(
      temp.data().get(), nbytes, begin_it, unique_out.data().get(), counts_out.data().get(),
      d_num_runs_out.data(), n_samples - beg_pos, ctx->CUDACtx()->Stream()));

  dh::PinnedMemory pinned_pool;
  auto pinned = pinned_pool.GetSpan<char>(sizeof(size_t) + sizeof(bst_node_t));
  curt::Stream copy_stream;
  size_t* h_num_runs = reinterpret_cast<size_t*>(pinned.subspan(0, sizeof(size_t)).data());

  curt::Event e;
  e.Record(cuctx->Stream());
  copy_stream.View().Wait(e);
  // flag for whether there's ignored position
  bst_node_t* h_first_unique =
      reinterpret_cast<bst_node_t*>(pinned.subspan(sizeof(size_t), sizeof(bst_node_t)).data());
  dh::safe_cuda(cudaMemcpyAsync(h_num_runs, d_num_runs_out.data(), sizeof(size_t),
                                cudaMemcpyDeviceToHost, copy_stream.View()));
  dh::safe_cuda(cudaMemcpyAsync(h_first_unique, d_unique_out.data(), sizeof(bst_node_t),
                                cudaMemcpyDeviceToHost, copy_stream.View()));

  /**
   * copy node index (leaf index)
   */
  auto& nidx = *p_nidx;
  auto& nptr = *p_nptr;
  nidx.SetDevice(ctx->Device());
  nidx.Resize(n_leaf);
  auto d_node_idx = nidx.DeviceSpan();

  nptr.SetDevice(ctx->Device());
  nptr.Resize(n_leaf + 1, 0);
  auto d_node_ptr = nptr.DeviceSpan();

  dh::LaunchN(n_leaf, [=] XGBOOST_DEVICE(size_t i) {
    if (i >= d_num_runs_out[0]) {
      // d_num_runs_out <= max_n_unique
      // this omits all the leaf that are empty. A leaf can be empty when there's
      // missing data, which can be caused by sparse input and distributed training.
      return;
    }
    d_node_idx[i] = d_unique_out[i];
    d_node_ptr[i + 1] = d_counts_out[i];
    if (i == 0) {
      d_node_ptr[0] = beg_pos;
    }
  });
  thrust::inclusive_scan(cuctx->CTP(), dh::tbegin(d_node_ptr), dh::tend(d_node_ptr),
                         dh::tbegin(d_node_ptr));
  copy_stream.View().Sync();
  CHECK_GT(*h_num_runs, 0);
  CHECK_LE(*h_num_runs, n_leaf);

  if (*h_num_runs < n_leaf) {
    // shrink to omit the sampled nodes.
    nptr.Resize(*h_num_runs + 1);
    nidx.Resize(*h_num_runs);

    std::vector<bst_node_t> leaves;
    tree::WalkTree(tree, [&](auto const& tree, bst_node_t nidx) {
      if (tree.IsLeaf(nidx)) {
        leaves.push_back(nidx);
      }
      return true;
    });
    CHECK_EQ(leaves.size(), n_leaf);
    // Fill all the leaves that don't have any sample. This is hacky and inefficient. An
    // alternative is to leave the objective to handle missing leaf, which is more messy
    // as we need to take other distributed workers into account.
    auto& h_nidx = nidx.HostVector();
    auto& h_nptr = nptr.HostVector();
    detail::FillMissingLeaf(leaves, &h_nidx, &h_nptr);
    nidx.DevicePointer();
    nptr.DevicePointer();
  }
  CHECK_EQ(nidx.Size(), n_leaf);
  CHECK_EQ(nptr.Size(), n_leaf + 1);
}

namespace cuda_impl {
void UpdateTreeLeaf(Context const* ctx, common::Span<bst_node_t const> position,
                    bst_target_t group_idx, MetaInfo const& info, float learning_rate,
                    HostDeviceVector<float> const& predt, std::vector<float> const& h_alphas,
                    RegTree* p_tree) {
  dh::safe_cuda(cudaSetDevice(ctx->Ordinal()));
  dh::device_vector<size_t> ridx;
  HostDeviceVector<size_t> nptr;
  HostDeviceVector<bst_node_t> nidx;

  EncodeTreeLeafDevice(ctx, position, &ridx, &nptr, &nidx, *p_tree);

  if (nptr.Empty()) {
    std::vector<float> quantiles;
    detail::UpdateLeafValues(ctx, &quantiles, nidx.ConstHostVector(), info, learning_rate, p_tree);
  }

  predt.SetDevice(ctx->Device());
  auto d_predt = linalg::MakeTensorView(ctx, predt.ConstDeviceSpan(), info.num_row_,
                                        predt.Size() / info.num_row_);
  CHECK_LT(group_idx, d_predt.Shape(1));
  if (p_tree->IsMultiTarget()) {
    CHECK_EQ(d_predt.Shape(1), h_alphas.size());
  }
  HostDeviceVector<float> quantiles;

  auto d_row_index = dh::ToSpan(ridx);
  // node segments
  auto seg_beg = nptr.ConstDevicePointer();
  auto seg_end = seg_beg + nptr.Size();
  CHECK_EQ(nidx.Size() + 1, nptr.Size());

  collective::ApplyWithLabels(ctx, info, &quantiles, [&] {
    auto d_labels = info.labels.View(ctx->Device());

    auto values = [=] XGBOOST_DEVICE(std::size_t i, std::size_t j) {
      // If it's vector-leaf, group_idx is 0, j is used. Otherwise, j is 0, group idx is used.
      auto p_idx = cuda::std::max(j, static_cast<std::size_t>(group_idx));
      auto p = d_predt(d_row_index[i], p_idx);
      // label is a single column for quantile regression, but it's a matrix for MAE.
      auto y_idx = cuda::std::max(j, static_cast<std::size_t>(group_idx));
      y_idx = cuda::std::min(y_idx, d_labels.Shape(1) - 1);
      auto y = d_labels(d_row_index[i], y_idx);
      return y - p;
    };
    CHECK_EQ(d_labels.Shape(0), position.size());

    if (info.weights_.Empty()) {
      common::SegmentedQuantile(ctx, h_alphas, seg_beg, seg_end, values, info.num_row_, &quantiles);
    } else {
      info.weights_.SetDevice(ctx->Device());
      auto d_weights = info.weights_.ConstDeviceSpan();
      CHECK_EQ(d_weights.size(), d_row_index.size());
      auto w_it =
          thrust::make_permutation_iterator(dh::tcbegin(d_weights), dh::tcbegin(d_row_index));
      common::SegmentedWeightedQuantile(ctx, h_alphas, seg_beg, seg_end, values, w_it,
                                        w_it + d_weights.size(), &quantiles);
    }
  });

  if (p_tree->IsMultiTarget()) {
    linalg::VecScaMul(ctx, linalg::MakeVec(ctx->Device(), quantiles.DeviceSpan()), learning_rate);
    p_tree->SetLeaves(nidx.ConstHostVector(), quantiles.ConstHostSpan());
  } else {
    detail::UpdateLeafValues(ctx, &quantiles.HostVector(), nidx.ConstHostVector(), info,
                             learning_rate, p_tree);
  }
}
}  // namespace cuda_impl
}  // namespace xgboost::obj
