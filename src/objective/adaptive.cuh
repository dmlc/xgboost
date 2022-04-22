/*!
 * Copyright 2022 by XGBoost Contributors
 */
#pragma once
#include <thrust/sort.h>

#include <cub/cub.cuh>

#include "../common/device_helpers.cuh"
#include "xgboost/generic_parameters.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/tree_model.h"

namespace xgboost {
namespace obj {
namespace detail {
inline void FillMissingLeaf(std::vector<bst_node_t> const& maybe_missing,
                            HostDeviceVector<bst_node_t>* p_nidx,
                            HostDeviceVector<size_t>* p_nptr) {
  auto& h_node_idx = p_nidx->HostVector();
  auto& h_node_ptr = p_nptr->HostVector();

  for (auto leaf : maybe_missing) {
    if (std::binary_search(h_node_idx.cbegin(), h_node_idx.cend(), leaf)) {
      continue;
    }
    auto it = std::upper_bound(h_node_idx.cbegin(), h_node_idx.cend(), leaf);
    auto pos = it - h_node_idx.cbegin();
    h_node_idx.insert(h_node_idx.cbegin() + pos, leaf);
    h_node_ptr.insert(h_node_ptr.cbegin() + pos, h_node_ptr[pos]);
  }

  // push to device.
  p_nidx->ConstDevicePointer();
  p_nptr->ConstDevicePointer();
}

inline void EncodeTreeLeaf(Context const* ctx, common::Span<bst_node_t const> position,
                           HostDeviceVector<size_t>* p_nptr, HostDeviceVector<bst_node_t>* p_nidx,
                           RegTree const& tree) {
  // copy position to buffer
  dh::safe_cuda(cudaSetDevice(ctx->gpu_id));
  size_t n_samples = position.size();
  dh::XGBDeviceAllocator<char> alloc;
  dh::device_vector<bst_node_t> sorted_position(position.size());
  dh::safe_cuda(cudaMemcpyAsync(sorted_position.data().get(), position.data(),
                                position.size_bytes(), cudaMemcpyDeviceToDevice));
  dh::device_vector<size_t> ridx(position.size());
  dh::Iota(dh::ToSpan(ridx));
  // sort row index according to node index
  thrust::stable_sort_by_key(thrust::cuda::par(alloc), sorted_position.begin(),
                             sorted_position.begin() + n_samples, ridx.begin());

  size_t n_leaf = tree.GetNumLeaves();
  // +1 for subsample, which is set to an unique value in above kernel.
  size_t max_n_unique = n_leaf + 1;

  dh::caching_device_vector<size_t> counts_out(max_n_unique + 1, 0);
  auto d_counts_out = dh::ToSpan(counts_out).subspan(0, max_n_unique);
  auto d_num_runs_out = dh::ToSpan(counts_out).subspan(max_n_unique, 1);
  dh::caching_device_vector<bst_node_t> unique_out(max_n_unique, 0);
  auto d_unique_out = dh::ToSpan(unique_out);

  size_t nbytes;
  cub::DeviceRunLengthEncode::Encode(nullptr, nbytes, sorted_position.begin(),
                                     unique_out.data().get(), counts_out.data().get(),
                                     d_num_runs_out.data(), n_samples);
  dh::TemporaryArray<char> temp(nbytes);
  cub::DeviceRunLengthEncode::Encode(temp.data().get(), nbytes, sorted_position.begin(),
                                     unique_out.data().get(), counts_out.data().get(),
                                     d_num_runs_out.data(), n_samples);

  dh::XGBCachingDeviceAllocator<char> caching;
  dh::PinnedMemory pinned_pool;
  auto pinned = pinned_pool.GetSpan<char>(sizeof(size_t) + sizeof(bst_node_t));
  dh::CUDAStream copy_stream;
  size_t* h_num_runs = reinterpret_cast<size_t*>(pinned.subspan(0, sizeof(size_t)).data());
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
  nidx.SetDevice(ctx->gpu_id);
  nidx.Resize(n_leaf);
  auto d_node_idx = nidx.DeviceSpan();

  nptr.SetDevice(ctx->gpu_id);
  nptr.Resize(n_leaf + 1, 0);
  auto d_node_ptr = nptr.DeviceSpan();

  dh::LaunchN(n_leaf, [=] XGBOOST_DEVICE(size_t i) {
    if (i >= d_num_runs_out[0]) {
      // d_num_runs_out <= max_n_unique
      // this omits all the leaf that are empty. A leaf can be empty when there's
      // missing data, which can be caused by sparse input and distributed training.
      return;
    }
    if (d_unique_out[0] < 0) {
      // shift 1 to the left
      // some rows are ignored due to sampling, `kIgnoredTreePosition` is -1 so it's the
      // smallest value and is sorted to the left.
      // d_unique_out.size() == n_leaf + 1.
      d_node_idx[i] = d_unique_out[i + 1];
      d_node_ptr[i + 1] = d_counts_out[i + 1];
      if (i == 0) {
        d_node_ptr[0] = d_counts_out[0];
      }
    } else {
      d_node_idx[i] = d_unique_out[i];
      d_node_ptr[i + 1] = d_counts_out[i];
      if (i == 0) {
        d_node_ptr[0] = 0;
      }
    }
  });
  thrust::inclusive_scan(thrust::cuda::par(caching), dh::tbegin(d_node_ptr), dh::tend(d_node_ptr),
                         dh::tbegin(d_node_ptr));
  copy_stream.View().Sync();
  if (*h_first_unique < 0) {
    *h_num_runs -= 1;  // sampled.
  }
  CHECK_GT(*h_num_runs, 0);
  CHECK_LE(*h_num_runs, n_leaf);

  if (*h_num_runs < n_leaf) {
    // shrink to omit the `kIgnoredTreePosition`.
    nptr.Resize(*h_num_runs + 1);
    nidx.Resize(*h_num_runs);

    std::vector<bst_node_t> leaves;
    tree.WalkTree([&](bst_node_t nidx) {
      if (tree[nidx].IsLeaf()) {
        leaves.push_back(nidx);
      }
      return true;
    });
    CHECK_EQ(leaves.size(), n_leaf);
    // Fill all the leaves that don't have any sample. This is hacky and inefficient. An
    // alternative is to leave the objective to handle missing leaf, which is more messy
    // as we need to take other distributed workers into account.
    FillMissingLeaf(leaves, &nidx, &nptr);
  }
  CHECK_EQ(nidx.Size(), n_leaf);
  CHECK_EQ(nptr.Size(), n_leaf + 1);
}
}  // namespace detail
}  // namespace obj
}  // namespace xgboost
