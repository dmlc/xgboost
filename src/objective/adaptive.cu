/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include <thrust/sort.h>

#include <cub/cub.cuh>

#include "../common/device_helpers.cuh"
#include "../common/stats.cuh"
#include "adaptive.h"

namespace xgboost {
namespace obj {
namespace detail {
void EncodeTreeLeafDevice(Context const* ctx, common::Span<bst_node_t const> position,
                          dh::device_vector<size_t>* p_ridx, HostDeviceVector<size_t>* p_nptr,
                          HostDeviceVector<bst_node_t>* p_nidx, RegTree const& tree) {
  // copy position to buffer
  dh::safe_cuda(cudaSetDevice(ctx->gpu_id));
  size_t n_samples = position.size();
  dh::XGBDeviceAllocator<char> alloc;
  dh::device_vector<bst_node_t> sorted_position(position.size());
  dh::safe_cuda(cudaMemcpyAsync(sorted_position.data().get(), position.data(),
                                position.size_bytes(), cudaMemcpyDeviceToDevice));

  p_ridx->resize(position.size());
  dh::Iota(dh::ToSpan(*p_ridx));
  // sort row index according to node index
  thrust::stable_sort_by_key(thrust::cuda::par(alloc), sorted_position.begin(),
                             sorted_position.begin() + n_samples, p_ridx->begin());
  dh::XGBCachingDeviceAllocator<char> caching;
  size_t beg_pos =
      thrust::find_if(thrust::cuda::par(caching), sorted_position.cbegin(), sorted_position.cend(),
                      [] XGBOOST_DEVICE(bst_node_t nidx) { return nidx >= 0; }) -
      sorted_position.cbegin();
  if (beg_pos == sorted_position.size()) {
    auto& leaf = p_nidx->HostVector();
    tree.WalkTree([&](bst_node_t nidx) {
      if (tree[nidx].IsLeaf()) {
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
  dh::safe_cuda(cub::DeviceRunLengthEncode::Encode(nullptr, nbytes, begin_it,
                                                   unique_out.data().get(), counts_out.data().get(),
                                                   d_num_runs_out.data(), n_samples - beg_pos));
  dh::TemporaryArray<char> temp(nbytes);
  dh::safe_cuda(cub::DeviceRunLengthEncode::Encode(temp.data().get(), nbytes, begin_it,
                                                   unique_out.data().get(), counts_out.data().get(),
                                                   d_num_runs_out.data(), n_samples - beg_pos));

  dh::PinnedMemory pinned_pool;
  auto pinned = pinned_pool.GetSpan<char>(sizeof(size_t) + sizeof(bst_node_t));
  dh::CUDAStream copy_stream;
  size_t* h_num_runs = reinterpret_cast<size_t*>(pinned.subspan(0, sizeof(size_t)).data());

  dh::CUDAEvent e;
  e.Record(dh::DefaultStream());
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
    d_node_idx[i] = d_unique_out[i];
    d_node_ptr[i + 1] = d_counts_out[i];
    if (i == 0) {
      d_node_ptr[0] = beg_pos;
    }
  });
  thrust::inclusive_scan(thrust::cuda::par(caching), dh::tbegin(d_node_ptr), dh::tend(d_node_ptr),
                         dh::tbegin(d_node_ptr));
  copy_stream.View().Sync();
  CHECK_GT(*h_num_runs, 0);
  CHECK_LE(*h_num_runs, n_leaf);

  if (*h_num_runs < n_leaf) {
    // shrink to omit the sampled nodes.
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
    auto& h_nidx = nidx.HostVector();
    auto& h_nptr = nptr.HostVector();
    FillMissingLeaf(leaves, &h_nidx, &h_nptr);
    nidx.DevicePointer();
    nptr.DevicePointer();
  }
  CHECK_EQ(nidx.Size(), n_leaf);
  CHECK_EQ(nptr.Size(), n_leaf + 1);
}

void UpdateTreeLeafDevice(Context const* ctx, common::Span<bst_node_t const> position,
                          MetaInfo const& info, HostDeviceVector<float> const& predt, float alpha,
                          RegTree* p_tree) {
  dh::safe_cuda(cudaSetDevice(ctx->gpu_id));
  dh::device_vector<size_t> ridx;
  HostDeviceVector<size_t> nptr;
  HostDeviceVector<bst_node_t> nidx;

  EncodeTreeLeafDevice(ctx, position, &ridx, &nptr, &nidx, *p_tree);

  if (nptr.Empty()) {
    std::vector<float> quantiles;
    UpdateLeafValues(&quantiles, nidx.ConstHostVector(), p_tree);
  }

  HostDeviceVector<float> quantiles;
  predt.SetDevice(ctx->gpu_id);
  auto d_predt = predt.ConstDeviceSpan();
  auto d_labels = info.labels.View(ctx->gpu_id);

  auto d_row_index = dh::ToSpan(ridx);
  auto seg_beg = nptr.DevicePointer();
  auto seg_end = seg_beg + nptr.Size();
  auto val_beg = dh::MakeTransformIterator<float>(thrust::make_counting_iterator(0ul),
                                                  [=] XGBOOST_DEVICE(size_t i) {
                                                    auto predt = d_predt[d_row_index[i]];
                                                    auto y = d_labels(d_row_index[i]);
                                                    return y - predt;
                                                  });
  auto val_end = val_beg + d_labels.Size();
  CHECK_EQ(nidx.Size() + 1, nptr.Size());
  if (info.weights_.Empty()) {
    common::SegmentedQuantile(ctx, alpha, seg_beg, seg_end, val_beg, val_end, &quantiles);
  } else {
    info.weights_.SetDevice(ctx->gpu_id);
    auto d_weights = info.weights_.ConstDeviceSpan();
    CHECK_EQ(d_weights.size(), d_row_index.size());
    auto w_it = thrust::make_permutation_iterator(dh::tcbegin(d_weights), dh::tcbegin(d_row_index));
    common::SegmentedWeightedQuantile(ctx, alpha, seg_beg, seg_end, val_beg, val_end, w_it,
                                      w_it + d_weights.size(), &quantiles);
  }

  UpdateLeafValues(&quantiles.HostVector(), nidx.ConstHostVector(), p_tree);
}
}  // namespace detail
}  // namespace obj
}  // namespace xgboost
