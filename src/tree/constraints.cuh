/**
 * Copyright 2019-2026, XGBoost contributors
 *
 * \file Various constraints used in GPU_Hist.
 */
#ifndef XGBOOST_TREE_CONSTRAINTS_CUH_
#define XGBOOST_TREE_CONSTRAINTS_CUH_

#include <thrust/scan.h>  // for exclusive_scan

#include <algorithm>                   // for max
#include <cstddef>                     // for size_t
#include <cstdint>                     // for uint32_t
#include <cub/block/block_reduce.cuh>  // for BlockReduce
#include <cub/block/block_scan.cuh>    // for BlockScan
#include <vector>                      // for vector

#include "../common/bitfield.h"          // for LBitField64
#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/device_helpers.cuh"  // for LaunchKernel
#include "../common/device_vector.cuh"   // for CachingDeviceUVector, device_vector
#include "param.h"                       // for TrainParam
#include "xgboost/span.h"                // for Span

namespace xgboost {
namespace detail {
// Each node owns one row of words in device memory.
struct NodeBitFields {
  common::Span<LBitField64::value_type> words;
  std::size_t words_per_node;

  [[nodiscard]] XGBOOST_DEVICE LBitField64 operator[](bst_node_t nidx) const {
    return LBitField64{words.subspan(nidx * words_per_node, words_per_node)};
  }
};

XGBOOST_DEVICE inline bool HasConstrainedSet(bst_feature_t n_outputs, bst_feature_t n_inputs) {
  return n_outputs != n_inputs;
}

// Count eligible features and the space each node needs in the packed result.
template <std::uint32_t kBlockSize, typename Input>
__global__ void CountEligibleFeatures(common::Span<Input> inputs, NodeBitFields masks,
                                      common::Span<bst_feature_t> counts,
                                      common::Span<std::size_t> sizes) {
  auto const& input = inputs[blockIdx.x];
  if (input.nidx == 0) {
    if (threadIdx.x == 0) {
      counts[blockIdx.x] = input.feature_set.size();
      sizes[blockIdx.x] = 0;
    }
    return;
  }

  auto allowed = masks[input.nidx];
  bst_feature_t count = 0;
  for (auto i : dh::BlockStrideRange(std::size_t{0}, input.feature_set.size())) {
    count += allowed.Check(input.feature_set[i]);
  }
  using Reduce = cub::BlockReduce<bst_feature_t, kBlockSize>;
  __shared__ typename Reduce::TempStorage storage;
  auto total = Reduce{storage}.Sum(count);
  if (threadIdx.x == 0) {
    counts[blockIdx.x] = total;
    sizes[blockIdx.x] = total == input.feature_set.size() ? 0 : total;
  }
}

// One block per node. The block scan gives each retained feature its position in sampled order.
template <std::uint32_t kBlockSize, typename Input>
__global__ void CompactEligibleFeatures(common::Span<Input> inputs, NodeBitFields masks,
                                        common::Span<bst_feature_t const> counts,
                                        common::Span<std::size_t const> offsets,
                                        common::Span<bst_feature_t> result) {
  auto& input = inputs[blockIdx.x];
  auto sampled = input.feature_set;
  auto count = counts[blockIdx.x];
  if (!HasConstrainedSet(count, sampled.size())) {
    return;  // Keep the original span.
  }
  auto out = result.subspan(offsets[blockIdx.x], count);
  auto allowed = masks[input.nidx];
  using Scan = cub::BlockScan<bst_feature_t, kBlockSize>;
  __shared__ typename Scan::TempStorage storage;
  // How many eligible features the current node’s CUDA block has already copied into its
  // output slice.
  bst_feature_t n_written = 0;
  // A full tile is a block (kBlockSize).
  for (std::size_t tile = 0; tile < sampled.size() && n_written < count; tile += kBlockSize) {
    auto i = tile + threadIdx.x;
    auto fidx = i < sampled.size() ? sampled[i] : bst_feature_t{0};
    bst_feature_t keep = i < sampled.size() && allowed.Check(fidx);
    // rank is the sum of the flags before this thread.
    // tile_count is the sum of all flags in the tile.
    bst_feature_t rank, tile_count;
    Scan{storage}.ExclusiveSum(keep, rank, tile_count);
    if (keep) {
      out[n_written + rank] = fidx;
    }
    n_written += tile_count;
    __syncthreads();  // Reuse the scan storage for the next tile.
  }
  // Also synchronize an empty result before replacing the span that the block read.
  __syncthreads();
  if (threadIdx.x == 0) {
    KERNEL_CHECK(n_written == count);
    input.feature_set = out;
  }
}
}  // namespace detail

// Feature interaction constraints built for GPU Hist updater.
struct FeatureInteractionConstraintDevice {
  using GroupId = std::uint32_t;

 protected:
  // Whether interaction constraint is used.
  bool has_constraint_;
  // n interaction sets.
  std::size_t n_sets_;

  // The parsed feature interaction constraints as CSR.
  dh::device_vector<bst_feature_t> d_fconstraints_;
  dh::device_vector<size_t> d_fconstraints_ptr_;
  /* Interaction sets for each feature as CSR.  For an input like:
   * [[0, 1], [1, 2]], this will have values:
   *
   * fid:                                |0 | 1  | 2|
   * sets a feature belongs to(d_sets_): |0 |0, 1| 1|
   *
   * h_sets_ptr_:                        |0, 1, 3, 4|
   */
  dh::device_vector<GroupId> d_sets_;
  // Host offsets into d_sets_ for looking up the groups containing a split feature.
  std::vector<std::size_t> h_sets_ptr_;

  // One fixed-width feature mask per node, stored consecutively on the device.
  dh::device_vector<LBitField64::value_type> node_constraints_;
  bst_node_t n_nodes_{0};

  [[nodiscard]] detail::NodeBitFields FeatureMasks() {
    return {dh::ToSpan(node_constraints_), LBitField64::ComputeStorageSize(this->Features())};
  }

  // Reused across batches. Each node gets a disjoint slice for its filtered features.
  dh::CachingDeviceUVector<bst_feature_t> result_buffer_;
  // One bit per interaction group. A set bit means the group contains the entire path.
  // Logical size is n_groups x n_nodes bits.
  dh::CachingDeviceUVector<LBitField64::value_type> node_groups_;

 private:
  template <typename Input>
  static auto MaxActiveFeature(common::Span<Input> inputs) {
    bst_feature_t max_active = 0;
    for (auto const& input : inputs) {
      max_active = std::max(max_active, static_cast<bst_feature_t>(input.feature_set.size()));
    }
    return max_active;
  }

 public:
  [[nodiscard]] bst_feature_t Features() const;
  FeatureInteractionConstraintDevice() = default;
  void Configure(tree::TrainParam const& param, int32_t const n_features);
  FeatureInteractionConstraintDevice(tree::TrainParam const& param, int32_t const n_features);
  FeatureInteractionConstraintDevice(FeatureInteractionConstraintDevice const& that) = delete;
  FeatureInteractionConstraintDevice(FeatureInteractionConstraintDevice&& that) = default;
  /*! \brief Reset before constructing a new tree. */
  void Reset(Context const* ctx);
  /** @brief Upload split inputs and filter their sorted, unique sampled features in place.
   *  Returns the largest eligible count. Filtered spans remain valid until the next batch.
   */
  template <typename Input>
  bst_feature_t Query(Context const* ctx, common::Span<Input> inputs,
                      common::Span<Input> d_inputs) {
    CHECK_EQ(inputs.size(), d_inputs.size());
    if (inputs.empty()) {
      return 0;
    }
    auto stream = ctx->CUDACtx()->Stream();
    dh::safe_cuda(cudaMemcpyAsync(d_inputs.data(), inputs.data(), inputs.size_bytes(),
                                  cudaMemcpyHostToDevice, stream));
    if (!has_constraint_) {
      return MaxActiveFeature(inputs);
    }

    auto masks = this->FeatureMasks();
    dh::CachingDeviceUVector<bst_feature_t> d_counts(inputs.size());
    dh::CachingDeviceUVector<std::size_t> d_offsets(inputs.size());
    std::uint32_t constexpr kBlockSize = 256;
    dh::LaunchKernel{static_cast<std::uint32_t>(inputs.size()), kBlockSize, 0, stream}(  // NOLINT
        detail::CountEligibleFeatures<kBlockSize, Input>, d_inputs, masks, dh::ToSpan(d_counts),
        dh::ToSpan(d_offsets));
    thrust::exclusive_scan(ctx->CUDACtx()->CTP(), d_offsets.begin(), d_offsets.end(),
                           d_offsets.begin(), std::size_t{0});
    std::vector<bst_feature_t> counts;
    dh::CopyTo(d_counts, &counts, stream);
    stream.Sync();

    std::size_t n_output = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if (!detail::HasConstrainedSet(counts[i], inputs[i].feature_set.size())) {
        // Every sampled feature survived, the node can reuse its original span
        continue;
      }
      n_output += counts[i];
    }
    result_buffer_.clear();
    result_buffer_.resize(n_output, stream);
    auto result = dh::ToSpan(result_buffer_);
    dh::LaunchKernel{static_cast<std::uint32_t>(inputs.size()), kBlockSize, 0, stream}(
        detail::CompactEligibleFeatures<kBlockSize, Input>, d_inputs, masks, dh::ToSpan(d_counts),
        dh::ToSpan(d_offsets), result);
    // Keep the host spans in sync with the split inputs passed to the device evaluator.
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if (detail::HasConstrainedSet(counts[i], inputs[i].feature_set.size())) {
        // Replace the feature set with the constrained set of features.
        inputs[i].feature_set = result.first(counts[i]);
        result = result.subspan(counts[i]);
      }
    }
    return MaxActiveFeature(inputs);
  }
  /*! \brief Apply split for node_id. */
  void Split(Context const* ctx, bst_node_t node_id, bst_feature_t feature_id, bst_node_t left_id,
             bst_node_t right_id);
};

}  // namespace xgboost
#endif  // XGBOOST_TREE_CONSTRAINTS_CUH_
