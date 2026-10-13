/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#pragma once

#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <memory>   // for unique_ptr
#include <vector>   // for vector

#include "../../common/cuda_context.cuh"    // for CUDAContext
#include "../../common/device_helpers.cuh"  // for LaunchN
#include "../../common/device_vector.cuh"   // for device_vector
#include "../../data/ellpack_page.cuh"      // for EllpackDeviceAccessor
#include "feature_groups.cuh"               // for FeatureGroups
#include "xgboost/base.h"                   // for GradientPair, GradientPairInt64
#include "xgboost/context.h"                // for Context
#include "xgboost/span.h"                   // for Span

namespace xgboost::tree {
// Shared-memory budget per block for the architecture's histogram launch bounds.
[[nodiscard]] std::size_t HistShmemBytes(std::int32_t device);

/**
 * @brief An atomicAdd designed for gradient pair with better performance.  For general
 *        int64_t atomicAdd, one can simply cast it to unsigned long long. Exposed for testing.
 */
XGBOOST_DEV_INLINE void AtomicAdd64As32(int64_t* dst, int64_t src) {
  uint32_t* y_low = reinterpret_cast<uint32_t*>(dst);
  uint32_t* y_high = y_low + 1;

  auto cast_src = reinterpret_cast<uint64_t*>(&src);

  uint32_t const x_low = static_cast<uint32_t>(src);
  uint32_t const x_high = (*cast_src) >> 32;

  auto const old = atomicAdd(y_low, x_low);
  uint32_t const carry = old > (std::numeric_limits<uint32_t>::max() - x_low) ? 1 : 0;
  uint32_t const sig = x_high + carry;
  atomicAdd(y_high, sig);
}

namespace cuda_impl {
// Start with about 16mb
std::size_t constexpr DftReserveSize() { return 1 << 22; }

// An entry is one stored ELLPACK bin index or missing marker.
// A segment contains one node's entries for one feature group, shared across targets.
// A sub-segment is a contiguous range of entries processed by a single CUDA block.
// A wave is enough blocks to fill the GPU's estimated concurrent block capacity once.
// A flush adds a block's shared-memory histogram to the global histogram.

// Wave target for load balance; determines the minimum entries per sub-segment.
inline constexpr std::size_t kTargetWaves = 32;
// Minimum wave target; limits entries per sub-segment even when the flush budget cannot be met.
inline constexpr std::size_t kMinWaves = 4;
// Histogram payload flushed across targets, as a percent of ELLPACK entry bytes counted once.
inline constexpr std::size_t kMaxFlushPercent = 25;

/**
 * @brief Choose entries per sub-segment to amortize histogram flushes The result is
 * rounded to whole tiles.
 *
 * @param n_total_entries            Entries across all nodes and feature groups, counted once for all
 *                                   targets.
 * @param entries_per_tile           Entries per full tile; equal to threads per block.
 * @param n_resident_blks_per_target Estimated share of concurrent GPU blocks per target, at least one.
 * @param n_targets                  Number of outputs being trained.
 * @param hist_bytes_per_block       Largest feature group's bin count times sizeof(GradientPairInt64),
 *                                   for one target; zero for global accumulation.
 * @param symbol_bits                Bits per ELLPACK symbol.
 */
[[nodiscard]] bst_idx_t SliceSegment(bst_idx_t n_total_entries, std::size_t entries_per_tile,
                                     std::size_t n_resident_blks_per_target, bst_target_t n_targets,
                                     std::size_t hist_bytes_per_block, std::uint32_t symbol_bits);

}  // namespace cuda_impl

/**
 * @brief Data storage for node histograms on device. Automatically expands.
 *
 * @author  Rory
 * @date    28/07/2018
 */
class DeviceHistogramStorage {
 private:
  using GradientSumT = GradientPairInt64;
  std::size_t stop_growing_size_{0};
  /** @brief Map nidx to starting index of its histogram. */
  std::map<int, size_t> nidx_map_;
  // Large buffer of zeroed memory, caches histograms
  dh::device_vector<typename GradientSumT::ValueT> data_;
  // If we run out of storage allocate one histogram at a time in overflow. Not cached,
  // overwritten when a new histogram is requested
  dh::device_vector<typename GradientSumT::ValueT> overflow_;
  std::map<int, size_t> overflow_nidx_map_;
  // The total number of bins across all features and targets
  bst_bin_t n_total_bins_;
  static constexpr std::size_t kNumItemsInGradientSum =
      sizeof(GradientSumT) / sizeof(typename GradientSumT::ValueT);
  static_assert(kNumItemsInGradientSum == 2, "Number of items in gradient type should be 2.");

 public:
  explicit DeviceHistogramStorage() { data_.reserve(cuda_impl::DftReserveSize()); }

  void Reset(Context const* ctx, bst_bin_t n_total_bins, std::size_t max_cached_nodes) {
    this->n_total_bins_ = n_total_bins;
    auto d_data = data_.data().get();
    dh::LaunchN(data_.size(), ctx->CUDACtx()->Stream(),
                [=] __device__(size_t idx) { d_data[idx] = 0.0f; });
    nidx_map_.clear();
    overflow_nidx_map_.clear();

    auto max_cached_bin_values =
        static_cast<std::size_t>(n_total_bins) * max_cached_nodes * kNumItemsInGradientSum;
    this->stop_growing_size_ = max_cached_bin_values;
  }

  [[nodiscard]] bool HistogramExists(bst_node_t nidx) const {
    return nidx_map_.find(nidx) != nidx_map_.cend() ||
           overflow_nidx_map_.find(nidx) != overflow_nidx_map_.cend();
  }
  [[nodiscard]] int Bins() const { return n_total_bins_; }
  [[nodiscard]] size_t HistogramSize() const { return n_total_bins_ * kNumItemsInGradientSum; }
  dh::device_vector<typename GradientSumT::ValueT>& Data() { return data_; }

  void AllocateHistograms(Context const* ctx, std::vector<bst_node_t> const& new_nidxs) {
    for (int nidx : new_nidxs) {
      CHECK(!HistogramExists(nidx));
    }
    // Number of items currently used in data
    const size_t used_size = nidx_map_.size() * HistogramSize();
    const size_t new_used_size = used_size + HistogramSize() * new_nidxs.size();
    CHECK_GE(this->stop_growing_size_, kNumItemsInGradientSum);
    if (used_size >= this->stop_growing_size_) {
      // Use overflow
      // Delete previous entries
      overflow_nidx_map_.clear();
      overflow_.resize(HistogramSize() * new_nidxs.size());
      // Zero memory
      auto d_data = overflow_.data().get();
      dh::LaunchN(overflow_.size(), ctx->CUDACtx()->Stream(),
                  [=] __device__(size_t idx) { d_data[idx] = 0.0; });
      // Append new histograms
      for (int nidx : new_nidxs) {
        overflow_nidx_map_[nidx] = overflow_nidx_map_.size() * HistogramSize();
      }
    } else {
      CHECK_GE(data_.size(), used_size);
      // Expand if necessary
      if (data_.size() < new_used_size) {
        data_.resize(std::max(data_.size() * 2, new_used_size));
      }
      // Append new histograms
      for (int nidx : new_nidxs) {
        nidx_map_[nidx] = nidx_map_.size() * HistogramSize();
      }
    }

    CHECK_GE(data_.size(), nidx_map_.size() * HistogramSize());
  }

  /**
   * \summary   Return pointer to histogram memory for a given node.
   * \param nidx    Tree node index.
   * \return    hist pointer.
   */
  common::Span<GradientSumT> GetNodeHistogram(int nidx) {
    CHECK(this->HistogramExists(nidx));

    if (nidx_map_.find(nidx) != nidx_map_.cend()) {
      // Fetch from normal cache
      auto ptr = data_.data().get() + nidx_map_.at(nidx);
      return {reinterpret_cast<GradientSumT*>(ptr), static_cast<std::size_t>(n_total_bins_)};
    } else {
      // Fetch from overflow
      auto ptr = overflow_.data().get() + overflow_nidx_map_.at(nidx);
      return {reinterpret_cast<GradientSumT*>(ptr), static_cast<std::size_t>(n_total_bins_)};
    }
  }
};

struct HistKernel;

class DeviceHistogramBuilder {
  std::unique_ptr<HistKernel> p_impl_;
  DeviceHistogramStorage hist_;
  common::Monitor monitor_;

 public:
  explicit DeviceHistogramBuilder();
  ~DeviceHistogramBuilder();
  // TODO(jiamingy): use a type larger than bst_bin_t since we need to support multi-target.
  void Reset(Context const* ctx, std::size_t max_cached_hist_nodes, bst_bin_t n_total_bins,
             bool force_global_memory);
  // Build histogram for single target and single node, a wrapper of the batched version.
  void BuildHistogram(Context const* ctx, EllpackAccessor const& matrix,
                      FeatureGroups const& feature_groups,
                      common::Span<GradientPairInt64 const> gpair,
                      common::Span<std::uint32_t const> ridx,
                      common::Span<GradientPairInt64> histogram);
  /**
   * @brief Build histograms for multiple nodes and multiple targets.
   *
   * @param ridxs One span of row indices for each node, empty nodes are allowed.
   * @param hists One histogram for each node, must match `ridxs`.
   */
  void BuildHistogram(Context const* ctx, EllpackAccessor const& matrix,
                      FeatureGroups const& feature_groups,
                      linalg::MatrixView<GradientPairInt64 const> gpair,
                      std::vector<common::Span<std::uint32_t const>> const& ridxs,
                      std::vector<common::Span<GradientPairInt64>> const& hists);

  [[nodiscard]] auto GetNodeHistogram(bst_node_t nidx) { return hist_.GetNodeHistogram(nidx); }

  // num histograms is the number of contiguous histograms in memory to reduce over
  void AllReduceHist(Context const* ctx, bst_node_t nidx, std::size_t num_histograms);

  [[nodiscard]] bool CanSubtract(bst_node_t nidx_parent, bst_node_t nidx_histogram) const {
    return hist_.HistogramExists(nidx_parent) && hist_.HistogramExists(nidx_histogram);
  }
  // Attempt to do subtraction trick
  // return true if succeeded
  [[nodiscard]] bool SubtractionTrick(Context const* ctx, bst_node_t nidx_parent,
                                      bst_node_t nidx_histogram, bst_node_t nidx_subtraction) {
    if (!this->CanSubtract(nidx_parent, nidx_histogram)) {
      return false;
    }
    auto d_node_hist_parent = hist_.GetNodeHistogram(nidx_parent);
    auto d_node_hist_histogram = hist_.GetNodeHistogram(nidx_histogram);
    auto d_node_hist_subtraction = hist_.GetNodeHistogram(nidx_subtraction);

    dh::LaunchN(d_node_hist_parent.size(), ctx->CUDACtx()->Stream(), [=] __device__(size_t idx) {
      d_node_hist_subtraction[idx] = d_node_hist_parent[idx] - d_node_hist_histogram[idx];
    });
    return true;
  }

  template <typename ExpandEntry>
  [[nodiscard]] auto SubtractHist(Context const* ctx, std::vector<ExpandEntry> const& candidates,
                                  std::vector<bst_node_t> const& build_nidx,
                                  std::vector<bst_node_t> const& subtraction_nidx) {
    this->monitor_.Start(__func__);
    std::vector<bst_node_t> need_build;
    for (std::size_t i = 0; i < subtraction_nidx.size(); i++) {
      auto build_hist_nidx = build_nidx.at(i);
      auto subtraction_trick_nidx = subtraction_nidx.at(i);
      auto parent_nidx = candidates.at(i).nidx;

      if (!this->SubtractionTrick(ctx, parent_nidx, build_hist_nidx, subtraction_trick_nidx)) {
        need_build.push_back(subtraction_trick_nidx);
      }
    }
    this->monitor_.Stop(__func__);
    return need_build;
  }

  void AllocateHistograms(Context const* ctx, std::vector<bst_node_t> const& nodes_to_build,
                          std::vector<bst_node_t> const& nodes_to_sub) {
    this->monitor_.Start(__func__);
    std::vector<bst_node_t> all_new = nodes_to_build;
    all_new.insert(all_new.end(), nodes_to_sub.cbegin(), nodes_to_sub.cend());
    // Allocate the histograms
    // Guaranteed contiguous memory
    this->AllocateHistograms(ctx, all_new);
    this->monitor_.Stop(__func__);
  }

  void AllocateHistograms(Context const* ctx, std::vector<int> const& new_nidxs) {
    this->hist_.AllocateHistograms(ctx, new_nidxs);
  }
};
}  // namespace xgboost::tree
