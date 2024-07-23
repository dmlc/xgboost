/**
 * Copyright 2020-2024, XGBoost Contributors
 */
#ifndef HISTOGRAM_CUH_
#define HISTOGRAM_CUH_
#include <memory>  // for unique_ptr

#include "../../common/cuda_context.cuh"    // for CUDAContext
#include "../../common/device_helpers.cuh"  // for LaunchN
#include "../../common/device_vector.cuh"   // for device_vector
#include "../../data/ellpack_page.cuh"      // for EllpackDeviceAccessor
#include "feature_groups.cuh"               // for FeatureGroupsAccessor
#include "xgboost/base.h"                   // for GradientPair, GradientPairInt64
#include "xgboost/context.h"                // for Context
#include "xgboost/span.h"                   // for Span

namespace xgboost::tree {
/**
 * \brief An atomicAdd designed for gradient pair with better performance.  For general
 *        int64_t atomicAdd, one can simply cast it to unsigned long long. Exposed for testing.
 */
XGBOOST_DEV_INLINE void AtomicAdd64As32(int64_t* dst, int64_t src) {
  uint32_t* y_low = reinterpret_cast<uint32_t*>(dst);
  uint32_t* y_high = y_low + 1;

  auto cast_src = reinterpret_cast<uint64_t *>(&src);

  uint32_t const x_low = static_cast<uint32_t>(src);
  uint32_t const x_high = (*cast_src) >> 32;

  auto const old = atomicAdd(y_low, x_low);
  uint32_t const carry = old > (std::numeric_limits<uint32_t>::max() - x_low) ? 1 : 0;
  uint32_t const sig = x_high + carry;
  atomicAdd(y_high, sig);
}

class GradientQuantiser {
 private:
  /* Convert gradient to fixed point representation. */
  GradientPairPrecise to_fixed_point_;
  /* Convert fixed point representation back to floating point. */
  GradientPairPrecise to_floating_point_;

 public:
  GradientQuantiser(Context const* ctx, common::Span<GradientPair const> gpair, MetaInfo const& info);
  [[nodiscard]] XGBOOST_DEVICE GradientPairInt64 ToFixedPoint(GradientPair const& gpair) const {
    auto adjusted = GradientPairInt64(gpair.GetGrad() * to_fixed_point_.GetGrad(),
                                      gpair.GetHess() * to_fixed_point_.GetHess());
    return adjusted;
  }
  [[nodiscard]] XGBOOST_DEVICE GradientPairInt64
  ToFixedPoint(GradientPairPrecise const& gpair) const {
    auto adjusted = GradientPairInt64(gpair.GetGrad() * to_fixed_point_.GetGrad(),
                                      gpair.GetHess() * to_fixed_point_.GetHess());
    return adjusted;
  }
  [[nodiscard]] XGBOOST_DEVICE GradientPairPrecise
  ToFloatingPoint(const GradientPairInt64& gpair) const {
    auto g = gpair.GetQuantisedGrad() * to_floating_point_.GetGrad();
    auto h = gpair.GetQuantisedHess() * to_floating_point_.GetHess();
    return {g,h};
  }
};

/**
 * @brief Data storage for node histograms on device. Automatically expands.
 *
 * @tparam kStopGrowingSize  Do not grow beyond this size
 *
 * @author  Rory
 * @date    28/07/2018
 */
template <size_t kStopGrowingSize = 1 << 28>
class DeviceHistogramStorage {
 private:
  using GradientSumT = GradientPairInt64;
  /** @brief Map nidx to starting index of its histogram. */
  std::map<int, size_t> nidx_map_;
  // Large buffer of zeroed memory, caches histograms
  dh::device_vector<typename GradientSumT::ValueT> data_;
  // If we run out of storage allocate one histogram at a time
  // in overflow. Not cached, overwritten when a new histogram
  // is requested
  dh::device_vector<typename GradientSumT::ValueT> overflow_;
  std::map<int, size_t> overflow_nidx_map_;
  int n_bins_;
  DeviceOrd device_id_;
  static constexpr size_t kNumItemsInGradientSum =
      sizeof(GradientSumT) / sizeof(typename GradientSumT::ValueT);
  static_assert(kNumItemsInGradientSum == 2, "Number of items in gradient type should be 2.");

 public:
  // Start with about 16mb
  DeviceHistogramStorage() { data_.reserve(1 << 22); }
  void Init(DeviceOrd device_id, int n_bins) {
    this->n_bins_ = n_bins;
    this->device_id_ = device_id;
  }

  void Reset(Context const* ctx) {
    auto d_data = data_.data().get();
    dh::LaunchN(data_.size(), ctx->CUDACtx()->Stream(),
                [=] __device__(size_t idx) { d_data[idx] = 0.0f; });
    nidx_map_.clear();
    overflow_nidx_map_.clear();
  }
  [[nodiscard]] bool HistogramExists(int nidx) const {
    return nidx_map_.find(nidx) != nidx_map_.cend() ||
           overflow_nidx_map_.find(nidx) != overflow_nidx_map_.cend();
  }
  [[nodiscard]] int Bins() const { return n_bins_; }
  [[nodiscard]] size_t HistogramSize() const { return n_bins_ * kNumItemsInGradientSum; }
  dh::device_vector<typename GradientSumT::ValueT>& Data() { return data_; }

  void AllocateHistograms(Context const* ctx, const std::vector<int>& new_nidxs) {
    for (int nidx : new_nidxs) {
      CHECK(!HistogramExists(nidx));
    }
    // Number of items currently used in data
    const size_t used_size = nidx_map_.size() * HistogramSize();
    const size_t new_used_size = used_size + HistogramSize() * new_nidxs.size();
    if (used_size >= kStopGrowingSize) {
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
      return {reinterpret_cast<GradientSumT*>(ptr), static_cast<std::size_t>(n_bins_)};
    } else {
      // Fetch from overflow
      auto ptr = overflow_.data().get() + overflow_nidx_map_.at(nidx);
      return {reinterpret_cast<GradientSumT*>(ptr), static_cast<std::size_t>(n_bins_)};
    }
  }
};

class DeviceHistogramBuilderImpl;

class DeviceHistogramBuilder {
  std::unique_ptr<DeviceHistogramBuilderImpl> p_impl_;

 public:
  DeviceHistogramBuilder();
  ~DeviceHistogramBuilder();

  void Reset(Context const* ctx, FeatureGroupsAccessor const& feature_groups,
             bool force_global_memory);
  void BuildHistogram(CUDAContext const* ctx, EllpackDeviceAccessor const& matrix,
                      FeatureGroupsAccessor const& feature_groups,
                      common::Span<GradientPair const> gpair,
                      common::Span<const std::uint32_t> ridx,
                      common::Span<GradientPairInt64> histogram, GradientQuantiser rounding);
};
}  // namespace xgboost::tree
#endif  // HISTOGRAM_CUH_
