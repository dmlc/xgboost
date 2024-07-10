/**
 * Copyright 2020-2024, XGBoost Contributors
 */
#ifndef HISTOGRAM_CUH_
#define HISTOGRAM_CUH_
#include <memory>  // for unique_ptr

#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../data/ellpack_page.cuh"    // for EllpackDeviceAccessor
#include "feature_groups.cuh"             // for FeatureGroupsAccessor
#include "xgboost/base.h"                 // for GradientPair, GradientPairInt64
#include "xgboost/context.h"              // for Context
#include "xgboost/span.h"                 // for Span

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
