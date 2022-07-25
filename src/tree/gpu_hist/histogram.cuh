/*!
 * Copyright 2020-2021 by XGBoost Contributors
 */
#ifndef HISTOGRAM_CUH_
#define HISTOGRAM_CUH_
#include <thrust/transform.h>

#include "feature_groups.cuh"

#include "../../data/ellpack_page.cuh"

namespace xgboost {
namespace tree {

template <typename T, typename U>
XGBOOST_DEV_INLINE T TruncateWithRoundingFactor(T const rounding_factor, U const x) {
  static_assert(sizeof(T) >= sizeof(U), "Rounding must have higher or equal precision.");
  return (rounding_factor + static_cast<T>(x)) - rounding_factor;
}

/**
 * Truncation factor for gradient, see comments in `CreateRoundingFactor()` for details.
 */
template <typename GradientSumT>
struct HistRounding {
  /* Factor to truncate the gradient before building histogram for deterministic result. */
  GradientSumT rounding;
  /* Convert gradient to fixed point representation. */
  GradientSumT to_fixed_point;
  /* Convert fixed point representation back to floating point. */
  GradientSumT to_floating_point;

  /* Type used in shared memory. */
  using SharedSumT = GradientPairInt32;
  using GlobalSumT = GradientPairInt64;
  using T = typename GradientSumT::ValueT;

  XGBOOST_DEV_INLINE GlobalSumT ToFixedPoint(GradientPair const& gpair) const {
    auto adjusted = GlobalSumT(T(gpair.GetGrad() * to_fixed_point.GetGrad()),
                               T(gpair.GetHess() * to_fixed_point.GetHess()));
    return adjusted;
  }
  XGBOOST_DEV_INLINE GradientSumT ToFloatingPoint(GlobalSumT const &gpair) const {
    auto g = gpair.GetGrad() * to_floating_point.GetGrad();
    auto h = gpair.GetHess() * to_floating_point.GetHess();
    GradientSumT truncated{
        TruncateWithRoundingFactor<T>(rounding.GetGrad(), g),
        TruncateWithRoundingFactor<T>(rounding.GetHess(), h),
    };
    return truncated;
  }
};

template <typename GradientSumT>
HistRounding<GradientSumT> CreateRoundingFactor(common::Span<GradientPair const> gpair);

XGBOOST_DEV_INLINE void AtomicAddGpairWithOverflow(
    xgboost::GradientPairInt32* dst_shared, xgboost::GradientPairInt64 const& gpair,
    xgboost::GradientPairPrecise* dst_global, const HistRounding<GradientPairPrecise>& rounding) {
  auto dst_ptr = reinterpret_cast<typename xgboost::GradientPairInt32::ValueT*>(dst_shared);
  int old_grad = atomicAdd(dst_ptr, static_cast<int>(gpair.GetGrad()));
  int64_t grad_diff = (old_grad + gpair.GetGrad()) - (old_grad + static_cast<int>(gpair.GetGrad()));

  int old_hess = atomicAdd(dst_ptr + 1, static_cast<int>(gpair.GetHess()));
  int64_t hess_diff = (old_hess + gpair.GetHess()) - (old_hess + static_cast<int>(gpair.GetHess()));

  if (grad_diff != 0 || hess_diff != 0) {
    auto truncated = rounding.ToFloatingPoint({grad_diff, hess_diff});
    dh::AtomicAddGpair(dst_global, truncated);
  }
}

template <typename GradientSumT>
void BuildGradientHistogram(EllpackDeviceAccessor const& matrix,
                            FeatureGroupsAccessor const& feature_groups,
                            common::Span<GradientPair const> gpair,
                            common::Span<const uint32_t> ridx,
                            common::Span<GradientSumT> histogram,
                            HistRounding<GradientSumT> rounding,
                            bool force_global_memory = false);
}  // namespace tree
}  // namespace xgboost

#endif  // HISTOGRAM_CUH_
