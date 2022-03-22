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
  using SharedSumT = std::conditional_t<
      std::is_same<typename GradientSumT::ValueT, float>::value,
      GradientPairInt32, GradientPairInt64>;
  using T = typename GradientSumT::ValueT;

  XGBOOST_DEV_INLINE SharedSumT ToFixedPoint(GradientPair const& gpair) const {
    auto adjusted = SharedSumT(T(gpair.GetGrad() * to_fixed_point.GetGrad()),
                               T(gpair.GetHess() * to_fixed_point.GetHess()));
    return adjusted;
  }
  XGBOOST_DEV_INLINE GradientSumT ToFloatingPoint(SharedSumT const &gpair) const {
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
