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
};

template <typename GradientSumT>
HistRounding<GradientSumT> CreateRoundingFactor(common::Span<GradientPair const> gpair);

template <typename T, typename U>
XGBOOST_DEV_INLINE T TruncateWithRoundingFactor(T const rounding_factor, U const x) {
  static_assert(sizeof(T) >= sizeof(U), "Rounding must have higher or equal precision.");
  return (rounding_factor + static_cast<T>(x)) - rounding_factor;
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
