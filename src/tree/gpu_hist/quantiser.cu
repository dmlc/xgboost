/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <thrust/copy.h>                         // for copy_n
#include <thrust/iterator/transform_iterator.h>  // for make_transform_iterator

#include <algorithm>            // for clamp, min, max
#include <cmath>                // for frexp, ldexp, isfinite
#include <cstdint>              // uint32_t, int32_t
#include <cuda/functional>      // for proclaim_copyable_arguments
#include <cuda/std/functional>  // for plus
#include <limits>               // for numeric_limits
#include <utility>              // for pair
#include <vector>               // for vector

#include "../../collective/aggregator.h"
#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../common/deterministic.cuh"
#include "../../common/device_helpers.cuh"
#include "../../common/linalg_op.cuh"  // for tbegin, tcbegin
#include "quantiser.cuh"
#include "xgboost/base.h"

namespace xgboost::tree {
namespace {
struct Pair {
  GradientPairPrecise first;
  GradientPairPrecise second;
};
__host__ XGBOOST_DEV_INLINE Pair operator+(Pair const& lhs, Pair const& rhs) {
  return {lhs.first + rhs.first, lhs.second + rhs.second};
}

// Sums of the positive and negative values, and the largest magnitudes.
struct Stats {
  Pair sums;
  float max_grad;
  float max_hess;
};
struct CombineStats {
  __host__ XGBOOST_DEV_INLINE Stats operator()(Stats const& lhs, Stats const& rhs) const {
    return {lhs.sums + rhs.sums, fmaxf(lhs.max_grad, rhs.max_grad),
            fmaxf(lhs.max_hess, rhs.max_hess)};
  }
};

struct Clip {
  static XGBOOST_DEV_INLINE float Pclip(float v) { return v > 0 ? v : 0; }
  static XGBOOST_DEV_INLINE float Nclip(float v) { return v < 0 ? abs(v) : 0; }

  XGBOOST_DEV_INLINE Stats operator()(GradientPair x) const {
    auto pg = Pclip(x.GetGrad());
    auto ph = Pclip(x.GetHess());

    auto ng = Nclip(x.GetGrad());
    auto nh = Nclip(x.GetHess());

    return {{GradientPairPrecise{pg, ph}, GradientPairPrecise{ng, nh}},
            fabsf(x.GetGrad()),
            fabsf(x.GetHess())};
  }
};

/**
 * In algorithm 5 (see common::CreateRoundingFactor) the bound is calculated as
 * $max(|v_i|) * n$.  Here we use the bound:
 *
 * @begin{equation}
 *   max( fl(\sum^{V}_{v_i>0}{v_i}), fl(\sum^{V}_{v_i<0}|v_i|) )
 * @end{equation}
 *
 * to avoid outliers, as the full reduction is reproducible on GPU with reduction tree.
 */
Stats MakeQuantiserForTarget(Context const* ctx, linalg::VectorView<GradientPair const> gpair) {
  auto beg = thrust::make_transform_iterator(linalg::tcbegin(gpair), Clip{});
  return dh::Reduce(ctx->CUDACtx()->CTP(), beg, beg + gpair.Size(), Stats{}, CombineStats{});
}

// A gradient larger than this multiple of the mean magnitude goes straight to the global
// histogram at full precision. At most 1 / kHeavyMeanMultiple of the rows can be that large.
constexpr double kHeavyMeanMultiple = 32.0;

/**
 * @brief Choose the rounding of one channel for 32-bit shared bins.
 *
 * @param unit      The fixed-point unit, a power of two.
 * @param max_abs   The largest magnitude.
 * @param mean_abs  The mean magnitude.
 * @param window_rows The most rows a bin holds between flushes.
 * @param bin_limit The largest value of a bin: INT32_MAX for gradients, UINT32_MAX for Hessians.
 *                  Both fit `window_rows * 2^bits` magnitudes.
 *
 * @return The shift, and the bits of narrow magnitudes in shifted units.
 */
std::pair<std::int8_t, std::int8_t> NarrowChannel(double unit, double max_abs, double mean_abs,
                                                  std::int64_t window_rows,
                                                  std::int64_t bin_limit) {
  if (!(max_abs > 0.0) || !std::isfinite(max_abs) || !std::isfinite(mean_abs)) {
    // All zero, or not finite: only zero is narrow.
    return {0, 0};
  }
  // A bin holds at most window_rows values, so magnitudes up to 2^bits can't overflow it. The
  // bound of a negative int32 is one larger than that of a positive one.
  std::int8_t bits = 0;
  while (window_rows * (std::int64_t{2} << bits) <= bin_limit + 1) {
    ++bits;
  }
  // Values up to the clip are narrow. CreateRoundingFactor returns a power of two strictly
  // greater than its argument, so every value up to the clip is below 2^bits narrow units.
  auto clip = std::min(max_abs, kHeavyMeanMultiple * mean_abs);
  auto narrow_unit = common::CreateRoundingFactor<double>(clip / std::ldexp(1.0, bits), 1);
  int unit_exp, narrow_exp;
  std::frexp(unit, &unit_exp);
  std::frexp(narrow_unit, &narrow_exp);
  // A narrow unit finer than the fixed-point unit keeps the fixed-point unit: the values are then
  // smaller still.
  auto shift = static_cast<std::int8_t>(std::clamp(narrow_exp - unit_exp, 1, 62));
  return {shift, bits};
}

GradientQuantiser BuildQuantiserFromPair(Pair const& p, float max_grad, float max_hess,
                                         std::size_t total_rows) {
  using GradientSumT = GradientPairPrecise;
  using T = typename GradientSumT::ValueT;

  GradientSumT positive_sum{p.first}, negative_sum{p.second};

  auto histogram_rounding =
      GradientSumT{common::CreateRoundingFactor<T>(
                       std::max(positive_sum.GetGrad(), negative_sum.GetGrad()), total_rows),
                   common::CreateRoundingFactor<T>(
                       std::max(positive_sum.GetHess(), negative_sum.GetHess()), total_rows)};

  using IntT = typename GradientPairInt64::ValueT;

  auto to_floating_point =
      histogram_rounding /
      static_cast<T>(static_cast<IntT>(1)
                     << (sizeof(typename GradientSumT::ValueT) * 8 - 2));  // keep 1 for sign bit
  auto to_fixed_point = GradientSumT{static_cast<T>(1) / to_floating_point.GetGrad(),
                                     static_cast<T>(1) / to_floating_point.GetHess()};

  auto n = static_cast<double>(std::max<std::size_t>(total_rows, 1));
  // A block can't hold more rows than the data has, which gives small data a finer unit.
  auto window_rows =
      std::clamp<std::int64_t>(static_cast<std::int64_t>(total_rows), 2, kMaxSharedRows);
  auto [shift_grad, bits_grad] = NarrowChannel(
      to_floating_point.GetGrad(), max_grad, (positive_sum.GetGrad() + negative_sum.GetGrad()) / n,
      window_rows, std::numeric_limits<std::int32_t>::max());
  auto [shift_hess, bits_hess] = NarrowChannel(
      to_floating_point.GetHess(), max_hess, (positive_sum.GetHess() + negative_sum.GetHess()) / n,
      window_rows, std::numeric_limits<std::uint32_t>::max());
  SharedBinRounding narrow{shift_grad, shift_hess, bits_grad, bits_hess,
                           static_cast<std::int32_t>(window_rows)};
  return GradientQuantiser{to_fixed_point, to_floating_point, narrow};
}
}  // anonymous namespace

GradientQuantiserGroup::GradientQuantiserGroup(Context const* ctx,
                                               linalg::MatrixView<GradientPair const> gpair) {
  auto n_targets = gpair.Shape(1);
  CHECK_GE(n_targets, 1);

  // Local reduction per target — these are fast device-local operations.
  using ReduceT = typename GradientPairPrecise::ValueT;
  std::vector<Pair> h_pairs(n_targets);
  // The largest gradient and Hessian of each target.
  std::vector<float> h_max(2 * n_targets);
  std::size_t n_samples = gpair.Shape(0);
  for (bst_target_t t = 0; t < n_targets; ++t) {
    auto stats = MakeQuantiserForTarget(ctx, gpair.Slice(linalg::All(), t));
    h_pairs[t] = stats.sums;
    h_max[2 * t] = stats.max_grad;
    h_max[2 * t + 1] = stats.max_hess;
  }

  auto rc = collective::Success() << [&]() {
    static_assert(sizeof(Pair) == sizeof(ReduceT) * 4);
    auto casted = linalg::MakeVec(reinterpret_cast<ReduceT*>(h_pairs.data()), 4 * n_targets);
    return collective::GlobalSum(ctx, casted);
  } << [&]() {
    return collective::Allreduce(ctx, linalg::MakeVec(h_max.data(), h_max.size()),
                                 collective::Op::kMax);
  } << [&] {
    // Single GlobalSum for total_rows (shared across targets).
    return collective::GlobalSum(ctx, linalg::MakeVec(&n_samples, 1));
  };
  collective::SafeColl(rc);

  // Build quantisers on host from the reduced pairs.
  h_quantizers_.resize(n_targets);
  for (bst_target_t t = 0; t < n_targets; ++t) {
    h_quantizers_[t] =
        BuildQuantiserFromPair(h_pairs[t], h_max[2 * t], h_max[2 * t + 1], n_samples);
  }

  // Copy to device.
  d_quantizers_.resize(n_targets);
  dh::safe_cuda(cudaMemcpyAsync(d_quantizers_.data(), h_quantizers_.data(),
                                n_targets * sizeof(GradientQuantiser), cudaMemcpyHostToDevice,
                                ctx->CUDACtx()->Stream()));
}

GradientQuantiserGroup::GradientQuantiserGroup(Context const* ctx,
                                               linalg::VectorView<GradientPair const> gpair)
    : GradientQuantiserGroup(
          ctx, linalg::MakeTensorView(ctx, gpair.Values(), gpair.Size(), bst_target_t{1})) {}

void CalcQuantizedGpairs(Context const* ctx, linalg::MatrixView<GradientPair const> gpairs,
                         common::Span<GradientQuantiser const> roundings,
                         linalg::Matrix<GradientPairInt64>* p_out) {
  auto shape = gpairs.Shape();
  if (p_out->Empty()) {
    *p_out = linalg::Matrix<GradientPairInt64>{shape, ctx->Device(), linalg::kF};
  } else {
    p_out->Reshape(shape);
  }

  auto out_gpair = p_out->View(ctx->Device());
  CHECK(out_gpair.FContiguous());
  auto it = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) {
    auto [ridx, target_idx] = linalg::UnravelIndex(i, gpairs.Shape());
    auto g = gpairs(ridx, target_idx);
    return roundings[target_idx].ToRoundedFixedPoint(g);
  });
  thrust::copy_n(ctx->CUDACtx()->CTP(), it, gpairs.Size(), linalg::tbegin(out_gpair));
}
}  // namespace xgboost::tree
