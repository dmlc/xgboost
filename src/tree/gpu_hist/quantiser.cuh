/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#pragma once
#include <cstdint>  // for int32_t, int64_t
#include <vector>   // for vector

#include "../../common/deterministic.cuh"   // for CreateRoundingFactor
#include "../../common/device_helpers.cuh"  // for ToSpan
#include "../../common/device_vector.cuh"   // for device_vector, DeviceUVector
#include "xgboost/base.h"                   // for GradientPairPrecise, GradientPairInt64
#include "xgboost/context.h"                // for Context
#include "xgboost/linalg.h"                 // for VectorView, MatrixView

namespace xgboost::tree {

/**
 * @brief A simple quantiser for single float values to enable deterministic summation.
 *
 * Similar to GradientQuantiser but for a single float channel.
 */
struct FloatQuantiser {
  double to_fixed_point;
  double to_floating_point;
  FloatQuantiser(double max_abs, bst_idx_t n) {
    auto rounding = common::CreateRoundingFactor<double>(max_abs, n);
    // See the gradient quantizer for details.
    constexpr std::int64_t kMaxInt = static_cast<std::int64_t>(1) << 62;
    to_floating_point = rounding / static_cast<double>(kMaxInt);
    to_fixed_point = static_cast<double>(1.0) / to_floating_point;
  }
};
// Functors that can be easily passed into thrust algorithms
struct ToFixedPointOp {
  double factor;
  explicit ToFixedPointOp(FloatQuantiser const& q) : factor{q.to_fixed_point} {}
  XGBOOST_DEVICE std::int64_t operator()(double val) const {
    return static_cast<std::int64_t>(val * factor);
  }
};
struct ToFloatingPointOp {
  double factor;
  explicit ToFloatingPointOp(FloatQuantiser const& q) : factor{q.to_floating_point} {}
  XGBOOST_DEVICE double operator()(std::int64_t val) const {
    return static_cast<double>(val) * factor;
  }
};

/**
 * @brief The most rows whose values a 32-bit shared-memory histogram bin may hold between
 *        flushes. A bin receives at most one value from each row.
 */
inline constexpr std::int64_t kMaxSharedRows = 8192;

/**
 * @brief How the histogram kernel narrows fixed-point gradients into 32-bit shared bins.
 *
 * A gradient pair is narrow when both values are multiples of their `2^shift`, the gradient
 * satisfies `-2^bits_grad <= grad >> shift < 2^bits_grad` and the Hessian
 * `0 <= hess >> shift < 2^bits_hess`. The kernel adds `value >> shift` to a 32-bit bin and adds
 * any other pair straight to the global histogram. A bin holds at most `window_rows` values
 * between flushes, and the bits keep `window_rows * 2^bits_grad` within int32 and
 * `window_rows * (2^bits_hess - 1)` within uint32, so no bin can overflow.
 *
 * `GradientQuantiser::ToRoundedFixedPoint` gives every pair that is not narrow an odd gradient.
 * The gradient shift is at least one, so narrow gradients are even, and the kernel tells the two
 * apart by the parity of the gradient alone. Gradients for the histograms must come from it.
 */
struct SharedBinRounding {
  // At least one.
  std::int8_t shift_grad{1};
  std::int8_t shift_hess{0};
  // The defaults leave almost no pair narrow.
  std::int8_t bits_grad{0};
  std::int8_t bits_hess{0};
  // At least 2.
  std::int32_t window_rows{static_cast<std::int32_t>(kMaxSharedRows)};

  /** @brief Whether the pair is narrow, see above. */
  [[nodiscard]] XGBOOST_DEVICE bool IsNarrow(std::int64_t grad, std::int64_t hess) const {
    auto narrow_grad = grad >> shift_grad, narrow_hess = hess >> shift_hess;
    // Shift back unsigned, a left shift of a negative value is undefined.
    auto back = [](std::int64_t v, std::int8_t shift) {
      return static_cast<std::int64_t>(static_cast<std::uint64_t>(v) << shift);
    };
    // Multiples of 2^shift; a gradient whose top bits are all zero or all one; a Hessian whose
    // top bits are zero.
    return back(narrow_grad, shift_grad) == grad && back(narrow_hess, shift_hess) == hess &&
           static_cast<std::uint64_t>((narrow_grad >> bits_grad) + 1) <= 1 &&
           (narrow_hess >> bits_hess) == 0;
  }
};

/**
 * @brief Per-target quantiser for converting gradients between floating-point and
 *        fixed-point representations.
 *
 * The gradients of the rows are converted with `ToRoundedFixedPoint`: most are rounded to a
 * multiple of `2^shift` fixed-point units so that the histogram kernel can sum them in 32-bit
 * shared memory (see @ref SharedBinRounding), and values too large for that keep the full
 * fixed-point precision. Every user of the quantised gradients reads the same values, so
 * histograms and node sums stay consistent.
 */
class GradientQuantiser {
 private:
  /* Convert gradient to fixed point representation. */
  GradientPairPrecise to_fixed_point_;
  /* Convert fixed point representation back to floating point. */
  GradientPairPrecise to_floating_point_;
  SharedBinRounding narrow_;

  template <typename GradientPairT>
  [[nodiscard]] XGBOOST_DEVICE GradientPairInt64
  ToFixedPointImpl(GradientPairT const& gpair) const {
    auto grad = static_cast<GradientPairInt64::ValueT>(gpair.GetGrad() * to_fixed_point_.GetGrad());
    auto hess = static_cast<GradientPairInt64::ValueT>(gpair.GetHess() * to_fixed_point_.GetHess());
    // Preserve positive curvature through fixed-point truncation.
    if (gpair.GetHess() > 0.0 && hess == 0) {
      hess = 1;
    }
    return {grad, hess};
  }

 public:
  GradientQuantiser() = default;
  // Used for test. No value is narrow.
  GradientQuantiser(GradientPairPrecise to_fixed, GradientPairPrecise to_float)
      : GradientQuantiser{to_fixed, to_float, SharedBinRounding{}} {}
  GradientQuantiser(GradientPairPrecise to_fixed, GradientPairPrecise to_float,
                    SharedBinRounding narrow)
      : to_fixed_point_{to_fixed}, to_floating_point_{to_float}, narrow_{narrow} {}
  [[nodiscard]] XGBOOST_DEVICE SharedBinRounding const& Narrow() const { return narrow_; }
  /**
   * @brief Convert a row's gradient for the histograms: a multiple of 2^shift fixed-point units
   *        when it is narrow, otherwise the full fixed-point value with the lowest bit of the
   *        gradient set. See @ref SharedBinRounding.
   */
  template <typename GradientPairT>
  [[nodiscard]] XGBOOST_DEVICE GradientPairInt64
  ToRoundedFixedPoint(GradientPairT const& gpair) const {
    using T = GradientPairInt64::ValueT;
    auto exact = this->ToFixedPointImpl(gpair);
    T grad = exact.GetQuantisedGrad(), hess = exact.GetQuantisedHess();
    // Truncate toward zero to a multiple of 2^shift, as `%` keeps the sign of the dividend.
    grad -= grad % (T{1} << narrow_.shift_grad);
    hess -= hess % (T{1} << narrow_.shift_hess);
    // Preserve positive curvature through truncation.
    if (gpair.GetHess() > 0.0 && hess == 0) {
      hess = T{1} << narrow_.shift_hess;
    }
    if (narrow_.IsNarrow(grad, hess)) {
      return {grad, hess};
    }
    // An odd gradient marks the pair for the global histogram. The narrow ones are even, since
    // the gradient shift is at least one.
    return {exact.GetQuantisedGrad() | 1, exact.GetQuantisedHess()};
  }
  [[nodiscard]] XGBOOST_DEVICE GradientPairInt64 ToFixedPoint(GradientPair const& gpair) const {
    return this->ToFixedPointImpl(gpair);
  }
  [[nodiscard]] XGBOOST_DEVICE GradientPairInt64
  ToFixedPoint(GradientPairPrecise const& gpair) const {
    return this->ToFixedPointImpl(gpair);
  }
  [[nodiscard]] XGBOOST_DEVICE GradientPairPrecise
  ToFloatingPoint(const GradientPairInt64& gpair) const {
    auto g = gpair.GetQuantisedGrad() * to_floating_point_.GetGrad();
    auto h = gpair.GetQuantisedHess() * to_floating_point_.GetHess();
    return {g, h};
  }
};

/**
 * @brief Unified quantiser group for single-target and multi-target gradient quantisation.
 */
class GradientQuantiserGroup {
 private:
  std::vector<GradientQuantiser> h_quantizers_;
  dh::DeviceUVector<GradientQuantiser> d_quantizers_;

 public:
  /** @brief Construct from a gradient matrix (n_samples x n_targets). */
  GradientQuantiserGroup(Context const* ctx, linalg::MatrixView<GradientPair const> gpair);
  /** @brief Convenience constructor from a vector (single-target). */
  GradientQuantiserGroup(Context const* ctx, linalg::VectorView<GradientPair const> gpair);

  [[nodiscard]] common::Span<GradientQuantiser const> DeviceSpan() const {
    return dh::ToSpan(this->d_quantizers_);
  }
  [[nodiscard]] GradientQuantiser const& operator[](bst_target_t t) const {
    return this->h_quantizers_[t];
  }
  [[nodiscard]] bst_target_t Size() const { return this->h_quantizers_.size(); }
};

void CalcQuantizedGpairs(Context const* ctx, linalg::MatrixView<GradientPair const> gpairs,
                         common::Span<GradientQuantiser const> roundings,
                         linalg::Matrix<GradientPairInt64>* p_out);
}  // namespace xgboost::tree
