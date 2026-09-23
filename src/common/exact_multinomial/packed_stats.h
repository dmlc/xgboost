/**
 * Copyright 2026, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_EXACT_MULTINOMIAL_PACKED_STATS_H_
#define XGBOOST_COMMON_EXACT_MULTINOMIAL_PACKED_STATS_H_

#include <xgboost/base.h>      // for XGBOOST_DEVICE
#include <xgboost/gradient.h>  // for ExactHessian
#include <xgboost/logging.h>   // for CHECK_EQ, CHECK_LT
#include <xgboost/span.h>      // for Span

#include <cstddef>      // for size_t
#include <type_traits>  // for remove_const_t, is_const_v, is_same_v, enable_if_t

namespace xgboost::common {
/**
 * @brief Number of free logits for a `n_classes` way softmax.
 *
 * The K x K multinomial Hessian `diag(p) - p p^T` is singular: adding a constant to every
 * logit leaves the softmax unchanged, so `1` spans its null space. Pinning the last class
 * to zero removes that direction and leaves `K - 1` free logits.
 */
[[nodiscard]] XGBOOST_DEVICE constexpr std::size_t NumFreeClasses(std::size_t n_classes) {
  return n_classes - 1;
}

/**
 * @brief Number of scalars needed for the lower triangle of a `n_free x n_free` Hessian.
 *
 * Equal to `(K - 1) * K / 2` for `K` classes. Forwards to @ref ExactHessian::PackedSize so
 * that the per-row transport and the packed views here cannot disagree about the layout.
 */
[[nodiscard]] constexpr std::size_t PackedHessianSize(std::size_t n_free) {
  return ExactHessian::PackedSize(n_free);
}

/**
 * @brief Number of scalars for one packed gradient and Hessian pair.
 */
[[nodiscard]] XGBOOST_DEVICE constexpr std::size_t PackedStatsStride(std::size_t n_free) {
  return n_free + PackedHessianSize(n_free);
}

/**
 * @brief Offset of the Hessian entry `(i, j)` inside the packed lower triangle.
 *
 * The Hessian is symmetric, so an upper triangle index is mirrored onto the lower one and
 * both orderings address the same scalar. Row `i` starts at `i * (i + 1) / 2`:
 *
 *   row 0 | 0
 *   row 1 | 1  2
 *   row 2 | 3  4  5
 */
[[nodiscard]] XGBOOST_DEVICE constexpr std::size_t PackedHessianIndex(std::size_t i,
                                                                     std::size_t j) {
  auto row = i < j ? j : i;
  auto col = i < j ? i : j;
  return row * (row + 1) / 2 + col;
}

/**
 * @brief View over the packed lower triangle of one symmetric Hessian.
 *
 * The storage is not owned. Both triangles address the same scalar through
 * @ref PackedHessianIndex, so a caller can neither transpose the matrix by accident nor
 * leave one half of it unwritten. `T` may be const to obtain a read only view.
 *
 * This is the layout shared by the per-row transport (@ref xgboost::ExactHessian) and the
 * Hessian half of @ref PackedMultinomialStats, so a row can later be accumulated into a
 * histogram bin without translating between layouts.
 */
template <typename T>
class PackedHessianView {
 public:
  using ValueT = std::remove_const_t<T>;  // NOLINT

  PackedHessianView() = default;
  /**
   * @param data   Storage of exactly `PackedHessianSize(n_free)` scalars.
   * @param n_free Number of free logits, `K - 1`.
   */
  PackedHessianView(Span<T> data, std::size_t n_free) : data_{data}, n_free_{n_free} {
    CHECK_EQ(data.size(), PackedHessianSize(n_free));
  }
  /** @brief Implicit conversion from a mutable view to a read only one. */
  template <typename U, typename = std::enable_if_t<std::is_const_v<T> &&
                                                    std::is_same_v<U, std::remove_const_t<T>>>>
  PackedHessianView(PackedHessianView<U> const& that)  // NOLINT
      : data_{that.Data()}, n_free_{that.NumFree()} {}

  [[nodiscard]] std::size_t NumFree() const { return n_free_; }
  [[nodiscard]] Span<T> Data() const { return data_; }

  /** @brief Read `H(i, j)`. Both triangles map onto the same scalar. */
  [[nodiscard]] ValueT Get(std::size_t i, std::size_t j) const {
    return data_[PackedHessianIndex(i, j)];
  }
  /** @brief Write `H(i, j)`, which also writes `H(j, i)`. */
  void Set(std::size_t i, std::size_t j, ValueT v) const { data_[PackedHessianIndex(i, j)] = v; }
  void Add(std::size_t i, std::size_t j, ValueT v) const { data_[PackedHessianIndex(i, j)] += v; }

  /**
   * @brief Total curvature over all `K` classes, not just the free ones.
   *
   * Only the `K-1` free block is stored, but the reference class's curvature is recoverable
   * from it exactly. Summing every entry of the symmetric free block gives
   * `p_ref * (1 - p_ref)`, because row `i` sums to `p_i * p_ref`. Writing `S` for the sum of
   * the packed lower triangle, the full symmetric sum is `2S - trace`, so
   *
   *   trace(H_full) = trace(H_free) + (2S - trace(H_free)) = 2S
   *
   * The identity is linear, so it survives accumulation over rows and sample weights, and
   * the result equals `sum_rows w * (1 - sum_k p_k^2)` -- the weighted Gini impurity of the
   * predicted distribution.
   *
   * Unlike the trace of the free block alone, this value does not depend on which class was
   * chosen as the reference, which is what makes it usable as a split constraint.
   */
  [[nodiscard]] ValueT TotalCurvature() const {
    ValueT sum{0};
    for (auto v : data_) {
      sum += v;
    }
    return sum + sum;
  }

 private:
  Span<T> data_;
  std::size_t n_free_{0};
};

/**
 * @brief View the `row`-th packed Hessian inside a flat buffer.
 */
template <typename T>
[[nodiscard]] PackedHessianView<T> PackedHessianAtRow(Span<T> buffer, std::size_t n_free,
                                                      std::size_t row) {
  auto size = PackedHessianSize(n_free);
  return {buffer.subspan(row * size, size), n_free};
}

/**
 * @brief View over the packed gradient and Hessian of one exact multinomial statistic.
 *
 * The last class is the reference class, leaving `n_free = K - 1` free logits. A single
 * contiguous buffer holds the gradient followed by the lower triangle of the Hessian:
 *
 *   [ g_0 .. g_{n-1} | h_00 | h_10 h_11 | h_20 h_21 h_22 | .. ]
 *
 * The view does not own its storage. A histogram allocates `n_bins * PackedStatsStride()`
 * scalars once and hands each bin the matching sub-span, so accumulation never allocates
 * per bin. `T` may be const to obtain a read only view.
 */
template <typename T>
class PackedMultinomialStats {
 public:
  using ValueT = std::remove_const_t<T>;  // NOLINT

  PackedMultinomialStats() = default;
  /**
   * @param data   Storage of exactly `PackedStatsStride(n_free)` scalars.
   * @param n_free Number of free logits, `K - 1`.
   */
  PackedMultinomialStats(Span<T> data, std::size_t n_free) : data_{data}, n_free_{n_free} {
    CHECK_EQ(data.size(), PackedStatsStride(n_free));
  }
  /** @brief Implicit conversion from a mutable view to a read only one. */
  template <typename U, typename = std::enable_if_t<std::is_const_v<T> &&
                                                    std::is_same_v<U, std::remove_const_t<T>>>>
  PackedMultinomialStats(PackedMultinomialStats<U> const& that)  // NOLINT
      : data_{that.Data()}, n_free_{that.NumFree()} {}

  [[nodiscard]] std::size_t NumFree() const { return n_free_; }
  /** @brief The full packed buffer, gradient first. */
  [[nodiscard]] Span<T> Data() const { return data_; }
  [[nodiscard]] Span<T> Gradient() const { return data_.subspan(0, n_free_); }
  /** @brief The packed lower triangle of the Hessian. */
  [[nodiscard]] Span<T> Hessian() const {
    return data_.subspan(n_free_, PackedHessianSize(n_free_));
  }
  /** @brief The Hessian half as a symmetric view over this statistic's storage. */
  [[nodiscard]] PackedHessianView<T> HessianView() const { return {this->Hessian(), n_free_}; }

  [[nodiscard]] ValueT GetGradient(std::size_t i) const { return data_[i]; }
  void SetGradient(std::size_t i, ValueT v) const { data_[i] = v; }
  void AddGradient(std::size_t i, ValueT v) const { data_[i] += v; }

  /** @brief Read `H(i, j)`. Both triangles map onto the same scalar. */
  [[nodiscard]] ValueT GetHessian(std::size_t i, std::size_t j) const {
    return this->HessianView().Get(i, j);
  }
  /** @brief Write `H(i, j)`, which also writes `H(j, i)`. */
  void SetHessian(std::size_t i, std::size_t j, ValueT v) const {
    this->HessianView().Set(i, j, v);
  }
  void AddHessian(std::size_t i, std::size_t j, ValueT v) const {
    this->HessianView().Add(i, j, v);
  }
  /** @brief @see PackedHessianView::TotalCurvature. */
  [[nodiscard]] ValueT TotalCurvature() const { return this->HessianView().TotalCurvature(); }

  /** @brief Accumulate `that` into this statistic. */
  void Add(PackedMultinomialStats<T const> that) const {
    CHECK_EQ(that.NumFree(), n_free_);
    auto rhs = that.Data();
    for (std::size_t i = 0, n = data_.size(); i < n; ++i) {
      data_[i] += rhs[i];
    }
  }
  /** @brief Subtract `that` from this statistic, used for sibling histograms. */
  void Subtract(PackedMultinomialStats<T const> that) const {
    CHECK_EQ(that.NumFree(), n_free_);
    auto rhs = that.Data();
    for (std::size_t i = 0, n = data_.size(); i < n; ++i) {
      data_[i] -= rhs[i];
    }
  }
  /** @brief Reset the gradient and the Hessian to zero. */
  void Zero() const {
    for (std::size_t i = 0, n = data_.size(); i < n; ++i) {
      data_[i] = ValueT{0};
    }
  }

  [[nodiscard]] PackedMultinomialStats<T const> AsConst() const { return {data_, n_free_}; }

 private:
  Span<T> data_;
  std::size_t n_free_{0};
};

/**
 * @brief View the `bin`-th statistic inside a flat histogram buffer.
 */
template <typename T>
[[nodiscard]] PackedMultinomialStats<T> PackedStatsAtBin(Span<T> buffer, std::size_t n_free,
                                                         std::size_t bin) {
  auto stride = PackedStatsStride(n_free);
  return {buffer.subspan(bin * stride, stride), n_free};
}
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_EXACT_MULTINOMIAL_PACKED_STATS_H_
