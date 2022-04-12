/*!
 * Copyright 2022 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_STATS_H_
#define XGBOOST_COMMON_STATS_H_
#include <iterator>
#include <limits>
#include <vector>

#include "common.h"
#include "xgboost/linalg.h"

namespace xgboost {
namespace common {

template <typename Fn>
class IndexTransformIter {
  size_t iter_{0};
  Fn fn_;

 public:
  using iterator_category = std::random_access_iterator_tag;  // NOLINT
  using value_type = std::result_of_t<Fn(size_t)>;            // NOLINT
  using difference_type = detail::ptrdiff_t;                  // NOLINT
  using reference = std::add_lvalue_reference_t<value_type>;  // NOLINT
  using pointer = std::add_pointer_t<value_type>;             // NOLINT

 public:
  XGBOOST_DEVICE explicit IndexTransformIter(Fn&& fn) : fn_{fn} {}
  IndexTransformIter(IndexTransformIter const&) = default;

  value_type operator*() const { return fn_(iter_); }

  XGBOOST_DEVICE auto operator-(IndexTransformIter const& that) const { return iter_ - that.iter_; }

  XGBOOST_DEVICE IndexTransformIter& operator++() {
    iter_++;
    return *this;
  }
  XGBOOST_DEVICE IndexTransformIter operator++(int) {
    auto ret = *this;
    ++(*this);
    return ret;
  }
  XGBOOST_DEVICE IndexTransformIter& operator+=(difference_type n) {
    iter_ += n;
    return *this;
  }
  XGBOOST_DEVICE IndexTransformIter& operator-=(difference_type n) {
    (*this) += -n;
    return *this;
  }
  XGBOOST_DEVICE IndexTransformIter operator+(difference_type n) const {
    auto ret = *this;
    return ret += n;
  }
  XGBOOST_DEVICE IndexTransformIter operator-(difference_type n) const {
    auto ret = *this;
    return ret -= n;
  }
};

template <typename Fn>
auto MakeIndexTransformIter(Fn&& fn) {
  return IndexTransformIter<Fn>(std::forward<Fn>(fn));
}

/**
 * \brief Percentile with masked array using linear interpolation.
 *
 *   https://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm
 *
 * \param alpha Percentile, must be in range [0, 1].
 * \param begin Iterator begin for input array.
 * \param end   Iterator end for input array.
 *
 * \return The result of interpolation.
 */
template <typename Iter>
float Percentile(double alpha, Iter const& begin, Iter const& end) {
  CHECK(alpha >= 0 && alpha <= 1);
  auto n = static_cast<double>(std::distance(begin, end));
  if (n == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  std::vector<size_t> sorted_idx(n);
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                   [&](size_t l, size_t r) { return *(begin + l) < *(begin + r); });
  std::cout << "CPU" << std::endl;
  for (auto v : sorted_idx) {
    std::cout << v << ", ";
  }
  std::cout << std::endl;

  auto val = [&](size_t i) { return *(begin + sorted_idx[i]); };
  static_assert(std::is_same<decltype(val(0)), float>::value, "");

  if (alpha <= (1 / (n + 1))) {
    return val(0);
  }
  if (alpha >= (n / (n + 1))) {
    return val(sorted_idx.size() - 1);
  }

  double x = alpha * static_cast<double>((n + 1));
  double k = std::floor(x) - 1;
  CHECK_GE(k, 0);
  double d = (x - 1) - k;

  auto v0 = val(static_cast<size_t>(k));
  auto v1 = val(static_cast<size_t>(k) + 1);
  return v0 + d * (v1 - v0);
}

inline float WeightedPercentile(float quantile, common::Span<size_t const> row_set,
                                linalg::VectorView<float const> labels,
                                linalg::VectorView<float const> weights) {
  std::vector<size_t> sorted_idx(row_set.size());
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                   [&](size_t i, size_t j) { return labels(row_set[i]) < labels(row_set[j]); });
  std::vector<float> weighted_cdf(row_set.size());
  weighted_cdf[0] = weights(row_set[sorted_idx[0]]);
  for (size_t i = 1; i < row_set.size(); ++i) {
    weighted_cdf[i] = weighted_cdf[i - 1] + weights(row_set[sorted_idx[i]]);
  }
  float thresh = weighted_cdf.back() * quantile;
  size_t pos =
      std::upper_bound(weighted_cdf.cbegin(), weighted_cdf.cend(), thresh) - weighted_cdf.cbegin();
  pos = std::min(pos, static_cast<size_t>(row_set.size() - 1));
  if (pos == 0 || pos == static_cast<size_t>(row_set.size() - 1)) {
    return labels(row_set[sorted_idx[pos]]);
  }
  CHECK_GE(thresh, weighted_cdf[pos - 1]);
  CHECK_LT(thresh, weighted_cdf[pos]);
  float v1 = labels(row_set[sorted_idx[pos - 1]]);
  float v2 = labels(row_set[sorted_idx[pos]]);
  if (weighted_cdf[pos + 1] - weighted_cdf[pos] >= 1.0f) {
    return (thresh - weighted_cdf[pos]) / (weighted_cdf[pos + 1] - weighted_cdf[pos]) * (v2 - v2) +
           v1;
  } else {
    return v2;
  }
}
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_STATS_H_
