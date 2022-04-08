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
  XGBOOST_DEVICE IndexTransformIter(IndexTransformIter const&) = default;

  XGBOOST_DEVICE value_type operator*() const { return fn_(iter_); }

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
 * \param alpha percentile, must be in range [0, 1].
 * \param index The index of valid elements in arr.
 * \param arr   Input values.
 *
 * \return The result of interpolation.
 */
template <typename Iter>
float Percentile(double alpha, Iter begin, Iter end) {
  CHECK(alpha >= 0 && alpha <= 1);
  auto n = static_cast<double>(std::distance(begin, end));
  if (n == 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  std::vector<size_t> sorted_idx(n);
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                   [&](size_t l, size_t r) { return *(begin + l) < *(begin + r); });

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
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_STATS_H_
