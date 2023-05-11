/**
 * Copyright 2022 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_TRANSFORM_ITERATOR_H_
#define XGBOOST_COMMON_TRANSFORM_ITERATOR_H_

#include <cstddef>      // std::size_t
#include <iterator>     // std::random_access_iterator_tag
#include <type_traits>  // std::result_of_t, std::add_pointer_t, std::add_lvalue_reference_t
#include <utility>      // std::forward

#include "xgboost/span.h"  // ptrdiff_t

namespace xgboost {
namespace common {
/**
 * \brief Transform iterator that takes an index and calls transform operator.
 *
 *   This is CPU-only right now as taking host device function as operator complicates the
 *   code.  For device side one can use `thrust::transform_iterator` instead.
 */
template <typename Fn>
class IndexTransformIter {
  std::size_t iter_{0};
  Fn fn_;

 public:
  using iterator_category = std::random_access_iterator_tag;  // NOLINT
  using reference = std::result_of_t<Fn(std::size_t)>;        // NOLINT
  using value_type = std::remove_cv_t<std::remove_reference_t<reference>>; // NOLINT
  using difference_type = detail::ptrdiff_t;                  // NOLINT
  using pointer = std::add_pointer_t<value_type>;             // NOLINT

 public:
  /**
   * \param op Transform operator, takes a size_t index as input.
   */
  explicit IndexTransformIter(Fn &&op) : fn_{op} {}
  IndexTransformIter(IndexTransformIter const &) = default;
  IndexTransformIter &operator=(IndexTransformIter &&) = default;
  IndexTransformIter &operator=(IndexTransformIter const &that) {
    iter_ = that.iter_;
    return *this;
  }

  reference operator*() const { return fn_(iter_); }
  reference operator[](std::size_t i) const {
    auto iter = *this + i;
    return *iter;
  }

  auto operator-(IndexTransformIter const &that) const { return iter_ - that.iter_; }
  bool operator==(IndexTransformIter const &that) const { return iter_ == that.iter_; }
  bool operator!=(IndexTransformIter const &that) const { return !(*this == that); }

  IndexTransformIter &operator++() {
    iter_++;
    return *this;
  }
  IndexTransformIter operator++(int) {
    auto ret = *this;
    ++(*this);
    return ret;
  }
  IndexTransformIter &operator+=(difference_type n) {
    iter_ += n;
    return *this;
  }
  IndexTransformIter &operator-=(difference_type n) {
    (*this) += -n;
    return *this;
  }
  IndexTransformIter operator+(difference_type n) const {
    auto ret = *this;
    return ret += n;
  }
  IndexTransformIter operator-(difference_type n) const {
    auto ret = *this;
    return ret -= n;
  }
};

template <typename Fn>
auto MakeIndexTransformIter(Fn &&fn) {
  return IndexTransformIter<Fn>(std::forward<Fn>(fn));
}
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_TRANSFORM_ITERATOR_H_
