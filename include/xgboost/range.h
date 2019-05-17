/*!
 * Copyright 2015-2019 XGBoost contributors
 */
#ifndef XGBOOST_RANGE_H_
#define XGBOOST_RANGE_H_

#include <xgboost/base.h>
#include <cinttypes>

namespace xgboost {
namespace common {

/*
 * Range iterator
 */
class Range {
 public:
  using DifferenceType = int64_t;

  class Iterator {
    friend class Range;

   public:
    XGBOOST_DEVICE DifferenceType operator*() const { return i_; }
    XGBOOST_DEVICE const Iterator &operator++() {
      i_ += step_;
      return *this;
    }
    XGBOOST_DEVICE Iterator operator++(int) {
      Iterator res {*this};
      i_ += step_;
      return res;
    }

    XGBOOST_DEVICE bool operator==(const Iterator &other) const {
      return i_ >= other.i_;
    }
    XGBOOST_DEVICE bool operator!=(const Iterator &other) const {
      return i_ < other.i_;
    }

    XGBOOST_DEVICE void Step(DifferenceType s) { step_ = s; }

   protected:
    XGBOOST_DEVICE explicit Iterator(DifferenceType start) : i_(start) {}
    XGBOOST_DEVICE explicit Iterator(DifferenceType start, DifferenceType step) :
        i_{start}, step_{step} {}

   public:
    int64_t i_;
    DifferenceType step_ = 1;
  };

  XGBOOST_DEVICE Iterator begin() const { return begin_; }  // NOLINT
  XGBOOST_DEVICE Iterator end() const { return end_; }      // NOLINT

  XGBOOST_DEVICE Range(DifferenceType begin, DifferenceType end)
      : begin_(begin), end_(end) {}
  XGBOOST_DEVICE Range(DifferenceType begin, DifferenceType end,
                       DifferenceType step)
      : begin_(begin, step), end_(end) {}

  XGBOOST_DEVICE bool operator==(const Range& other) const {
    return *begin_ == *other.begin_ && *end_ == *other.end_;
  }
  XGBOOST_DEVICE bool operator!=(const Range& other) const {
    return !(*this == other);
  }

  XGBOOST_DEVICE void Step(DifferenceType s) { begin_.Step(s); }

 private:
  Iterator begin_;
  Iterator end_;
};

}      // namespace common
}      // namespace xgboost
#endif  // XGBOOST_RANGE_H_
