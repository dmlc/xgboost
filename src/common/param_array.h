/**
 * Copyright 2023-2025, XGBoost contributors
 */
#pragma once

#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <vector>   // for vector

namespace xgboost::common {
// A shim to enable ADL for parameter parsing. Alternatively, we can put the stream
// operators in std namespace, which seems to be less ideal.
template <typename T>
class ParamArray {
  std::vector<T> values_;

 public:
  using size_type = typename decltype(values_)::size_type;              // NOLINT
  using const_reference = typename decltype(values_)::const_reference;  // NOLINT

 public:
  template <typename... Args>
  explicit ParamArray(Args&&... args) : values_{std::forward<Args>(args)...} {}

  [[nodiscard]] std::vector<T>& Get() { return values_; }
  [[nodiscard]] std::vector<T> const& Get() const { return values_; }
  const_reference operator[](size_type i) const { return values_[i]; }
  [[nodiscard]] bool empty() const { return values_.empty(); }       // NOLINT
  [[nodiscard]] std::size_t size() const { return values_.size(); }  // NOLINT

  ParamArray& operator=(std::vector<T> const& that) {
    this->values_ = that;
    return *this;
  }
};

// For parsing quantile parameters. Input can be a string to a single float or a list of
// floats.
std::ostream& operator<<(std::ostream& os, const ParamArray<float>& t);
std::istream& operator>>(std::istream& is, ParamArray<float>& t);
}  // namespace xgboost::common
