/**
 * Copyright 2023-2025, XGBoost contributors
 */
#pragma once

#include <istream>  // for istream
#include <ostream>  // for ostream
#include <string>   // for string
#include <utility>  // for forward
#include <vector>   // for vector

#include "xgboost/string_view.h"  // for StringView

namespace xgboost::common {
/**
 * @brief A shim to enable ADL for parameter parsing. Alternatively, we can put the stream
 * operators in std namespace, which seems to be less ideal.
 *
 * @tparam scalar_compatible To help avoid breaking change for parameters that are used to be
 * scalars.
 */
template <typename T, bool scalar_compatible>
class ParamArray {
  std::string name_;
  std::vector<T> values_;

 public:
  using size_type = typename decltype(values_)::size_type;              // NOLINT
  using const_reference = typename decltype(values_)::const_reference;  // NOLINT

 public:
  ParamArray() = default;

  ParamArray(ParamArray const& that) = default;
  ParamArray& operator=(ParamArray const& that) = default;

  ParamArray(ParamArray&& that) = default;
  ParamArray& operator=(ParamArray&& that) = default;

  template <typename... Args>
  explicit ParamArray(StringView name, Args&&... args)
      : name_{name}, values_{std::forward<Args>(args)...} {}

  [[nodiscard]] std::vector<T>& Get() { return values_; }
  [[nodiscard]] std::vector<T> const& Get() const { return values_; }
  const_reference operator[](size_type i) const { return values_[i]; }
  [[nodiscard]] bool empty() const { return values_.empty(); }       // NOLINT
  [[nodiscard]] std::size_t size() const { return values_.size(); }  // NOLINT
  [[nodiscard]] auto data() const { return values_.data(); }         // NOLINT
  ParamArray& operator=(std::vector<T> const& that) {
    this->values_ = that;
    return *this;
  }
  [[nodiscard]] StringView Name() const { return this->name_; }
};

// For parsing array-based parameters inside DMLC parameter. Input can be a string to a
// single float or a list of floats.
std::ostream& operator<<(std::ostream& os, const ParamArray<float, false>& t);
std::ostream& operator<<(std::ostream& os, const ParamArray<float, true>& t);
std::istream& operator>>(std::istream& is, ParamArray<float, false>& t);
std::istream& operator>>(std::istream& is, ParamArray<float, true>& t);
}  // namespace xgboost::common
