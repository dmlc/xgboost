/**
 * Copyright 2021-2023, XGBoost Contributors
 */
#ifndef XGBOOST_STRING_VIEW_H_
#define XGBOOST_STRING_VIEW_H_
#include <xgboost/logging.h>  // CHECK_LT
#include <xgboost/span.h>     // Span

#include <algorithm>  // for equal, min
#include <cstddef>    // for size_t
#include <iterator>   // for reverse_iterator
#include <ostream>    // for ostream
#include <string>     // for char_traits, string

namespace xgboost {
struct StringView {
 private:
  using CharT = char;
  using Traits = std::char_traits<CharT>;
  CharT const* str_{nullptr};
  std::size_t size_{0};

 public:
  using value_type = CharT;                                        // NOLINT
  using iterator = const CharT*;                                   // NOLINT
  using const_iterator = iterator;                                 // NOLINT
  using reverse_iterator = std::reverse_iterator<const_iterator>;  // NOLINT
  using const_reverse_iterator = reverse_iterator;                 // NOLINT

 public:
  constexpr StringView() = default;
  constexpr StringView(value_type const* str, std::size_t size) : str_{str}, size_{size} {}
  StringView(std::string const& str) : str_{str.c_str()}, size_{str.size()} {}  // NOLINT
  constexpr StringView(value_type const* str)                                   // NOLINT
      : str_{str}, size_{str == nullptr ? 0ul : Traits::length(str)} {}

  [[nodiscard]] value_type const& operator[](std::size_t p) const { return str_[p]; }
  [[nodiscard]] explicit operator std::string() const { return {this->c_str(), this->size()}; }
  [[nodiscard]] value_type const& at(std::size_t p) const {  // NOLINT
    CHECK_LT(p, size_);
    return str_[p];
  }
  [[nodiscard]] constexpr std::size_t size() const { return size_; }       // NOLINT
  [[nodiscard]] constexpr bool empty() const { return size() == 0; }       // NOLINT
  [[nodiscard]] StringView substr(std::size_t beg, std::size_t n) const {  // NOLINT
    CHECK_LE(beg, size_);
    std::size_t len = std::min(n, size_ - beg);
    return {str_ + beg, len};
  }
  [[nodiscard]] value_type const* c_str() const { return str_; }  // NOLINT

  [[nodiscard]] constexpr const_iterator cbegin() const { return str_; }         // NOLINT
  [[nodiscard]] constexpr const_iterator cend() const { return str_ + size(); }  // NOLINT
  [[nodiscard]] constexpr iterator begin() const { return str_; }                // NOLINT
  [[nodiscard]] constexpr iterator end() const { return str_ + size(); }         // NOLINT

  [[nodiscard]] const_reverse_iterator rbegin() const noexcept {  // NOLINT
    return const_reverse_iterator(this->end());
  }
  [[nodiscard]] const_reverse_iterator crbegin() const noexcept {  // NOLINT
    return const_reverse_iterator(this->end());
  }
  [[nodiscard]] const_reverse_iterator rend() const noexcept {  // NOLINT
    return const_reverse_iterator(this->begin());
  }
  [[nodiscard]] const_reverse_iterator crend() const noexcept {  // NOLINT
    return const_reverse_iterator(this->begin());
  }
};

inline std::ostream& operator<<(std::ostream& os, StringView const v) {
  for (auto c : v) {
    os.put(c);
  }
  return os;
}

inline bool operator==(StringView l, StringView r) {
  if (l.size() != r.size()) {
    return false;
  }
  return std::equal(l.cbegin(), l.cend(), r.cbegin());
}

inline bool operator!=(StringView l, StringView r) { return !(l == r); }

inline bool operator<(StringView l, StringView r) {
  return common::Span<StringView::value_type const>{l.c_str(), l.size()} <
         common::Span<StringView::value_type const>{r.c_str(), r.size()};
}

inline bool operator<(std::string const& l, StringView r) { return StringView{l} < r; }

inline bool operator<(StringView l, std::string const& r) { return l < StringView{r}; }
}  // namespace xgboost
#endif  // XGBOOST_STRING_VIEW_H_
