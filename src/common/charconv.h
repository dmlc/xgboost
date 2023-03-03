/*!
 * Copyright 2019 by XGBoost Contributors
 *
 * \brief Implement `std::to_chars` and `std::from_chars` for float.  Only base 10 with
 *        scientific format is supported.  The implementation guarantees roundtrip
 *        reproducibility.
 */
#ifndef XGBOOST_COMMON_CHARCONV_H_
#define XGBOOST_COMMON_CHARCONV_H_

#include <cstddef>
#include <system_error>
#include <iterator>
#include <limits>

#include "xgboost/logging.h"

namespace xgboost {

struct to_chars_result {  // NOLINT
  char* ptr;
  std::errc ec;
};

struct from_chars_result {  // NOLINT
  const char *ptr;
  std::errc ec;
};

namespace detail {
int32_t ToCharsFloatImpl(float f, char * const result);
to_chars_result ToCharsUnsignedImpl(char *first, char *last,
                                    uint64_t const value);
from_chars_result FromCharFloatImpl(const char *buffer, const int len,
                                    float *result);
}  // namespace detail

template <typename T>
struct NumericLimits;

template <> struct NumericLimits<float> {
  // Unlike std::numeric_limit<float>::max_digits10, which represents the **minimum**
  // length of base10 digits that are necessary to uniquely represent all distinct values.
  // This value is used to represent the maximum length.  As sign bit occupies 1 character:
  // sign + len(str(2^24)) + decimal point + `E` + sign + len(str(2^8)) + '\0'
  static constexpr size_t kToCharsSize = 16;
};

template <> struct NumericLimits<int64_t> {
  // From llvm libcxx: numeric_limits::digits10 returns value less on 1 than desired for
  // unsigned numbers.  For example, for 1-byte unsigned value digits10 is 2 (999 can not
  // be represented), so we need +1 here.
  static constexpr size_t kToCharsSize =
      std::numeric_limits<int64_t>::digits10 +
      3;  // +1 for minus, +1 for digits10, +1 for '\0' just to be safe.
};

inline to_chars_result to_chars(char  *first, char *last, float value) {  // NOLINT
  if (XGBOOST_EXPECT(!(static_cast<size_t>(last - first) >=
                       NumericLimits<float>::kToCharsSize),
                     false)) {
    return {first, std::errc::value_too_large};
  }
  auto index = detail::ToCharsFloatImpl(value, first);
  to_chars_result ret;
  ret.ptr = first + index;

  if (XGBOOST_EXPECT(ret.ptr < last, true)) {
    ret.ec = std::errc();
  } else {
    ret.ec =  std::errc::value_too_large;
    ret.ptr = last;
  }
  return ret;
}

inline to_chars_result to_chars(char *first, char *last, int64_t value) { // NOLINT
  if (XGBOOST_EXPECT(first == last, false)) {
    return {first, std::errc::value_too_large};
  }
  // first write '-' and convert to unsigned, then write the rest.
  if (value == 0) {
    *first = '0';
    return {std::next(first), std::errc()};
  }
  uint64_t unsigned_value = value;
  if (value < 0) {
    *first = '-';
    std::advance(first, 1);
    unsigned_value = static_cast<uint64_t>(~value) + static_cast<uint64_t>(1);
  }
  return detail::ToCharsUnsignedImpl(first, last, unsigned_value);
}

inline from_chars_result from_chars(const char *buffer, const char *end, // NOLINT
                                    float &value) {  // NOLINT
  from_chars_result res =
      detail::FromCharFloatImpl(buffer, std::distance(buffer, end), &value);
  return res;
}
}  // namespace xgboost

#endif   // XGBOOST_COMMON_CHARCONV_H_
