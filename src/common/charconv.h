/*!
 * Copyright 2019 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_CHARCONV_H_
#define XGBOOST_COMMON_CHARCONV_H_

#include <cstddef>
#include <system_error>
#include <iterator>
#include <limits>

#include "xgboost/logging.h"

namespace xgboost {
int32_t ToCharsFloatImpl(float f, char * const result);

struct to_chars_result {  // NOLINT
  char* ptr;
  std::errc ec;
};

template <typename T>
struct NumericLimits;

template <> struct NumericLimits<float> {
  // Unlike std::numeric_limit<float>::max_digits10, which represents the **minimum**
  // length of base10 digits that are necessary to uniquely represent all distinct values.
  // This value is used to represent the maximum length.  As sign bit occupies 1 character:
  // sign + len(str(2^24)) + decimal point + `E` + sign + len(str(2^8)) + '\0'
  static constexpr size_t kMaxDigit10Len = 16;
};

template <> struct NumericLimits<int64_t> {
  static constexpr size_t kDigit10 = 21;
};

inline to_chars_result to_chars(char  *first, char *last, float value) {  // NOLINT
  if (XGBOOST_EXPECT(!(static_cast<size_t>(last - first) >=
                       NumericLimits<float>::kMaxDigit10Len),
                     false)) {
    return {first, std::errc::value_too_large};
  }
  auto index = ToCharsFloatImpl(value, first);
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

to_chars_result ToCharsUnsignedImpl(char *first, char *last,
                                    uint64_t const value);

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
    unsigned_value = uint64_t(~value) + uint64_t(1);
  }
  return ToCharsUnsignedImpl(first, last, unsigned_value);
}
}  // namespace xgboost

#endif   // XGBOOST_COMMON_CHARCONV_H_
