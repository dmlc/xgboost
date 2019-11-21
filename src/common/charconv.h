/*!
 * Copyright 2019 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_CHARCONV_H_
#define XGBOOST_COMMON_CHARCONV_H_

#include <system_error>
#include <iterator>
#include <limits>

#include "ryu.h"
#include "luts.h"
#include "xgboost/logging.h"

namespace xgboost {

struct to_chars_result {
  char* ptr;
  std::errc ec;
};

namespace {

constexpr uint32_t ShortestDigit10Impl(uint64_t value, uint32_t n) {
  // Should trigger tail recursion optimization.
  return value < 10 ? n :
      (value < Tens(2) ? n + 1 :
       (value < Tens(3) ? n + 2 :
        (value < Tens(4) ? n + 3 :
         ShortestDigit10Impl(value / Tens(4), n + 4))));
}

constexpr uint32_t ShortestDigit10(uint64_t value) {
  return ShortestDigit10Impl(value, 1);
}

// This is an implementation for base 10 inspired by the one in libstdc++v3.  The general
// scheme is by decomposing the value into multiple combination of base (which is 10) by
// mod, until the value is lesser than 10, then last char is just char '0' (ascii 48) plus
// that value.  Other popular implementations can be found in RapidJson and libc++ (in
// llvm-project), which uses the same general work flow with the same look up table, but
// probably with better performance as they are more complicated.
inline void itoaUnsignedImpl(char* first, uint32_t length, uint64_t value) {
  uint32_t position = length - 1;
  while (value > Tens(2)) {
    auto const num = (value % Tens(2)) * 2;
    value /= Tens(2);
    first[position] = kItoaLut[num + 1];
    first[position - 1] = kItoaLut[num];
    position -= 2;
  }
  if (value > 10) {
    auto const num = value * 2;
    first[0] = kItoaLut[num];
    first[1] = kItoaLut[num + 1];
  } else {
    first[0]= '0' + static_cast<char>(value);
  }
}

inline to_chars_result toCharsUnsignedImpl(char *first, char *last,
                                           uint64_t const value) {
  const uint32_t output_len = ShortestDigit10(value);
  to_chars_result ret;
  if (XGBOOST_EXPECT(std::distance(first, last) == 0, false)) {
    ret.ec = std::errc::value_too_large;
    ret.ptr = last;
    return ret;
  }

  itoaUnsignedImpl(first, output_len, value);
  ret.ptr = first + output_len;
  ret.ec = std::errc();
  return ret;
}
}  // anonymous namespace

template <typename T>
struct NumericLimits;

template <> struct NumericLimits<int64_t> {
  static constexpr size_t kDigit10 = 21;
};

template <> struct NumericLimits<float> {
  static constexpr size_t kDigit10 = 16;
};

inline to_chars_result to_chars(char *first, char *last, int64_t value) {
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
  return toCharsUnsignedImpl(first, last, unsigned_value);
}

inline to_chars_result to_chars(char  *first, char *last, float value) {
  if (XGBOOST_EXPECT(!(last - first >= static_cast<std::ptrdiff_t>(
                                       NumericLimits<float>::kDigit10)),
                     false)) {
    return {first, std::errc::value_too_large};
  }
  auto index = toCharsFloatImpl(value, first);
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
}  // namespace xgboost

#endif   // XGBOOST_COMMON_CHARCONV_H_
