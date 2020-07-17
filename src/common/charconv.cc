/*!
 * Copyright 2020 by XGBoost Contributors
 *
 * \brief An implemenation of Ryu algorithm:
 *
 * https://dl.acm.org/citation.cfm?id=3192369
 *
 * The code is adopted from original (half) c implementation:
 * https://github.com/ulfjack/ryu.git with some more comments and tidying.  License is
 * attached below.
 *
 * Copyright 2018 Ulf Adams
 *
 * The contents of this file may be used under the terms of the Apache License,
 * Version 2.0.
 *
 *    (See accompanying file LICENSE-Apache or copy at
 *     http: *www.apache.org/licenses/LICENSE-2.0)
 *
 * Alternatively, the contents of this file may be used under the terms of
 * the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE-Boost or copy at
 *     https://www.boost.org/LICENSE_1_0.txt)
 *
 * Unless required by applicable law or agreed to in writing, this software
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.
 */
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <cmath>

#include "xgboost/logging.h"
#include "charconv.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

/*
 * We did some cleanup from the original implementation instead of doing line to line
 * port.
 *
 * The basic concept of floating rounding is, for a floating point number, we need to
 * convert base2 to base10.  During which we need to implement correct rounding.  Hence on
 * base2 we have:
 *
 * {low, value, high}
 *
 * 3 values, representing round down, no rounding, and round up.  In the original
 * implementation and paper, variables representing these 3 values are typically postfixed
 * with m, r, p like {vr, vm, vp}.  Here we name them more verbosely.
 */

namespace xgboost {
namespace detail {
static constexpr char kItoaLut[200] = {
    '0', '0', '0', '1', '0', '2', '0', '3', '0', '4', '0', '5', '0', '6', '0',
    '7', '0', '8', '0', '9', '1', '0', '1', '1', '1', '2', '1', '3', '1', '4',
    '1', '5', '1', '6', '1', '7', '1', '8', '1', '9', '2', '0', '2', '1', '2',
    '2', '2', '3', '2', '4', '2', '5', '2', '6', '2', '7', '2', '8', '2', '9',
    '3', '0', '3', '1', '3', '2', '3', '3', '3', '4', '3', '5', '3', '6', '3',
    '7', '3', '8', '3', '9', '4', '0', '4', '1', '4', '2', '4', '3', '4', '4',
    '4', '5', '4', '6', '4', '7', '4', '8', '4', '9', '5', '0', '5', '1', '5',
    '2', '5', '3', '5', '4', '5', '5', '5', '6', '5', '7', '5', '8', '5', '9',
    '6', '0', '6', '1', '6', '2', '6', '3', '6', '4', '6', '5', '6', '6', '6',
    '7', '6', '8', '6', '9', '7', '0', '7', '1', '7', '2', '7', '3', '7', '4',
    '7', '5', '7', '6', '7', '7', '7', '8', '7', '9', '8', '0', '8', '1', '8',
    '2', '8', '3', '8', '4', '8', '5', '8', '6', '8', '7', '8', '8', '8', '9',
    '9', '0', '9', '1', '9', '2', '9', '3', '9', '4', '9', '5', '9', '6', '9',
    '7', '9', '8', '9', '9'};

constexpr uint32_t Tens(uint32_t n) { return n == 1 ? 10 : (Tens(n - 1) * 10); }

struct UnsignedFloatBase2;

struct UnsignedFloatBase10 {
  uint32_t mantissa;
  // Decimal exponent's range is -45 to 38
  // inclusive, and can fit in a short if needed.
  int32_t exponent;
};

template <typename To, typename From>
To BitCast(From&& from) {
  static_assert(sizeof(From) == sizeof(To), "Bit cast doesn't change output size.");
  To t;
  std::memcpy(&t, &from, sizeof(To));
  return t;
}

struct IEEE754 {
  static constexpr uint32_t kFloatMantissaBits = 23;
  static constexpr uint32_t kFloatBias = 127;
  static constexpr uint32_t kFloatExponentBits = 8;

  static void Decode(float f, UnsignedFloatBase2* uf, bool* signbit);
  static float Encode(UnsignedFloatBase2 const& uf, bool signbit);

  static float Infinity(bool sign) {
    uint32_t f =
        ((static_cast<uint32_t>(sign))
         << (IEEE754::kFloatExponentBits + IEEE754::kFloatMantissaBits)) |
        (0xffu << IEEE754::kFloatMantissaBits);
    float result = BitCast<float>(f);
    return result;
  }
};

struct UnsignedFloatBase2 {
  uint32_t mantissa;
  // Decimal exponent's range is -45 to 38
  // inclusive, and can fit in a short if needed.
  uint32_t exponent;

  bool Infinite() const {
    return exponent == ((1u << IEEE754::kFloatExponentBits) - 1u);
  }
  bool Zero() const {
    return mantissa == 0 && exponent == 0;
  }
};

inline void IEEE754::Decode(float f, UnsignedFloatBase2 *uf, bool *signbit) {
  auto bits = BitCast<uint32_t>(f);
  // Decode bits into sign, mantissa, and exponent.
  *signbit = std::signbit(f);
  uf->mantissa = bits & ((1u << kFloatMantissaBits) - 1);
  uf->exponent = (bits >> IEEE754::kFloatMantissaBits) &
                 ((1u << IEEE754::kFloatExponentBits) - 1);  // remove signbit
}

inline float IEEE754::Encode(UnsignedFloatBase2 const &uf, bool signbit) {
  uint32_t f =
      ((((static_cast<uint32_t>(signbit)) << IEEE754::kFloatExponentBits) |
        static_cast<uint32_t>(uf.exponent))
       << IEEE754::kFloatMantissaBits) |
      uf.mantissa;
  return BitCast<float>(f);
}

// Represents the interval of information-preserving outputs.
struct MantissaInteval {
  int32_t exponent;
  // low: smaller half way point
  uint32_t mantissa_low;
  // correct: f
  uint32_t mantissa_correct;
  // high: larger half way point
  uint32_t mantissa_high;
};

struct RyuPowLogUtils {
  // This table is generated by PrintFloatLookupTable from ryu.  We adopted only the float
  // 32 table instead of double full table.
  // f2s_full_table.h
  uint32_t constexpr static kFloatPow5InvBitcount = 59;
  static constexpr uint64_t kFloatPow5InvSplit[55] = {
      576460752303423489u, 461168601842738791u, 368934881474191033u,
      295147905179352826u, 472236648286964522u, 377789318629571618u,
      302231454903657294u, 483570327845851670u, 386856262276681336u,
      309485009821345069u, 495176015714152110u, 396140812571321688u,
      316912650057057351u, 507060240091291761u, 405648192073033409u,
      324518553658426727u, 519229685853482763u, 415383748682786211u,
      332306998946228969u, 531691198313966350u, 425352958651173080u,
      340282366920938464u, 544451787073501542u, 435561429658801234u,
      348449143727040987u, 557518629963265579u, 446014903970612463u,
      356811923176489971u, 570899077082383953u, 456719261665907162u,
      365375409332725730u, 292300327466180584u, 467680523945888934u,
      374144419156711148u, 299315535325368918u, 478904856520590269u,
      383123885216472215u, 306499108173177772u, 490398573077084435u,
      392318858461667548u, 313855086769334039u, 502168138830934462u,
      401734511064747569u, 321387608851798056u, 514220174162876889u,
      411376139330301511u, 329100911464241209u, 526561458342785934u,
      421249166674228747u, 336999333339382998u, 539198933343012796u,
      431359146674410237u, 345087317339528190u, 552139707743245103u,
      441711766194596083u};

  uint32_t constexpr static kFloatPow5Bitcount = 61;
  static constexpr uint64_t kFloatPow5Split[47] = {
      1152921504606846976u, 1441151880758558720u, 1801439850948198400u,
      2251799813685248000u, 1407374883553280000u, 1759218604441600000u,
      2199023255552000000u, 1374389534720000000u, 1717986918400000000u,
      2147483648000000000u, 1342177280000000000u, 1677721600000000000u,
      2097152000000000000u, 1310720000000000000u, 1638400000000000000u,
      2048000000000000000u, 1280000000000000000u, 1600000000000000000u,
      2000000000000000000u, 1250000000000000000u, 1562500000000000000u,
      1953125000000000000u, 1220703125000000000u, 1525878906250000000u,
      1907348632812500000u, 1192092895507812500u, 1490116119384765625u,
      1862645149230957031u, 1164153218269348144u, 1455191522836685180u,
      1818989403545856475u, 2273736754432320594u, 1421085471520200371u,
      1776356839400250464u, 2220446049250313080u, 1387778780781445675u,
      1734723475976807094u, 2168404344971008868u, 1355252715606880542u,
      1694065894508600678u, 2117582368135750847u, 1323488980084844279u,
      1654361225106055349u, 2067951531382569187u, 1292469707114105741u,
      1615587133892632177u, 2019483917365790221u};

  static uint32_t Pow5Factor(uint32_t value) noexcept(true) {
    uint32_t count = 0;
    for (;;) {
      const uint32_t q = value / 5;
      const uint32_t r = value % 5;
      if (r != 0) {
        break;
      }
      value = q;
      ++count;
    }
    return count;
  }

  // Returns true if value is divisible by 5^p.
  static bool MultipleOfPowerOf5(const uint32_t value, const uint32_t p) noexcept(true) {
    return Pow5Factor(value) >= p;
  }

  // Returns true if value is divisible by 2^p.
  static bool MultipleOfPowerOf2(const uint32_t value, const uint32_t p) noexcept(true) {
#ifdef __GNUC__
    return static_cast<uint32_t>(__builtin_ctz(value)) >= p;
#else
    return (value & ((1u << p) - 1)) == 0;
#endif  //  __GNUC__
  }

  // Returns e == 0 ? 1 : ceil(log_2(5^e)).
  static uint32_t Pow5Bits(const int32_t e) noexcept(true) {
    return static_cast<uint32_t>(((e * 163391164108059ull) >> 46) + 1);
  }

  static int32_t Log2Pow5(const int32_t e) {
    // This approximation works up to the point that the multiplication
    // overflows at e = 3529. If the multiplication were done in 64 bits, it
    // would fail at 5^4004 which is just greater than 2^9297.
    assert(e >= 0);
    assert(e <= 3528);
    return static_cast<int32_t>(((static_cast<uint32_t>(e)) * 1217359) >> 19);
  }

  static int32_t CeilLog2Pow5(const int32_t e) {
    return RyuPowLogUtils::Log2Pow5(e) + 1;
  }

  /*
   * \brief Multiply 32-bit and 64-bit -> 128 bit, then access the higher bits.
   */
  static uint32_t MulShift(const uint32_t x, const uint64_t y,
                           const int32_t shift) noexcept(true) {
    // For 32-bit * 64-bit: x * y, it can be decomposed into:
    //
    //   x * (y_high + y_low) = (x * y_high) + (x * y_low)
    //
    // For more general case 64-bit * 64-bit, see https://stackoverflow.com/a/1541458
    const uint32_t y_low = static_cast<uint32_t>(y);
    const uint32_t y_high = static_cast<uint32_t>(y >> 32);

    const uint64_t low = static_cast<uint64_t>(x) * y_low;
    const uint64_t high = static_cast<uint64_t>(x) * y_high;

    const uint64_t sum = (low >> 32) + high;
    const uint64_t shifted_sum = sum >> (shift - 32);

    return static_cast<uint32_t>(shifted_sum);
  }

  /*
   * \brief floor(5^q/2*k) and shift by j
   */
  static uint32_t MulPow5InvDivPow2(const uint32_t m, const uint32_t q,
                                    const int32_t j) noexcept(true) {
    return MulShift(m, kFloatPow5InvSplit[q], j);
  }

  /*
   * \brief floor(2^k/5^q) + 1 and shift by j
   */
  static uint32_t MulPow5divPow2(const uint32_t m, const uint32_t i,
                                 const int32_t j) noexcept(true) {
    // clang-tidy makes false assumption that can lead to i >= 47, which is impossible.
    // Can be verified by enumerating all float32 values.
    return MulShift(m, kFloatPow5Split[i], j);  // NOLINT
  }

  static uint32_t FloorLog2(const uint32_t value) {
#if defined(_MSC_VER)
    unsigned long index;  // NOLINT
    return _BitScanReverse(&index, value) ? index : 32;
#else
    return 31 - __builtin_clz(value);
#endif
  }

  /*
   * \brief floor(e * log_10(2)).
   */
  static uint32_t Log10Pow2(const int32_t e) noexcept(true) {
    // The first value this approximation fails for is 2^1651 which is just
    // greater than 10^297.
    assert(e >= 0);
    assert(e <= 1 << 15);
    return static_cast<uint32_t>((static_cast<uint64_t>(e) * 169464822037455ull) >> 49);
  }

  // Returns floor(e * log_10(5)).
  static uint32_t Log10Pow5(const int32_t expoent) noexcept(true) {
    // The first value this approximation fails for is 5^2621 which is just
    // greater than 10^1832.
    assert(expoent >= 0);
    assert(expoent <= 1 << 15);
    return static_cast<uint32_t>(
        ((static_cast<uint64_t>(expoent)) * 196742565691928ull) >> 48);
  }
};

constexpr uint64_t RyuPowLogUtils::kFloatPow5InvSplit[55];
constexpr uint64_t RyuPowLogUtils::kFloatPow5Split[47];

class PowerBaseComputer {
 private:
  static uint8_t
  ToDecimalBase(bool const accept_bounds, uint32_t const mantissa_low_shift,
                MantissaInteval const base2, MantissaInteval *base10,
                bool *mantissa_low_is_trailing_zeros,
                bool *mantissa_out_is_trailing_zeros) noexcept(true) {
    uint8_t last_removed_digit = 0;
    if (base2.exponent >= 0) {
      const uint32_t q = RyuPowLogUtils::Log10Pow2(base2.exponent);
      base10->exponent = static_cast<int32_t>(q);
      const int32_t k = RyuPowLogUtils::kFloatPow5InvBitcount +
                        RyuPowLogUtils::Pow5Bits(static_cast<int32_t>(q)) - 1;
      const int32_t i = -base2.exponent + static_cast<int32_t>(q) + k;
      base10->mantissa_low =
          RyuPowLogUtils::MulPow5InvDivPow2(base2.mantissa_low, q, i);
      base10->mantissa_correct =
          RyuPowLogUtils::MulPow5InvDivPow2(base2.mantissa_correct, q, i);
      base10->mantissa_high =
          RyuPowLogUtils::MulPow5InvDivPow2(base2.mantissa_high, q, i);

      if (q != 0 &&
          (base10->mantissa_high - 1) / 10 <= base10->mantissa_low / 10) {
        // We need to know one removed digit even if we are not going to loop
        // below. We could use q = X - 1 above, except that would require 33
        // bits for the result, and we've found that 32-bit arithmetic is
        // faster even on 64-bit machines.
        const int32_t l =
            RyuPowLogUtils::kFloatPow5InvBitcount +
            RyuPowLogUtils::Pow5Bits(static_cast<int32_t>(q - 1)) - 1;
        last_removed_digit = static_cast<uint8_t>(
            RyuPowLogUtils::MulPow5InvDivPow2(
                base2.mantissa_correct, q - 1,
                -base2.exponent + static_cast<int32_t>(q) - 1 + l) %
            10);
      }
      if (q <= 9) {
        // The largest power of 5 that fits in 24 bits is 5^10, but q <= 9 seems to be
        // safe as well. Only one of mantissa_high, mantissa_correct, and mantissa_low can
        // be a multiple of 5, if any.
        if (base2.mantissa_correct % 5 == 0) {
          *mantissa_out_is_trailing_zeros =
              RyuPowLogUtils::MultipleOfPowerOf5(base2.mantissa_correct, q);
        } else if (accept_bounds) {
          *mantissa_low_is_trailing_zeros =
              RyuPowLogUtils::MultipleOfPowerOf5(base2.mantissa_low, q);
        } else {
          base10->mantissa_high -=
              RyuPowLogUtils::MultipleOfPowerOf5(base2.mantissa_high, q);
        }
      }
    } else {
      const uint32_t q = RyuPowLogUtils::Log10Pow5(-base2.exponent);
      base10->exponent = static_cast<int32_t>(q) + base2.exponent;
      const int32_t i = -base2.exponent - static_cast<int32_t>(q);
      const int32_t k =
          RyuPowLogUtils::Pow5Bits(i) - RyuPowLogUtils::kFloatPow5Bitcount;
      int32_t j = static_cast<int32_t>(q) - k;
      base10->mantissa_correct = RyuPowLogUtils::MulPow5divPow2(
          base2.mantissa_correct, static_cast<uint32_t>(i), j);
      base10->mantissa_high = RyuPowLogUtils::MulPow5divPow2(
          base2.mantissa_high, static_cast<uint32_t>(i), j);
      base10->mantissa_low = RyuPowLogUtils::MulPow5divPow2(
          base2.mantissa_low, static_cast<uint32_t>(i), j);

      if (q != 0 &&
          (base10->mantissa_high - 1) / 10 <= base10->mantissa_low / 10) {
        j = static_cast<int32_t>(q) - 1 -
            (RyuPowLogUtils::Pow5Bits(i + 1) -
             RyuPowLogUtils::kFloatPow5Bitcount);
        last_removed_digit = static_cast<uint8_t>(
            RyuPowLogUtils::MulPow5divPow2(base2.mantissa_correct,
                                           static_cast<uint32_t>(i + 1), j) %
            10);
      }
      if (q <= 1) {
        // {mantissa_out, mantissa_out_high, mantissa_out_low} is trailing zeros if
        // {mantissa_correct,mantissa_high,mantissa_low} has at least q trailing 0
        // bits.mantissa_correct = 4 * m2, so it always has at least two trailing 0 bits.
        *mantissa_out_is_trailing_zeros = true;
        if (accept_bounds) {
          // mantissa_low = mantissa_correct - 1 - mantissa_low_shift, so it has 1
          // trailing 0 bit iff mmShift == 1.
          *mantissa_low_is_trailing_zeros = mantissa_low_shift == 1;
        } else {
          // mantissa_high = mantissa_correct + 2, so it always has at least one trailing
          // 0 bit.
          --base10->mantissa_high;
        }
      } else if (q < 31) {
        *mantissa_out_is_trailing_zeros =
            RyuPowLogUtils::MultipleOfPowerOf2(base2.mantissa_correct, q - 1);
      }
    }
    return last_removed_digit;
  }

  /*
   * \brief A varient of extended euclidean GCD algorithm.
   */
  static UnsignedFloatBase10
  ShortestRepresentation(bool mantissa_low_is_trailing_zeros,
                         bool mantissa_out_is_trailing_zeros,
                         uint8_t last_removed_digit, bool const accept_bounds,
                         MantissaInteval base10) noexcept(true) {
    int32_t removed {0};
    uint32_t output {0};

    if (mantissa_low_is_trailing_zeros || mantissa_out_is_trailing_zeros) {
      // General case, which happens rarely (~4.0%).
      while (base10.mantissa_high / 10 > base10.mantissa_low / 10) {
        mantissa_low_is_trailing_zeros &= base10.mantissa_low % 10 == 0;
        mantissa_out_is_trailing_zeros &= last_removed_digit == 0;
        last_removed_digit = static_cast<uint8_t>(base10.mantissa_correct % 10);
        base10.mantissa_correct /= 10;
        base10.mantissa_high /= 10;
        base10.mantissa_low /= 10;
        ++removed;
      }

      if (mantissa_low_is_trailing_zeros) {
        while (base10.mantissa_low % 10 == 0) {
          mantissa_out_is_trailing_zeros &= last_removed_digit == 0;
          last_removed_digit = static_cast<uint8_t>(base10.mantissa_correct % 10);
          base10.mantissa_correct /= 10;
          base10.mantissa_high /= 10;
          base10.mantissa_low /= 10;
          ++removed;
        }
      }

      if (mantissa_out_is_trailing_zeros && last_removed_digit == 5 &&
          base10.mantissa_correct % 2 == 0) {
        // Round even if the exact number is .....50..0.
        last_removed_digit = 4;
      }
      // We need to take mantissa_out + 1 if mantissa_out is outside bounds or we need to
      // round up.
      output = base10.mantissa_correct +
               ((base10.mantissa_correct == base10.mantissa_low &&
                 (!accept_bounds || !mantissa_low_is_trailing_zeros)) ||
                last_removed_digit >= 5);
    } else {
      // Specialized for the common case (~96.0%). Percentages below are
      // relative to this. Loop iterations below (approximately): 0: 13.6%,
      // 1: 70.7%, 2: 14.1%, 3: 1.39%, 4: 0.14%, 5+: 0.01%
      while (base10.mantissa_high / 10 > base10.mantissa_low / 10) {
        last_removed_digit = static_cast<uint8_t>(base10.mantissa_correct % 10);
        base10.mantissa_correct /= 10;
        base10.mantissa_high /= 10;
        base10.mantissa_low /= 10;
        ++removed;
      }

      // We need to take mantissa_out + 1 if mantissa_out is outside bounds or we need to
      // round up.
      output = base10.mantissa_correct +
               (base10.mantissa_correct == base10.mantissa_low ||
                last_removed_digit >= 5);
    }
    const int32_t exp = base10.exponent + removed;

    UnsignedFloatBase10 fd;
    fd.exponent = exp;
    fd.mantissa = output;
    return fd;
  }

 public:
  static UnsignedFloatBase10 Binary2Decimal(UnsignedFloatBase2 const f) noexcept(true) {
    MantissaInteval base2_range;
    uint32_t mantissa_base2;
    if (f.exponent == 0) {
      // We subtract 2 so that the bounds computation has 2 additional bits.
      base2_range.exponent = static_cast<int32_t>(1) -
                             static_cast<int32_t>(IEEE754::kFloatBias) -
                             static_cast<int32_t>(IEEE754::kFloatMantissaBits) -
                             static_cast<int32_t>(2);
      static_assert(static_cast<int32_t>(1) -
                            static_cast<int32_t>(IEEE754::kFloatBias) -
                            static_cast<int32_t>(IEEE754::kFloatMantissaBits) -
                            static_cast<int32_t>(2) ==
                        -151,
                    "");
      mantissa_base2 = f.mantissa;
    } else {
      base2_range.exponent = static_cast<int32_t>(f.exponent) - IEEE754::kFloatBias -
                             IEEE754::kFloatMantissaBits - 2;
      mantissa_base2 = (1u << IEEE754::kFloatMantissaBits) | f.mantissa;
    }
    const bool even = (mantissa_base2 & 1) == 0;
    const bool accept_bounds = even;

    // Step 2: Determine the interval of valid decimal representations.
    base2_range.mantissa_correct = 4 * mantissa_base2;
    base2_range.mantissa_high = 4 * mantissa_base2 + 2;
    // Implicit bool -> int conversion. True is 1, false is 0.
    const uint32_t mantissa_low_shift = f.mantissa != 0 || f.exponent <= 1;
    base2_range.mantissa_low = 4 * mantissa_base2 - 1 - mantissa_low_shift;

    // Step 3: Convert to a decimal power base using 64-bit arithmetic.
    MantissaInteval base10_range;
    bool mantissa_low_is_trailing_zeros = false;
    bool mantissa_out_is_trailing_zeros = false;
    auto last_removed_digit = PowerBaseComputer::ToDecimalBase(
        accept_bounds, mantissa_low_shift, base2_range, &base10_range,
        &mantissa_low_is_trailing_zeros, &mantissa_out_is_trailing_zeros);

    // Step 4: Find the shortest decimal representation in the interval of valid
    // representations.
    auto out = ShortestRepresentation(mantissa_low_is_trailing_zeros,
                                      mantissa_out_is_trailing_zeros,
                                      last_removed_digit,
                                      accept_bounds, base10_range);
    return out;
  }
};

/*
 * \brief Print the floating point number in base 10.
 */
class RyuPrinter {
 private:
  static inline uint32_t OutputLength(const uint32_t v) noexcept(true) {
    // Function precondition: v is not a 10-digit number.
    // (f2s: 9 digits are sufficient for round-tripping.)
    // (d2fixed: We print 9-digit blocks.)
    static_assert(100000000 == Tens(8), "");
    assert(v < Tens(9));
    if (v >= Tens(8)) {
      return 9;
    }
    if (v >= Tens(7)) {
      return 8;
    }
    if (v >= Tens(6)) {
      return 7;
    }
    if (v >= Tens(5)) {
      return 6;
    }
    if (v >= Tens(4)) {
      return 5;
    }
    if (v >= Tens(3)) {
      return 4;
    }
    if (v >= Tens(2)) {
      return 3;
    }
    if (v >= Tens(1)) {
      return 2;
    }
    return 1;
  }

 public:
  static int32_t PrintBase10Float(UnsignedFloatBase10 v, const bool sign,
                                  char *const result) noexcept(true) {
    // Step 5: Print the decimal representation.
    int index = 0;
    if (sign) {
      result[index++] = '-';
    }

    uint32_t output = v.mantissa;
    const uint32_t out_length = OutputLength(output);

    // Print the decimal digits.
    // The following code is equivalent to:
    // for (uint32_t i = 0; i < olength - 1; ++i) {
    //   const uint32_t c = output % 10; output /= 10;
    //   result[index + olength - i] = (char) ('0' + c);
    // }
    // result[index] = '0' + output % 10;
    uint32_t i = 0;
    while (output >= Tens(4)) {
      const uint32_t c = output % Tens(4);
      output /= Tens(4);
      const uint32_t c0 = (c % 100) << 1;
      const uint32_t c1 = (c / 100) << 1;
      // This is used to speed up decimal digit generation by copying
      // pairs of digits into the final output.
      std::memcpy(result + index + out_length - i - 1, kItoaLut + c0, 2);
      std::memcpy(result + index + out_length - i - 3, kItoaLut + c1, 2);
      i += 4;
    }
    if (output >= 100) {
      const uint32_t c = (output % 100) << 1;
      output /= 100;
      std::memcpy(result + index + out_length - i - 1, kItoaLut + c, 2);
      i += 2;
    }
    if (output >= 10) {
      const uint32_t c = output << 1;
      // We can't use std::memcpy here: the decimal dot goes between these two
      // digits.
      result[index + out_length - i] = kItoaLut[c + 1];
      result[index] = kItoaLut[c];
    } else {
      result[index] = static_cast<char>('0' + output);
    }

    // Print decimal point if needed.
    if (out_length > 1) {
      result[index + 1] = '.';
      index += out_length + 1;
    } else {
      ++index;
    }

    // Print the exponent.
    result[index++] = 'E';
    int32_t exp = v.exponent + static_cast<int32_t>(out_length) - 1;
    if (exp < 0) {
      result[index++] = '-';
      exp = -exp;
    }

    if (exp >= 10) {
      std::memcpy(result + index, kItoaLut + 2 * exp, 2);
      index += 2;
    } else {
      result[index++] = static_cast<char>('0' + exp);
    }

    return index;
  }

  static int32_t PrintSpecialFloat(const bool sign, UnsignedFloatBase2 f,
                                   char *const result) noexcept(true) {
    if (f.mantissa) {
      std::memcpy(result, u8"NaN", 3);
      return 3;
    }
    if (sign) {
      result[0] = '-';
    }
    if (f.exponent) {
      std::memcpy(result + sign, u8"Infinity", 8);
      return sign + 8;
    }
    std::memcpy(result + sign, u8"0E0", 3);
    return sign + 3;
  }
};

int32_t ToCharsFloatImpl(float f, char * const result) {
  // Step 1: Decode the floating-point number, and unify normalized and
  // subnormal cases.
  UnsignedFloatBase2 uf32;
  bool sign;
  IEEE754::Decode(f, &uf32, &sign);

  // Case distinction; exit early for the easy cases.
  if (uf32.Infinite() || uf32.Zero()) {
    return RyuPrinter::PrintSpecialFloat(sign, uf32, result);
  }

  const UnsignedFloatBase10 v = PowerBaseComputer::Binary2Decimal(uf32);
  const auto index = RyuPrinter::PrintBase10Float(v, sign, result);
  return index;
}


// ====================== Integer ==================

// This is an implementation for base 10 inspired by the one in libstdc++v3.  The general
// scheme is by decomposing the value into multiple combination of base (which is 10) by
// mod, until the value is lesser than 10, then last char is just char '0' (ascii 48) plus
// that value.  Other popular implementations can be found in RapidJson and libc++ (in
// llvm-project), which uses the same general work flow with the same look up table, but
// probably with better performance as they are more complicated.
void ItoaUnsignedImpl(char *first, uint32_t length, uint64_t value) {
  uint32_t position = length - 1;
  while (value >= Tens(2)) {
    auto const num = (value % Tens(2)) * 2;
    value /= Tens(2);
    first[position] = kItoaLut[num + 1];
    first[position - 1] = kItoaLut[num];
    position -= 2;
  }
  if (value >= 10) {
    auto const num = value * 2;
    first[0] = kItoaLut[num];
    first[1] = kItoaLut[num + 1];
  } else {
    first[0]= '0' + value;
  }
}

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

to_chars_result ToCharsUnsignedImpl(char *first, char *last,
                                    uint64_t const value) {
  const uint32_t output_len = ShortestDigit10(value);
  to_chars_result ret;
  if (XGBOOST_EXPECT(std::distance(first, last) == 0, false)) {
    ret.ec = std::errc::value_too_large;
    ret.ptr = last;
    return ret;
  }

  ItoaUnsignedImpl(first, output_len, value);
  ret.ptr = first + output_len;
  ret.ec = std::errc();
  return ret;
}

/*
 * The parsing is also part of ryu.  As of writing, the implementation in ryu uses full
 * double table.  But here we optimize the table size with float table instead.  The
 * result is exactly the same.
 */
from_chars_result FromCharFloatImpl(const char *buffer, const int len,
                                    float *result) {
  if (len == 0) {
    return {buffer, std::errc::invalid_argument};
  }
  int32_t m10digits = 0;
  int32_t e10digits = 0;
  int32_t dot_ind = len;
  int32_t e_ind = len;
  uint32_t mantissa_b10 = 0;
  int32_t exp_b10 = 0;
  bool signed_mantissa = false;
  bool signed_exp = false;
  int32_t i = 0;
  if (buffer[i] == '-') {
    signed_mantissa = true;
    i++;
  }
  for (; i < len; i++) {
    char c = buffer[i];
    if (c == '.') {
      if (dot_ind != len) {
        return {buffer + i, std::errc::invalid_argument};
      }
      dot_ind = i;
      continue;
    }
    if ((c < '0') || (c > '9')) {
      break;
    }
    if (m10digits >= 9) {
      return {buffer + i, std::errc::result_out_of_range};
    }
    mantissa_b10 = 10 * mantissa_b10 + (c - '0');
    if (mantissa_b10 != 0) {
      m10digits++;
    }
  }

  if (i < len && ((buffer[i] == 'e') || (buffer[i] == 'E'))) {
    e_ind = i;
    i++;
    if (i < len && ((buffer[i] == '-') || (buffer[i] == '+'))) {
      signed_exp = buffer[i] == '-';
      i++;
    }
    for (; i < len; i++) {
      char c = buffer[i];
      if ((c < '0') || (c > '9')) {
        return {buffer + i, std::errc::invalid_argument};
      }
      if (e10digits > 3) {
        return {buffer + i, std::errc::result_out_of_range};
      }
      exp_b10 = 10 * exp_b10 + (c - '0');
      if (exp_b10 != 0) {
        e10digits++;
      }
    }
  }
  if (i < len) {
    return {buffer + i, std::errc::invalid_argument};
  }
  if (signed_exp) {
    exp_b10 = -exp_b10;
  }
  exp_b10 -= dot_ind < e_ind ? e_ind - dot_ind - 1 : 0;
  if (mantissa_b10 == 0) {
    *result = signed_mantissa ? -0.0f : 0.0f;
    return {};
  }

  if ((m10digits + exp_b10 <= -46) || (mantissa_b10 == 0)) {
    // Number is less than 1e-46, which should be rounded down to 0; return
    // +/-0.0.
    uint32_t ieee =
        (static_cast<uint32_t>(signed_mantissa))
        << (IEEE754::kFloatExponentBits + IEEE754::kFloatMantissaBits);
    *result = BitCast<float>(ieee);
    return {};
  }
  if (m10digits + exp_b10 >= 40) {
    // Number is larger than 1e+39, which should be rounded to +/-Infinity.
    *result = IEEE754::Infinity(signed_mantissa);
    return {};
  }

  // Convert to binary float m2 * 2^e2, while retaining information about
  // whether the conversion was exact (trailingZeros).
  int32_t exp_b2;
  uint32_t mantissa_b2;
  bool trailing_zeros;
  if (exp_b10 >= 0) {
    // The length of m * 10^e in bits is:
    //   log2(m10 * 10^e10) = log2(m10) + e10 log2(10) = log2(m10) + e10 + e10 *
    //   log2(5)
    //
    // We want to compute the IEEE754::kFloatMantissaBits + 1 top-most bits (+1 for the
    // implicit leading one in IEEE format). We therefore choose a binary output
    // exponent of
    //   log2(m10 * 10^e10) - (IEEE754::kFloatMantissaBits + 1).
    //
    // We use floor(log2(5^e10)) so that we get at least this many bits; better
    // to have an additional bit than to not have enough bits.
    exp_b2 = RyuPowLogUtils::FloorLog2(mantissa_b10) + exp_b10 +
             RyuPowLogUtils::Log2Pow5(exp_b10) -
             (IEEE754::kFloatMantissaBits + 1);

    // We now compute [m10 * 10^e10 / 2^e2] = [m10 * 5^e10 / 2^(e2-e10)].
    // To that end, we use the RyuPowLogUtils::kFloatPow5Bitcount table.
    int j = exp_b2 - exp_b10 - RyuPowLogUtils::CeilLog2Pow5(exp_b10) +
            RyuPowLogUtils::kFloatPow5Bitcount;
    assert(j >= 0);
    mantissa_b2 = RyuPowLogUtils::MulPow5divPow2(mantissa_b10, exp_b10, j);

    // We also compute if the result is exact, i.e.,
    //   [m10 * 10^e10 / 2^e2] == m10 * 10^e10 / 2^e2.
    // This can only be the case if 2^e2 divides m10 * 10^e10, which in turn
    // requires that the largest power of 2 that divides m10 + e10 is greater
    // than e2. If e2 is less than e10, then the result must be exact. Otherwise
    // we use the existing multipleOfPowerOf2 function.
    trailing_zeros =
        exp_b2 < exp_b10 ||
        (exp_b2 - exp_b10 < 32 &&
         RyuPowLogUtils::MultipleOfPowerOf2(mantissa_b10, exp_b2 - exp_b10));
  } else {
    exp_b2 = RyuPowLogUtils::FloorLog2(mantissa_b10) + exp_b10 -
             RyuPowLogUtils::CeilLog2Pow5(-exp_b10) -
             (IEEE754::kFloatMantissaBits + 1);

    // We now compute [m10 * 10^e10 / 2^e2] = [m10 / (5^(-e10) 2^(e2-e10))].
    int j = exp_b2 - exp_b10 + RyuPowLogUtils::CeilLog2Pow5(-exp_b10) - 1 +
            RyuPowLogUtils::kFloatPow5InvBitcount;
    mantissa_b2 = RyuPowLogUtils::MulPow5InvDivPow2(mantissa_b10, -exp_b10, j);

    // We also compute if the result is exact, i.e.,
    //   [m10 / (5^(-e10) 2^(e2-e10))] == m10 / (5^(-e10) 2^(e2-e10))
    //
    // If e2-e10 >= 0, we need to check whether (5^(-e10) 2^(e2-e10)) divides
    // m10, which is the case iff pow5(m10) >= -e10 AND pow2(m10) >= e2-e10.
    //
    // If e2-e10 < 0, we have actually computed [m10 * 2^(e10 e2) / 5^(-e10)]
    // above, and we need to check whether 5^(-e10) divides (m10 * 2^(e10-e2)),
    // which is the case iff pow5(m10 * 2^(e10-e2)) = pow5(m10) >= -e10.
    trailing_zeros =
        (exp_b2 < exp_b10 ||
         (exp_b2 - exp_b10 < 32 && RyuPowLogUtils::MultipleOfPowerOf2(
                                       mantissa_b10, exp_b2 - exp_b10))) &&
        RyuPowLogUtils::MultipleOfPowerOf5(mantissa_b10, -exp_b10);
  }

  // Compute the final IEEE exponent.
  uint32_t f_e2 =
      std::max(static_cast<int32_t>(0),
               static_cast<int32_t>(exp_b2 + IEEE754::kFloatBias +
                                    RyuPowLogUtils::FloorLog2(mantissa_b2)));

  if (f_e2 > 0xfe) {
    // Final IEEE exponent is larger than the maximum representable; return
    // +/-Infinity.
    *result = IEEE754::Infinity(signed_mantissa);
    return {};
  }

  // We need to figure out how much we need to shift m2. The tricky part is that
  // we need to take the final IEEE exponent into account, so we need to reverse
  // the bias and also special-case the value 0.
  int32_t shift = (f_e2 == 0 ? 1 : f_e2) - exp_b2 - IEEE754::kFloatBias -
                  IEEE754::kFloatMantissaBits;
  assert(shift >= 0);

  // We need to round up if the exact value is more than 0.5 above the value we
  // computed. That's equivalent to checking if the last removed bit was 1 and
  // either the value was not just trailing zeros or the result would otherwise
  // be odd.
  //
  // We need to update trailingZeros given that we have the exact output
  // exponent ieee_e2 now.
  trailing_zeros &= (mantissa_b2 & ((1u << (shift - 1)) - 1)) == 0;
  uint32_t lastRemovedBit = (mantissa_b2 >> (shift - 1)) & 1;
  bool roundup = (lastRemovedBit != 0) &&
                 (!trailing_zeros || (((mantissa_b2 >> shift) & 1) != 0));

  uint32_t f_m2 = (mantissa_b2 >> shift) + roundup;
  assert(f_m2 <= (1u << (IEEE754::kFloatMantissaBits + 1)));
  f_m2 &= (1u << IEEE754::kFloatMantissaBits) - 1;
  if (f_m2 == 0 && roundup) {
    // Rounding up may overflow the mantissa.
    // In this case we move a trailing zero of the mantissa into the exponent.
    // Due to how the IEEE represents +/-Infinity, we don't need to check for
    // overflow here.
    f_e2++;
  }
  *result = IEEE754::Encode({f_m2, f_e2}, signed_mantissa);
  return {};
}
}  // namespace detail
}  // namespace xgboost
