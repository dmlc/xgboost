/*!
 * Copyright 2017 by Contributors
 * \author Rory Mitchell
 */
#pragma once
#include <algorithm>
#include "xgboost/base.h"

#ifdef XGBOOST_USE_AVX
namespace avx {
/**
 * \struct  Float8
 *
 * \brief Helper class for processing a vector of eight floats using AVX
 * instructions. Implements basic math operators.
 */

struct Float8 {
  __m256 x;
  explicit Float8(const __m256& x) : x(x) {}
  explicit Float8(const float& val) : x(_mm256_broadcast_ss(&val)) {}
  explicit Float8(const float* vec) : x(_mm256_loadu_ps(vec)) {}
  Float8() : x() {}
  Float8& operator+=(const Float8& rhs) {
    x = _mm256_add_ps(x, rhs.x);
    return *this;
  }
  Float8& operator-=(const Float8& rhs) {
    x = _mm256_sub_ps(x, rhs.x);
    return *this;
  }
  Float8& operator*=(const Float8& rhs) {
    x = _mm256_mul_ps(x, rhs.x);
    return *this;
  }
  Float8& operator/=(const Float8& rhs) {
    x = _mm256_div_ps(x, rhs.x);
    return *this;
  }
  void Print() {
    float* f = reinterpret_cast<float*>(&x);
    printf("%f %f %f %f %f %f %f %f\n", f[0], f[1], f[2], f[3], f[4], f[5],
           f[6], f[7]);
  }
};

inline Float8 operator+(Float8 lhs, const Float8& rhs) {
  lhs += rhs;
  return lhs;
}
inline Float8 operator-(Float8 lhs, const Float8& rhs) {
  lhs -= rhs;
  return lhs;
}
inline Float8 operator*(Float8 lhs, const Float8& rhs) {
  lhs *= rhs;
  return lhs;
}
inline Float8 operator/(Float8 lhs, const Float8& rhs) {
  lhs /= rhs;
  return lhs;
}

inline Float8 round(const Float8& x) {
  return Float8(_mm256_round_ps(x.x, _MM_FROUND_TO_NEAREST_INT));
}
}  // namespace avx

// Overload std::max/min
namespace std {
inline avx::Float8 max(const avx::Float8& a, const avx::Float8& b) {  // NOLINT
  return avx::Float8(_mm256_max_ps(a.x, b.x));
}
inline avx::Float8 min(const avx::Float8& a, const avx::Float8& b) {  // NOLINT
  return avx::Float8(_mm256_min_ps(a.x, b.x));
}
}  // namespace std

namespace avx {

// https://codingforspeed.com/using-faster-exponential-approximation/
inline Float8 Exp4096(Float8 x) {
  x *= Float8(1.0f / 4096.0f);
  x += Float8(1.0f);
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  return x;
}

inline Float8 pow2n(Float8 const& n) {
  const float pow2_23 = 8388608.0;  // 2^23
  const float bias = 127.0;         // bias in exponent
  Float8 a =
      n + Float8(bias + pow2_23);  // put n + bias in least significant bits
  __m256i b = _mm256_castps_si256(a.x);

  // Do bit shift in SSE so we don't have to use AVX2 instructions
  __m128i c1 = _mm256_castsi256_si128(b);
  b = _mm256_permute2f128_si256(b, b, 1);
  __m128i c2 = _mm256_castsi256_si128(b);
  c1 = _mm_slli_epi32(c1, 23);
  c2 = _mm_slli_epi32(c2, 23);

  __m256i c = _mm256_insertf128_si256(_mm256_castsi128_si256(c1), (c2), 0x1);
  return Float8(_mm256_castsi256_ps(c));
}

inline Float8 polynomial_5(Float8 const& x, const float c0, const float c1,
                           const float c2, const float c3, const float c4,
                           const float c5) {
  // calculates polynomial c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
  Float8 x2 = x * x;
  Float8 x4 = x2 * x2;
  return (Float8(c2) + Float8(c3) * x) * x2 +
         ((Float8(c4) + Float8(c5) * x) * x4 + (Float8(c0) + Float8(c1) * x));
}

// AVX exp Function based off Agner Fog's vector library
// https://github.com/darealshinji/vectorclass/blob/master/vectormath_exp.h
// Modified so it doesn't require AVX2 instructions
// Clamps input values to the range -87.3f, +87.3f
inline Float8 ExpAgner(Float8 x) {
  // Clamp input values
  float max_x = 87.3f;
  x = std::min(x, Float8(max_x));
  x = std::max(x, Float8(-max_x));

  // 1/log(2)
  const float log2e = 1.44269504088896340736f;

  // Taylor coefficients
  const float P0expf = 1.f / 2.f;
  const float P1expf = 1.f / 6.f;
  const float P2expf = 1.f / 24.f;
  const float P3expf = 1.f / 120.f;
  const float P4expf = 1.f / 720.f;
  const float P5expf = 1.f / 5040.f;

  const float ln2f_hi = 0.693359375f;
  const float ln2f_lo = -2.12194440e-4f;

  Float8 r = round(x * Float8(log2e));
  x -= r * Float8(ln2f_hi);
  x -= r * Float8(ln2f_lo);

  Float8 x2 = x * x;
  Float8 z = polynomial_5(x, P0expf, P1expf, P2expf, P3expf, P4expf, P5expf);
  z *= x2;
  z += x;

  // multiply by power of 2
  Float8 n2 = pow2n(r);

  z = (z + Float8(1.0f)) * n2;
  return z;
}

inline Float8 Sigmoid(Float8 x) {
  Float8 exp = ExpAgner(x * Float8(-1.0f));
  x = Float8(1.0f) + exp;
  return Float8(_mm256_rcp_ps(x.x));
}

// Store 8 gradient pairs given vectors containing gradient and Hessian
inline void StoreGpair(xgboost::GradientPair* dst, const Float8& grad,
                       const Float8& hess) {
  float* ptr = reinterpret_cast<float*>(dst);
  __m256 gpair_low = _mm256_unpacklo_ps(grad.x, hess.x);
  __m256 gpair_high = _mm256_unpackhi_ps(grad.x, hess.x);
  _mm256_storeu_ps(ptr, _mm256_permute2f128_ps(gpair_low, gpair_high, 0x20));
  _mm256_storeu_ps(ptr + 8,
                   _mm256_permute2f128_ps(gpair_low, gpair_high, 0x31));
}
}  // namespace avx
#else
namespace avx {
/**
 * \struct  Float8
 *
 * \brief Fallback implementation not using AVX.
 */

struct Float8 {  // NOLINT
  float x[8];
  explicit Float8(const float& val) {
    for (float & i : x) {
      i = val;
    }
  }
  explicit Float8(const float* vec) {
    for (int i = 0; i < 8; i++) {
      x[i] = vec[i];
    }
  }
  Float8() = default;
  Float8& operator+=(const Float8& rhs) {
    for (int i = 0; i < 8; i++) {
      x[i] += rhs.x[i];
    }
    return *this;
  }
  Float8& operator-=(const Float8& rhs) {
    for (int i = 0; i < 8; i++) {
      x[i] -= rhs.x[i];
    }
    return *this;
  }
  Float8& operator*=(const Float8& rhs) {
    for (int i = 0; i < 8; i++) {
      x[i] *= rhs.x[i];
    }
    return *this;
  }
  Float8& operator/=(const Float8& rhs) {
    for (int i = 0; i < 8; i++) {
      x[i] /= rhs.x[i];
    }
    return *this;
  }
  void Print() {
    auto* f = reinterpret_cast<float*>(&x);
    printf("%f %f %f %f %f %f %f %f\n", f[0], f[1], f[2], f[3], f[4], f[5],
           f[6], f[7]);
  }
};

inline Float8 operator+(Float8 lhs, const Float8& rhs) {
  lhs += rhs;
  return lhs;
}
inline Float8 operator-(Float8 lhs, const Float8& rhs) {
  lhs -= rhs;
  return lhs;
}
inline Float8 operator*(Float8 lhs, const Float8& rhs) {
  lhs *= rhs;
  return lhs;
}
inline Float8 operator/(Float8 lhs, const Float8& rhs) {
  lhs /= rhs;
  return lhs;
}

// Store 8 gradient pairs given vectors containing gradient and Hessian
inline void StoreGpair(xgboost::GradientPair* dst, const Float8& grad,
                       const Float8& hess) {
  for (int i = 0; i < 8; i++) {
    dst[i] = xgboost::GradientPair(grad.x[i], hess.x[i]);
  }
}

inline Float8 Sigmoid(Float8 x) {
  Float8 sig;
  for (int i = 0; i < 8; i++) {
    sig.x[i] = 1.0f / (1.0f + std::exp(-x.x[i]));
  }
  return sig;
}
}  // namespace avx

namespace std {
inline avx::Float8 max(const avx::Float8& a, const avx::Float8& b) {  // NOLINT
  avx::Float8 max;
  for (int i = 0; i < 8; i++) {
    max.x[i] = std::max(a.x[i], b.x[i]);
  }
  return max;
}
inline avx::Float8 min(const avx::Float8& a, const avx::Float8& b) {  // NOLINT
  avx::Float8 min;
  for (int i = 0; i < 8; i++) {
    min.x[i] = std::min(a.x[i], b.x[i]);
  }
  return min;
}
}  // namespace std
#endif
