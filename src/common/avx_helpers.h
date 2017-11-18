/*!
 * Copyright 2017 by Contributors
 * \author Rory Mitchell
 */
#pragma once
#include <algorithm>
#include "xgboost/base.h"

namespace avx {

#ifdef XGBOOST_USE_AVX
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

inline Float8 ApproximateSigmoid(Float8 x) {
  Float8 exp = Exp4096(x * Float8(-1.0f));
  x = Float8(1.0f) + exp;
  return Float8(_mm256_rcp_ps(x.x));
}

// Store 8 gradient pairs given vectors containing gradient and Hessian
inline void StoreGpair(xgboost::bst_gpair* dst, const Float8& grad,
                       const Float8& hess) {
  float* ptr = reinterpret_cast<float*>(dst);
  __m256 gpair_low = _mm256_unpacklo_ps(grad.x, hess.x);
  __m256 gpair_high = _mm256_unpackhi_ps(grad.x, hess.x);
  _mm256_storeu_ps(ptr, _mm256_permute2f128_ps(gpair_low, gpair_high, 0x20));
  _mm256_storeu_ps(ptr + 8,
                   _mm256_permute2f128_ps(gpair_low, gpair_high, 0x31));
}
#else
/**
 * \struct  Float8
 *
 * \brief Fallback implementation not using AVX.
 */

struct Float8 {
  float x[8];
  explicit Float8(const float& val) {
    for (int i = 0; i < 8; i++) {
      x[i] = val;
    }
  }
  explicit Float8(const float* vec) {
    for (int i = 0; i < 8; i++) {
      x[i] = vec[i];
    }
  }
  Float8() {}
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

// Store 8 gradient pairs given vectors containing gradient and Hessian
inline void StoreGpair(xgboost::bst_gpair* dst, const Float8& grad,
                       const Float8& hess) {
  for (int i = 0; i < 8; i++) {
    dst[i] = xgboost::bst_gpair(grad.x[i], hess.x[i]);
  }
}

inline Float8 ApproximateSigmoid(Float8 x) {
  Float8 sig;
  for (int i = 0; i < 8; i++) {
    sig.x[i] = 1.0f / (1.0f + std::exp(-x.x[i]));
  }
  return sig;
}
#endif
}  // namespace avx

// Overload std::max
namespace std {
#ifdef XGBOOST_USE_AVX
inline avx::Float8 max(const avx::Float8& a, const avx::Float8& b) {
  return avx::Float8(_mm256_max_ps(a.x, b.x));
}
#else
inline avx::Float8 max(const avx::Float8& a, const avx::Float8& b) {
  avx::Float8 max;
  for (int i = 0; i < 8; i++) {
    max.x[i] = std::max(a.x[i], b.x[i]);
  }
  return max;
}
#endif
}  // namespace std
