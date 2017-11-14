/*!
 * Copyright 2017 by Contributors
 * \author Rory Mitchell
 */
#pragma once
#include <algorithm>

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

// https://codingforspeed.com/using-faster-exponential-approximation/
inline Float8 Exp4096(Float8 x) {
  Float8 fraction(1.0 / 4096.0);
  Float8 ones(1.0f);
  x.x = _mm256_fmadd_ps(x.x, fraction.x, ones.x);  // x = 1.0 + x / 4096
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

inline Float8 max(const Float8& a, const Float8& b) {
  return Float8(_mm256_max_ps(a.x, b.x));
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

}  // namespace avx
