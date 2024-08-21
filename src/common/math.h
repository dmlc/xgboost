/**
 * Copyright 2015-2023 by XGBoost Contributors
 * \file math.h
 * \brief additional math utils
 * \author Tianqi Chen
 */
#ifndef XGBOOST_COMMON_MATH_H_
#define XGBOOST_COMMON_MATH_H_

#include <xgboost/base.h>  // for XGBOOST_DEVICE

#include <algorithm>    // for max
#include <cmath>        // for exp, abs, log, lgamma
#include <limits>       // for numeric_limits
#include <type_traits>  // for is_floating_point_v, conditional, is_signed, is_same, declval
#include <utility>      // for pair

namespace xgboost {
namespace common {

template <typename T> XGBOOST_DEVICE T Sqr(T const &w) { return w * w; }

/*!
 * \brief calculate the sigmoid of the input.
 * \param x input parameter
 * \return the transformed value.
 */
XGBOOST_DEVICE inline float Sigmoid(float x) {
  float constexpr kEps = 1e-16;  // avoid 0 div
  x = std::min(-x, 88.7f);       // avoid exp overflow
  auto denom = expf(x) + 1.0f + kEps;
  auto y = 1.0f / denom;
  return y;
}

XGBOOST_DEVICE inline double Sigmoid(double x) {
  auto denom = std::exp(-x) + 1.0;
  auto y = 1.0 / denom;
  return y;
}
/*!
 * \brief Equality test for both integer and floating point.
 */
template <typename T, typename U>
XGBOOST_DEVICE constexpr bool CloseTo(T a, U b) {
  using Casted = typename std::conditional_t<
      std::is_floating_point_v<T> || std::is_floating_point_v<U>, double,
      typename std::conditional_t<std::is_signed_v<T> || std::is_signed_v<U>, std::int64_t,
                                  std::uint64_t>>;
  return std::is_floating_point_v<Casted> ?
      std::abs(static_cast<Casted>(a) -static_cast<Casted>(b)) < 1e-6 : a == b;
}

/*!
 * \brief Do inplace softmax transformaton on start to end
 *
 * \tparam Iterator Input iterator type
 *
 * \param start Start iterator of input
 * \param end end iterator of input
 */
template <typename Iterator>
XGBOOST_DEVICE inline void Softmax(Iterator start, Iterator end) {
  static_assert(
      std::is_same_v<
          float, typename std::remove_reference_t<decltype(std::declval<Iterator>().operator*())>>,
      "Values should be of type bst_float");
  bst_float wmax = *start;
  for (Iterator i = start+1; i != end; ++i) {
    wmax = fmaxf(*i, wmax);
  }
  double wsum = 0.0f;
  for (Iterator i = start; i != end; ++i) {
    *i = expf(*i - wmax);
    wsum += *i;
  }
  for (Iterator i = start; i != end; ++i) {
    *i /= static_cast<float>(wsum);
  }
}

/*!
 * \brief Find the maximum iterator within the iterators
 * \param begin The beginning iterator.
 * \param end The end iterator.
 * \return the iterator point to the maximum value.
 * \tparam Iterator The type of the iterator.
 */
template<typename Iterator>
XGBOOST_DEVICE inline Iterator FindMaxIndex(Iterator begin, Iterator end) {
  Iterator maxit = begin;
  for (Iterator it = begin; it != end; ++it) {
    if (*it > *maxit) maxit = it;
  }
  return maxit;
}

/*!
 * \brief perform numerically safe logsum
 * \param x left input operand
 * \param y right input operand
 * \return  log(exp(x) + exp(y))
 */
inline float LogSum(float x, float y) {
  if (x < y) {
    return y + std::log(std::exp(x - y) + 1.0f);
  } else {
    return x + std::log(std::exp(y - x) + 1.0f);
  }
}

/*!
 * \brief perform numerically safe logsum
 * \param begin The beginning iterator.
 * \param end The end iterator.
 * \return the iterator point to the maximum value.
 * \tparam Iterator The type of the iterator.
 */
template<typename Iterator>
inline float LogSum(Iterator begin, Iterator end) {
  float mx = *begin;
  for (Iterator it = begin; it != end; ++it) {
    mx = std::max(mx, *it);
  }
  float sum = 0.0f;
  for (Iterator it = begin; it != end; ++it) {
    sum += std::exp(*it - mx);
  }
  return mx + std::log(sum);
}

// Redefined here to workaround a VC bug that doesn't support overloading for integer
// types.
template <typename T>
XGBOOST_DEVICE typename std::enable_if_t<std::numeric_limits<T>::is_integer, bool> CheckNAN(T) {
  return false;
}

#if XGBOOST_STRICT_R_MODE && !defined(__CUDA_ARCH__)

bool CheckNAN(double v);

#else

XGBOOST_DEVICE bool inline CheckNAN(float x) {
#if defined(__CUDA_ARCH__)
  return isnan(x);
#else
  return std::isnan(x);
#endif  // defined(__CUDA_ARCH__)
}

XGBOOST_DEVICE bool inline CheckNAN(double x) {
#if defined(__CUDA_ARCH__)
  return isnan(x);
#else
  return std::isnan(x);
#endif  // defined(__CUDA_ARCH__)
}

#endif  // XGBOOST_STRICT_R_MODE && !defined(__CUDA_ARCH__)
// GPU version is not uploaded in CRAN anyway.
// Specialize only when using R with CPU.
#if XGBOOST_STRICT_R_MODE && !defined(XGBOOST_USE_CUDA)
double LogGamma(double v);

#else  // Not R or R with GPU.

template<typename T>
XGBOOST_DEVICE inline T LogGamma(T v) {
#ifdef _MSC_VER

#if _MSC_VER >= 1800
  return lgamma(v);
#else
#pragma message("Warning: lgamma function was not available until VS2013"\
                ", poisson regression will be disabled")
  utils::Error("lgamma function was not available until VS2013");
  return static_cast<T>(1.0);
#endif  // _MSC_VER >= 1800

#else
  return lgamma(v);
#endif  // _MSC_VER
}

#endif  // XGBOOST_STRICT_R_MODE && !defined(XGBOOST_USE_CUDA)

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_MATH_H_
