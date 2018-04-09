/*!
 * Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief defines configuration macros of xgboost.
 */
#ifndef XGBOOST_BASE_H_
#define XGBOOST_BASE_H_

#include <dmlc/base.h>
#include <dmlc/omp.h>
#include <cmath>
#include <iostream>

/*!
 * \brief string flag for R library, to leave hooks when needed.
 */
#ifndef XGBOOST_STRICT_R_MODE
#define XGBOOST_STRICT_R_MODE 0
#endif

/*!
 * \brief Whether always log console message with time.
 *  It will display like, with timestamp appended to head of the message.
 *  "[21:47:50] 6513x126 matrix with 143286 entries loaded from
 * ../data/agaricus.txt.train"
 */
#ifndef XGBOOST_LOG_WITH_TIME
#define XGBOOST_LOG_WITH_TIME 1
#endif

/*!
 * \brief Whether customize the logger outputs.
 */
#ifndef XGBOOST_CUSTOMIZE_LOGGER
#define XGBOOST_CUSTOMIZE_LOGGER XGBOOST_STRICT_R_MODE
#endif

/*!
 * \brief Whether to customize global PRNG.
 */
#ifndef XGBOOST_CUSTOMIZE_GLOBAL_PRNG
#define XGBOOST_CUSTOMIZE_GLOBAL_PRNG XGBOOST_STRICT_R_MODE
#endif

/*!
 * \brief Check if alignas(*) keyword is supported. (g++ 4.8 or higher)
 */
#if defined(__GNUC__) && ((__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || __GNUC__ > 4)
#define XGBOOST_ALIGNAS(X) alignas(X)
#else
#define XGBOOST_ALIGNAS(X)
#endif

#if defined(__GNUC__) && ((__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || __GNUC__ > 4) && \
    !defined(__CUDACC__)
#include <parallel/algorithm>
#define XGBOOST_PARALLEL_SORT(X, Y, Z) __gnu_parallel::sort((X), (Y), (Z))
#define XGBOOST_PARALLEL_STABLE_SORT(X, Y, Z) \
  __gnu_parallel::stable_sort((X), (Y), (Z))
#elif defined(_MSC_VER) && (!__INTEL_COMPILER)
#include <ppl.h>
#define XGBOOST_PARALLEL_SORT(X, Y, Z) concurrency::parallel_sort((X), (Y), (Z))
#define XGBOOST_PARALLEL_STABLE_SORT(X, Y, Z) std::stable_sort((X), (Y), (Z))
#else
#define XGBOOST_PARALLEL_SORT(X, Y, Z) std::sort((X), (Y), (Z))
#define XGBOOST_PARALLEL_STABLE_SORT(X, Y, Z) std::stable_sort((X), (Y), (Z))
#endif

/*!
 * \brief Tag function as usable by device
 */
#ifdef __NVCC__
#define XGBOOST_DEVICE __host__ __device__
#else
#define XGBOOST_DEVICE
#endif

/*! \brief namespace of xgboost*/
namespace xgboost {
/*!
 * \brief unsigned integer type used in boost,
 *  used for feature index and row index.
 */
using bst_uint = uint32_t;  // NOLINT
using bst_int = int32_t;    // NOLINT
/*! \brief long integers */
typedef uint64_t bst_ulong;  // NOLINT(*)
/*! \brief float type, used for storing statistics */
using bst_float = float;


namespace detail {
/*! \brief Implementation of gradient statistics pair. Template specialisation
 * may be used to overload different gradients types e.g. low precision, high
 * precision, integer, floating point. */
template <typename T>
class BstGpairInternal {
  /*! \brief gradient statistics */
  T grad_;
  /*! \brief second order gradient statistics */
  T hess_;

  XGBOOST_DEVICE void SetGrad(float g) { grad_ = g; }
  XGBOOST_DEVICE void SetHess(float h) { hess_ = h; }

 public:
  using value_t = T;

  XGBOOST_DEVICE BstGpairInternal() : grad_(0), hess_(0) {}

  XGBOOST_DEVICE BstGpairInternal(float grad, float hess) {
    SetGrad(grad);
    SetHess(hess);
  }

  // Copy constructor if of same value type
  XGBOOST_DEVICE BstGpairInternal(const BstGpairInternal<T> &g) = default;

  // Copy constructor if different value type - use getters and setters to
  // perform conversion
  template <typename T2>
  XGBOOST_DEVICE explicit BstGpairInternal(const BstGpairInternal<T2> &g) {
    SetGrad(g.GetGrad());
    SetHess(g.GetHess());
  }

  XGBOOST_DEVICE float GetGrad() const { return grad_; }
  XGBOOST_DEVICE float GetHess() const { return hess_; }

  XGBOOST_DEVICE BstGpairInternal<T> &operator+=(
      const BstGpairInternal<T> &rhs) {
    grad_ += rhs.grad_;
    hess_ += rhs.hess_;
    return *this;
  }

  XGBOOST_DEVICE BstGpairInternal<T> operator+(
      const BstGpairInternal<T> &rhs) const {
    BstGpairInternal<T> g;
    g.grad_ = grad_ + rhs.grad_;
    g.hess_ = hess_ + rhs.hess_;
    return g;
  }

  XGBOOST_DEVICE BstGpairInternal<T> &operator-=(
      const BstGpairInternal<T> &rhs) {
    grad_ -= rhs.grad_;
    hess_ -= rhs.hess_;
    return *this;
  }

  XGBOOST_DEVICE BstGpairInternal<T> operator-(
      const BstGpairInternal<T> &rhs) const {
    BstGpairInternal<T> g;
    g.grad_ = grad_ - rhs.grad_;
    g.hess_ = hess_ - rhs.hess_;
    return g;
  }

  XGBOOST_DEVICE explicit BstGpairInternal(int value) {
    *this = BstGpairInternal<T>(static_cast<float>(value),
                                  static_cast<float>(value));
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const BstGpairInternal<T> &g) {
    os << g.GetGrad() << "/" << g.GetHess();
    return os;
  }
};

template<>
inline XGBOOST_DEVICE float BstGpairInternal<int64_t>::GetGrad() const {
  return grad_ * 1e-4f;
}
template<>
inline XGBOOST_DEVICE float BstGpairInternal<int64_t>::GetHess() const {
  return hess_ * 1e-4f;
}
template<>
inline XGBOOST_DEVICE void BstGpairInternal<int64_t>::SetGrad(float g) {
  grad_ = static_cast<int64_t>(std::round(g * 1e4));
}
template<>
inline XGBOOST_DEVICE void BstGpairInternal<int64_t>::SetHess(float h) {
  hess_ = static_cast<int64_t>(std::round(h * 1e4));
}

}  // namespace detail

/*! \brief gradient statistics pair usually needed in gradient boosting */
using bst_gpair = detail::BstGpairInternal<float>;

/*! \brief High precision gradient statistics pair */
using bst_gpair_precise = detail::BstGpairInternal<double>;

/*! \brief High precision gradient statistics pair with integer backed
 * storage. Operators are associative where floating point versions are not
 * associative. */
using bst_gpair_integer = detail::BstGpairInternal<int64_t>;

/*! \brief small eps gap for minimum split decision. */
const bst_float rt_eps = 1e-6f;

/*! \brief define unsigned long for openmp loop */
using omp_ulong = dmlc::omp_ulong;
/*! \brief define unsigned int for openmp loop */
using bst_omp_uint = dmlc::omp_uint;

/*!
 * \brief define compatible keywords in g++
 *  Used to support g++-4.6 and g++4.7
 */
#if DMLC_USE_CXX11 && defined(__GNUC__) && !defined(__clang_version__)
#if __GNUC__ == 4 && __GNUC_MINOR__ < 8
#define override
#define final
#endif
#endif
}  // namespace xgboost
#endif  // XGBOOST_BASE_H_
