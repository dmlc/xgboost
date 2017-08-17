/*!
 * Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief defines configuration macros of xgboost.
 */
#ifndef XGBOOST_BASE_H_
#define XGBOOST_BASE_H_

#include <dmlc/base.h>
#include <dmlc/omp.h>

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
typedef uint32_t bst_uint;
typedef int32_t bst_int;
/*! \brief long integers */
typedef uint64_t bst_ulong;  // NOLINT(*)
/*! \brief float type, used for storing statistics */
typedef float bst_float;


/*! \brief Implementation of gradient statistics pair */
template <typename T>
struct bst_gpair_internal {
  /*! \brief gradient statistics */
  T grad;
  /*! \brief second order gradient statistics */
  T hess;

  XGBOOST_DEVICE bst_gpair_internal() : grad(0), hess(0) {}

  XGBOOST_DEVICE bst_gpair_internal(T grad, T hess)
      : grad(grad), hess(hess) {}

  template <typename T2>
  XGBOOST_DEVICE bst_gpair_internal(bst_gpair_internal<T2>&g)
      : grad(g.grad), hess(g.hess) {}

  XGBOOST_DEVICE bst_gpair_internal<T> &operator+=(const bst_gpair_internal<T> &rhs) {
    grad += rhs.grad;
    hess += rhs.hess;
    return *this;
  }

  XGBOOST_DEVICE bst_gpair_internal<T> operator+(const bst_gpair_internal<T> &rhs) const {
    bst_gpair_internal<T> g;
    g.grad = grad + rhs.grad;
    g.hess = hess + rhs.hess;
    return g;
  }

  XGBOOST_DEVICE bst_gpair_internal<T> &operator-=(const bst_gpair_internal<T> &rhs) {
    grad -= rhs.grad;
    hess -= rhs.hess;
    return *this;
  }

  XGBOOST_DEVICE bst_gpair_internal<T> operator-(const bst_gpair_internal<T> &rhs) const {
    bst_gpair_internal<T> g;
    g.grad = grad - rhs.grad;
    g.hess = hess - rhs.hess;
    return g;
  }

  XGBOOST_DEVICE bst_gpair_internal(int value) {
    *this = bst_gpair_internal<T>(static_cast<float>(value), static_cast<float>(value));
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const bst_gpair_internal<T> &g) {
    os << g.grad << "/" << g.hess;
    return os;
  }
};

/*! \brief gradient statistics pair usually needed in gradient boosting */
typedef bst_gpair_internal<float> bst_gpair;

/*! \brief High precision gradient statistics pair */
typedef bst_gpair_internal<double> bst_gpair_precise;

/*! \brief small eps gap for minimum split decision. */
const bst_float rt_eps = 1e-6f;

/*! \brief define unsigned long for openmp loop */
typedef dmlc::omp_ulong omp_ulong;
/*! \brief define unsigned int for openmp loop */
typedef dmlc::omp_uint bst_omp_uint;

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
