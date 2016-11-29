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
 *  "[21:47:50] 6513x126 matrix with 143286 entries loaded from ../data/agaricus.txt.train"
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
#define XGBOOST_CUSTOMIZE_GLOBAL_PRNG  XGBOOST_STRICT_R_MODE
#endif

/*! \brief namespace of xgboo st*/
namespace xgboost {
/*!
 * \brief unsigned integer type used in boost,
 *  used for feature index and row index.
 */
typedef uint32_t bst_uint;
/*! \brief long integers */
typedef uint64_t bst_ulong;  // NOLINT(*)
/*! \brief float type, used for storing statistics */
typedef float bst_float;

/*! \brief gradient statistics pair usually needed in gradient boosting */
struct bst_gpair {
  /*! \brief gradient statistics */
  bst_float grad;
  /*! \brief second order gradient statistics */
  bst_float hess;
  bst_gpair() {}
  bst_gpair(bst_float grad, bst_float hess) : grad(grad), hess(hess) {}
};

/*! \brief small eps gap for minimum split decision. */
const float rt_eps = 1e-6f;

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
