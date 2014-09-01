#ifndef XGBOOST_UTILS_OMP_H_
#define XGBOOST_UTILS_OMP_H_
/*!
 * \file omp.h
 * \brief header to handle OpenMP compatibility issues
 * \author Tianqi Chen
 */
#if defined(_OPENMP)
#include <omp.h>
#else
#ifndef DISABLE_OPENMP
// use pragma message instead of warning
#pragma message ("Warning: OpenMP is not available, xgboost will be compiled into single-thread code. Use OpenMP-enabled compiler to get benefit of multi-threading")
#endif
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline void omp_set_num_threads(int nthread) {}
#endif

// loop variable used in openmp
namespace xgboost {
#ifdef _MSC_VER
typedef int bst_omp_uint;
#else
typedef unsigned bst_omp_uint;
#endif
} // namespace xgboost

#endif  // XGBOOST_UTILS_OMP_H_
