#ifndef XGBOOST_UTILS_OMP_H_
#define XGBOOST_UTILS_OMP_H_
/*!
 * \file omp.h
 * \brief header to handle OpenMP compatibility issues
 * \author Tianqi Chen
 */
#if defined(_OPENMP)
#include <omp.h>
namespace xgboost {
// loop variable used in openmp
#ifdef _MSC_VER
typedef int bst_omp_uint;
#else
typedef unsigned bst_omp_uint;
#endif
} // namespace xgboost

#else
#ifndef DISABLE_OPENMP
#ifndef _MSC_VER
#warning "OpenMP is not available, compile to single thread code."\
		 "You may want to ungrade your compiler to enable OpenMP support,"\
		 "to get benefit of multi-threading."
#else
// TODO add warning for msvc
#endif
#endif
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline void omp_set_num_threads(int nthread) {}
#endif
#endif  // XGBOOST_UTILS_OMP_H_
