#ifndef XGBOOST_OMP_H
#define XGBOOST_OMP_H
/*!
 * \file xgboost_omp.h
 * \brief header to handle OpenMP compatibility issues
 *
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */

#if defined(_OPENMP)
#include <omp.h>
#else
#warning "OpenMP is not available, compile to single thread code"
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline void omp_set_num_threads(int nthread) {}
#endif
#endif
