/*!
 * Copyright 2017-2019 by Contributors
 * \file hist_util.cc
 */
#include <dmlc/timer.h>
#include <dmlc/omp.h>

#include <rabit/rabit.h>
#include <numeric>
#include <vector>

#include "xgboost/base.h"
#include "./hist_util_sycl.h"
#include "./../../src/tree/updater_quantile_hist.h"

#include "CL/sycl.hpp"

#if defined(XGBOOST_MM_PREFETCH_PRESENT)
  #include <xmmintrin.h>
  #define PREFETCH_READ_T0(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#elif defined(XGBOOST_BUILTIN_PREFETCH_PRESENT)
  #define PREFETCH_READ_T0(addr) __builtin_prefetch(reinterpret_cast<const char*>(addr), 0, 3)
#else  // no SW pre-fetching available; PREFETCH_READ_T0 is no-op
  #define PREFETCH_READ_T0(addr) do {} while (0)
#endif  // defined(XGBOOST_MM_PREFETCH_PRESENT)

namespace xgboost {
namespace common {

void IncrementHistSycl(cl::sycl::queue qu, GHistRowSycl dst, const GHistRowSycl add, size_t begin, size_t end) {
  using FPType = decltype(tree::GradStats::sum_grad);
  FPType* pdst = reinterpret_cast<FPType*>(dst.data());
  const FPType* padd = reinterpret_cast<const FPType*>(add.data());

  qu.submit([&](cl::sycl::handler& cgh) {
  	cgh.parallel_for<class IncrementHist>(cl::sycl::range<1>(2 * (end - begin)), [=](cl::sycl::id<1> pid) {
  	  pdst[pid[0]] += padd[pid[0]];	
  	});
  }).wait();
}

void CopyHistSycl(cl::sycl::queue qu, GHistRowSycl dst, const GHistRowSycl src, size_t begin, size_t end) {
  using FPType = decltype(tree::GradStats::sum_grad);
  FPType* pdst = reinterpret_cast<FPType*>(dst.data());
  const FPType* psrc = reinterpret_cast<const FPType*>(src.data());

  qu.submit([&](cl::sycl::handler& cgh) {
  	cgh.parallel_for<class CopyHist>(cl::sycl::range<1>(2 * (end - begin)), [=](cl::sycl::id<1> pid) {
  	  pdst[pid[0]] = psrc[pid[0]];	
  	});
  }).wait();
}


void SubtractionHistSycl(cl::sycl::queue qu, GHistRowSycl dst, const GHistRowSycl src1, const GHistRowSycl src2,
                     size_t begin, size_t end) {
  using FPType = decltype(tree::GradStats::sum_grad);
  FPType* pdst = reinterpret_cast<FPType*>(dst.data());
  const FPType* psrc1 = reinterpret_cast<const FPType*>(src1.data());
  const FPType* psrc2 = reinterpret_cast<const FPType*>(src2.data());

  qu.submit([&](cl::sycl::handler& cgh) {
  	cgh.parallel_for<class SubtractionHist>(cl::sycl::range<1>(2 * (end - begin)), [=](cl::sycl::id<1> pid) {
  	  pdst[pid[0]] = psrc1[pid[0]] - psrc2[pid[0]];	
  	});
  }).wait();
}

}  // namespace common
}  // namespace xgboost
