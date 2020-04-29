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

struct Prefetch {
 public:
  static constexpr size_t kCacheLineSize = 64;
  static constexpr size_t kPrefetchOffset = 10;

 private:
  static constexpr size_t kNoPrefetchSize =
      kPrefetchOffset + kCacheLineSize /
      sizeof(decltype(GHistIndexMatrix::row_ptr)::value_type);

 public:
  static size_t NoPrefetchSize(size_t rows) {
    return std::min(rows, kNoPrefetchSize);
  }

  template <typename T>
  static constexpr size_t GetPrefetchStep() {
    return Prefetch::kCacheLineSize / sizeof(T);
  }
};

constexpr size_t Prefetch::kNoPrefetchSize;

template<typename FPType, bool do_prefetch, typename BinIdxType>
void BuildHistDenseKernel(const std::vector<GradientPair>& gpair,
                          const RowSetCollection::Elem row_indices,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          GHistRowSycl hist) {
  const size_t size = row_indices.Size();
  const size_t* rid = row_indices.begin;
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  FPType* hist_data = reinterpret_cast<FPType*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array

  for (size_t i = 0; i < size; ++i) {
    const size_t icol_start = rid[i] * n_features;
    const size_t idx_gh = two * rid[i];

    if (do_prefetch) {
      const size_t icol_start_prefetch = rid[i + Prefetch::kPrefetchOffset] * n_features;

      PREFETCH_READ_T0(pgh + two * rid[i + Prefetch::kPrefetchOffset]);
      for (size_t j = icol_start_prefetch; j < icol_start_prefetch + n_features;
           j += Prefetch::GetPrefetchStep<BinIdxType>()) {
        PREFETCH_READ_T0(gradient_index + j);
      }
    }
    const BinIdxType* gr_index_local = gradient_index + icol_start;
    for (size_t j = 0; j < n_features; ++j) {
      const uint32_t idx_bin = two * (static_cast<uint32_t>(gr_index_local[j]) +
                                      offsets[j]);

      hist_data[idx_bin]   += pgh[idx_gh];
      hist_data[idx_bin+1] += pgh[idx_gh+1];
    }
  }
}

template<typename FPType, bool do_prefetch>
void BuildHistSparseKernel(const std::vector<GradientPair>& gpair,
                           const RowSetCollection::Elem row_indices,
                           const GHistIndexMatrix& gmat,
                           GHistRowSycl hist) {
  const size_t size = row_indices.Size();
  const size_t* rid = row_indices.begin;
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const uint32_t* gradient_index = gmat.index.data<uint32_t>();
  const size_t* row_ptr =  gmat.row_ptr.data();
  FPType* hist_data = reinterpret_cast<FPType*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array

  for (size_t i = 0; i < size; ++i) {
    const size_t icol_start = row_ptr[rid[i]];
    const size_t icol_end = row_ptr[rid[i]+1];
    const size_t idx_gh = two * rid[i];

    if (do_prefetch) {
      const size_t icol_start_prftch = row_ptr[rid[i+Prefetch::kPrefetchOffset]];
      const size_t icol_end_prefect = row_ptr[rid[i+Prefetch::kPrefetchOffset]+1];

      PREFETCH_READ_T0(pgh + two * rid[i + Prefetch::kPrefetchOffset]);
      for (size_t j = icol_start_prftch; j < icol_end_prefect;
        j+=Prefetch::GetPrefetchStep<uint32_t>()) {
        PREFETCH_READ_T0(gradient_index + j);
      }
    }
    for (size_t j = icol_start; j < icol_end; ++j) {
      const uint32_t idx_bin = two * gradient_index[j];
      hist_data[idx_bin]   += pgh[idx_gh];
      hist_data[idx_bin+1] += pgh[idx_gh+1];
    }
  }
}


template<typename FPType, bool do_prefetch, typename BinIdxType>
void BuildHistDispatchKernel(const std::vector<GradientPair>& gpair,
                     const RowSetCollection::Elem row_indices,
                     const GHistIndexMatrix& gmat, GHistRowSycl hist, bool isDense) {
  if (isDense) {
    const size_t* row_ptr =  gmat.row_ptr.data();
    const size_t n_features = row_ptr[row_indices.begin[0]+1] - row_ptr[row_indices.begin[0]];
    BuildHistDenseKernel<FPType, do_prefetch, BinIdxType>(gpair, row_indices,
                                                       gmat, n_features, hist);
  } else {
    BuildHistSparseKernel<FPType, do_prefetch>(gpair, row_indices,
                                                        gmat, hist);
  }
}

template<typename FPType, bool do_prefetch>
void BuildHistKernel(const std::vector<GradientPair>& gpair,
                     const RowSetCollection::Elem row_indices,
                     const GHistIndexMatrix& gmat, const bool isDense, GHistRowSycl hist) {
  const bool is_dense = row_indices.Size() && isDense;
  switch (gmat.index.GetBinTypeSize()) {
    case kUint8BinsTypeSize:
      BuildHistDispatchKernel<FPType, do_prefetch, uint8_t>(gpair, row_indices,
                                                            gmat, hist, is_dense);
      break;
    case kUint16BinsTypeSize:
      BuildHistDispatchKernel<FPType, do_prefetch, uint16_t>(gpair, row_indices,
                                                             gmat, hist, is_dense);
      break;
    case kUint32BinsTypeSize:
      BuildHistDispatchKernel<FPType, do_prefetch, uint32_t>(gpair, row_indices,
                                                             gmat, hist, is_dense);
      break;
    default:
      CHECK(false);  // no default behavior
  }
}

void GHistBuilderSycl::BuildHist(const std::vector<GradientPair>& gpair,
                             const RowSetCollection::Elem row_indices,
                             const GHistIndexMatrix& gmat,
                             GHistRowSycl hist,
                             bool isDense) {
  using FPType = decltype(tree::GradStats::sum_grad);
  const size_t nrows = row_indices.Size();
  const size_t no_prefetch_size = Prefetch::NoPrefetchSize(nrows);

  // if need to work with all rows from bin-matrix (e.g. root node)
  const bool contiguousBlock = (row_indices.begin[nrows - 1] - row_indices.begin[0]) == (nrows - 1);

  if (contiguousBlock) {
    // contiguous memory access, built-in HW prefetching is enough
    BuildHistKernel<FPType, false>(gpair, row_indices, gmat, isDense, hist);
  } else {
    const RowSetCollection::Elem span1(row_indices.begin, row_indices.end - no_prefetch_size);
    const RowSetCollection::Elem span2(row_indices.end - no_prefetch_size, row_indices.end);

    BuildHistKernel<FPType, true>(gpair, span1, gmat, isDense, hist);
    // no prefetching to avoid loading extra memory
    BuildHistKernel<FPType, false>(gpair, span2, gmat, isDense, hist);
  }
}

void GHistBuilderSycl::BuildBlockHist(const std::vector<GradientPair>& gpair,
                                  const RowSetCollection::Elem row_indices,
                                  const GHistIndexBlockMatrix& gmatb,
                                  GHistRowSycl hist) {
  constexpr int kUnroll = 8;  // loop unrolling factor
  const size_t nblock = gmatb.GetNumBlock();
  const size_t nrows = row_indices.end - row_indices.begin;
  const size_t rest = nrows % kUnroll;
#if defined(_OPENMP)
  const auto nthread = static_cast<bst_omp_uint>(this->nthread_);  // NOLINT
#endif  // defined(_OPENMP)
  tree::GradStats* p_hist = hist.data();

#pragma omp parallel for num_threads(nthread) schedule(guided)
  for (bst_omp_uint bid = 0; bid < nblock; ++bid) {
    auto gmat = gmatb[bid];

    for (size_t i = 0; i < nrows - rest; i += kUnroll) {
      size_t rid[kUnroll];
      size_t ibegin[kUnroll];
      size_t iend[kUnroll];
      GradientPair stat[kUnroll];

      for (int k = 0; k < kUnroll; ++k) {
        rid[k] = row_indices.begin[i + k];
        ibegin[k] = gmat.row_ptr[rid[k]];
        iend[k] = gmat.row_ptr[rid[k] + 1];
        stat[k] = gpair[rid[k]];
      }
      for (int k = 0; k < kUnroll; ++k) {
        for (size_t j = ibegin[k]; j < iend[k]; ++j) {
          const uint32_t bin = gmat.index[j];
          p_hist[bin].Add(stat[k]);
        }
      }
    }
    for (size_t i = nrows - rest; i < nrows; ++i) {
      const size_t rid = row_indices.begin[i];
      const size_t ibegin = gmat.row_ptr[rid];
      const size_t iend = gmat.row_ptr[rid + 1];
      const GradientPair stat = gpair[rid];
      for (size_t j = ibegin; j < iend; ++j) {
        const uint32_t bin = gmat.index[j];
        p_hist[bin].Add(stat);
      }
    }
  }
}

void GHistBuilderSycl::SubtractionTrick(GHistRowSycl self, GHistRowSycl sibling, GHistRowSycl parent) {
  const size_t size = self.size();
  CHECK_EQ(sibling.size(), size);
  CHECK_EQ(parent.size(), size);

  const size_t block_size = 1024;  // aproximatly 1024 values per block
  size_t n_blocks = size/block_size + !!(size%block_size);

#pragma omp parallel for
  for (omp_ulong iblock = 0; iblock < n_blocks; ++iblock) {
    const size_t ibegin = iblock*block_size;
    const size_t iend = (((iblock+1)*block_size > size) ? size : ibegin + block_size);
    SubtractionHist(self, parent, sibling, ibegin, iend);
  }
}

}  // namespace common
}  // namespace xgboost
