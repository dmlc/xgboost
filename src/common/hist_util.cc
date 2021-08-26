/*!
 * Copyright 2017-2020 by Contributors
 * \file hist_util.cc
 */
#include <dmlc/timer.h>
#include <dmlc/omp.h>

#include <rabit/rabit.h>
#include <numeric>
#include <vector>

#include "xgboost/base.h"
#include "../common/common.h"
#include "hist_util.h"
#include "random.h"
#include "column_matrix.h"
#include "quantile.h"
#include "./../tree/updater_quantile_hist.h"
#include "../data/gradient_index.h"

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

HistogramCuts::HistogramCuts() {
  cut_ptrs_.HostVector().emplace_back(0);
}

/*!
 * \brief fill a histogram by zeros in range [begin, end)
 */
template<typename GradientSumT>
void InitilizeHistByZeroes(GHistRow<GradientSumT> hist, size_t begin, size_t end) {
#if defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  std::fill(hist.begin() + begin, hist.begin() + end,
            xgboost::detail::GradientPairInternal<GradientSumT>());
#else  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  memset(hist.data() + begin, '\0', (end-begin)*
         sizeof(xgboost::detail::GradientPairInternal<GradientSumT>));
#endif  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
}
template void InitilizeHistByZeroes(GHistRow<float> hist, size_t begin,
                                    size_t end);
template void InitilizeHistByZeroes(GHistRow<double> hist, size_t begin,
                                    size_t end);

/*!
 * \brief Increment hist as dst += add in range [begin, end)
 */
template<typename GradientSumT>
void IncrementHist(GHistRow<GradientSumT> dst, const GHistRow<GradientSumT> add,
                   size_t begin, size_t end) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst.data());
  const GradientSumT* padd = reinterpret_cast<const GradientSumT*>(add.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] += padd[i];
  }
}
template void IncrementHist(GHistRow<float> dst, const GHistRow<float> add,
                            size_t begin, size_t end);
template void IncrementHist(GHistRow<double> dst, const GHistRow<double> add,
                            size_t begin, size_t end);

/*!
 * \brief Copy hist from src to dst in range [begin, end)
 */
template<typename GradientSumT>
void CopyHist(GHistRow<GradientSumT> dst, const GHistRow<GradientSumT> src,
              size_t begin, size_t end) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst.data());
  const GradientSumT* psrc = reinterpret_cast<const GradientSumT*>(src.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc[i];
  }
}
template void CopyHist(GHistRow<float> dst, const GHistRow<float> src,
                       size_t begin, size_t end);
template void CopyHist(GHistRow<double> dst, const GHistRow<double> src,
                       size_t begin, size_t end);

/*!
 * \brief Compute Subtraction: dst = src1 - src2 in range [begin, end)
 */
template<typename GradientSumT>
void SubtractionHist(GradientSumT* dst, const GradientSumT* src1,
                     const GradientSumT* src2,
                     size_t begin, size_t end) {
  for (size_t i = begin; i < end; ++i) {
    dst[i] = src1[i] - src2[i];
  }
}
template void SubtractionHist(float* dst, const float* src1,
                              const float* src2,
                              size_t begin, size_t end);
template void SubtractionHist(double* dst, const double* src1,
                              const double* src2,
                              size_t begin, size_t end);

template<typename GradientSumT>
void ClearHist(GradientSumT* dest_hist,
                size_t begin, size_t end) {
  for (size_t bin_id = begin; bin_id < end; ++bin_id) {
    dest_hist[bin_id]  = 0;
  }
}
template void ClearHist(float* dest_hist,
                        size_t begin, size_t end);
template void ClearHist(double* dest_hist,
                        size_t begin, size_t end);

template<typename GradientSumT>
void ReduceHist(GradientSumT* dest_hist, GradientSumT* hist0,
                std::vector<std::vector<std::vector<GradientSumT>>>* histograms,
                const size_t node_id,
                const std::vector<uint16_t>& threads_id_for_node,
                size_t begin, size_t end) {
  for (size_t bin_id = begin; bin_id < end; ++bin_id) {
    dest_hist[bin_id] = hist0[bin_id];
    hist0[bin_id] = 0;
  }
  for (size_t tid = 1; tid < threads_id_for_node.size(); ++tid) {
    const size_t thread_id = threads_id_for_node[tid];
    GradientSumT* hist =  (*histograms)[thread_id][node_id].data();
    for (size_t bin_id = begin; bin_id < end; ++bin_id) {
      dest_hist[bin_id] += hist[bin_id];
      hist[bin_id] = 0;
    }
  }
}
template void ReduceHist(float* dest_hist, float* hist0,
                         std::vector<std::vector<std::vector<float>>>* histograms,
                         const size_t node_displace,
                         const std::vector<uint16_t>& threads_id_for_node,
                         size_t begin, size_t end);
template void ReduceHist(double* dest_hist, double* hist0,
                         std::vector<std::vector<std::vector<double>>>* histograms,
                         const size_t node_displace,
                         const std::vector<uint16_t>& threads_id_for_node,
                         size_t begin, size_t end);

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


template<typename FPType, bool do_prefetch, typename BinIdxType, bool any_missing = true>
void BuildHistKernel(const std::vector<GradientPair>& gpair,
                     const RowSetCollection::Elem row_indices,
                     const GHistIndexMatrix& gmat,
                     GHistRow<FPType> hist) {
  const size_t size = row_indices.Size();
  const size_t* rid = row_indices.begin;
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const BinIdxType* gradient_index = gmat.index.Data<BinIdxType>();
  const size_t* row_ptr =  gmat.row_ptr.data();
  const uint32_t* offsets = gmat.index.Offset();
  const size_t n_features = row_ptr[row_indices.begin[0]+1] - row_ptr[row_indices.begin[0]];
  FPType* hist_data = reinterpret_cast<FPType*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array

  for (size_t i = 0; i < size; ++i) {
    const size_t icol_start = any_missing ? row_ptr[rid[i]] : rid[i] * n_features;
    const size_t icol_end =  any_missing ? row_ptr[rid[i]+1] : icol_start + n_features;
    const size_t row_size = icol_end - icol_start;
    const size_t idx_gh = two * rid[i];

    if (do_prefetch) {
      const size_t icol_start_prefetch = any_missing ? row_ptr[rid[i+Prefetch::kPrefetchOffset]] :
                                       rid[i + Prefetch::kPrefetchOffset] * n_features;
      const size_t icol_end_prefetch = any_missing ?  row_ptr[rid[i+Prefetch::kPrefetchOffset]+1] :
                                      icol_start_prefetch + n_features;

      PREFETCH_READ_T0(pgh + two * rid[i + Prefetch::kPrefetchOffset]);
      for (size_t j = icol_start_prefetch; j < icol_end_prefetch;
        j+=Prefetch::GetPrefetchStep<uint32_t>()) {
        PREFETCH_READ_T0(gradient_index + j);
      }
    }
    const BinIdxType* gr_index_local = gradient_index + icol_start;

    for (size_t j = 0; j < row_size; ++j) {
      const uint32_t idx_bin = two * (static_cast<uint32_t>(gr_index_local[j]) + (
                                      any_missing ? 0 : offsets[j]));

      hist_data[idx_bin]   += pgh[idx_gh];
      hist_data[idx_bin+1] += pgh[idx_gh+1];
    }
  }
}

template<typename FPType, bool do_prefetch, bool any_missing>
void BuildHistDispatch(const std::vector<GradientPair>& gpair,
                       const RowSetCollection::Elem row_indices,
                       const GHistIndexMatrix& gmat, GHistRow<FPType> hist) {
  switch (gmat.index.GetBinTypeSize()) {
    case kUint8BinsTypeSize:
      BuildHistKernel<FPType, do_prefetch, uint8_t, any_missing>(gpair, row_indices,
                                                                      gmat, hist);
      break;
    case kUint16BinsTypeSize:
      BuildHistKernel<FPType, do_prefetch, uint16_t, any_missing>(gpair, row_indices,
                                                                       gmat, hist);
      break;
    case kUint32BinsTypeSize:
      BuildHistKernel<FPType, do_prefetch, uint32_t, any_missing>(gpair, row_indices,
                                                                       gmat, hist);
      break;
    default:
      CHECK(false);  // no default behavior
  }
}

template <typename GradientSumT>
template <bool any_missing>
void GHistBuilder<GradientSumT>::BuildHist(
    const std::vector<GradientPair> &gpair,
    const RowSetCollection::Elem row_indices,
    const GHistIndexMatrix &gmat,
    GHistRowT hist) {
  const size_t nrows = row_indices.Size();
  const size_t no_prefetch_size = Prefetch::NoPrefetchSize(nrows);

  // if need to work with all rows from bin-matrix (e.g. root node)
  const bool contiguousBlock = (row_indices.begin[nrows - 1] - row_indices.begin[0]) == (nrows - 1);

  if (contiguousBlock) {
    // contiguous memory access, built-in HW prefetching is enough
    BuildHistDispatch<GradientSumT, false, any_missing>(gpair, row_indices, gmat, hist);
  } else {
    const RowSetCollection::Elem span1(row_indices.begin, row_indices.end - no_prefetch_size);
    const RowSetCollection::Elem span2(row_indices.end - no_prefetch_size, row_indices.end);

    BuildHistDispatch<GradientSumT, true, any_missing>(gpair, span1, gmat, hist);
    // no prefetching to avoid loading extra memory
    BuildHistDispatch<GradientSumT, false, any_missing>(gpair, span2, gmat, hist);
  }
}
template void
GHistBuilder<float>::BuildHist<true>(const std::vector<GradientPair> &gpair,
                                     const RowSetCollection::Elem row_indices,
                                     const GHistIndexMatrix &gmat,
                                     GHistRow<float> hist);
template void
GHistBuilder<float>::BuildHist<false>(const std::vector<GradientPair> &gpair,
                                      const RowSetCollection::Elem row_indices,
                                      const GHistIndexMatrix &gmat,
                                      GHistRow<float> hist);
template void
GHistBuilder<double>::BuildHist<true>(const std::vector<GradientPair> &gpair,
                                      const RowSetCollection::Elem row_indices,
                                      const GHistIndexMatrix &gmat,
                                      GHistRow<double> hist);
template void
GHistBuilder<double>::BuildHist<false>(const std::vector<GradientPair> &gpair,
                                       const RowSetCollection::Elem row_indices,
                                       const GHistIndexMatrix &gmat,
                                       GHistRow<double> hist);

}  // namespace common
}  // namespace xgboost
