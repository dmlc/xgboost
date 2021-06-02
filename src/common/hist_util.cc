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

void GHistIndexMatrix::ResizeIndex(const size_t n_index,
                                   const bool isDense) {
  if ((max_num_bins - 1 <= static_cast<int>(std::numeric_limits<uint8_t>::max())) && isDense) {
    index.SetBinTypeSize(kUint8BinsTypeSize);
    index.Resize((sizeof(uint8_t)) * n_index);
  } else if ((max_num_bins - 1 > static_cast<int>(std::numeric_limits<uint8_t>::max())  &&
    max_num_bins - 1 <= static_cast<int>(std::numeric_limits<uint16_t>::max())) && isDense) {
    index.SetBinTypeSize(kUint16BinsTypeSize);
    index.Resize((sizeof(uint16_t)) * n_index);
  } else {
    index.SetBinTypeSize(kUint32BinsTypeSize);
    index.Resize((sizeof(uint32_t)) * n_index);
  }
}

HistogramCuts::HistogramCuts() {
  cut_ptrs_.HostVector().emplace_back(0);
}

void GHistIndexMatrix::Init(DMatrix* p_fmat, int max_bins) {
  cut = SketchOnDMatrix(p_fmat, max_bins);

  max_num_bins = max_bins;
  const int32_t nthread = omp_get_max_threads();
  const uint32_t nbins = cut.Ptrs().back();
  hit_count.resize(nbins, 0);
  hit_count_tloc_.resize(nthread * nbins, 0);

  this->p_fmat = p_fmat;
  size_t new_size = 1;
  for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
    new_size += batch.Size();
  }

  row_ptr.resize(new_size);
  row_ptr[0] = 0;

  size_t rbegin = 0;
  size_t prev_sum = 0;
  const bool isDense = p_fmat->IsDense();
  this->isDense_ = isDense;

  for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
    // The number of threads is pegged to the batch size. If the OMP
    // block is parallelized on anything other than the batch/block size,
    // it should be reassigned
    const size_t batch_threads = std::max(
        size_t(1),
        std::min(batch.Size(), static_cast<size_t>(omp_get_max_threads())));
    auto page = batch.GetView();
    MemStackAllocator<size_t, 128> partial_sums(batch_threads);
    size_t* p_part = partial_sums.Get();

    size_t block_size =  batch.Size() / batch_threads;

    dmlc::OMPException exc;
    #pragma omp parallel num_threads(batch_threads)
    {
      #pragma omp for
      for (omp_ulong tid = 0; tid < batch_threads; ++tid) {
        exc.Run([&]() {
          size_t ibegin = block_size * tid;
          size_t iend = (tid == (batch_threads-1) ? batch.Size() : (block_size * (tid+1)));

          size_t sum = 0;
          for (size_t i = ibegin; i < iend; ++i) {
            sum += page[i].size();
            row_ptr[rbegin + 1 + i] = sum;
          }
        });
      }

      #pragma omp single
      {
        exc.Run([&]() {
          p_part[0] = prev_sum;
          for (size_t i = 1; i < batch_threads; ++i) {
            p_part[i] = p_part[i - 1] + row_ptr[rbegin + i*block_size];
          }
        });
      }

      #pragma omp for
      for (omp_ulong tid = 0; tid < batch_threads; ++tid) {
        exc.Run([&]() {
          size_t ibegin = block_size * tid;
          size_t iend = (tid == (batch_threads-1) ? batch.Size() : (block_size * (tid+1)));

          for (size_t i = ibegin; i < iend; ++i) {
            row_ptr[rbegin + 1 + i] += p_part[tid];
          }
        });
      }
    }
    exc.Rethrow();

    const size_t n_offsets = cut.Ptrs().size() - 1;
    const size_t n_index = row_ptr[rbegin + batch.Size()];
    ResizeIndex(n_index, isDense);

    CHECK_GT(cut.Values().size(), 0U);

    uint32_t* offsets = nullptr;
    if (isDense) {
      index.ResizeOffset(n_offsets);
      offsets = index.Offset();
      for (size_t i = 0; i < n_offsets; ++i) {
        offsets[i] = cut.Ptrs()[i];
      }
    }

    if (isDense) {
      BinTypeSize curent_bin_size = index.GetBinTypeSize();
      if (curent_bin_size == kUint8BinsTypeSize) {
        common::Span<uint8_t> index_data_span = {index.data<uint8_t>(),
                                                 n_index};
        SetIndexData(index_data_span, batch_threads, batch, rbegin, nbins,
                     [offsets](auto idx, auto j) {
                       return static_cast<uint8_t>(idx - offsets[j]);
                     });

      } else if (curent_bin_size == kUint16BinsTypeSize) {
        common::Span<uint16_t> index_data_span = {index.data<uint16_t>(),
                                                  n_index};
        SetIndexData(index_data_span, batch_threads, batch, rbegin, nbins,
                     [offsets](auto idx, auto j) {
                       return static_cast<uint16_t>(idx - offsets[j]);
                     });
      } else {
        CHECK_EQ(curent_bin_size, kUint32BinsTypeSize);
        common::Span<uint32_t> index_data_span = {index.data<uint32_t>(),
                                                  n_index};
        SetIndexData(index_data_span, batch_threads, batch, rbegin, nbins,
                     [offsets](auto idx, auto j) {
                       return static_cast<uint32_t>(idx - offsets[j]);
                     });
      }

    /* For sparse DMatrix we have to store index of feature for each bin
       in index field to chose right offset. So offset is nullptr and index is not reduced */
    } else {
      common::Span<uint32_t> index_data_span = {index.data<uint32_t>(), n_index};
      SetIndexData(index_data_span, batch_threads, batch, rbegin, nbins,
                   [](auto idx, auto) { return idx; });
    }

    ParallelFor(bst_omp_uint(nbins), nthread, [&](bst_omp_uint idx) {
      for (int32_t tid = 0; tid < nthread; ++tid) {
        hit_count[idx] += hit_count_tloc_[tid * nbins + idx];
        hit_count_tloc_[tid * nbins + idx] = 0;  // reset for next batch
      }
    });

    prev_sum = row_ptr[rbegin + batch.Size()];
    rbegin += batch.Size();
  }
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
void SubtractionHist(GHistRow<GradientSumT> dst, const GHistRow<GradientSumT> src1,
                     const GHistRow<GradientSumT> src2,
                     size_t begin, size_t end) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst.data());
  const GradientSumT* psrc1 = reinterpret_cast<const GradientSumT*>(src1.data());
  const GradientSumT* psrc2 = reinterpret_cast<const GradientSumT*>(src2.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc1[i] - psrc2[i];
  }
}
template void SubtractionHist(GHistRow<float> dst, const GHistRow<float> src1,
                              const GHistRow<float> src2,
                              size_t begin, size_t end);
template void SubtractionHist(GHistRow<double> dst, const GHistRow<double> src1,
                              const GHistRow<double> src2,
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


template<typename FPType, bool do_prefetch, typename BinIdxType>
void BuildHistDenseKernel(const std::vector<GradientPair>& gpair,
                          const RowSetCollection::Elem row_indices,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          GHistRow<FPType> hist) {
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
                           GHistRow<FPType> hist) {
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
                     const GHistIndexMatrix& gmat, GHistRow<FPType> hist, bool isDense) {
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
                     const GHistIndexMatrix& gmat, const bool isDense, GHistRow<FPType> hist) {
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

template <typename GradientSumT>
void GHistBuilder<GradientSumT>::BuildHist(
    const std::vector<GradientPair> &gpair,
    const RowSetCollection::Elem row_indices, const GHistIndexMatrix &gmat,
    GHistRowT hist, bool isDense) {
  const size_t nrows = row_indices.Size();
  const size_t no_prefetch_size = Prefetch::NoPrefetchSize(nrows);

  // if need to work with all rows from bin-matrix (e.g. root node)
  const bool contiguousBlock = (row_indices.begin[nrows - 1] - row_indices.begin[0]) == (nrows - 1);

  if (contiguousBlock) {
    // contiguous memory access, built-in HW prefetching is enough
    BuildHistKernel<GradientSumT, false>(gpair, row_indices, gmat, isDense, hist);
  } else {
    const RowSetCollection::Elem span1(row_indices.begin, row_indices.end - no_prefetch_size);
    const RowSetCollection::Elem span2(row_indices.end - no_prefetch_size, row_indices.end);

    BuildHistKernel<GradientSumT, true>(gpair, span1, gmat, isDense, hist);
    // no prefetching to avoid loading extra memory
    BuildHistKernel<GradientSumT, false>(gpair, span2, gmat, isDense, hist);
  }
}
template
void GHistBuilder<float>::BuildHist(const std::vector<GradientPair>& gpair,
                             const RowSetCollection::Elem row_indices,
                             const GHistIndexMatrix& gmat,
                             GHistRow<float> hist,
                             bool isDense);
template
void GHistBuilder<double>::BuildHist(const std::vector<GradientPair>& gpair,
                             const RowSetCollection::Elem row_indices,
                             const GHistIndexMatrix& gmat,
                             GHistRow<double> hist,
                             bool isDense);

template<typename GradientSumT>
void GHistBuilder<GradientSumT>::SubtractionTrick(GHistRowT self,
                                                  GHistRowT sibling,
                                                  GHistRowT parent) {
  const size_t size = self.size();
  CHECK_EQ(sibling.size(), size);
  CHECK_EQ(parent.size(), size);

  const size_t block_size = 1024;  // aproximatly 1024 values per block
  size_t n_blocks = size/block_size + !!(size%block_size);

  ParallelFor(omp_ulong(n_blocks), [&](omp_ulong iblock) {
    const size_t ibegin = iblock*block_size;
    const size_t iend = (((iblock+1)*block_size > size) ? size : ibegin + block_size);
    SubtractionHist(self, parent, sibling, ibegin, iend);
  });
}
template
void GHistBuilder<float>::SubtractionTrick(GHistRow<float> self,
                                           GHistRow<float> sibling,
                                           GHistRow<float> parent);
template
void GHistBuilder<double>::SubtractionTrick(GHistRow<double> self,
                                            GHistRow<double> sibling,
                                            GHistRow<double> parent);

}  // namespace common
}  // namespace xgboost
