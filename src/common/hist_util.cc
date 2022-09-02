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

HistogramCuts SketchOnDMatrix(DMatrix *m, int32_t max_bins, int32_t n_threads, bool use_sorted,
                              Span<float> const hessian) {
  HistogramCuts out;
  auto const& info = m->Info();
  std::vector<bst_row_t> reduced(info.num_col_, 0);
  for (auto const &page : m->GetBatches<SparsePage>()) {
    auto const &entries_per_column =
        CalcColumnSize(data::SparsePageAdapterBatch{page.GetView()}, info.num_col_, n_threads,
                       [](auto) { return true; });
    CHECK_EQ(entries_per_column.size(), info.num_col_);
    for (size_t i = 0; i < entries_per_column.size(); ++i) {
      reduced[i] += entries_per_column[i];
    }
  }

  if (!use_sorted) {
    HostSketchContainer container(max_bins, m->Info().feature_types.ConstHostSpan(), reduced,
                                  HostSketchContainer::UseGroup(info), n_threads);
    for (auto const& page : m->GetBatches<SparsePage>()) {
      container.PushRowPage(page, info, hessian);
    }
    container.MakeCuts(&out);
  } else {
    SortedSketchContainer container{max_bins, m->Info().feature_types.ConstHostSpan(), reduced,
                                    HostSketchContainer::UseGroup(info), n_threads};
    for (auto const& page : m->GetBatches<SortedCSCPage>()) {
      container.PushColPage(page, info, hessian);
    }
    container.MakeCuts(&out);
  }

  return out;
}

/*!
 * \brief fill a histogram by zeros in range [begin, end)
 */
void InitilizeHistByZeroes(GHistRow hist, size_t begin, size_t end) {
#if defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  std::fill(hist.begin() + begin, hist.begin() + end, xgboost::GradientPairPrecise());
#else  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  memset(hist.data() + begin, '\0', (end - begin) * sizeof(xgboost::GradientPairPrecise));
#endif  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
}

/*!
 * \brief Increment hist as dst += add in range [begin, end)
 */
void IncrementHist(GHistRow dst, const GHistRow add, size_t begin, size_t end) {
  double* pdst = reinterpret_cast<double*>(dst.data());
  const double *padd = reinterpret_cast<const double *>(add.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] += padd[i];
  }
}

/*!
 * \brief Copy hist from src to dst in range [begin, end)
 */
void CopyHist(GHistRow dst, const GHistRow src, size_t begin, size_t end) {
  double *pdst = reinterpret_cast<double *>(dst.data());
  const double *psrc = reinterpret_cast<const double *>(src.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc[i];
  }
}

/*!
 * \brief Compute Subtraction: dst = src1 - src2 in range [begin, end)
 */
void SubtractionHist(GHistRow dst, const GHistRow src1, const GHistRow src2, size_t begin,
                     size_t end) {
  double* pdst = reinterpret_cast<double*>(dst.data());
  const double* psrc1 = reinterpret_cast<const double*>(src1.data());
  const double* psrc2 = reinterpret_cast<const double*>(src2.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc1[i] - psrc2[i];
  }
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

template <bool do_prefetch, typename BinIdxType, bool first_page, bool any_missing = true>
void BuildHistKernel(const std::vector<GradientPair> &gpair,
                     const RowSetCollection::Elem row_indices, const GHistIndexMatrix &gmat,
                     GHistRow hist) {
  const size_t size = row_indices.Size();
  const size_t *rid = row_indices.begin;
  auto const *pgh = reinterpret_cast<const float *>(gpair.data());
  const BinIdxType *gradient_index = gmat.index.data<BinIdxType>();

  auto const &row_ptr = gmat.row_ptr.data();
  auto base_rowid = gmat.base_rowid;
  const uint32_t *offsets = gmat.index.Offset();
  auto get_row_ptr = [&](size_t ridx) {
    return first_page ? row_ptr[ridx] : row_ptr[ridx - base_rowid];
  };
  auto get_rid = [&](size_t ridx) {
    return first_page ? ridx : (ridx - base_rowid);
  };

  const size_t n_features =
      get_row_ptr(row_indices.begin[0] + 1) - get_row_ptr(row_indices.begin[0]);
  auto hist_data = reinterpret_cast<double *>(hist.data());
  const uint32_t two{2};  // Each element from 'gpair' and 'hist' contains
                          // 2 FP values: gradient and hessian.
                          // So we need to multiply each row-index/bin-index by 2
                          // to work with gradient pairs as a singe row FP array

  for (size_t i = 0; i < size; ++i) {
    const size_t icol_start =
        any_missing ? get_row_ptr(rid[i]) : get_rid(rid[i]) * n_features;
    const size_t icol_end =
        any_missing ? get_row_ptr(rid[i] + 1) : icol_start + n_features;

    const size_t row_size = icol_end - icol_start;
    const size_t idx_gh = two * rid[i];

    if (do_prefetch) {
      const size_t icol_start_prefetch =
          any_missing
              ? get_row_ptr(rid[i + Prefetch::kPrefetchOffset])
              : get_rid(rid[i + Prefetch::kPrefetchOffset]) * n_features;
      const size_t icol_end_prefetch =
          any_missing ? get_row_ptr(rid[i + Prefetch::kPrefetchOffset] + 1)
                      : icol_start_prefetch + n_features;

      PREFETCH_READ_T0(pgh + two * rid[i + Prefetch::kPrefetchOffset]);
      for (size_t j = icol_start_prefetch; j < icol_end_prefetch;
           j += Prefetch::GetPrefetchStep<uint32_t>()) {
        PREFETCH_READ_T0(gradient_index + j);
      }
    }
    const BinIdxType *gr_index_local = gradient_index + icol_start;

    // The trick with pgh_t buffer helps the compiler to generate faster binary.
    const float pgh_t[] = {pgh[idx_gh], pgh[idx_gh + 1]};
    for (size_t j = 0; j < row_size; ++j) {
      const uint32_t idx_bin = two * (static_cast<uint32_t>(gr_index_local[j]) +
                                      (any_missing ? 0 : offsets[j]));
      auto hist_local = hist_data + idx_bin;
      *(hist_local)     += pgh_t[0];
      *(hist_local + 1) += pgh_t[1];
    }
  }
}

template <bool do_prefetch, bool any_missing>
void BuildHistDispatch(const std::vector<GradientPair> &gpair,
                       const RowSetCollection::Elem row_indices, const GHistIndexMatrix &gmat,
                       GHistRow hist) {
  auto first_page = gmat.base_rowid == 0;
  if (first_page) {
    switch (gmat.index.GetBinTypeSize()) {
    case kUint8BinsTypeSize:
      BuildHistKernel<do_prefetch, uint8_t, true, any_missing>(gpair, row_indices, gmat, hist);
      break;
    case kUint16BinsTypeSize:
      BuildHistKernel<do_prefetch, uint16_t, true, any_missing>(gpair, row_indices, gmat, hist);
      break;
    case kUint32BinsTypeSize:
      BuildHistKernel<do_prefetch, uint32_t, true, any_missing>(gpair, row_indices, gmat, hist);
      break;
    default:
      CHECK(false);  // no default behavior
    }
  } else {
    switch (gmat.index.GetBinTypeSize()) {
    case kUint8BinsTypeSize:
      BuildHistKernel<do_prefetch, uint8_t, false, any_missing>(gpair, row_indices, gmat, hist);
      break;
    case kUint16BinsTypeSize:
      BuildHistKernel<do_prefetch, uint16_t, false, any_missing>(gpair, row_indices, gmat, hist);
      break;
    case kUint32BinsTypeSize:
      BuildHistKernel<do_prefetch, uint32_t, false, any_missing>(gpair, row_indices, gmat, hist);
      break;
    default:
      CHECK(false);  // no default behavior
    }
  }
}

template <bool any_missing>
void GHistBuilder::BuildHist(const std::vector<GradientPair> &gpair,
                             const RowSetCollection::Elem row_indices, const GHistIndexMatrix &gmat,
                             GHistRow hist) const {
  const size_t nrows = row_indices.Size();
  const size_t no_prefetch_size = Prefetch::NoPrefetchSize(nrows);

  // if need to work with all rows from bin-matrix (e.g. root node)
  const bool contiguousBlock =
      (row_indices.begin[nrows - 1] - row_indices.begin[0]) == (nrows - 1);

  if (contiguousBlock) {
    // contiguous memory access, built-in HW prefetching is enough
    BuildHistDispatch<false, any_missing>(gpair, row_indices,
                                                        gmat, hist);
  } else {
    const RowSetCollection::Elem span1(row_indices.begin,
                                       row_indices.end - no_prefetch_size);
    const RowSetCollection::Elem span2(row_indices.end - no_prefetch_size,
                                       row_indices.end);

    BuildHistDispatch<true, any_missing>(gpair, span1, gmat, hist);
    // no prefetching to avoid loading extra memory
    BuildHistDispatch<false, any_missing>(gpair, span2, gmat, hist);
  }
}

template void GHistBuilder::BuildHist<true>(const std::vector<GradientPair> &gpair,
                                            const RowSetCollection::Elem row_indices,
                                            const GHistIndexMatrix &gmat, GHistRow hist) const;

template void GHistBuilder::BuildHist<false>(const std::vector<GradientPair> &gpair,
                                             const RowSetCollection::Elem row_indices,
                                             const GHistIndexMatrix &gmat, GHistRow hist) const;
}  // namespace common
}  // namespace xgboost
