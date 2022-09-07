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

template <bool _column_sampling, bool _read_by_column>
struct GHistBuildingManager {
  GHistBuildingManager(std::shared_ptr<common::ColumnSampler> column_sampler, int depth) {
    if (column_sampling && read_by_column) {
      const size_t n_sampled_features = column_sampler->GetFeatureSet(depth)->Size();
      fids.resize(n_sampled_features);
      for (size_t i = 0; i < n_sampled_features; ++i) {
        fids[i] = column_sampler->GetFeatureSet(depth)->ConstHostVector()[i];
      }
    }
  }

  std::vector<int> fids;
  constexpr static bool column_sampling = _column_sampling;
  constexpr static bool read_by_column = _read_by_column;
};

template <bool do_prefetch, typename BinIdxType, bool first_page, bool any_missing = true>
void RowsWiseBuildHistKernel(const std::vector<GradientPair> &gpair,
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

template <typename BinIdxType, bool first_page, bool any_missing, class GHistBuildingManager>
void ColsWiseBuildHistKernel(const std::vector<GradientPair> &gpair,
                            const RowSetCollection::Elem row_indices, const GHistIndexMatrix &gmat,
                            GHistRow hist, const GHistBuildingManager& hbm) {
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

  const size_t n_features = gmat.cut.Ptrs().size() - 1;
  const size_t n_columns = GHistBuildingManager::column_sampling ? hbm.fids.size() : n_features;
  auto hist_data = reinterpret_cast<double *>(hist.data());
  const uint32_t two{2};  // Each element from 'gpair' and 'hist' contains
                          // 2 FP values: gradient and hessian.
                          // So we need to multiply each row-index/bin-index by 2
                          // to work with gradient pairs as a singe row FP array
  for (size_t cid = 0; cid < n_columns; ++cid) {
    const size_t local_cid = GHistBuildingManager::column_sampling ? hbm.fids[cid] : cid;
    for (size_t i = 0; i < size; ++i) {
      const size_t row_id = rid[i];
      const size_t icol_start =
          any_missing ? get_row_ptr(row_id) : get_rid(row_id) * n_features;

      const BinIdxType *gr_index_local = gradient_index + icol_start;
      const uint32_t idx_bin = two * (static_cast<uint32_t>(gr_index_local[local_cid]) +
                                      (any_missing ? 0 : offsets[local_cid]));
      auto hist_local = hist_data + idx_bin;

      const size_t idx_gh = two * row_id;
      // The trick with pgh_t buffer helps the compiler to generate faster binary.
      const float pgh_t[] = {pgh[idx_gh], pgh[idx_gh + 1]};
      *(hist_local)     += pgh_t[0];
      *(hist_local + 1) += pgh_t[1];
    }
  }
}

template <bool do_prefetch, typename BinIdxType, bool first_page,
          bool any_missing, class GHistBuildingManager>
void BuildHistKernel(const std::vector<GradientPair> &gpair,
                     const RowSetCollection::Elem row_indices, const GHistIndexMatrix &gmat,
                     GHistRow hist, const GHistBuildingManager& hbm) {
  if (GHistBuildingManager::read_by_column) {
    ColsWiseBuildHistKernel<BinIdxType, first_page, any_missing>
                           (gpair, row_indices, gmat, hist, hbm);
  } else {
    RowsWiseBuildHistKernel<do_prefetch, BinIdxType, first_page, any_missing>
                           (gpair, row_indices, gmat, hist);
  }
}

template <bool do_prefetch, bool first_page, bool any_missing, class GHistBuildingManager>
void BuildHistDispatch(const std::vector<GradientPair> &gpair,
                       const RowSetCollection::Elem row_indices, const GHistIndexMatrix &gmat,
                       GHistRow hist, const GHistBuildingManager& hbm) {
  switch (gmat.index.GetBinTypeSize()) {
  case kUint8BinsTypeSize:
    BuildHistKernel<do_prefetch, uint8_t, first_page, any_missing>
                   (gpair, row_indices, gmat, hist, hbm);
    break;
  case kUint16BinsTypeSize:
    BuildHistKernel<do_prefetch, uint16_t, first_page, any_missing>
                   (gpair, row_indices, gmat, hist, hbm);
    break;
  case kUint32BinsTypeSize:
    BuildHistKernel<do_prefetch, uint32_t, first_page, any_missing>
                   (gpair, row_indices, gmat, hist, hbm);
    break;
  default:
    CHECK(false);  // no default behavior
  }
}

template <bool do_prefetch, bool any_missing, class GHistBuildingManager>
void BuildHistDispatch(const std::vector<GradientPair> &gpair,
                       const RowSetCollection::Elem row_indices, const GHistIndexMatrix &gmat,
                       GHistRow hist, const GHistBuildingManager& hbm) {
  auto first_page = gmat.base_rowid == 0;
  if (first_page) {
    BuildHistDispatch<do_prefetch, true, any_missing>(gpair, row_indices, gmat, hist, hbm);
  } else {
    BuildHistDispatch<do_prefetch, false, any_missing>(gpair, row_indices, gmat, hist, hbm);
  }
}

template <bool any_missing, class GHistBuildingManager>
void BuildHistDispatch(const std::vector<GradientPair> &gpair,
                       const RowSetCollection::Elem row_indices, const GHistIndexMatrix &gmat,
                       GHistRow hist, const GHistBuildingManager& hbm) {
  const size_t nrows = row_indices.Size();
  const size_t no_prefetch_size = Prefetch::NoPrefetchSize(nrows);
  // if need to work with all rows from bin-matrix (e.g. root node)
  const bool contiguousBlock =
      (row_indices.begin[nrows - 1] - row_indices.begin[0]) == (nrows - 1);

  if (contiguousBlock) {
    // contiguous memory access, built-in HW prefetching is enough
    BuildHistDispatch<false, any_missing>(gpair, row_indices, gmat, hist, hbm);
  } else {
    const RowSetCollection::Elem span1(row_indices.begin,
                                       row_indices.end - no_prefetch_size);
    const RowSetCollection::Elem span2(row_indices.end - no_prefetch_size,
                                       row_indices.end);

    BuildHistDispatch<true, any_missing>(gpair, span1, gmat, hist, hbm);
    // no prefetching to avoid loading extra memory
    BuildHistDispatch<false, any_missing>(gpair, span2, gmat, hist, hbm);
  }
}

template <bool any_missing>
void GHistBuilder::BuildHist(const std::vector<GradientPair> &gpair,
                             const RowSetCollection::Elem row_indices,
                             const GHistIndexMatrix &gmat,
                             GHistRow hist, std::shared_ptr<ColumnSampler> column_sampler,
                             int depth, bool column_sampling) const {
  constexpr double adhoc_l2_size = 1024 * 1024 * 0.8;
  const bool hist_fit_to_l2 = adhoc_l2_size > 2*sizeof(float)*gmat.cut.Ptrs().back();
  bool read_by_column = column_sampling ? true : !hist_fit_to_l2 && !any_missing;

  if (read_by_column) {
    if (column_sampling) {
      GHistBuildingManager<true, true> hbm(column_sampler, depth);
      BuildHistDispatch<any_missing>(gpair, row_indices, gmat, hist, hbm);
    } else {
      GHistBuildingManager<false, true> hbm(column_sampler, depth);
      BuildHistDispatch<any_missing>(gpair, row_indices, gmat, hist, hbm);
    }
  } else {
    // column_sampling doesn't matter in this case
    GHistBuildingManager<false, false> hbm(column_sampler, depth);
    BuildHistDispatch<any_missing>(gpair, row_indices, gmat, hist, hbm);
  }
}

template void GHistBuilder::BuildHist<true>(const std::vector<GradientPair> &gpair,
                                            const RowSetCollection::Elem row_indices,
                                            const GHistIndexMatrix &gmat, GHistRow hist,
                                            std::shared_ptr<ColumnSampler> column_sampler,
                                            int depth, bool column_sampling) const;

template void GHistBuilder::BuildHist<false>(const std::vector<GradientPair> &gpair,
                                            const RowSetCollection::Elem row_indices,
                                            const GHistIndexMatrix &gmat, GHistRow hist,
                                            std::shared_ptr<ColumnSampler> column_sampler,
                                            int depth, bool column_sampling) const;
}  // namespace common
}  // namespace xgboost
