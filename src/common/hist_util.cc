/**
 * Copyright 2017-2025, XGBoost Contributors
 * \file hist_util.cc
 */
#include "hist_util.h"

#include <dmlc/timer.h>

#include <vector>

#include "../data/adapter.h"         // for SparsePageAdapterBatch
#include "../data/gradient_index.h"  // for GHistIndexMatrix
#include "cache_manager.h"           // for CacheManager
#include "io.h"                      // for AlignedResourceReadStream, AlignedFileWriteStream
#include "quantile.h"
#include "xgboost/base.h"
#include "xgboost/context.h"  // for Context
#include "xgboost/data.h"     // for SparsePage, SortedCSCPage

#if defined(XGBOOST_MM_PREFETCH_PRESENT)
#include <xmmintrin.h>
#define PREFETCH_READ_T0(addr) _mm_prefetch(reinterpret_cast<const char *>(addr), _MM_HINT_T0)
#elif defined(XGBOOST_BUILTIN_PREFETCH_PRESENT)
#define PREFETCH_READ_T0(addr) __builtin_prefetch(reinterpret_cast<const char *>(addr), 0, 3)
#else  // no SW pre-fetching available; PREFETCH_READ_T0 is no-op
#define PREFETCH_READ_T0(addr) \
  do {                         \
  } while (0)
#endif  // defined(XGBOOST_MM_PREFETCH_PRESENT)

namespace xgboost::common {
HistogramCuts::HistogramCuts(bst_feature_t n_features)
    : cut_ptrs_(static_cast<std::size_t>(n_features) + 1, 0) {}

void HistogramCuts::Save(common::AlignedFileWriteStream *fo) const {
  auto const &ptrs = this->Ptrs();
  CHECK_LE(Span{ptrs}.size_bytes(), WriteVec(fo, ptrs));
  auto const &vals = this->Values();
  CHECK_LE(Span{vals}.size_bytes(), WriteVec(fo, vals));
  CHECK_GE(fo->Write(has_categorical_), sizeof(has_categorical_));
  CHECK_GE(fo->Write(max_cat_), sizeof(max_cat_));
}

[[nodiscard]] HistogramCuts *HistogramCuts::Load(common::AlignedResourceReadStream *fi) {
  auto p_cuts = new HistogramCuts{0};
  CHECK(ReadVec(fi, &p_cuts->cut_ptrs_.HostVector()));
  CHECK(ReadVec(fi, &p_cuts->cut_values_.HostVector()));
  CHECK(fi->Read(&p_cuts->has_categorical_));
  CHECK(fi->Read(&p_cuts->max_cat_));
  return p_cuts;
}

HistogramCuts SketchOnDMatrix(Context const *ctx, DMatrix *m, bst_bin_t max_bins, bool use_sorted,
                              Span<float const> hessian) {
  auto const &info = m->Info();
  auto n_threads = ctx->Threads();
  std::vector<bst_idx_t> reduced(info.num_col_, 0);
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
    HostSketchContainer container(ctx, max_bins, m->Info().feature_types.ConstHostSpan(), reduced,
                                  HostSketchContainer::UseGroup(info));
    for (auto const &page : m->GetBatches<SparsePage>()) {
      container.PushRowPage(page, info, hessian);
    }
    return container.MakeCuts(ctx, m->Info());
  } else {
    HostSketchContainer container{ctx, max_bins, m->Info().feature_types.ConstHostSpan(), reduced,
                                  HostSketchContainer::UseGroup(info)};
    for (auto const &page : m->GetBatches<SortedCSCPage>(ctx)) {
      container.PushColPage(page, info, hessian);
    }
    return container.MakeCuts(ctx, m->Info());
  }
}

/*!
 * \brief Increment hist as dst += add in range [begin, end)
 */
void IncrementHist(GHistRow dst, ConstGHistRow add, std::size_t begin, std::size_t end) {
  double *pdst = reinterpret_cast<double *>(dst.data());
  const double *padd = reinterpret_cast<const double *>(add.data());

  for (std::size_t i = 2 * begin; i < 2 * end; ++i) {
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
  double *pdst = reinterpret_cast<double *>(dst.data());
  const double *psrc1 = reinterpret_cast<const double *>(src1.data());
  const double *psrc2 = reinterpret_cast<const double *>(src2.data());

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
      kPrefetchOffset + kCacheLineSize / sizeof(decltype(GHistIndexMatrix::row_ptr)::value_type);

 public:
  static size_t NoPrefetchSize(size_t rows) { return std::min(rows, kNoPrefetchSize); }

  template <typename T>
  static constexpr size_t GetPrefetchStep() {
    return Prefetch::kCacheLineSize / sizeof(T);
  }
};

constexpr size_t Prefetch::kNoPrefetchSize;

struct RuntimeFlags {
  const bool first_page;
  const bool read_by_column;
  const BinTypeSize bin_type_size;
};

template <bool _any_missing, bool _first_page = false, bool _read_by_column = false,
          typename BinIdxTypeName = uint8_t>
class GHistBuildingManager {
 public:
  constexpr static bool kAnyMissing = _any_missing;
  constexpr static bool kFirstPage = _first_page;
  constexpr static bool kReadByColumn = _read_by_column;
  using BinIdxType = BinIdxTypeName;

 private:
  template <bool new_first_page>
  struct SetFirstPage {
    using Type = GHistBuildingManager<kAnyMissing, new_first_page, kReadByColumn, BinIdxType>;
  };

  template <bool new_read_by_column>
  struct SetReadByColumn {
    using Type = GHistBuildingManager<kAnyMissing, kFirstPage, new_read_by_column, BinIdxType>;
  };

  template <typename NewBinIdxType>
  struct SetBinIdxType {
    using Type = GHistBuildingManager<kAnyMissing, kFirstPage, kReadByColumn, NewBinIdxType>;
  };

  using Type = GHistBuildingManager<kAnyMissing, kFirstPage, kReadByColumn, BinIdxType>;

 public:
  /* Entry point to dispatcher
   * This function check matching run time flags to compile time flags.
   * In case of difference, it creates a Manager with different template parameters
   *  and forward the call there.
   */
  template <typename Fn>
  static void DispatchAndExecute(const RuntimeFlags &flags, Fn &&fn) {
    if (flags.first_page != kFirstPage) {
      SetFirstPage<true>::Type::DispatchAndExecute(flags, std::forward<Fn>(fn));
    } else if (flags.read_by_column != kReadByColumn) {
      SetReadByColumn<true>::Type::DispatchAndExecute(flags, std::forward<Fn>(fn));
    } else if (flags.bin_type_size != sizeof(BinIdxType)) {
      DispatchBinType(flags.bin_type_size, [&](auto t) {
        using NewBinIdxType = decltype(t);
        SetBinIdxType<NewBinIdxType>::Type::DispatchAndExecute(flags, std::forward<Fn>(fn));
      });
    } else {
      fn(Type());
    }
  }
};

// Build the sparse histogram one column block at a time, accumulating into a
// thread-local buffer that fits in cache. Returns false (fall through to the
// direct-scatter loop) when the histogram fits in cache or per-leaf scatter
// work does not amortize the per-block overhead.
template <class BuildingManager>
bool BuildSparseHistByBlocks(Span<GradientPair const> gpair, Span<bst_idx_t const> row_indices,
                             const GHistIndexMatrix &gmat, GHistRow hist) {
  constexpr bool kFirstPage = BuildingManager::kFirstPage;
  using BinIdxType = typename BuildingManager::BinIdxType;

  constexpr size_t kColBlockSize = 32;

  auto const &cut_ptrs = gmat.cut.Ptrs();
  const size_t n_features = cut_ptrs.size() - 1;
  const size_t total_hist_bins = cut_ptrs.back();

  // Check 1: histogram size vs. per-thread cache budget. Detected once per
  // process via CacheManager (CPUID + sane defaults).
  static const CacheManager kCacheManager{};
  const size_t hist_bytes = 2 * sizeof(double) * total_hist_bins;
  const double l3_per_thread =
      static_cast<double>(kCacheManager.L3Size()) / std::max(1, omp_get_max_threads());
  const double usable_cache = 0.8 * (kCacheManager.L2Size() + l3_per_thread);
  if (static_cast<double>(hist_bytes) <= usable_cache) {
    return false;
  }

  const size_t size = row_indices.size();
  bst_idx_t const *rid = row_indices.data();
  auto const &row_ptr = gmat.row_ptr.data();
  auto base_rowid = gmat.base_rowid;
  auto get_row_ptr = [&](bst_idx_t ridx) {
    return kFirstPage ? row_ptr[ridx] : row_ptr[ridx - base_rowid];
  };

  // Check 2: per-node nnz must amortize the tile path's fixed per-block cost.
  size_t nnz = 0;
  for (size_t i = 0; i < size; ++i) {
    nnz += get_row_ptr(rid[i] + 1) - get_row_ptr(rid[i]);
  }
  const size_t n_blocks = (n_features + kColBlockSize - 1) / kColBlockSize;
  const size_t tile_overhead = size * n_blocks + total_hist_bins;
  if (nnz <= 2 * tile_overhead) {
    return false;
  }

  size_t max_block_bins = 0;
  for (size_t jj = 0; jj < n_features; jj += kColBlockSize) {
    size_t jj_end = std::min(jj + kColBlockSize, n_features);
    size_t bins = cut_ptrs[jj_end] - cut_ptrs[jj];
    max_block_bins = std::max(max_block_bins, bins);
  }

  std::vector<double> tl_buf(max_block_bins * 2);
  std::vector<size_t> tl_row_starts(size);
  std::vector<size_t> tl_row_sizes(size);
  std::vector<uint32_t> tl_cursors(size, 0);

  for (size_t i = 0; i < size; ++i) {
    tl_row_starts[i] = get_row_ptr(rid[i]);
    tl_row_sizes[i] = get_row_ptr(rid[i] + 1) - tl_row_starts[i];
  }

  auto const *p_gpair = reinterpret_cast<const float *>(gpair.data());
  const BinIdxType *gradient_index = gmat.index.data<BinIdxType>();
  auto hist_data = reinterpret_cast<double *>(hist.data());
  const uint32_t two{2};

  for (size_t cid_begin = 0; cid_begin < n_features; cid_begin += kColBlockSize) {
    const size_t cid_end = std::min(cid_begin + kColBlockSize, n_features);
    const uint32_t bin_lo = cut_ptrs[cid_begin];
    const uint32_t bin_hi = cut_ptrs[cid_end];
    const size_t block_n_bins = bin_hi - bin_lo;

    double *local_hist = tl_buf.data();
    std::fill_n(local_hist, block_n_bins * 2, 0.0);

    for (size_t i = 0; i < size; ++i) {
      const BinIdxType *gr_row = gradient_index + tl_row_starts[i];
      const size_t row_size = tl_row_sizes[i];
      const size_t idx_gh = two * rid[i];
      const float pgh_t[] = {p_gpair[idx_gh], p_gpair[idx_gh + 1]};

      size_t j = tl_cursors[i];
      while (j < row_size && static_cast<uint32_t>(gr_row[j]) < bin_lo) ++j;
      while (j < row_size && static_cast<uint32_t>(gr_row[j]) < bin_hi) {
        const uint32_t gidx = static_cast<uint32_t>(gr_row[j]);
        const uint32_t local_bin = two * (gidx - bin_lo);
        *(local_hist + local_bin) += pgh_t[0];
        *(local_hist + local_bin + 1) += pgh_t[1];
        ++j;
      }
      tl_cursors[i] = j;
    }

    double *dst = hist_data + two * bin_lo;
    for (size_t j = 0; j < block_n_bins * 2; ++j) {
      dst[j] += local_hist[j];
    }
  }
  return true;
}

template <bool do_prefetch, class BuildingManager>
void RowsWiseBuildHistKernel(Span<GradientPair const> gpair, Span<bst_idx_t const> row_indices,
                             const GHistIndexMatrix &gmat, GHistRow hist) {
  constexpr bool kAnyMissing = BuildingManager::kAnyMissing;
  constexpr bool kFirstPage = BuildingManager::kFirstPage;
  using BinIdxType = typename BuildingManager::BinIdxType;

  if constexpr (kAnyMissing) {
    if (BuildSparseHistByBlocks<BuildingManager>(gpair, row_indices, gmat, hist)) {
      return;
    }
  }

  const size_t size = row_indices.size();
  bst_idx_t const *rid = row_indices.data();
  auto const *p_gpair = reinterpret_cast<const float *>(gpair.data());
  const BinIdxType *gradient_index = gmat.index.data<BinIdxType>();

  auto const &row_ptr = gmat.row_ptr.data();
  auto base_rowid = gmat.base_rowid;
  std::uint32_t const *offsets = gmat.index.Offset();
  // There's no feature-based compression if missing value is present.
  if (kAnyMissing) {
    CHECK(!offsets);
  } else {
    CHECK(offsets);
  }

  auto get_row_ptr = [&](bst_idx_t ridx) {
    return kFirstPage ? row_ptr[ridx] : row_ptr[ridx - base_rowid];
  };
  auto get_rid = [&](bst_idx_t ridx) {
    return kFirstPage ? ridx : (ridx - base_rowid);
  };

  CHECK_NE(row_indices.size(), 0);
  const size_t n_features =
      get_row_ptr(row_indices.data()[0] + 1) - get_row_ptr(row_indices.data()[0]);
  auto hist_data = reinterpret_cast<double *>(hist.data());
  const uint32_t two{2};  // Each element from 'gpair' and 'hist' contains
                          // 2 FP values: gradient and hessian.
                          // So we need to multiply each row-index/bin-index by 2
                          // to work with gradient pairs as a singe row FP array

  for (std::size_t i = 0; i < size; ++i) {
    const size_t icol_start = kAnyMissing ? get_row_ptr(rid[i]) : get_rid(rid[i]) * n_features;
    const size_t icol_end = kAnyMissing ? get_row_ptr(rid[i] + 1) : icol_start + n_features;

    const size_t row_size = icol_end - icol_start;
    const size_t idx_gh = two * rid[i];

    if (do_prefetch) {
      const size_t icol_start_prefetch =
          kAnyMissing ? get_row_ptr(rid[i + Prefetch::kPrefetchOffset])
                      : get_rid(rid[i + Prefetch::kPrefetchOffset]) * n_features;
      const size_t icol_end_prefetch = kAnyMissing
                                           ? get_row_ptr(rid[i + Prefetch::kPrefetchOffset] + 1)
                                           : icol_start_prefetch + n_features;

      PREFETCH_READ_T0(p_gpair + two * rid[i + Prefetch::kPrefetchOffset]);
      for (size_t j = icol_start_prefetch; j < icol_end_prefetch;
           j += Prefetch::GetPrefetchStep<uint32_t>()) {
        PREFETCH_READ_T0(gradient_index + j);
      }
    }
    const BinIdxType *gr_index_local = gradient_index + icol_start;

    // The trick with pgh_t buffer helps the compiler to generate faster binary.
    const float pgh_t[] = {p_gpair[idx_gh], p_gpair[idx_gh + 1]};
    for (size_t j = 0; j < row_size; ++j) {
      const uint32_t idx_bin =
          two * (static_cast<uint32_t>(gr_index_local[j]) + (kAnyMissing ? 0 : offsets[j]));
      auto hist_local = hist_data + idx_bin;
      *(hist_local) += pgh_t[0];
      *(hist_local + 1) += pgh_t[1];
    }
  }
}

template <class BuildingManager>
void ColsWiseBuildHistKernel(Span<GradientPair const> gpair, Span<bst_idx_t const> row_indices,
                             const GHistIndexMatrix &gmat, GHistRow hist) {
  constexpr bool kAnyMissing = BuildingManager::kAnyMissing;
  constexpr bool kFirstPage = BuildingManager::kFirstPage;
  using BinIdxType = typename BuildingManager::BinIdxType;
  const size_t size = row_indices.size();
  bst_idx_t const *rid = row_indices.data();
  auto const *pgh = reinterpret_cast<const float *>(gpair.data());
  const BinIdxType *gradient_index = gmat.index.data<BinIdxType>();

  auto const &row_ptr = gmat.row_ptr.data();
  auto base_rowid = gmat.base_rowid;
  const uint32_t *offsets = gmat.index.Offset();
  auto get_row_ptr = [&](bst_idx_t ridx) {
    return kFirstPage ? row_ptr[ridx] : row_ptr[ridx - base_rowid];
  };
  auto get_rid = [&](bst_idx_t ridx) {
    return kFirstPage ? ridx : (ridx - base_rowid);
  };

  const size_t n_features = gmat.cut.Ptrs().size() - 1;
  const size_t n_columns = n_features;
  auto hist_data = reinterpret_cast<double *>(hist.data());
  const uint32_t two{2};  // Each element from 'gpair' and 'hist' contains
                          // 2 FP values: gradient and hessian.
                          // So we need to multiply each row-index/bin-index by 2
                          // to work with gradient pairs as a singe row FP array

  // Column-block tiling with local buffer:
  // Process kColBlockSize columns at a time, accumulating into a small local
  // buffer that fits in L1/L2. This amortizes gpair loads across multiple
  // columns per row visit and localizes histogram writes.
  auto const &cut_ptrs = gmat.cut.Ptrs();
  constexpr size_t kColBlockSize = 32;

  // Pre-allocate thread-local buffer sized for the largest column block
  size_t max_block_bins = 0;
  for (size_t jj = 0; jj < n_columns; jj += kColBlockSize) {
    size_t jj_end = std::min(jj + kColBlockSize, n_columns);
    size_t bins = cut_ptrs[jj_end] - cut_ptrs[jj];
    max_block_bins = std::max(max_block_bins, bins);
  }

  std::vector<double> tl_cols_buf(max_block_bins * 2);

  for (size_t cid_begin = 0; cid_begin < n_columns; cid_begin += kColBlockSize) {
    const size_t cid_end = std::min(cid_begin + kColBlockSize, n_columns);
    const size_t chunk_bin_begin = cut_ptrs[cid_begin];
    const size_t chunk_bin_end = cut_ptrs[cid_end];
    const size_t chunk_n_bins = chunk_bin_end - chunk_bin_begin;

    double *local_hist = tl_cols_buf.data();
    std::fill_n(local_hist, chunk_n_bins * 2, 0.0);

    for (size_t i = 0; i < size; ++i) {
      const size_t row_id = rid[i];
      const size_t icol_start = kAnyMissing ? get_row_ptr(row_id) : get_rid(row_id) * n_features;
      const size_t icol_end = kAnyMissing ? get_row_ptr(rid[i] + 1) : icol_start + n_features;
      const size_t row_size = icol_end - icol_start;

      const size_t idx_gh = two * row_id;
      const float pgh_t[] = {pgh[idx_gh], pgh[idx_gh + 1]};
      const BinIdxType *gr_index_local = gradient_index + icol_start;

      for (size_t cid = cid_begin; cid < cid_end; ++cid) {
        if (cid < row_size) {
          const uint32_t offset = kAnyMissing ? 0 : offsets[cid];
          const uint32_t global_bin = static_cast<uint32_t>(gr_index_local[cid]) + offset;
          const uint32_t local_bin = two * (global_bin - static_cast<uint32_t>(chunk_bin_begin));
          *(local_hist + local_bin) += pgh_t[0];
          *(local_hist + local_bin + 1) += pgh_t[1];
        }
      }
    }

    // Flush local buffer to full histogram
    double *dst = hist_data + two * chunk_bin_begin;
    for (size_t j = 0; j < chunk_n_bins * 2; ++j) {
      dst[j] += local_hist[j];
    }
  }
}

template <class BuildingManager>
void BuildHistDispatch(Span<GradientPair const> gpair, Span<bst_idx_t const> row_indices,
                       const GHistIndexMatrix &gmat, GHistRow hist) {
  if (BuildingManager::kReadByColumn) {
    ColsWiseBuildHistKernel<BuildingManager>(gpair, row_indices, gmat, hist);
  } else {
    if (row_indices.empty()) {
      return;
    }

    const size_t nrows = row_indices.size();
    const size_t no_prefetch_size = Prefetch::NoPrefetchSize(nrows);
    // if need to work with all rows from bin-matrix (e.g. root node)
    const bool contiguousBlock =
        (row_indices.begin()[nrows - 1] - row_indices.begin()[0]) == (nrows - 1);

    if (contiguousBlock) {
      // contiguous memory access, built-in HW prefetching is enough
      RowsWiseBuildHistKernel<false, BuildingManager>(gpair, row_indices, gmat, hist);
    } else {
      auto span1 = row_indices.subspan(0, row_indices.size() - no_prefetch_size);
      if (!span1.empty()) {
        RowsWiseBuildHistKernel<true, BuildingManager>(gpair, span1, gmat, hist);
      }
      // no prefetching to avoid loading extra memory
      auto span2 = row_indices.subspan(row_indices.size() - no_prefetch_size);
      if (!span2.empty()) {
        RowsWiseBuildHistKernel<false, BuildingManager>(gpair, span2, gmat, hist);
      }
    }
  }
}

template <bool any_missing>
void BuildHist(Span<GradientPair const> gpair, Span<bst_idx_t const> row_indices,
               const GHistIndexMatrix &gmat, GHistRow hist, bool read_by_column) {
  bool first_page = gmat.base_rowid == 0;
  auto bin_type_size = gmat.index.GetBinTypeSize();

  GHistBuildingManager<any_missing>::DispatchAndExecute(
      {first_page, read_by_column, bin_type_size}, [&](auto t) {
        using BuildingManager = decltype(t);
        BuildHistDispatch<BuildingManager>(gpair, row_indices, gmat, hist);
      });
}

template void BuildHist<true>(Span<GradientPair const> gpair, Span<bst_idx_t const> row_indices,
                              const GHistIndexMatrix &gmat, GHistRow hist, bool read_by_column);

template void BuildHist<false>(Span<GradientPair const> gpair, Span<bst_idx_t const> row_indices,
                               const GHistIndexMatrix &gmat, GHistRow hist, bool read_by_column);
}  // namespace xgboost::common
