/*!
 * Copyright 2017-2021 by Contributors
 * \file hist_builder.h
 */
#ifndef XGBOOST_COMMON_HIST_BUILDER_H_
#define XGBOOST_COMMON_HIST_BUILDER_H_

#include <algorithm>
#include <vector>
#include "hist_util.h"
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

template<typename FPType, bool do_prefetch,
         typename BinIdxType, bool is_root,
         bool any_missing>
void BuildHistKernel(const std::vector<GradientPair>& gpair,
                     const uint32_t* rows,
                     const size_t row_begin,
                     const size_t row_end,
                     const GHistIndexMatrix& gmat,
                     const BinIdxType* numa_data,
                     uint16_t* nodes_ids,
                     std::vector<std::vector<FPType>>* p_hists,
                     const uint16_t* mapping_ids, uint32_t base_rowid = 0) {
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const BinIdxType* gradient_index = numa_data;
  const size_t* row_ptr =  gmat.row_ptr.data();
  const uint32_t* offsets = gmat.index.Offset();
  const size_t n_features = row_ptr[1] - row_ptr[0];
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array
  std::vector<std::vector<FPType>>& hists = *p_hists;
  for (size_t i = row_begin; i < row_end; ++i) {
    const size_t ri = is_root ? i : rows[i];
    const size_t icol_start = any_missing ? row_ptr[ri] : ri * n_features;
    const size_t icol_end =  any_missing ? row_ptr[ri+1] : icol_start + n_features;
    const size_t row_size = icol_end - icol_start;
    const size_t idx_gh = two * (ri + base_rowid);
    const uint32_t nid = is_root ? 0 : mapping_ids[nodes_ids[ri]];
    if (do_prefetch) {
      const size_t icol_start_prefetch = any_missing ? row_ptr[rows[i+Prefetch::kPrefetchOffset]] :
                                       rows[i + Prefetch::kPrefetchOffset] * n_features;
      const size_t icol_end_prefetch = any_missing ?  row_ptr[rows[i+Prefetch::kPrefetchOffset]+1] :
                                      icol_start_prefetch + n_features;

      PREFETCH_READ_T0(pgh + two * rows[i + Prefetch::kPrefetchOffset]);
      for (size_t j = icol_start_prefetch; j < icol_end_prefetch;
        j+=Prefetch::GetPrefetchStep<uint32_t>()) {
        PREFETCH_READ_T0(gradient_index + j);
      }
    } else if (is_root) {
      nodes_ids[ri] = 0;
    }

    const BinIdxType* gr_index_local = gradient_index + icol_start;
    FPType* hist_data = hists[nid].data();

    for (size_t j = 0; j < row_size; ++j) {
      const uint32_t idx_bin = two * (static_cast<uint32_t>(gr_index_local[j]) + (
                                      any_missing ? 0 : offsets[j]));
      hist_data[idx_bin]   += pgh[idx_gh];
      hist_data[idx_bin+1] += pgh[idx_gh+1];
    }
  }
}

/*!
 * \brief builder for histograms of gradient statistics
 */
// template<typename GradientSumT>
class GHistBuilder {
 public:
  // using GHistRowT = GHistRow<GradientSumT>;
  GHistBuilder() = default;

  // construct a histogram via histogram aggregation
  template <typename BinIdxType, bool any_missing, bool is_root>
  void BuildHist(const std::vector<GradientPair>& gpair,
                 const uint32_t* rows,
                 const size_t row_begin,
                 const size_t row_end,
                 const GHistIndexMatrix& gmat,
                 const BinIdxType* numa_data,
                 uint16_t* nodes_ids,
                 std::vector<std::vector<double>>* p_hists,
                 const uint16_t* mapping_ids, uint32_t base_rowid = 0) {
    const size_t nrows = row_end - row_begin;
    const size_t no_prefetch_size = Prefetch::NoPrefetchSize(nrows);
    if (is_root) {
        // contiguous memory access, built-in HW prefetching is enough
        BuildHistKernel<double, false, BinIdxType, true, any_missing>(
          gpair, rows, row_begin, row_end, gmat, numa_data, nodes_ids, p_hists,
          mapping_ids, base_rowid);
    } else {
        BuildHistKernel<double, true, BinIdxType, false, any_missing>(
          gpair, rows, row_begin, row_end - no_prefetch_size,
          gmat, numa_data, nodes_ids, p_hists, mapping_ids, base_rowid);
        BuildHistKernel<double, false, BinIdxType, false, any_missing>(
          gpair, rows,  row_end - no_prefetch_size, row_end,
          gmat, numa_data, nodes_ids, p_hists, mapping_ids, base_rowid);
    }
  }
};

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_HIST_BUILDER_H_
