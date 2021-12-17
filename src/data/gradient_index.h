/*!
 * Copyright 2017-2021 by Contributors
 * \brief Data type for fast histogram aggregation.
 */
#ifndef XGBOOST_DATA_GRADIENT_INDEX_H_
#define XGBOOST_DATA_GRADIENT_INDEX_H_
#include <vector>
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "../common/categorical.h"
#include "../common/hist_util.h"
#include "../common/threading_utils.h"

namespace xgboost {
/*!
 * \brief preprocessed global index matrix, in CSR format
 *
 *  Transform floating values to integer index in histogram This is a global histogram
 *  index for CPU histogram.  On GPU ellpack page is used.
 */
class GHistIndexMatrix {
  void PushBatch(SparsePage const &batch, common::Span<FeatureType const> ft,
                 size_t rbegin, size_t prev_sum, uint32_t nbins,
                 int32_t n_threads);

 public:
  /*! \brief row pointer to rows by element position */
  std::vector<size_t> row_ptr;
  /*! \brief The index data */
  common::Index index;
  /*! \brief hit count of each index */
  std::vector<size_t> hit_count;
  /*! \brief The corresponding cuts */
  common::HistogramCuts cut;
  DMatrix* p_fmat;
  size_t max_num_bins;
  size_t base_rowid{0};

  GHistIndexMatrix() = default;
  GHistIndexMatrix(DMatrix* x, int32_t max_bin, bool sorted_sketch, common::Span<float> hess = {}) {
    this->Init(x, max_bin, sorted_sketch, hess);
  }
  // Create a global histogram matrix, given cut
  void Init(DMatrix* p_fmat, int max_num_bins, bool sorted_sketch, common::Span<float> hess);
  void Init(SparsePage const& page, common::Span<FeatureType const> ft,
            common::HistogramCuts const& cuts, int32_t max_bins_per_feat, bool is_dense,
            int32_t n_threads);

  // specific method for sparse data as no possibility to reduce allocated memory
  template <typename BinIdxType, typename GetOffset>
  void SetIndexData(common::Span<BinIdxType> index_data_span,
                    common::Span<FeatureType const> ft,
                    size_t batch_threads, const SparsePage &batch,
                    size_t rbegin, size_t nbins, GetOffset get_offset) {
    const xgboost::Entry *data_ptr = batch.data.HostVector().data();
    const std::vector<bst_row_t> &offset_vec = batch.offset.HostVector();
    const size_t batch_size = batch.Size();
    CHECK_LT(batch_size, offset_vec.size());
    BinIdxType* index_data = index_data_span.data();
    auto const& ptrs = cut.Ptrs();
    auto const& values = cut.Values();
    common::ParallelFor(batch_size, batch_threads, [&](omp_ulong i) {
      const int tid = omp_get_thread_num();
      size_t ibegin = row_ptr[rbegin + i];
      size_t iend = row_ptr[rbegin + i + 1];
      const size_t size = offset_vec[i + 1] - offset_vec[i];
      SparsePage::Inst inst = {data_ptr + offset_vec[i], size};
      CHECK_EQ(ibegin + inst.size(), iend);
      for (bst_uint j = 0; j < inst.size(); ++j) {
        auto e = inst[j];
        if (common::IsCat(ft, e.index)) {
          auto bin_idx = cut.SearchCatBin(e);
          index_data[ibegin + j] = get_offset(bin_idx, j);
          ++hit_count_tloc_[tid * nbins + bin_idx];
        } else {
          uint32_t idx = cut.SearchBin(inst[j].fvalue, inst[j].index, ptrs, values);
          index_data[ibegin + j] = get_offset(idx, j);
          ++hit_count_tloc_[tid * nbins + idx];
        }
      }
    });
  }

  void ResizeIndex(const size_t n_index,
                   const bool isDense);

  inline void GetFeatureCounts(size_t* counts) const {
    auto nfeature = cut.Ptrs().size() - 1;
    for (unsigned fid = 0; fid < nfeature; ++fid) {
      auto ibegin = cut.Ptrs()[fid];
      auto iend = cut.Ptrs()[fid + 1];
      for (auto i = ibegin; i < iend; ++i) {
        counts[fid] += hit_count[i];
      }
    }
  }
  inline bool IsDense() const {
    return isDense_;
  }
  void SetDense(bool is_dense) { isDense_ = is_dense; }

  bst_row_t Size() const {
    return row_ptr.empty() ? 0 : row_ptr.size() - 1;
  }

 private:
  std::vector<size_t> hit_count_tloc_;
  bool isDense_;
};
}      // namespace xgboost
#endif  // XGBOOST_DATA_GRADIENT_INDEX_H_
