/*!
 * Copyright 2017-2022 by XGBoost Contributors
 * \brief Data type for fast histogram aggregation.
 */
#ifndef XGBOOST_DATA_GRADIENT_INDEX_H_
#define XGBOOST_DATA_GRADIENT_INDEX_H_
#include <memory>
#include <vector>

#include "../common/categorical.h"
#include "../common/hist_util.h"
#include "../common/threading_utils.h"
#include "xgboost/base.h"
#include "xgboost/data.h"

namespace xgboost {
/*!
 * \brief preprocessed global index matrix, in CSR format
 *
 *  Transform floating values to integer index in histogram This is a global histogram
 *  index for CPU histogram.  On GPU ellpack page is used.
 */
class GHistIndexMatrix {
  /**
   * \brief Push a page into index matrix, the function is only necessary because hist has
   *        partial support for external memory.
   *
   * \param rbegin The beginning row index of current page. (total rows in previous pages)
   * \param prev_sum Total number of entries in previous pages.
   */
  void PushBatch(SparsePage const& batch, common::Span<FeatureType const> ft, size_t rbegin,
                 size_t prev_sum, uint32_t nbins, int32_t n_threads);

 public:
  /*! \brief row pointer to rows by element position */
  std::vector<size_t> row_ptr;
  /*! \brief The index data */
  common::Index index;
  /*! \brief hit count of each index */
  std::vector<size_t> hit_count;
  /*! \brief The corresponding cuts */
  common::HistogramCuts cut;
  /*! \brief max_bin for each feature. */
  size_t max_num_bins;
  /*! \brief base row index for current page (used by external memory) */
  size_t base_rowid{0};

  GHistIndexMatrix();
  GHistIndexMatrix(DMatrix* x, int32_t max_bin, double sparse_thresh, bool sorted_sketch,
                   int32_t n_threads, common::Span<float> hess = {});
  ~GHistIndexMatrix();

  // Create a global histogram matrix, given cut
  void Init(DMatrix* p_fmat, int max_bins, double sparse_thresh, bool sorted_sketch,
            int32_t n_threads, common::Span<float> hess);
  void Init(SparsePage const& page, common::Span<FeatureType const> ft,
            common::HistogramCuts const& cuts, int32_t max_bins_per_feat, bool is_dense,
            double sparse_thresh, int32_t n_threads);

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
    common::ParallelFor(batch_size, batch_threads, [&](omp_ulong ridx) {
      const int tid = omp_get_thread_num();
      size_t ibegin = row_ptr[rbegin + ridx];    // index of first entry for current block
      size_t iend = row_ptr[rbegin + ridx + 1];  // first entry for next block
      const size_t size = offset_vec[ridx + 1] - offset_vec[ridx];
      SparsePage::Inst inst = {data_ptr + offset_vec[ridx], size};
      CHECK_EQ(ibegin + inst.size(), iend);
      for (bst_uint j = 0; j < inst.size(); ++j) {
        auto e = inst[j];
        if (common::IsCat(ft, e.index)) {
          auto bin_idx = cut.SearchCatBin(e);
          index_data[ibegin + j] = get_offset(bin_idx, j);
          ++hit_count_tloc_[tid * nbins + bin_idx];
        } else {
          uint32_t idx = cut.SearchBin(e.fvalue, e.index, ptrs, values);
          index_data[ibegin + j] = get_offset(idx, j);
          ++hit_count_tloc_[tid * nbins + idx];
        }
      }
    });
  }

  void ResizeIndex(const size_t n_index, const bool isDense);

  void GetFeatureCounts(size_t* counts) const {
    auto nfeature = cut.Ptrs().size() - 1;
    for (unsigned fid = 0; fid < nfeature; ++fid) {
      auto ibegin = cut.Ptrs()[fid];
      auto iend = cut.Ptrs()[fid + 1];
      for (auto i = ibegin; i < iend; ++i) {
        counts[fid] += hit_count[i];
      }
    }
  }

  bool IsDense() const {
    return isDense_;
  }
  void SetDense(bool is_dense) { isDense_ = is_dense; }
  /**
   * \brief Get the local row index.
   */
  size_t RowIdx(size_t ridx) const { return row_ptr[ridx - base_rowid]; }

  bst_row_t Size() const {
    return row_ptr.empty() ? 0 : row_ptr.size() - 1;
  }

  bool ReadColumnPage(dmlc::SeekStream* fi);
  size_t WriteColumnPage(dmlc::Stream* fo) const;

  common::ColumnMatrix const& Transpose() const;

 private:
  std::unique_ptr<common::ColumnMatrix> columns_;
  std::vector<size_t> hit_count_tloc_;
  bool isDense_;
};

/**
 * \brief Should we regenerate the gradient index?
 *
 * \param old Parameter stored in DMatrix.
 * \param p   New parameter passed in by caller.
 */
inline bool RegenGHist(BatchParam old, BatchParam p) {
  // parameter is renewed or caller requests a regen
  if (p == BatchParam{}) {
    // empty parameter is passed in, don't regenerate so that we can use gindex in
    // predictor, which doesn't have any training parameter.
    return false;
  }

  // Avoid comparing nan values.
  bool l_nan = std::isnan(old.sparse_thresh);
  bool r_nan = std::isnan(p.sparse_thresh);
  // regenerate if parameter is changed.
  bool st_chg = (l_nan != r_nan) || (!l_nan && !r_nan && (old.sparse_thresh != p.sparse_thresh));
  bool param_chg = old.gpu_id != p.gpu_id || old.max_bin != p.max_bin;
  return p.regen || param_chg || st_chg;
}
}      // namespace xgboost
#endif  // XGBOOST_DATA_GRADIENT_INDEX_H_
