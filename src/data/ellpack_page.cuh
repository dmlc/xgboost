/*!
 * Copyright 2019 by XGBoost Contributors
 */

#ifndef XGBOOST_DATA_ELLPACK_PAGE_H_
#define XGBOOST_DATA_ELLPACK_PAGE_H_

#include <xgboost/data.h>

#include "../common/compressed_iterator.h"
#include "../common/device_helpers.cuh"
#include "../common/hist_util.h"
#include <thrust/binary_search.h>

namespace xgboost {
/** \brief Struct for accessing and manipulating an ellpack matrix on the
 * device. Does not own underlying memory and may be trivially copied into
 * kernels.*/
struct EllpackDeviceAccessor {
  /*! \brief Whether or not if the matrix is dense. */
  bool is_dense;
  /*! \brief Row length for ELLPack, equal to number of features. */
  size_t row_stride;
  size_t base_rowid{};
  size_t n_rows{};
  common::CompressedIterator<uint32_t> gidx_iter;
  /*! \brief Minimum value for each feature. Size equals to number of features. */
  common::Span<const bst_float> min_fvalue;
  /*! \brief Histogram cut pointers. Size equals to (number of features + 1). */
  common::Span<const uint32_t> feature_segments;
  /*! \brief Histogram cut values. Size equals to (bins per feature * number of features). */
  common::Span<const bst_float> gidx_fvalue_map;

  EllpackDeviceAccessor(int device, const common::HistogramCuts& cuts,
                        bool is_dense, size_t row_stride, size_t base_rowid,
                        size_t n_rows,common::CompressedIterator<uint32_t> gidx_iter)
      : is_dense(is_dense),
        row_stride(row_stride),
        base_rowid(base_rowid),
        n_rows(n_rows) ,gidx_iter(gidx_iter){
    cuts.cut_values_.SetDevice(device);
    cuts.cut_ptrs_.SetDevice(device);
    cuts.min_vals_.SetDevice(device);
    gidx_fvalue_map = cuts.cut_values_.ConstDeviceSpan();
    feature_segments = cuts.cut_ptrs_.ConstDeviceSpan();
    min_fvalue = cuts.min_vals_.ConstDeviceSpan();
  }
  // Get a matrix element, uses binary search for look up Return NaN if missing
  // Given a row index and a feature index, returns the corresponding cut value
  __device__ int32_t GetBinIndex(size_t ridx, size_t fidx) const {
    ridx -= base_rowid;
    auto row_begin = row_stride * ridx;
    auto row_end = row_begin + row_stride;
    auto gidx = -1;
    if (is_dense) {
      gidx = gidx_iter[row_begin + fidx];
    } else {
      gidx = common::BinarySearchBin(row_begin,
                                     row_end,
                                     gidx_iter,
                                     feature_segments[fidx],
                                     feature_segments[fidx + 1]);
    }
    return gidx;
  }

  __device__ uint32_t SearchBin(float value, size_t column_id) const {
    auto beg = feature_segments[column_id];
    auto end = feature_segments[column_id + 1];
    auto it =
        thrust::upper_bound(thrust::seq, gidx_fvalue_map.cbegin()+ beg, gidx_fvalue_map.cbegin() + end, value);
    uint32_t idx = it - gidx_fvalue_map.cbegin();
    if (idx == end) {
      idx -= 1;
    }
    return idx;
  }

  __device__ bst_float GetFvalue(size_t ridx, size_t fidx) const {
    auto gidx = GetBinIndex(ridx, fidx);
    if (gidx == -1) {
      return nan("");
    }
    return gidx_fvalue_map[gidx];
  }

  // Check if the row id is withing range of the current batch.
  __device__ bool IsInRange(size_t row_id) const {
    return row_id >= base_rowid && row_id < base_rowid + n_rows;
  }
  /*! \brief Return the total number of symbols (total number of bins plus 1 for
   * not found). */
  XGBOOST_DEVICE size_t NumSymbols() const { return gidx_fvalue_map.size() + 1; }

  XGBOOST_DEVICE size_t NullValue() const { return gidx_fvalue_map.size(); }

  XGBOOST_DEVICE size_t NumBins() const { return gidx_fvalue_map.size(); }

  XGBOOST_DEVICE size_t NumFeatures() const { return min_fvalue.size(); }
};


class EllpackPageImpl {
 public:
  /*!
   * \brief Default constructor.
   *
   * This is used in the external memory case. An empty ELLPACK page is constructed with its content
   * set later by the reader.
   */
  EllpackPageImpl() = default;

  /*!
   * \brief Constructor from an existing EllpackInfo.
   *
   * This is used in the sampling case. The ELLPACK page is constructed from an existing EllpackInfo
   * and the given number of rows.
   */
  EllpackPageImpl(int device, common::HistogramCuts cuts, bool is_dense,
                  size_t row_stride, size_t n_rows);
  /*!
   * \brief Constructor used for external memory.
   */
  EllpackPageImpl(int device, common::HistogramCuts cuts,
                  const SparsePage &page, bool is_dense, size_t row_stride,
                  common::Span<FeatureType const> feature_types);

  /*!
   * \brief Constructor from an existing DMatrix.
   *
   * This is used in the in-memory case. The ELLPACK page is constructed from an existing DMatrix
   * in CSR format.
   */
  explicit EllpackPageImpl(DMatrix* dmat, const BatchParam& parm);

  template <typename AdapterBatch>
  explicit EllpackPageImpl(AdapterBatch batch, float missing, int device, bool is_dense, int nthread,
                           common::Span<size_t> row_counts_span,
                           size_t row_stride, size_t n_rows, size_t n_cols,
                           common::HistogramCuts const& cuts);

  /*! \brief Copy the elements of the given ELLPACK page into this page.
   *
   * @param device The GPU device to use.
   * @param page The ELLPACK page to copy from.
   * @param offset The number of elements to skip before copying.
   * @returns The number of elements copied.
   */
  size_t Copy(int device, EllpackPageImpl* page, size_t offset);

  /*! \brief Compact the given ELLPACK page into the current page.
   *
   * @param device The GPU device to use.
   * @param page The ELLPACK page to compact from.
   * @param row_indexes Row indexes for the compacted page.
   */
  void Compact(int device, EllpackPageImpl* page, common::Span<size_t> row_indexes);


  /*! \return Number of instances in the page. */
  size_t Size() const;

  /*! \brief Set the base row id for this page. */
  void SetBaseRowId(size_t row_id) {
    base_rowid = row_id;
  }

  common::HistogramCuts& Cuts() { return cuts_; }
  common::HistogramCuts const& Cuts() const { return cuts_; }

  /*! \return Estimation of memory cost of this page. */
  static size_t MemCostBytes(size_t num_rows, size_t row_stride, const common::HistogramCuts&cuts) ;


  /*! \brief Return the total number of symbols (total number of bins plus 1 for
   * not found). */
  size_t NumSymbols() const { return cuts_.TotalBins() + 1; }

  EllpackDeviceAccessor GetDeviceAccessor(int device) const;

 private:
  /*!
   * \brief Compress a single page of CSR data into ELLPACK.
   *
   * @param device The GPU device to use.
   * @param row_batch The CSR page.
   */
  void CreateHistIndices(int device,
                         const SparsePage& row_batch,
                         common::Span<FeatureType const> feature_types);
  /*!
   * \brief Initialize the buffer to store compressed features.
   */
  void InitCompressedData(int device);


public:
  /*! \brief Whether or not if the matrix is dense. */
  bool is_dense;
  /*! \brief Row length for ELLPack. */
  size_t row_stride;
  size_t base_rowid{0};
  size_t n_rows{};
  /*! \brief global index of histogram, which is stored in ELLPack format. */
  HostDeviceVector<common::CompressedByteT> gidx_buffer;

 private:
  common::HistogramCuts cuts_;
  common::Monitor monitor_;
};

inline size_t GetRowStride(DMatrix* dmat) {
  if (dmat->IsDense()) return dmat->Info().num_col_;

  size_t row_stride = 0;
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    const auto& row_offset = batch.offset.ConstHostVector();
    for (auto i = 1ull; i < row_offset.size(); i++) {
      row_stride = std::max(
        row_stride, static_cast<size_t>(row_offset[i] - row_offset[i - 1]));
    }
  }
  return row_stride;
}
}  // namespace xgboost

#endif  // XGBOOST_DATA_ELLPACK_PAGE_H_
