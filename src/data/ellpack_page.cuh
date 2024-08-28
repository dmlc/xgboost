/**
 * Copyright 2019-2024, XGBoost Contributors
 */
#ifndef XGBOOST_DATA_ELLPACK_PAGE_CUH_
#define XGBOOST_DATA_ELLPACK_PAGE_CUH_

#include <thrust/binary_search.h>

#include <limits>  // for numeric_limits

#include "../common/categorical.h"
#include "../common/compressed_iterator.h"
#include "../common/device_helpers.cuh"
#include "../common/hist_util.h"
#include "../common/ref_resource_view.h"  // for RefResourceView
#include "ellpack_page.h"
#include "xgboost/data.h"

namespace xgboost {
/**
 * @brief Struct for accessing and manipulating an ELLPACK matrix on the device.
 *
 * Does not own underlying memory and may be trivially copied into kernels.
 */
struct EllpackDeviceAccessor {
  /** @brief Whether or not if the matrix is dense. */
  bool is_dense;
  /** @brief Row length for ELLPACK, equal to number of features when the data is dense. */
  bst_idx_t row_stride;
  /** @brief Starting index of the rows. Used for external memory. */
  bst_idx_t base_rowid;
  /** @brief Number of rows in this batch. */
  bst_idx_t n_rows;
  /** @brief Acessor for the gradient index. */
  common::CompressedIterator<std::uint32_t> gidx_iter;
  /** @brief Minimum value for each feature. Size equals to number of features. */
  common::Span<const float> min_fvalue;
  /** @brief Histogram cut pointers. Size equals to (number of features + 1). */
  common::Span<const std::uint32_t> feature_segments;
  /** @brief Histogram cut values. Size equals to (bins per feature * number of features). */
  common::Span<const float> gidx_fvalue_map;
  /** @brief Type of each feature, categorical or numerical. */
  common::Span<const FeatureType> feature_types;

  EllpackDeviceAccessor() = delete;
  EllpackDeviceAccessor(DeviceOrd device, std::shared_ptr<const common::HistogramCuts> cuts,
                        bool is_dense, size_t row_stride, size_t base_rowid, size_t n_rows,
                        common::CompressedIterator<uint32_t> gidx_iter,
                        common::Span<FeatureType const> feature_types)
      : is_dense(is_dense),
        row_stride(row_stride),
        base_rowid(base_rowid),
        n_rows(n_rows),
        gidx_iter(gidx_iter),
        feature_types{feature_types} {
    if (device.IsCUDA()) {
      cuts->cut_values_.SetDevice(device);
      cuts->cut_ptrs_.SetDevice(device);
      cuts->min_vals_.SetDevice(device);
      gidx_fvalue_map = cuts->cut_values_.ConstDeviceSpan();
      feature_segments = cuts->cut_ptrs_.ConstDeviceSpan();
      min_fvalue = cuts->min_vals_.ConstDeviceSpan();
    } else {
      gidx_fvalue_map = cuts->cut_values_.ConstHostSpan();
      feature_segments = cuts->cut_ptrs_.ConstHostSpan();
      min_fvalue = cuts->min_vals_.ConstHostSpan();
    }
  }

  /**
   * @brief Given a row index and a feature index, returns the corresponding cut value.
   *
   * Uses binary search for look up. Returns NaN if missing.
   *
   * @tparam global_ridx Whether the row index is global to all ellpack batches or it's
   *                     local to the current batch.
   */
  template <bool global_ridx = true>
  [[nodiscard]] __device__ bst_bin_t GetBinIndex(bst_idx_t ridx, size_t fidx) const {
    if (global_ridx) {
      ridx -= base_rowid;
    }
    auto row_begin = row_stride * ridx;
    auto row_end = row_begin + row_stride;
    bst_bin_t gidx = -1;
    if (is_dense) {
      gidx = gidx_iter[row_begin + fidx];
    } else {
      gidx = common::BinarySearchBin(row_begin, row_end, gidx_iter, feature_segments[fidx],
                                     feature_segments[fidx + 1]);
    }
    return gidx;
  }

  template <bool is_cat>
  [[nodiscard]] __device__ uint32_t SearchBin(float value, size_t column_id) const {
    auto beg = feature_segments[column_id];
    auto end = feature_segments[column_id + 1];
    uint32_t idx = 0;
    if (is_cat) {
      auto it = dh::MakeTransformIterator<bst_cat_t>(
          gidx_fvalue_map.cbegin(), [](float v) { return common::AsCat(v); });
      idx = thrust::lower_bound(thrust::seq, it + beg, it + end,
                                common::AsCat(value)) -
            it;
    } else {
      auto it = thrust::upper_bound(thrust::seq, gidx_fvalue_map.cbegin() + beg,
                                    gidx_fvalue_map.cbegin() + end, value);
      idx = it - gidx_fvalue_map.cbegin();
    }

    if (idx == end) {
      idx -= 1;
    }
    return idx;
  }

  [[nodiscard]] __device__ float GetFvalue(bst_idx_t ridx, size_t fidx) const {
    auto gidx = GetBinIndex(ridx, fidx);
    if (gidx == -1) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return gidx_fvalue_map[gidx];
  }

  // Check if the row id is withing range of the current batch.
  [[nodiscard]] __device__ bool IsInRange(size_t row_id) const {
    return row_id >= base_rowid && row_id < base_rowid + n_rows;
  }
  /*! \brief Return the total number of symbols (total number of bins plus 1 for
   * not found). */
  [[nodiscard]] XGBOOST_DEVICE size_t NumSymbols() const { return gidx_fvalue_map.size() + 1; }

  [[nodiscard]] XGBOOST_DEVICE size_t NullValue() const { return this->NumBins(); }

  [[nodiscard]] XGBOOST_DEVICE size_t NumBins() const { return gidx_fvalue_map.size(); }

  [[nodiscard]] XGBOOST_DEVICE size_t NumFeatures() const { return min_fvalue.size(); }
};


class GHistIndexMatrix;

class EllpackPageImpl {
 public:
  /**
   * @brief Default constructor.
   *
   * This is used in the external memory case. An empty ELLPACK page is constructed with its content
   * set later by the reader.
   */
  EllpackPageImpl() = default;

  /**
   * @brief Constructor from an existing EllpackInfo.
   *
   * This is used in the sampling case. The ELLPACK page is constructed from an existing
   * Ellpack page and the given number of rows.
   */
  EllpackPageImpl(Context const* ctx, std::shared_ptr<common::HistogramCuts const> cuts,
                  bool is_dense, bst_idx_t row_stride, bst_idx_t n_rows);
  /**
   * @brief Constructor used for external memory.
   */
  EllpackPageImpl(Context const* ctx, std::shared_ptr<common::HistogramCuts const> cuts,
                  const SparsePage& page, bool is_dense, size_t row_stride,
                  common::Span<FeatureType const> feature_types);

  /**
   * @brief Constructor from an existing DMatrix.
   *
   * This is used in the in-memory case. The ELLPACK page is constructed from an existing DMatrix
   * in CSR format.
   */
  explicit EllpackPageImpl(Context const* ctx, DMatrix* dmat, const BatchParam& parm);

  template <typename AdapterBatch>
  explicit EllpackPageImpl(Context const* ctx, AdapterBatch batch, float missing, bool is_dense,
                           common::Span<size_t> row_counts_span,
                           common::Span<FeatureType const> feature_types, size_t row_stride,
                           size_t n_rows, std::shared_ptr<common::HistogramCuts const> cuts);
  /**
   * @brief Constructor from an existing CPU gradient index.
   */
  explicit EllpackPageImpl(Context const* ctx, GHistIndexMatrix const& page,
                           common::Span<FeatureType const> ft);

  /**
   * @brief Copy the elements of the given ELLPACK page into this page.
   *
   * @param ctx The GPU context.
   * @param page The ELLPACK page to copy from.
   * @param offset The number of elements to skip before copying.
   * @returns The number of elements copied.
   */
  bst_idx_t Copy(Context const* ctx, EllpackPageImpl const* page, bst_idx_t offset);

  /**
   * @brief Compact the given ELLPACK page into the current page.
   *
   * @param ctx The GPU context.
   * @param page The ELLPACK page to compact from.
   * @param row_indexes Row indexes for the compacted page.
   */
  void Compact(Context const* ctx, EllpackPageImpl const* page, common::Span<size_t> row_indexes);

  /** @return Number of instances in the page. */
  [[nodiscard]] bst_idx_t Size() const;

  /** @brief Set the base row id for this page. */
  void SetBaseRowId(std::size_t row_id) {
    base_rowid = row_id;
  }

  [[nodiscard]] common::HistogramCuts const& Cuts() const { return *cuts_; }
  [[nodiscard]] std::shared_ptr<common::HistogramCuts const> CutsShared() const { return cuts_; }
  void SetCuts(std::shared_ptr<common::HistogramCuts const> cuts) { cuts_ = cuts; }

  [[nodiscard]] bool IsDense() const { return is_dense; }
  /** @return Estimation of memory cost of this page. */
  std::size_t MemCostBytes() const;

  /**
   * @brief Return the total number of symbols (total number of bins plus 1 for not
   *        found).
   */
  [[nodiscard]] std::size_t NumSymbols() const { return cuts_->TotalBins() + 1; }
  /**
   * @brief Get an accessor that can be passed into CUDA kernels.
   */
  [[nodiscard]] EllpackDeviceAccessor GetDeviceAccessor(
      DeviceOrd device, common::Span<FeatureType const> feature_types = {}) const;
  /**
   * @brief Get an accessor for host code.
   */
  [[nodiscard]] EllpackDeviceAccessor GetHostAccessor(
      Context const* ctx, std::vector<common::CompressedByteT>* h_gidx_buffer,
      common::Span<FeatureType const> feature_types = {}) const;
  /**
   * @brief Calculate the number of non-missing values.
   */
  [[nodiscard]] bst_idx_t NumNonMissing(Context const* ctx,
                                        common::Span<FeatureType const> feature_types) const;

 private:
  /**
   * @brief Compress a single page of CSR data into ELLPACK.
   *
   * @param device The GPU device to use.
   * @param row_batch The CSR page.
   */
  void CreateHistIndices(DeviceOrd device, const SparsePage& row_batch,
                         common::Span<FeatureType const> feature_types);
  /**
   * @brief Initialize the buffer to store compressed features.
   */
  void InitCompressedData(Context const* ctx);

 public:
  /** @brief Whether or not if the matrix is dense. */
  bool is_dense;
  /** @brief Row length for ELLPACK. */
  bst_idx_t row_stride;
  bst_idx_t base_rowid{0};
  bst_idx_t n_rows{0};
  /**
   * @brief Index of the gradient histogram, which is stored in ELLPACK format.
   *
   * This can be backed by various storage types.
   */
  common::RefResourceView<common::CompressedByteT> gidx_buffer;

 private:
  std::shared_ptr<common::HistogramCuts const> cuts_;
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

#endif  // XGBOOST_DATA_ELLPACK_PAGE_CUH_
