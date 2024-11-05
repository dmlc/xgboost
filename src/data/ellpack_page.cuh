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
  bst_idx_t null_value;
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
  std::uint32_t const* feature_segments;
  /** @brief Histogram cut values. Size equals to (bins per feature * number of features). */
  common::Span<const float> gidx_fvalue_map;
  /** @brief Type of each feature, categorical or numerical. */
  common::Span<const FeatureType> feature_types;

  EllpackDeviceAccessor() = delete;
  EllpackDeviceAccessor(Context const* ctx, std::shared_ptr<const common::HistogramCuts> cuts,
                        bst_idx_t row_stride, bst_idx_t base_rowid, bst_idx_t n_rows,
                        common::CompressedIterator<uint32_t> gidx_iter, bst_idx_t null_value,
                        common::Span<FeatureType const> feature_types)
      : null_value{null_value},
        row_stride{row_stride},
        base_rowid{base_rowid},
        n_rows{n_rows},
        gidx_iter{gidx_iter},
        feature_types{feature_types} {
    if (ctx->IsCUDA()) {
      cuts->cut_values_.SetDevice(ctx->Device());
      cuts->cut_ptrs_.SetDevice(ctx->Device());
      cuts->min_vals_.SetDevice(ctx->Device());
      gidx_fvalue_map = cuts->cut_values_.ConstDeviceSpan();
      feature_segments = cuts->cut_ptrs_.ConstDevicePointer();
      min_fvalue = cuts->min_vals_.ConstDeviceSpan();
    } else {
      gidx_fvalue_map = cuts->cut_values_.ConstHostSpan();
      feature_segments = cuts->cut_ptrs_.ConstHostPointer();
      min_fvalue = cuts->min_vals_.ConstHostSpan();
    }
  }

  [[nodiscard]] XGBOOST_HOST_DEV_INLINE bool IsDenseCompressed() const {
    return this->row_stride == this->NumFeatures();
  }
  /**
   * @brief Given a row index and a feature index, returns the corresponding bin index.
   *
   * Uses binary search for look up.
   *
   * @tparam global_ridx Whether the row index is global to all ellpack batches or it's
   *                     local to the current batch.
   *
   * @return -1 if it's a missing value.
   */
  template <bool global_ridx = true>
  [[nodiscard]] __device__ bst_bin_t GetBinIndex(bst_idx_t ridx, std::size_t fidx) const {
    if (global_ridx) {
      ridx -= base_rowid;
    }
    auto row_begin = row_stride * ridx;
    if (!this->IsDenseCompressed()) {
      // binary search returns -1 if it's missing
      auto row_end = row_begin + row_stride;
      bst_bin_t gidx = common::BinarySearchBin(row_begin, row_end, gidx_iter,
                                               feature_segments[fidx], feature_segments[fidx + 1]);
      return gidx;
    }
    bst_bin_t gidx = gidx_iter[row_begin + fidx];
    if (gidx == this->NullValue()) {
      // Missing value in a dense ellpack
      return -1;
    }
    // Dense ellpack
    gidx += this->feature_segments[fidx];
    return gidx;
  }
  /**
   * @brief Find a bin to place the value in. Used during construction of the Ellpack.
   */
  template <bool is_cat>
  [[nodiscard]] __device__ bst_bin_t SearchBin(float value, std::size_t fidx) const {
    auto beg = feature_segments[fidx];
    auto end = feature_segments[fidx + 1];
    bst_bin_t gidx = 0;
    if (is_cat) {
      auto it = dh::MakeTransformIterator<bst_cat_t>(gidx_fvalue_map.cbegin(),
                                                     [](float v) { return common::AsCat(v); });
      gidx = thrust::lower_bound(thrust::seq, it + beg, it + end, common::AsCat(value)) - it;
    } else {
      auto it = thrust::upper_bound(thrust::seq, gidx_fvalue_map.cbegin() + beg,
                                    gidx_fvalue_map.cbegin() + end, value);
      gidx = it - gidx_fvalue_map.cbegin();
    }

    if (gidx == end) {
      gidx -= 1;
    }
    return gidx;
  }

  [[nodiscard]] __device__ float GetFvalue(bst_idx_t ridx, size_t fidx) const {
    auto gidx = GetBinIndex(ridx, fidx);
    if (gidx == -1) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return gidx_fvalue_map[gidx];
  }
  [[nodiscard]] XGBOOST_HOST_DEV_INLINE bst_idx_t NullValue() const { return this->null_value; }
  [[nodiscard]] XGBOOST_HOST_DEV_INLINE bst_idx_t NumBins() const { return gidx_fvalue_map.size(); }
  [[nodiscard]] XGBOOST_HOST_DEV_INLINE size_t NumFeatures() const { return min_fvalue.size(); }
};


class GHistIndexMatrix;

/**
 * @brief This is either an Ellpack format matrix or a dense matrix.
 *
 * When there's no compression can be made by using ellpack, we use this structure as a
 * simple dense matrix. For dense matrix, we can provide extra compression by counting the
 * histogram bin for each feature instead of for the entire dataset.
 */
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
   * @brief Constructor from existing ellpack matrics.
   *
   * This is used in the sampling case. The ELLPACK page is constructed from an existing
   * Ellpack page and the given number of rows.
   */
  EllpackPageImpl(Context const* ctx, std::shared_ptr<common::HistogramCuts const> cuts,
                  bool is_dense, bst_idx_t row_stride, bst_idx_t n_rows);
  /**
   * @brief Constructor used for external memory with DMatrix.
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
  /**
   * @brief Constructor for Quantile DMatrix using an adapter.
   */
  template <typename AdapterBatch>
  explicit EllpackPageImpl(Context const* ctx, AdapterBatch batch, float missing, bool is_dense,
                           common::Span<bst_idx_t const> row_counts_span,
                           common::Span<FeatureType const> feature_types, bst_idx_t row_stride,
                           bst_idx_t n_rows, std::shared_ptr<common::HistogramCuts const> cuts);
  /**
   * @brief Constructor from an existing CPU gradient index.
   */
  explicit EllpackPageImpl(Context const* ctx, GHistIndexMatrix const& page,
                           common::Span<FeatureType const> ft);

  EllpackPageImpl(EllpackPageImpl const& that) = delete;
  EllpackPageImpl& operator=(EllpackPageImpl const& that) = delete;

  EllpackPageImpl(EllpackPageImpl&& that) = default;
  EllpackPageImpl& operator=(EllpackPageImpl&& that) = default;

  ~EllpackPageImpl() noexcept(false);

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
  void SetCuts(std::shared_ptr<common::HistogramCuts const> cuts);
  /**
   * @brief Fully dense, there's not a single missing value.
   */
  [[nodiscard]] bool IsDense() const { return this->is_dense; }
  /**
   * @brief Stored as a dense matrix, but there might be missing values.
   */
  [[nodiscard]] bool IsDenseCompressed() const {
    return this->cuts_->NumFeatures() == this->info.row_stride;
  }

  /** @return Estimation of memory cost of this page. */
  std::size_t MemCostBytes() const;

  /**
   * @brief Return the total number of symbols (total number of bins plus 1 for not
   *        found).
   */
  [[nodiscard]] auto NumSymbols() const { return this->info.n_symbols; }
  void SetNumSymbols(bst_idx_t n_symbols) { this->info.n_symbols = n_symbols; }
  /**
   * @brief Copy basic shape from another page.
   */
  void CopyInfo(EllpackPageImpl const* page) {
    CHECK_NE(this, page);
    this->n_rows = page->Size();
    this->is_dense = page->IsDense();
    this->info.row_stride = page->info.row_stride;
    this->SetBaseRowId(page->base_rowid);
    this->SetNumSymbols(page->NumSymbols());
  }
  /**
   * @brief Get an accessor that can be passed into CUDA kernels.
   */
  [[nodiscard]] EllpackDeviceAccessor GetDeviceAccessor(
      Context const* ctx, common::Span<FeatureType const> feature_types = {}) const;
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
   * @param row_batch The CSR page.
   */
  void CreateHistIndices(Context const* ctx, const SparsePage& row_batch,
                         common::Span<FeatureType const> feature_types);
  /**
   * @brief Initialize the buffer to store compressed features.
   */
  void InitCompressedData(Context const* ctx);

  std::shared_ptr<common::HistogramCuts const> cuts_;

 public:
  bool is_dense{false};

  bst_idx_t base_rowid{0};
  bst_idx_t n_rows{0};
  /**
   * @brief Index of the gradient histogram, which is stored in ELLPACK format.
   *
   * This can be backed by various storage types.
   */
  common::RefResourceView<common::CompressedByteT> gidx_buffer;
  /**
   * @brief Compression infomation.
   */
  struct Info {
    /** @brief Row length for ELLPACK. */
    bst_idx_t row_stride{0};
    /** @brief The number of unique bins including missing. */
    bst_idx_t n_symbols{0};
  } info;

 private:
  common::Monitor monitor_;
};

[[nodiscard]] inline bst_idx_t GetRowStride(DMatrix* dmat) {
  if (dmat->IsDense()) {
    return dmat->Info().num_col_;
  }

  bst_idx_t row_stride = 0;
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    const auto& row_offset = batch.offset.ConstHostVector();
    for (auto i = 1ull; i < row_offset.size(); i++) {
      row_stride = std::max(row_stride, static_cast<size_t>(row_offset[i] - row_offset[i - 1]));
    }
  }
  return row_stride;
}

[[nodiscard]] EllpackPageImpl::Info CalcNumSymbols(
    Context const* ctx, bst_idx_t row_stride, bool is_dense,
    std::shared_ptr<common::HistogramCuts const> cuts);
}  // namespace xgboost

#endif  // XGBOOST_DATA_ELLPACK_PAGE_CUH_
