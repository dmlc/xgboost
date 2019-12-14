/*!
 * Copyright 2019 by XGBoost Contributors
 */

#ifndef XGBOOST_DATA_ELLPACK_PAGE_H_
#define XGBOOST_DATA_ELLPACK_PAGE_H_

#include <xgboost/data.h>

#include "../common/compressed_iterator.h"
#include "../common/device_helpers.cuh"
#include "../common/hist_util.h"

namespace xgboost {

// Find a gidx value for a given feature otherwise return -1 if not found
__forceinline__ __device__ int BinarySearchRow(
    bst_uint begin, bst_uint end,
    common::CompressedIterator<uint32_t> data,
    int const fidx_begin, int const fidx_end) {
  bst_uint previous_middle = UINT32_MAX;
  while (end != begin) {
    auto middle = begin + (end - begin) / 2;
    if (middle == previous_middle) {
      break;
    }
    previous_middle = middle;

    auto gidx = data[middle];

    if (gidx >= fidx_begin && gidx < fidx_end) {
      return gidx;
    } else if (gidx < fidx_begin) {
      begin = middle;
    } else {
      end = middle;
    }
  }
  // Value is missing
  return -1;
}

/** \brief Meta information about the ELLPACK matrix. */
struct EllpackInfo {
  /*! \brief Whether or not if the matrix is dense. */
  bool is_dense;
  /*! \brief Row length for ELLPack, equal to number of features. */
  size_t row_stride;
  /*! \brief Total number of bins, also used as the null index value, . */
  size_t n_bins;
  /*! \brief Minimum value for each feature. Size equals to number of features. */
  common::Span<bst_float> min_fvalue;
  /*! \brief Histogram cut pointers. Size equals to (number of features + 1). */
  common::Span<uint32_t> feature_segments;
  /*! \brief Histogram cut values. Size equals to (bins per feature * number of features). */
  common::Span<bst_float> gidx_fvalue_map;

  EllpackInfo() = default;

  /*!
   * \brief Constructor.
   *
   * @param device The GPU device to use.
   * @param is_dense Whether the matrix is dense.
   * @param row_stride The number of features between starts of consecutive rows.
   * @param hmat The histogram cuts of all the features.
   * @param ba The BulkAllocator that owns the GPU memory.
   */
  explicit EllpackInfo(int device,
                       bool is_dense,
                       size_t row_stride,
                       const common::HistogramCuts& hmat,
                       dh::BulkAllocator* ba);
};

/** \brief Struct for accessing and manipulating an ellpack matrix on the
 * device. Does not own underlying memory and may be trivially copied into
 * kernels.*/
struct EllpackMatrix {
  EllpackInfo info;
  size_t base_rowid{};
  size_t n_rows{};
  common::CompressedIterator<uint32_t> gidx_iter;

  // Get a matrix element, uses binary search for look up Return NaN if missing
  // Given a row index and a feature index, returns the corresponding cut value
  __device__ bst_float GetElement(size_t ridx, size_t fidx) const {
    ridx -= base_rowid;
    auto row_begin = info.row_stride * ridx;
    auto row_end = row_begin + info.row_stride;
    auto gidx = -1;
    if (info.is_dense) {
      gidx = gidx_iter[row_begin + fidx];
    } else {
      gidx = BinarySearchRow(row_begin,
                             row_end,
                             gidx_iter,
                             info.feature_segments[fidx],
                             info.feature_segments[fidx + 1]);
    }
    if (gidx == -1) {
      return nan("");
    }
    return info.gidx_fvalue_map[gidx];
  }

  // Check if the row id is withing range of the current batch.
  __device__ bool IsInRange(size_t row_id) const {
    return row_id >= base_rowid && row_id < base_rowid + n_rows;
  }
};

// Instances of this type are created while creating the histogram bins for the
// entire dataset across multiple sparse page batches. This keeps track of the number
// of rows to process from a batch and the position from which to process on each device.
struct RowStateOnDevice {
  // Number of rows assigned to this device
  size_t total_rows_assigned_to_device;
  // Number of rows processed thus far
  size_t total_rows_processed;
  // Number of rows to process from the current sparse page batch
  size_t rows_to_process_from_batch;
  // Offset from the current sparse page batch to begin processing
  size_t row_offset_in_current_batch;

  explicit RowStateOnDevice(size_t total_rows)
      : total_rows_assigned_to_device(total_rows), total_rows_processed(0),
        rows_to_process_from_batch(0), row_offset_in_current_batch(0) {
  }

  explicit RowStateOnDevice(size_t total_rows, size_t batch_rows)
      : total_rows_assigned_to_device(total_rows), total_rows_processed(0),
        rows_to_process_from_batch(batch_rows), row_offset_in_current_batch(0) {
  }

  // Advance the row state by the number of rows processed
  void Advance() {
    total_rows_processed += rows_to_process_from_batch;
    CHECK_LE(total_rows_processed, total_rows_assigned_to_device);
    rows_to_process_from_batch = row_offset_in_current_batch = 0;
  }
};

// An instance of this type is created which keeps track of total number of rows to process,
// rows processed thus far, rows to process and the offset from the current sparse page batch
// to begin processing on each device
class DeviceHistogramBuilderState {
 public:
  explicit DeviceHistogramBuilderState(size_t n_rows) : device_row_state_(n_rows) {}

  const RowStateOnDevice& GetRowStateOnDevice() const {
    return device_row_state_;
  }

  // This method is invoked at the beginning of each sparse page batch. This distributes
  // the rows in the sparse page to the device.
  // TODO(sriramch): Think of a way to utilize *all* the GPUs to build the compressed bins.
  void BeginBatch(const SparsePage &batch) {
    size_t rem_rows = batch.Size();
    size_t row_offset_in_current_batch = 0;

    // Do we have anymore left to process from this batch on this device?
    if (device_row_state_.total_rows_assigned_to_device > device_row_state_.total_rows_processed) {
      // There are still some rows that needs to be assigned to this device
      device_row_state_.rows_to_process_from_batch =
          std::min(
              device_row_state_.total_rows_assigned_to_device - device_row_state_.total_rows_processed,
              rem_rows);
    } else {
      // All rows have been assigned to this device
      device_row_state_.rows_to_process_from_batch = 0;
    }

    device_row_state_.row_offset_in_current_batch = row_offset_in_current_batch;
    row_offset_in_current_batch += device_row_state_.rows_to_process_from_batch;
    rem_rows -= device_row_state_.rows_to_process_from_batch;
  }

  // This method is invoked after completion of each sparse page batch
  void EndBatch() {
    device_row_state_.Advance();
  }

 private:
  RowStateOnDevice device_row_state_{0};
};

class EllpackPageImpl {
 public:
  EllpackMatrix matrix;
  /*! \brief global index of histogram, which is stored in ELLPack format. */
  common::Span<common::CompressedByteT> gidx_buffer;
  std::vector<common::CompressedByteT> idx_buffer;

  /*!
   * \brief Default constructor.
   *
   * This is used in the external memory case. An empty ELLPACK page is constructed with its content
   * set later by the reader.
   */
  EllpackPageImpl() = default;

  /*!
   * \brief Constructor from an existing DMatrix.
   *
   * This is used in the in-memory case. The ELLPACK page is constructed from an existing DMatrix
   * in CSR format.
   */
  explicit EllpackPageImpl(DMatrix* dmat, const BatchParam& parm);

  /*!
   * \brief Initialize the EllpackInfo contained in the EllpackMatrix.
   *
   * This is used in the in-memory case. The current page owns the BulkAllocator, which in turn owns
   * the GPU memory used by the EllpackInfo.
   *
   * @param device The GPU device to use.
   * @param is_dense Whether the matrix is dense.
   * @param row_stride The number of features between starts of consecutive rows.
   * @param hmat The histogram cuts of all the features.
   */
  void InitInfo(int device, bool is_dense, size_t row_stride, const common::HistogramCuts& hmat);

  /*!
   * \brief Initialize the buffer to store compressed features.
   *
   * @param device The GPU device to use.
   * @param num_rows The number of rows we are storing in the buffer.
   */
  void InitCompressedData(int device, size_t num_rows);

  /*!
   * \brief Compress a single page of CSR data into ELLPACK.
   *
   * @param device The GPU device to use.
   * @param row_batch The CSR page.
   * @param device_row_state On-device data for maintaining state.
   */
  void CreateHistIndices(int device,
                         const SparsePage& row_batch,
                         const RowStateOnDevice& device_row_state);

  /*! \return Number of instances in the page. */
  size_t Size() const;

  /*! \brief Set the base row id for this page. */
  inline void SetBaseRowId(size_t row_id) {
    matrix.base_rowid = row_id;
  }

  /*! \brief clear the page. */
  void Clear();

  /*!
   * \brief Push a sparse page.
   * \param batch The row page.
   */
  void Push(int device, const SparsePage& batch);

  /*! \return Estimation of memory cost of this page. */
  size_t MemCostBytes() const;

  /*!
   * \brief Copy the ELLPACK matrix to GPU.
   *
   * @param device The GPU device to use.
   * @param info The EllpackInfo for the matrix.
   */
  void InitDevice(int device, EllpackInfo info);

  /*! \brief Compress the accumulated SparsePage into ELLPACK format.
   *
   * @param device The GPU device to use.
   */
  void CompressSparsePage(int device);

 private:
  common::Monitor monitor_;
  dh::BulkAllocator ba_;
  bool device_initialized_{false};
  SparsePage sparse_page_{};
};

}  // namespace xgboost

#endif  // XGBOOST_DATA_ELLPACK_PAGE_H_
