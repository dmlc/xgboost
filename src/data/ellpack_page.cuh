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

/** \brief Struct for accessing and manipulating an ellpack matrix on the
 * device. Does not own underlying memory and may be trivially copied into
 * kernels.*/
struct ELLPackMatrix {
  common::Span<uint32_t> feature_segments;
  /*! \brief minimum value for each feature. */
  common::Span<bst_float> min_fvalue;
  /*! \brief Cut. */
  common::Span<bst_float> gidx_fvalue_map;
  /*! \brief row length for ELLPack. */
  size_t row_stride{0};
  common::CompressedIterator<uint32_t> gidx_iter;
  int null_gidx_value;

  XGBOOST_DEVICE size_t BinCount() const { return gidx_fvalue_map.size(); }

  // Get a matrix element, uses binary search for look up Return NaN if missing
  // Given a row index and a feature index, returns the corresponding cut value
  __device__ bst_float GetElement(size_t ridx, size_t fidx) const {
    auto row_begin = row_stride * ridx;
    auto row_end = row_begin + row_stride;
    auto gidx = -1;
    if (is_dense) {
      gidx = gidx_iter[row_begin + fidx];
    } else {
      gidx =
          BinarySearchRow(row_begin, row_end, gidx_iter, feature_segments[fidx],
                          feature_segments[fidx + 1]);
    }
    if (gidx == -1) {
      return nan("");
    }
    return gidx_fvalue_map[gidx];
  }
  void Init(common::Span<uint32_t> feature_segments,
            common::Span<bst_float> min_fvalue,
            common::Span<bst_float> gidx_fvalue_map, size_t row_stride,
            common::CompressedIterator<uint32_t> gidx_iter, bool is_dense,
            int null_gidx_value) {
    this->feature_segments = feature_segments;
    this->min_fvalue = min_fvalue;
    this->gidx_fvalue_map = gidx_fvalue_map;
    this->row_stride = row_stride;
    this->gidx_iter = gidx_iter;
    this->is_dense = is_dense;
    this->null_gidx_value = null_gidx_value;
  }

 private:
  bool is_dense;
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
  explicit DeviceHistogramBuilderState(int n_rows) : device_row_state_(n_rows) {}

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
  ELLPackMatrix ellpack_matrix;
  int n_bins{};
  /*! \brief global index of histogram, which is stored in ELLPack format. */
  common::Span<common::CompressedByteT> gidx_buffer;

  explicit EllpackPageImpl(DMatrix* dmat, const BatchParam& parm);
  void InitCompressedData(int device,
                          const common::HistogramCuts& hmat,
                          size_t row_stride,
                          bool is_dense);
  void CreateHistIndices(int device,
                         const SparsePage& row_batch,
                         const RowStateOnDevice& device_row_state);

  size_t Size() const;

  /*! \brief clear the page
   */
  void Clear();

  /*!
   * \brief Push a sparse page
   * \param batch the row page
   */
  void Push(size_t row_stride, const common::HistogramCuts& hmat, const SparsePage& batch);

  /*! \return estimation of memory cost of this page
   */
  size_t MemCostBytes() const;

 private:
  DMatrix* dmat_;
  common::Monitor monitor_;
  dh::BulkAllocator ba;
};

}  // namespace xgboost

#endif  // XGBOOST_DATA_ELLPACK_PAGE_H_
