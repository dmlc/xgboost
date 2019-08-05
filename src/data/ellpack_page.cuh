/*!
 * Copyright 2019 by XGBoost Contributors
 *
 * \file ellpack_page.cuh
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
  bool is_dense;
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
};

/**
 * \struct  DeviceHistogram
 *
 * \summary Data storage for node histograms on device. Automatically expands.
 *
 * \tparam GradientSumT      histogram entry type.
 * \tparam kStopGrowingSize  Do not grow beyond this size
 *
 * \author  Rory
 * \date    28/07/2018
 */
template <typename GradientSumT, size_t kStopGrowingSize = 1 << 26>
class DeviceHistogram {
 private:
  /*! \brief Map nidx to starting index of its histogram. */
  std::map<int, size_t> nidx_map_;
  dh::device_vector<typename GradientSumT::ValueT> data_;
  int n_bins_;
  int device_id_;
  static constexpr size_t kNumItemsInGradientSum =
      sizeof(GradientSumT) / sizeof(typename GradientSumT::ValueT);
  static_assert(kNumItemsInGradientSum == 2,
                "Number of items in gradient type should be 2.");

 public:
  void Init(int device_id, int n_bins) {
    this->n_bins_ = n_bins;
    this->device_id_ = device_id;
  }

  void Reset() {
    dh::safe_cuda(cudaMemsetAsync(
        data_.data().get(), 0,
        data_.size() * sizeof(typename decltype(data_)::value_type)));
    nidx_map_.clear();
  }
  bool HistogramExists(int nidx) const {
    return nidx_map_.find(nidx) != nidx_map_.cend();
  }
  size_t HistogramSize() const {
    return n_bins_ * kNumItemsInGradientSum;
  }

  dh::device_vector<typename GradientSumT::ValueT>& Data() {
    return data_;
  }

  void AllocateHistogram(int nidx) {
    if (HistogramExists(nidx)) return;
    // Number of items currently used in data
    const size_t used_size = nidx_map_.size() * HistogramSize();
    const size_t new_used_size = used_size + HistogramSize();
    dh::safe_cuda(cudaSetDevice(device_id_));
    if (data_.size() >= kStopGrowingSize) {
      // Recycle histogram memory
      if (new_used_size <= data_.size()) {
        // no need to remove old node, just insert the new one.
        nidx_map_[nidx] = used_size;
        // memset histogram size in bytes
        dh::safe_cuda(cudaMemsetAsync(data_.data().get() + used_size, 0,
                                      n_bins_ * sizeof(GradientSumT)));
      } else {
        std::pair<int, size_t> old_entry = *nidx_map_.begin();
        nidx_map_.erase(old_entry.first);
        dh::safe_cuda(cudaMemsetAsync(data_.data().get() + old_entry.second, 0,
                                      n_bins_ * sizeof(GradientSumT)));
        nidx_map_[nidx] = old_entry.second;
      }
    } else {
      // Append new node histogram
      nidx_map_[nidx] = used_size;
      size_t new_required_memory = std::max(data_.size() * 2, HistogramSize());
      if (data_.size() < new_required_memory) {
        data_.resize(new_required_memory);
      }
    }
  }

  /**
   * \summary   Return pointer to histogram memory for a given node.
   * \param nidx    Tree node index.
   * \return    hist pointer.
   */
  common::Span<GradientSumT> GetNodeHistogram(int nidx) {
    CHECK(this->HistogramExists(nidx));
    auto ptr = data_.data().get() + nidx_map_[nidx];
    return common::Span<GradientSumT>(
        reinterpret_cast<GradientSumT*>(ptr), n_bins_);
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
  explicit EllpackPageImpl(DMatrix* dmat);

  template<typename GradientSumT>
  void Init(int device, const tree::TrainParam& param, int gpu_batch_nrows);

 private:
  template<typename GradientSumT>
  void InitCompressedData(const common::HistogramCuts& hmat,
                          const tree::TrainParam& param,
                          size_t row_stride,
                          bool is_dense);

  template<typename GradientSumT>
  void CreateHistIndices(
      const SparsePage& row_batch, const common::HistogramCuts& hmat,
      const RowStateOnDevice& device_row_state, int rows_per_batch);

  bool initialised_{false};
  int device_{-1};
  int n_bins{};
  bool use_shared_memory_histograms {false};

  DMatrix* dmat_;
  common::HistogramCuts hmat_;
  common::Monitor monitor_;

  dh::BulkAllocator ba;
  ELLPackMatrix ellpack_matrix;

  /*! \brief row_ptr form HistogramCuts. */
  common::Span<uint32_t> feature_segments_;
  /*! \brief minimum value for each feature. */
  common::Span<bst_float> min_fvalue_;
  /*! \brief Cut. */
  common::Span<bst_float> gidx_fvalue_map_;
  /*! \brief global index of histogram, which is stored in ELLPack format. */
  common::Span<common::CompressedByteT> gidx_buffer_;
};

// Total number of nodes in tree, given depth
XGBOOST_DEVICE inline int MaxNodesDepth(int depth) {
  return (1 << (depth + 1)) - 1;
}

}  // namespace xgboost

#endif  // XGBOOST_DATA_ELLPACK_PAGE_H_
