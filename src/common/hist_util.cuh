/**
 * Copyright 2020-2023 by XGBoost contributors
 *
 * \brief Front end and utilities for GPU based sketching.  Works on sliding window
 *        instead of stream.
 */
#ifndef COMMON_HIST_UTIL_CUH_
#define COMMON_HIST_UTIL_CUH_

#include <thrust/host_vector.h>

#include <cstddef>  // for size_t

#include "../data/device_adapter.cuh"
#include "device_helpers.cuh"
#include "hist_util.h"
#include "quantile.cuh"
#include "timer.h"

namespace xgboost {
namespace common {
namespace cuda {
/**
 * copy and paste of the host version, we can't make it a __host__ __device__ function as
 * the fn might be a host only or device only callable object, which is not allowed by nvcc.
 */
template <typename Fn>
auto __device__ DispatchBinType(BinTypeSize type, Fn&& fn) {
  switch (type) {
    case kUint8BinsTypeSize: {
      return fn(uint8_t{});
    }
    case kUint16BinsTypeSize: {
      return fn(uint16_t{});
    }
    case kUint32BinsTypeSize: {
      return fn(uint32_t{});
    }
  }
  SPAN_CHECK(false);
  return fn(uint32_t{});
}
}  // namespace cuda

namespace detail {
struct EntryCompareOp {
  __device__ bool operator()(const Entry& a, const Entry& b) {
    if (a.index == b.index) {
      return a.fvalue < b.fvalue;
    }
    return a.index < b.index;
  }
};

// Get column size from adapter batch and for output cuts.
template <typename Iter>
void GetColumnSizesScan(int device, size_t num_columns, size_t num_cuts_per_feature,
                        Iter batch_iter, data::IsValidFunctor is_valid,
                        size_t begin, size_t end,
                        HostDeviceVector<SketchContainer::OffsetT> *cuts_ptr,
                        dh::caching_device_vector<size_t>* column_sizes_scan) {
  column_sizes_scan->resize(num_columns + 1, 0);
  cuts_ptr->SetDevice(device);
  cuts_ptr->Resize(num_columns + 1, 0);

  dh::XGBCachingDeviceAllocator<char> alloc;
  auto d_column_sizes_scan = column_sizes_scan->data().get();
  dh::LaunchN(end - begin, [=] __device__(size_t idx) {
    auto e = batch_iter[begin + idx];
    if (is_valid(e)) {
      atomicAdd(&d_column_sizes_scan[e.column_idx], static_cast<size_t>(1));
    }
  });
  // Calculate cuts CSC pointer
  auto cut_ptr_it = dh::MakeTransformIterator<size_t>(
      column_sizes_scan->begin(), [=] __device__(size_t column_size) {
        return thrust::min(num_cuts_per_feature, column_size);
      });
  thrust::exclusive_scan(thrust::cuda::par(alloc), cut_ptr_it,
                         cut_ptr_it + column_sizes_scan->size(),
                         cuts_ptr->DevicePointer());
  thrust::exclusive_scan(thrust::cuda::par(alloc), column_sizes_scan->begin(),
                         column_sizes_scan->end(), column_sizes_scan->begin());
}

inline size_t constexpr BytesPerElement(bool has_weight) {
  // Double the memory usage for sorting.  We need to assign weight for each element, so
  // sizeof(float) is added to all elements.
  return (has_weight ? sizeof(Entry) + sizeof(float) : sizeof(Entry)) * 2;
}

/* \brief Calcuate the length of sliding window. Returns `sketch_batch_num_elements`
 *        directly if it's not 0.
 */
size_t SketchBatchNumElements(size_t sketch_batch_num_elements,
                              bst_row_t num_rows, bst_feature_t columns,
                              size_t nnz, int device,
                              size_t num_cuts, bool has_weight);

// Compute number of sample cuts needed on local node to maintain accuracy
// We take more cuts than needed and then reduce them later
size_t RequiredSampleCutsPerColumn(int max_bins, size_t num_rows);

/* \brief Estimate required memory for each sliding window.
 *
 *   It's not precise as to obtain exact memory usage for sparse dataset we need to walk
 *   through the whole dataset first.  Also if data is from host DMatrix, we copy the
 *   weight, group and offset on first batch, which is not considered in the function.
 *
 * \param num_rows     Number of rows in this worker.
 * \param num_columns  Number of columns for this dataset.
 * \param nnz          Number of non-zero element.  Put in something greater than rows *
 *                     cols if nnz is unknown.
 * \param num_bins     Number of histogram bins.
 * \param with_weights Whether weight is used, works the same for ranking and other models.
 *
 * \return The estimated bytes
 */
size_t RequiredMemory(bst_row_t num_rows, bst_feature_t num_columns, size_t nnz,
                      size_t num_bins, bool with_weights);

// Count the valid entries in each column and copy them out.
template <typename AdapterBatch, typename BatchIter>
void MakeEntriesFromAdapter(AdapterBatch const& batch, BatchIter batch_iter,
                            Range1d range, float missing,
                            size_t columns, size_t cuts_per_feature, int device,
                            HostDeviceVector<SketchContainer::OffsetT>* cut_sizes_scan,
                            dh::caching_device_vector<size_t>* column_sizes_scan,
                            dh::device_vector<Entry>* sorted_entries) {
  auto entry_iter = dh::MakeTransformIterator<Entry>(
      thrust::make_counting_iterator(0llu), [=] __device__(size_t idx) {
        return Entry(batch.GetElement(idx).column_idx,
                     batch.GetElement(idx).value);
      });
  data::IsValidFunctor is_valid(missing);
  // Work out how many valid entries we have in each column
  GetColumnSizesScan(device, columns, cuts_per_feature,
                     batch_iter, is_valid,
                     range.begin(), range.end(),
                     cut_sizes_scan,
                     column_sizes_scan);
  size_t num_valid = column_sizes_scan->back();
  // Copy current subset of valid elements into temporary storage and sort
  sorted_entries->resize(num_valid);
  dh::CopyIf(entry_iter + range.begin(), entry_iter + range.end(),
             sorted_entries->begin(), is_valid);
}

void SortByWeight(dh::device_vector<float>* weights,
                  dh::device_vector<Entry>* sorted_entries);

void RemoveDuplicatedCategories(
    int32_t device, MetaInfo const &info, Span<bst_row_t> d_cuts_ptr,
    dh::device_vector<Entry> *p_sorted_entries,
    dh::caching_device_vector<size_t> *p_column_sizes_scan);
}  // namespace detail

// Compute sketch on DMatrix.
// sketch_batch_num_elements 0 means autodetect. Only modify this for testing.
HistogramCuts DeviceSketch(int device, DMatrix* dmat, int max_bins,
                           size_t sketch_batch_num_elements = 0);

template <typename AdapterBatch>
void ProcessSlidingWindow(AdapterBatch const &batch, MetaInfo const &info,
                          int device, size_t columns, size_t begin, size_t end,
                          float missing, SketchContainer *sketch_container,
                          int num_cuts) {
  // Copy current subset of valid elements into temporary storage and sort
  dh::device_vector<Entry> sorted_entries;
  dh::caching_device_vector<size_t> column_sizes_scan;
  auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(
      thrust::make_counting_iterator(0llu),
      [=] __device__(size_t idx) { return batch.GetElement(idx); });
  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;
  cuts_ptr.SetDevice(device);
  detail::MakeEntriesFromAdapter(batch, batch_iter, {begin, end}, missing,
                                 columns, num_cuts, device,
                                 &cuts_ptr,
                                 &column_sizes_scan,
                                 &sorted_entries);
  dh::XGBDeviceAllocator<char> alloc;
  thrust::sort(thrust::cuda::par(alloc), sorted_entries.begin(),
               sorted_entries.end(), detail::EntryCompareOp());

  if (sketch_container->HasCategorical()) {
    auto d_cuts_ptr = cuts_ptr.DeviceSpan();
    detail::RemoveDuplicatedCategories(device, info, d_cuts_ptr,
                                       &sorted_entries, &column_sizes_scan);
  }

  auto d_cuts_ptr = cuts_ptr.DeviceSpan();
  auto const &h_cuts_ptr = cuts_ptr.HostVector();
  // Extract the cuts from all columns concurrently
  sketch_container->Push(dh::ToSpan(sorted_entries),
                         dh::ToSpan(column_sizes_scan), d_cuts_ptr,
                         h_cuts_ptr.back());
  sorted_entries.clear();
  sorted_entries.shrink_to_fit();
}

template <typename Batch>
void ProcessWeightedSlidingWindow(Batch batch, MetaInfo const& info,
                                  int num_cuts_per_feature,
                                  bool is_ranking, float missing, int device,
                                  size_t columns, size_t begin, size_t end,
                                  SketchContainer *sketch_container) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  dh::safe_cuda(cudaSetDevice(device));
  info.weights_.SetDevice(device);
  auto weights = info.weights_.ConstDeviceSpan();

  auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(
    thrust::make_counting_iterator(0llu),
    [=] __device__(size_t idx) { return batch.GetElement(idx); });
  dh::device_vector<Entry> sorted_entries;
  dh::caching_device_vector<size_t> column_sizes_scan;
  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;
  detail::MakeEntriesFromAdapter(batch, batch_iter,
                                 {begin, end}, missing,
                                 columns, num_cuts_per_feature, device,
                                 &cuts_ptr,
                                 &column_sizes_scan,
                                 &sorted_entries);
  data::IsValidFunctor is_valid(missing);

  dh::device_vector<float> temp_weights(sorted_entries.size());
  auto d_temp_weights = dh::ToSpan(temp_weights);

  if (is_ranking) {
    if (!weights.empty()) {
      CHECK_EQ(weights.size(), info.group_ptr_.size() - 1);
    }
    dh::caching_device_vector<bst_group_t> group_ptr(info.group_ptr_);
    auto d_group_ptr = dh::ToSpan(group_ptr);
    auto const weight_iter = dh::MakeTransformIterator<float>(
        thrust::make_counting_iterator(0lu), [=] __device__(size_t idx) -> float {
          auto ridx = batch.GetElement(idx).row_idx;
          bst_group_t group_idx = dh::SegmentId(d_group_ptr, ridx);
          return weights[group_idx];
        });
    auto retit = thrust::copy_if(thrust::cuda::par(alloc),
                                 weight_iter + begin, weight_iter + end,
                                 batch_iter + begin,
                                 d_temp_weights.data(),  // output
                                 is_valid);
    CHECK_EQ(retit - d_temp_weights.data(), d_temp_weights.size());
  } else {
    CHECK_EQ(batch.NumRows(), weights.size());
    auto const weight_iter = dh::MakeTransformIterator<float>(
        thrust::make_counting_iterator(0lu),
        [=]__device__(size_t idx) -> float {
          return weights[batch.GetElement(idx).row_idx];
        });
    auto retit = thrust::copy_if(thrust::cuda::par(alloc),
                                 weight_iter + begin, weight_iter + end,
                                 batch_iter + begin,
                                 d_temp_weights.data(),  // output
                                 is_valid);
    CHECK_EQ(retit - d_temp_weights.data(), d_temp_weights.size());
  }

  detail::SortByWeight(&temp_weights, &sorted_entries);

  if (sketch_container->HasCategorical()) {
    auto d_cuts_ptr = cuts_ptr.DeviceSpan();
    detail::RemoveDuplicatedCategories(device, info, d_cuts_ptr,
                                       &sorted_entries, &column_sizes_scan);
  }

  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();
  auto d_cuts_ptr = cuts_ptr.DeviceSpan();

  // Extract cuts
  sketch_container->Push(dh::ToSpan(sorted_entries),
                         dh::ToSpan(column_sizes_scan), d_cuts_ptr,
                         h_cuts_ptr.back(), dh::ToSpan(temp_weights));
  sorted_entries.clear();
  sorted_entries.shrink_to_fit();
}

/*
 * \brief Perform sketching on GPU.
 *
 * \param batch            A batch from adapter.
 * \param num_bins         Bins per column.
 * \param info             Metainfo used for sketching.
 * \param missing          Floating point value that represents invalid value.
 * \param sketch_container Container for output sketch.
 * \param sketch_batch_num_elements Number of element per-sliding window, use it only for
 *                                  testing.
 */
template <typename Batch>
void AdapterDeviceSketch(Batch batch, int num_bins,
                         MetaInfo const& info,
                         float missing, SketchContainer* sketch_container,
                         size_t sketch_batch_num_elements = 0) {
  size_t num_rows = batch.NumRows();
  size_t num_cols = batch.NumCols();
  size_t num_cuts_per_feature = detail::RequiredSampleCutsPerColumn(num_bins, num_rows);
  int32_t device = sketch_container->DeviceIdx();
  bool weighted = !info.weights_.Empty();

  if (weighted) {
    sketch_batch_num_elements = detail::SketchBatchNumElements(
        sketch_batch_num_elements,
        num_rows, num_cols, std::numeric_limits<size_t>::max(),
        device, num_cuts_per_feature, true);
    for (auto begin = 0ull; begin < batch.Size(); begin += sketch_batch_num_elements) {
      size_t end =
          std::min(batch.Size(), static_cast<std::size_t>(begin + sketch_batch_num_elements));
      ProcessWeightedSlidingWindow(batch, info,
                                   num_cuts_per_feature,
                                   HostSketchContainer::UseGroup(info), missing, device, num_cols, begin, end,
                                   sketch_container);
    }
  } else {
    sketch_batch_num_elements = detail::SketchBatchNumElements(
        sketch_batch_num_elements,
        num_rows, num_cols, std::numeric_limits<size_t>::max(),
        device, num_cuts_per_feature, false);
    for (auto begin = 0ull; begin < batch.Size(); begin += sketch_batch_num_elements) {
      size_t end =
          std::min(batch.Size(), static_cast<std::size_t>(begin + sketch_batch_num_elements));
      ProcessSlidingWindow(batch, info, device, num_cols, begin, end, missing,
                           sketch_container, num_cuts_per_feature);
    }
  }
}
}      // namespace common
}      // namespace xgboost

#endif  // COMMON_HIST_UTIL_CUH_
