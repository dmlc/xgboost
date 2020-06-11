/*!
 * Copyright 2020 XGBoost contributors
 */
#ifndef COMMON_HIST_UTIL_CUH_
#define COMMON_HIST_UTIL_CUH_

#include <thrust/host_vector.h>

#include "hist_util.h"
#include "threading_utils.h"
#include "device_helpers.cuh"
#include "../data/device_adapter.cuh"

namespace xgboost {
namespace common {

using WQSketch = DenseCuts::WQSketch;
using SketchEntry = WQSketch::Entry;

/*!
 * \brief A container that holds the device sketches across all
 *  sparse page batches which are distributed to different devices.
 *  As sketches are aggregated by column, the mutex guards
 *  multiple devices pushing sketch summary for the same column
 *  across distinct rows.
 */
struct SketchContainer {
  std::vector<DenseCuts::WQSketch> sketches_;  // NOLINT
  static constexpr int kOmpNumColsParallelizeLimit = 1000;
  static constexpr float kFactor = 8;

  SketchContainer(int max_bin, size_t num_columns, size_t num_rows) {
    // Initialize Sketches for this dmatrix
    sketches_.resize(num_columns);
#pragma omp parallel for schedule(static) if (num_columns > kOmpNumColsParallelizeLimit)  // NOLINT
    for (int icol = 0; icol < num_columns; ++icol) {                 // NOLINT
      sketches_[icol].Init(num_rows, 1.0 / (8 * max_bin));
    }
  }

  /**
   * \brief Pushes cuts to the sketches.
   *
   * \param entries_per_column  The entries per column.
   * \param entries             Vector of cuts from all columns, length
   * entries_per_column * num_columns. \param column_scan         Exclusive scan
   * of column sizes. Used to detect cases where there are fewer entries than we
   * have storage for.
   */
  void Push(size_t entries_per_column,
            const thrust::host_vector<SketchEntry>& entries,
            const thrust::host_vector<size_t>& column_scan) {
#pragma omp parallel for schedule(static) if (sketches_.size() > SketchContainer::kOmpNumColsParallelizeLimit)  // NOLINT
    for (int icol = 0; icol < sketches_.size(); ++icol) {
      size_t column_size = column_scan[icol + 1] - column_scan[icol];
      if (column_size == 0) continue;
      WQuantileSketch<bst_float, bst_float>::SummaryContainer summary;
      size_t num_available_cuts =
          std::min(size_t(entries_per_column), column_size);
      summary.Reserve(num_available_cuts);
      summary.MakeFromSorted(&entries[entries_per_column * icol],
                             num_available_cuts);

      sketches_[icol].PushSummary(summary);
    }
  }

  // Prevent copying/assigning/moving this as its internals can't be
  // assigned/copied/moved
  SketchContainer(const SketchContainer&) = delete;
  SketchContainer(const SketchContainer&&) = delete;
  SketchContainer& operator=(const SketchContainer&) = delete;
  SketchContainer& operator=(const SketchContainer&&) = delete;
};

struct EntryCompareOp {
  __device__ bool operator()(const Entry& a, const Entry& b) {
    if (a.index == b.index) {
      return a.fvalue < b.fvalue;
    }
    return a.index < b.index;
  }
};

/**
 * \brief Extracts the cuts from sorted data.
 *
 * \param device                The device.
 * \param cuts                  Output cuts
 * \param num_cuts_per_feature  Number of cuts per feature.
 * \param sorted_data           Sorted entries in segments of columns
 * \param column_sizes_scan     Describes the boundaries of column segments in
 * sorted data
 */
void ExtractCuts(int device,
                 size_t num_cuts_per_feature,
                 Span<Entry const> sorted_data,
                 Span<size_t const> column_sizes_scan,
                 Span<SketchEntry> out_cuts);

// Count the entries in each column and exclusive scan
inline void GetColumnSizesScan(int device,
                               dh::caching_device_vector<size_t>* column_sizes_scan,
                               Span<const Entry> entries, size_t num_columns) {
  column_sizes_scan->resize(num_columns + 1, 0);
  auto d_column_sizes_scan = column_sizes_scan->data().get();
  auto d_entries = entries.data();
  dh::LaunchN(device, entries.size(), [=] __device__(size_t idx) {
    auto& e = d_entries[idx];
    atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
                  &d_column_sizes_scan[e.index]),
              static_cast<unsigned long long>(1));  // NOLINT
  });
  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::exclusive_scan(thrust::cuda::par(alloc), column_sizes_scan->begin(),
                         column_sizes_scan->end(), column_sizes_scan->begin());
}

// For adapter.
template <typename Iter>
void GetColumnSizesScan(int device, size_t num_columns,
                        Iter batch_iter, data::IsValidFunctor is_valid,
                        size_t begin, size_t end,
                        dh::caching_device_vector<size_t>* column_sizes_scan) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  column_sizes_scan->resize(num_columns + 1, 0);
  auto d_column_sizes_scan = column_sizes_scan->data().get();
  dh::LaunchN(device, end - begin, [=] __device__(size_t idx) {
    auto e = batch_iter[begin + idx];
    if (is_valid(e)) {
      atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
                    &d_column_sizes_scan[e.column_idx]),
                static_cast<unsigned long long>(1));  // NOLINT
    }
  });
  thrust::exclusive_scan(thrust::cuda::par(alloc), column_sizes_scan->begin(),
                         column_sizes_scan->end(), column_sizes_scan->begin());
}

inline size_t BytesPerElement(bool has_weight) {
  // Double the memory usage for sorting.  We need to assign weight for each element, so
  // sizeof(float) is added to all elements.
  return (has_weight ? sizeof(Entry) + sizeof(float) : sizeof(Entry)) * 2;
}

inline size_t SketchBatchNumElements(size_t sketch_batch_num_elements,
                                     size_t columns, int device,
                                     size_t num_cuts, bool has_weight) {
  if (sketch_batch_num_elements == 0) {
    size_t bytes_per_element = BytesPerElement(has_weight);
    size_t bytes_cuts = num_cuts * columns * sizeof(SketchEntry);
    size_t bytes_num_columns = (columns + 1) * sizeof(size_t);
    // use up to 80% of available space
    sketch_batch_num_elements = (dh::AvailableMemory(device) -
                                 bytes_cuts - bytes_num_columns) *
                                0.8 / bytes_per_element;
  }
  return sketch_batch_num_elements;
}


// Compute number of sample cuts needed on local node to maintain accuracy
// We take more cuts than needed and then reduce them later
inline size_t RequiredSampleCuts(int max_bins, size_t num_rows) {
  double eps = 1.0 / (SketchContainer::kFactor * max_bins);
  size_t dummy_nlevel;
  size_t num_cuts;
  WQuantileSketch<bst_float, bst_float>::LimitSizeLevel(
      num_rows, eps, &dummy_nlevel, &num_cuts);
  return std::min(num_cuts, num_rows);
}

// sketch_batch_num_elements 0 means autodetect. Only modify this for testing.
HistogramCuts DeviceSketch(int device, DMatrix* dmat, int max_bins,
                           size_t sketch_batch_num_elements = 0);


template <typename AdapterBatch, typename BatchIter>
void MakeEntriesFromAdapter(AdapterBatch const& batch, BatchIter batch_iter,
                            Range1d range, float missing,
                            size_t columns, int device,
                            thrust::host_vector<size_t>* host_column_sizes_scan,
                            dh::caching_device_vector<size_t>* column_sizes_scan,
                            dh::caching_device_vector<Entry>* sorted_entries) {
  auto entry_iter = dh::MakeTransformIterator<Entry>(
      thrust::make_counting_iterator(0llu), [=] __device__(size_t idx) {
        return Entry(batch.GetElement(idx).column_idx,
                     batch.GetElement(idx).value);
      });
  data::IsValidFunctor is_valid(missing);
  // Work out how many valid entries we have in each column
  GetColumnSizesScan(device, columns,
                     batch_iter, is_valid,
                     range.begin(), range.end(),
                     column_sizes_scan);
  host_column_sizes_scan->resize(column_sizes_scan->size());
  thrust::copy(column_sizes_scan->begin(), column_sizes_scan->end(),
               host_column_sizes_scan->begin());

  size_t num_valid = host_column_sizes_scan->back();

  // Copy current subset of valid elements into temporary storage and sort
  sorted_entries->resize(num_valid);
  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::copy_if(thrust::cuda::par(alloc), entry_iter + range.begin(),
                  entry_iter + range.end(), sorted_entries->begin(), is_valid);
}

template <typename AdapterBatch>
void ProcessSlidingWindow(AdapterBatch const& batch, int device, size_t columns,
                          size_t begin, size_t end, float missing,
                          SketchContainer* sketch_container, int num_cuts) {
  // Copy current subset of valid elements into temporary storage and sort
  dh::caching_device_vector<Entry> sorted_entries;
  dh::caching_device_vector<size_t> column_sizes_scan;
  thrust::host_vector<size_t> host_column_sizes_scan;
  auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(
      thrust::make_counting_iterator(0llu),
      [=] __device__(size_t idx) { return batch.GetElement(idx); });
  MakeEntriesFromAdapter(batch, batch_iter, {begin, end}, missing, columns, device,
                         &host_column_sizes_scan,
                         &column_sizes_scan,
                         &sorted_entries);
  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::sort(thrust::cuda::par(alloc), sorted_entries.begin(),
               sorted_entries.end(), EntryCompareOp());

  // Extract the cuts from all columns concurrently
  dh::caching_device_vector<SketchEntry> cuts(columns * num_cuts);
  ExtractCuts(device, num_cuts,
              dh::ToSpan(sorted_entries),
              dh::ToSpan(column_sizes_scan),
              dh::ToSpan(cuts));

  // Push cuts into sketches stored in host memory
  thrust::host_vector<SketchEntry> host_cuts(cuts);
  sketch_container->Push(num_cuts, host_cuts, host_column_sizes_scan);
}

void ExtractWeightedCuts(int device,
                         size_t num_cuts_per_feature,
                         Span<Entry> sorted_data,
                         Span<float> weights_scan,
                         Span<size_t> column_sizes_scan,
                         Span<SketchEntry> cuts);

void SortByWeight(dh::XGBCachingDeviceAllocator<char>* alloc,
                  dh::caching_device_vector<float>* weights,
                  dh::caching_device_vector<Entry>* sorted_entries);

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
  dh::caching_device_vector<bst_group_t> group_ptr(info.group_ptr_);
  auto d_group_ptr = dh::ToSpan(group_ptr);

  auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(
    thrust::make_counting_iterator(0llu),
    [=] __device__(size_t idx) { return batch.GetElement(idx); });
  dh::caching_device_vector<Entry> sorted_entries;
  dh::caching_device_vector<size_t> column_sizes_scan;
  thrust::host_vector<size_t> host_column_sizes_scan;
  MakeEntriesFromAdapter(batch, batch_iter,
                         {begin, end}, missing, columns, device,
                         &host_column_sizes_scan,
                         &column_sizes_scan,
                         &sorted_entries);
  data::IsValidFunctor is_valid(missing);

  dh::caching_device_vector<float> temp_weights(sorted_entries.size());
  auto d_temp_weights = dh::ToSpan(temp_weights);

  if (is_ranking) {
    auto const weight_iter = dh::MakeTransformIterator<float>(
        thrust::make_constant_iterator(0lu),
        [=]__device__(size_t idx) -> float {
          auto ridx = batch.GetElement(idx).row_idx;
          auto it = thrust::upper_bound(thrust::seq,
                                        d_group_ptr.cbegin(), d_group_ptr.cend(),
                                        ridx) - 1;
          bst_group_t group = thrust::distance(d_group_ptr.cbegin(), it);
          return weights[group];
        });
    auto retit = thrust::copy_if(thrust::cuda::par(alloc),
                                 weight_iter + begin, weight_iter + end,
                                 batch_iter + begin,
                                 d_temp_weights.data(),  // output
                                 is_valid);
    CHECK_EQ(retit - d_temp_weights.data(), d_temp_weights.size());
  } else {
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

  SortByWeight(&alloc, &temp_weights, &sorted_entries);
  // Extract cuts
  dh::caching_device_vector<SketchEntry> cuts(columns * num_cuts_per_feature);
  ExtractWeightedCuts(device, num_cuts_per_feature,
                      dh::ToSpan(sorted_entries),
                      dh::ToSpan(temp_weights),
                      dh::ToSpan(column_sizes_scan),
                      dh::ToSpan(cuts));

  // add cuts into sketches
  thrust::host_vector<SketchEntry> host_cuts(cuts);
  sketch_container->Push(num_cuts_per_feature, host_cuts, host_column_sizes_scan);
}

template <typename AdapterT>
HistogramCuts AdapterDeviceSketch(AdapterT* adapter, int num_bins,
                                  float missing,
                                  size_t sketch_batch_num_elements = 0) {
  size_t num_cuts = RequiredSampleCuts(num_bins, adapter->NumRows());
  CHECK(adapter->NumRows() != data::kAdapterUnknownSize);
  CHECK(adapter->NumColumns() != data::kAdapterUnknownSize);

  adapter->BeforeFirst();
  adapter->Next();
  auto& batch = adapter->Value();
  sketch_batch_num_elements = SketchBatchNumElements(
      sketch_batch_num_elements,
      adapter->NumColumns(), adapter->DeviceIdx(), num_cuts, false);

  // Enforce single batch
  CHECK(!adapter->Next());

  HistogramCuts cuts;
  DenseCuts dense_cuts(&cuts);
  SketchContainer sketch_container(num_bins, adapter->NumColumns(),
                                   adapter->NumRows());

  for (auto begin = 0ull; begin < batch.Size();
       begin += sketch_batch_num_elements) {
    size_t end = std::min(batch.Size(), size_t(begin + sketch_batch_num_elements));
    auto const& batch = adapter->Value();
    ProcessSlidingWindow(batch, adapter->DeviceIdx(), adapter->NumColumns(),
                         begin, end, missing, &sketch_container, num_cuts);
  }

  dense_cuts.Init(&sketch_container.sketches_, num_bins, adapter->NumRows());
  return cuts;
}

template <typename Batch>
void AdapterDeviceSketch(Batch batch, int num_bins,
                         float missing, int device,
                         SketchContainer* sketch_container,
                         size_t sketch_batch_num_elements = 0) {
  size_t num_rows = batch.NumRows();
  size_t num_cols = batch.NumCols();
  size_t num_cuts = RequiredSampleCuts(num_bins, num_rows);
  sketch_batch_num_elements = SketchBatchNumElements(
      sketch_batch_num_elements,
      num_cols, device, num_cuts, false);
  for (auto begin = 0ull; begin < batch.Size(); begin += sketch_batch_num_elements) {
    size_t end = std::min(batch.Size(), size_t(begin + sketch_batch_num_elements));
    ProcessSlidingWindow(batch, device, num_cols,
                         begin, end, missing, sketch_container, num_cuts);
  }
}

template <typename Batch>
void AdapterDeviceSketchWeighted(Batch batch, int num_bins,
                                 MetaInfo const& info,
                                 float missing,
                                 int device,
                                 SketchContainer* sketch_container,
                                 size_t sketch_batch_num_elements = 0) {
  size_t num_rows = batch.NumRows();
  size_t num_cols = batch.NumCols();
  size_t num_cuts = RequiredSampleCuts(num_bins, num_rows);
  sketch_batch_num_elements = SketchBatchNumElements(
      sketch_batch_num_elements,
      num_cols, device, num_cuts, true);
  for (auto begin = 0ull; begin < batch.Size(); begin += sketch_batch_num_elements) {
    size_t end = std::min(batch.Size(), size_t(begin + sketch_batch_num_elements));
    ProcessWeightedSlidingWindow(batch, info,
                                 num_cuts,
                                 CutsBuilder::UseGroup(info), missing, device, num_cols, begin, end,
                                 sketch_container);
  }
}
}      // namespace common
}      // namespace xgboost

#endif  // COMMON_HIST_UTIL_CUH_