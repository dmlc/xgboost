/*!
 * Copyright 2020 XGBoost contributors
 */
#ifndef COMMON_HIST_UTIL_CUH_
#define COMMON_HIST_UTIL_CUH_

#include <thrust/host_vector.h>

#include "hist_util.h"
#include "quantile.cuh"
#include "device_helpers.cuh"
#include "timer.h"
#include "../data/device_adapter.cuh"

namespace xgboost {
namespace common {

namespace detail {
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
 * \param cuts_ptr              Column pointers to CSC structured cuts
 * \param sorted_data           Sorted entries in segments of columns
 * \param column_sizes_scan     Describes the boundaries of column segments in sorted data
 * \param out_cuts              Output cut values
 */
void ExtractCutsSparse(int device, common::Span<size_t const> cuts_ptr,
                       Span<Entry const> sorted_data,
                       Span<size_t const> column_sizes_scan,
                       Span<SketchEntry> out_cuts);

/**
 * \brief Extracts the cuts from sorted data, considering weights.
 *
 * \param device                The device.
 * \param cuts_ptr              Column pointers to CSC structured cuts
 * \param sorted_data           Sorted entries in segments of columns.
 * \param weights_scan          Inclusive scan of weights for each entry in sorted_data.
 * \param column_sizes_scan     Describes the boundaries of column segments in sorted data.
 * \param cuts                  Output cuts.
 */
void ExtractWeightedCutsSparse(int device,
                               common::Span<size_t const> cuts_ptr,
                               Span<Entry> sorted_data,
                               Span<float> weights_scan,
                               Span<size_t> column_sizes_scan,
                               Span<SketchEntry> cuts);

// Get column size For adapter.
template <typename Iter>
void GetColumnSizesScan(int device, size_t num_columns, size_t num_cuts_per_feature,
                        Iter batch_iter, data::IsValidFunctor is_valid,
                        size_t begin, size_t end,
                        HostDeviceVector<size_t> *cuts_ptr,
                        dh::caching_device_vector<size_t>* column_sizes_scan) {
  column_sizes_scan->resize(num_columns + 1, 0);
  cuts_ptr->SetDevice(device);
  cuts_ptr->Resize(num_columns + 1, 0);

  dh::XGBCachingDeviceAllocator<char> alloc;
  auto d_column_sizes_scan = column_sizes_scan->data().get();
  dh::LaunchN(device, end - begin, [=] __device__(size_t idx) {
    auto e = batch_iter[begin + idx];
    if (is_valid(e)) {
      atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
                    &d_column_sizes_scan[e.column_idx]),
                static_cast<unsigned long long>(1));  // NOLINT
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

/* \brief Calcuate the length of sliding window. */
size_t SketchBatchNumElements(size_t sketch_batch_num_elements,
                              bst_row_t num_rows, size_t columns, size_t nnz, int device,
                              size_t num_cuts, bool has_weight);

// Compute number of sample cuts needed on local node to maintain accuracy
// We take more cuts than needed and then reduce them later
size_t RequiredSampleCutsPerColumn(int max_bins, size_t num_rows);
// Compute required memory for each sliding window.
size_t RequiredMemory(bst_row_t num_rows, bst_feature_t num_columns, size_t nnz,
                      size_t num_bins, bool with_weights);

template <typename AdapterBatch, typename BatchIter>
void MakeEntriesFromAdapter(AdapterBatch const& batch, BatchIter batch_iter,
                            Range1d range, float missing,
                            size_t columns, size_t cuts_per_feature, int device,
                            HostDeviceVector<size_t>* cut_sizes_scan,
                            dh::caching_device_vector<size_t>* column_sizes_scan,
                            dh::caching_device_vector<Entry>* sorted_entries) {
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
  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::copy_if(thrust::cuda::par(alloc), entry_iter + range.begin(),
                  entry_iter + range.end(), sorted_entries->begin(), is_valid);
}

void SortByWeight(dh::XGBCachingDeviceAllocator<char>* alloc,
                  dh::caching_device_vector<float>* weights,
                  dh::caching_device_vector<Entry>* sorted_entries);
}  // namespace detail

// Compute sketch on DMatrix.
// sketch_batch_num_elements 0 means autodetect. Only modify this for testing.
HistogramCuts DeviceSketch(int device, DMatrix* dmat, int max_bins,
                           size_t sketch_batch_num_elements = 0);

template <typename AdapterBatch>
void ProcessSlidingWindow(AdapterBatch const& batch, int device, size_t columns,
                          size_t begin, size_t end, float missing,
                          SketchContainer* sketch_container, int num_cuts) {
  // Copy current subset of valid elements into temporary storage and sort
  dh::caching_device_vector<Entry> sorted_entries;
  dh::caching_device_vector<size_t> column_sizes_scan;
  auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(
      thrust::make_counting_iterator(0llu),
      [=] __device__(size_t idx) { return batch.GetElement(idx); });
  HostDeviceVector<size_t> cuts_ptr;
  detail::MakeEntriesFromAdapter(batch, batch_iter, {begin, end}, missing,
                                 columns, num_cuts, device,
                                 &cuts_ptr,
                                 &column_sizes_scan,
                                 &sorted_entries);
  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::sort(thrust::cuda::par(alloc), sorted_entries.begin(),
               sorted_entries.end(), detail::EntryCompareOp());

  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();
  auto d_cuts_ptr = cuts_ptr.ConstDeviceSpan();
  dh::caching_device_vector<SketchEntry> cuts(h_cuts_ptr.back());
  // Extract the cuts from all columns concurrently
  detail::ExtractCutsSparse(device, d_cuts_ptr,
                            dh::ToSpan(sorted_entries),
                            dh::ToSpan(column_sizes_scan),
                            dh::ToSpan(cuts));
  sorted_entries.clear();
  sorted_entries.shrink_to_fit();

  // Push cuts into sketches stored in host memory
  sketch_container->Push(cuts_ptr.ConstDeviceSpan(), &cuts);
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
  dh::caching_device_vector<bst_group_t> group_ptr(info.group_ptr_);
  auto d_group_ptr = dh::ToSpan(group_ptr);

  auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(
    thrust::make_counting_iterator(0llu),
    [=] __device__(size_t idx) { return batch.GetElement(idx); });
  dh::caching_device_vector<Entry> sorted_entries;
  dh::caching_device_vector<size_t> column_sizes_scan;
  HostDeviceVector<size_t> cuts_ptr;
  detail::MakeEntriesFromAdapter(batch, batch_iter,
                                 {begin, end}, missing,
                                 columns, num_cuts_per_feature, device,
                                 &cuts_ptr,
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

  detail::SortByWeight(&alloc, &temp_weights, &sorted_entries);

  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();
  auto d_cuts_ptr = cuts_ptr.ConstDeviceSpan();

  // Extract cuts
  dh::caching_device_vector<SketchEntry> cuts(h_cuts_ptr.back());
  detail::ExtractWeightedCutsSparse(device, d_cuts_ptr,
                                    dh::ToSpan(sorted_entries),
                                    dh::ToSpan(temp_weights),
                                    dh::ToSpan(column_sizes_scan),
                                    dh::ToSpan(cuts));
  sorted_entries.clear();
  sorted_entries.shrink_to_fit();
  // add cuts into sketches
  sketch_container->Push(cuts_ptr.ConstDeviceSpan(), &cuts);
}

template <typename AdapterT>
HistogramCuts AdapterDeviceSketch(AdapterT* adapter, int num_bins,
                                  float missing,
                                  size_t sketch_batch_num_elements = 0) {
  size_t num_cuts_per_feature = detail::RequiredSampleCutsPerColumn(num_bins, adapter->NumRows());
  CHECK(adapter->NumRows() != data::kAdapterUnknownSize);
  CHECK(adapter->NumColumns() != data::kAdapterUnknownSize);

  adapter->BeforeFirst();
  adapter->Next();
  auto& batch = adapter->Value();
  sketch_batch_num_elements = detail::SketchBatchNumElements(
      sketch_batch_num_elements,
      adapter->NumRows(), adapter->NumColumns(), std::numeric_limits<size_t>::max(),
      adapter->DeviceIdx(),
      num_cuts_per_feature, false);

  // Enforce single batch
  CHECK(!adapter->Next());

  HistogramCuts cuts;
  SketchContainer sketch_container(num_bins, adapter->NumColumns(),
                                   adapter->NumRows(), adapter->DeviceIdx());

  for (auto begin = 0ull; begin < batch.Size(); begin += sketch_batch_num_elements) {
    size_t end = std::min(batch.Size(), size_t(begin + sketch_batch_num_elements));
    auto const& batch = adapter->Value();
    ProcessSlidingWindow(batch, adapter->DeviceIdx(), adapter->NumColumns(),
                         begin, end, missing, &sketch_container, num_cuts_per_feature);
  }

  sketch_container.MakeCuts(&cuts);
  return cuts;
}

template <typename Batch>
void AdapterDeviceSketch(Batch batch, int num_bins,
                         float missing, int device,
                         SketchContainer* sketch_container,
                         size_t sketch_batch_num_elements = 0) {
  size_t num_rows = batch.NumRows();
  size_t num_cols = batch.NumCols();
  size_t num_cuts_per_feature = detail::RequiredSampleCutsPerColumn(num_bins, num_rows);
  sketch_batch_num_elements = detail::SketchBatchNumElements(
      sketch_batch_num_elements,
      num_rows, num_cols, std::numeric_limits<size_t>::max(),
      device, num_cuts_per_feature, false);
  for (auto begin = 0ull; begin < batch.Size(); begin += sketch_batch_num_elements) {
    size_t end = std::min(batch.Size(), size_t(begin + sketch_batch_num_elements));
    ProcessSlidingWindow(batch, device, num_cols,
                         begin, end, missing, sketch_container, num_cuts_per_feature);
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
  size_t num_cuts_per_feature = detail::RequiredSampleCutsPerColumn(num_bins, num_rows);
  sketch_batch_num_elements = detail::SketchBatchNumElements(
      sketch_batch_num_elements,
      num_rows, num_cols, std::numeric_limits<size_t>::max(),
      device, num_cuts_per_feature, true);
  for (auto begin = 0ull; begin < batch.Size(); begin += sketch_batch_num_elements) {
    size_t end = std::min(batch.Size(), size_t(begin + sketch_batch_num_elements));
    ProcessWeightedSlidingWindow(batch, info,
                                 num_cuts_per_feature,
                                 CutsBuilder::UseGroup(info), missing, device, num_cols, begin, end,
                                 sketch_container);
  }
}
}      // namespace common
}      // namespace xgboost

#endif  // COMMON_HIST_UTIL_CUH_