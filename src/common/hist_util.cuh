/*!
 * Copyright 2020 XGBoost contributors
 *
 * \brief Front end and utilities for GPU based sketching.  Works on sliding window
 *        instead of stream.
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

template <typename Batch>
struct SortIdxOp {
  Batch batch;
  bool XGBOOST_DEVICE operator()(uint32_t l, uint32_t r) {
    auto le = batch.GetElement(l);
    auto re = batch.GetElement(r);
    // sort between columns
    if (le.column_idx != re.column_idx) {
      return le.column_idx < re.column_idx;
    }
    // sort within columns
    return le.value < re.value;
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
  return (has_weight ? sizeof(uint32_t) + sizeof(float) : sizeof(uint32_t)) * 2;
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
void MakeEntriesFromAdapter(AdapterBatch const& batch, BatchIter batch_iter, Range1d range,
                            float missing, size_t columns, size_t cuts_per_feature, int device,
                            HostDeviceVector<SketchContainer::OffsetT>* cut_sizes_scan,
                            dh::caching_device_vector<size_t>* column_sizes_scan,
                            dh::device_vector<uint32_t>* sorted_idx) {
  data::IsValidFunctor is_valid(missing);
  // Work out how many valid entries we have in each column
  GetColumnSizesScan(device, columns, cuts_per_feature, batch_iter, is_valid, range.begin(),
                     range.end(), cut_sizes_scan, column_sizes_scan);
  size_t num_valid = column_sizes_scan->back();
  // Copy current subset of valid elements into temporary storage and sort
  dh::XGBCachingDeviceAllocator<char> alloc;
  auto entry_iter = thrust::make_counting_iterator<uint32_t>(0u);
  sorted_idx->resize(num_valid);
  auto copyif_op = [=] XGBOOST_DEVICE(uint32_t idx) {
    auto check = data::IsValidFunctor{missing};
    return check(batch.GetElement(idx).value);
  };
  dh::CopyIf(entry_iter + range.begin(), entry_iter + range.end(), sorted_idx->begin(), copyif_op);
}

template <typename Batch>
void RemoveDuplicatedCategories(Batch batch, MetaInfo const& info, Span<bst_row_t> d_cuts_ptr,
                                  dh::device_vector<uint32_t>* p_sorted_idx,
                                  dh::caching_device_vector<size_t>* p_column_sizes_scan) {
  auto d_feature_types = info.feature_types.ConstDeviceSpan();
  CHECK(!d_feature_types.empty());
  auto& column_sizes_scan = *p_column_sizes_scan;
  auto& sorted_idx = *p_sorted_idx;
  // Removing duplicated entries in categorical features.
  dh::caching_device_vector<size_t> new_column_scan(column_sizes_scan.size());
  dh::SegmentedUnique(column_sizes_scan.data().get(),
                      column_sizes_scan.data().get() + column_sizes_scan.size(), sorted_idx.begin(),
                      sorted_idx.end(), new_column_scan.data().get(), sorted_idx.begin(),
                      [=] __device__(uint32_t const& l, uint32_t const& r) {
                        auto const& le = batch.GetElement(l);
                        auto const& re = batch.GetElement(r);
                        if (le.column_idx == batch.GetElement(r).column_idx) {
                          if (IsCat(d_feature_types, le.column_idx)) {
                            return le.value == re.value;
                          }
                        }
                        return false;
                      });

  // Renew the column scan and cut scan based on categorical data.
  auto d_old_column_sizes_scan = dh::ToSpan(column_sizes_scan);
  dh::caching_device_vector<SketchContainer::OffsetT> new_cuts_size(info.num_col_ + 1);
  CHECK_EQ(new_column_scan.size(), new_cuts_size.size());
  dh::LaunchN(new_column_scan.size(),
              [=, d_new_cuts_size = dh::ToSpan(new_cuts_size),
               d_old_column_sizes_scan = dh::ToSpan(column_sizes_scan),
               d_new_columns_ptr = dh::ToSpan(new_column_scan)] __device__(size_t idx) {
                d_old_column_sizes_scan[idx] = d_new_columns_ptr[idx];
                if (idx == d_new_columns_ptr.size() - 1) {
                  return;
                }
                if (IsCat(d_feature_types, idx)) {
                  // Cut size is the same as number of categories in input.
                  d_new_cuts_size[idx] = d_new_columns_ptr[idx + 1] - d_new_columns_ptr[idx];
                } else {
                  d_new_cuts_size[idx] = d_cuts_ptr[idx + 1] - d_cuts_ptr[idx];
                }
              });
  // Turn size into ptr.
  thrust::exclusive_scan(thrust::device, new_cuts_size.cbegin(), new_cuts_size.cend(),
                         d_cuts_ptr.data());
}

template <typename Batch>
void SortWithWeight(Batch batch, Span<uint32_t> sorted_idx, Span<float> weights);
}  // namespace detail

// Compute sketch on DMatrix.
// sketch_batch_num_elements 0 means autodetect. Only modify this for testing.
HistogramCuts DeviceSketch(int device, DMatrix* dmat, int max_bins,
                           size_t sketch_batch_num_elements = 0);

template <typename AdapterBatch>
void ProcessSlidingWindow(AdapterBatch const& batch, MetaInfo const& info, int device,
                          size_t columns, size_t begin, size_t end, float missing,
                          SketchContainer* sketch_container, int num_cuts) {
  dh::safe_cuda(cudaSetDevice(device));
  // Copy current subset of valid elements into temporary storage and sort
  dh::caching_device_vector<size_t> column_sizes_scan;
  auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(
      thrust::make_counting_iterator(0llu),
      [=] __device__(size_t idx) { return batch.GetElement(idx); });
  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;
  cuts_ptr.SetDevice(device);
  dh::device_vector<uint32_t> sorted_idx;
  detail::MakeEntriesFromAdapter(batch, batch_iter, {begin, end}, missing, columns, num_cuts,
                                 device, &cuts_ptr, &column_sizes_scan, &sorted_idx);
  dh::XGBDeviceAllocator<char> alloc;
  thrust::sort(thrust::cuda::par(alloc), sorted_idx.begin(), sorted_idx.end(),
               detail::SortIdxOp<AdapterBatch>{batch});

  if (sketch_container->HasCategorical()) {
    auto d_cuts_ptr = cuts_ptr.DeviceSpan();
    info.feature_types.SetDevice(device);
    detail::RemoveDuplicatedCategories(batch, info, d_cuts_ptr, &sorted_idx, &column_sizes_scan);
  }

  auto d_cuts_ptr = cuts_ptr.DeviceSpan();
  auto const& h_cuts_ptr = cuts_ptr.HostVector();
  // Extract the cuts from all columns concurrently
  sketch_container->Push(batch, dh::ToSpan(sorted_idx), dh::ToSpan(column_sizes_scan), d_cuts_ptr,
                         h_cuts_ptr.back());
  sorted_idx.clear();
  sorted_idx.shrink_to_fit();
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
  dh::device_vector<uint32_t> sorted_idx;
  dh::caching_device_vector<size_t> column_sizes_scan;
  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;
  detail::MakeEntriesFromAdapter(batch, batch_iter, {begin, end}, missing, columns,
                                 num_cuts_per_feature, device, &cuts_ptr, &column_sizes_scan,
                                 &sorted_idx);
  data::IsValidFunctor is_valid(missing);

  dh::device_vector<float> temp_weights(sorted_idx.size());
  auto d_temp_weights = dh::ToSpan(temp_weights);

  if (is_ranking) {
    auto const weight_iter = dh::MakeTransformIterator<float>(
        thrust::make_constant_iterator(0lu),
        [=]__device__(size_t idx) -> float {
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

  detail::SortWithWeight(batch, dh::ToSpan(sorted_idx), dh::ToSpan(temp_weights));

  if (sketch_container->HasCategorical()) {
    auto d_cuts_ptr = cuts_ptr.DeviceSpan();
    info.feature_types.SetDevice(device);
    detail::RemoveDuplicatedCategories(batch, info, d_cuts_ptr, &sorted_idx, &column_sizes_scan);
  }

  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();
  auto d_cuts_ptr = cuts_ptr.DeviceSpan();

  // Extract cuts
  sketch_container->Push(batch, dh::ToSpan(sorted_idx), dh::ToSpan(column_sizes_scan), d_cuts_ptr,
                         h_cuts_ptr.back(), dh::ToSpan(temp_weights));
  sorted_idx.clear();
  sorted_idx.shrink_to_fit();
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
  bool weighted = info.weights_.Size() != 0;

  if (weighted) {
    sketch_batch_num_elements = detail::SketchBatchNumElements(
        sketch_batch_num_elements,
        num_rows, num_cols, std::numeric_limits<size_t>::max(),
        device, num_cuts_per_feature, true);
    for (auto begin = 0ull; begin < batch.Size(); begin += sketch_batch_num_elements) {
      size_t end = std::min(batch.Size(), size_t(begin + sketch_batch_num_elements));
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
      size_t end = std::min(batch.Size(), size_t(begin + sketch_batch_num_elements));
      ProcessSlidingWindow(batch, info, device, num_cols, begin, end, missing,
                           sketch_container, num_cuts_per_feature);
    }
  }
}
}      // namespace common
}      // namespace xgboost

#endif  // COMMON_HIST_UTIL_CUH_
