/**
 * Copyright 2020-2023 by XGBoost contributors
 *
 * \brief Front end and utilities for GPU based sketching.  Works on sliding window
 *        instead of stream.
 */
#ifndef COMMON_HIST_UTIL_CUH_
#define COMMON_HIST_UTIL_CUH_

#include <thrust/host_vector.h>

#include <cstddef>  // for size_t, byte

#include "../cub_sort/device/device_radix_sort.cuh"
#include "../data/device_adapter.cuh"
#include "cuda_context.cuh"
#include "device_helpers.cuh"
#include "hist_util.h"
#include "quantile.cuh"
#include "timer.h"

namespace xgboost::common {
namespace cuda_impl {
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
}  // namespace cuda_impl

namespace detail {
// Get column size from adapter batch and for output cuts.
template <std::uint32_t kBlockThreads, typename CounterT, typename BatchIt>
__global__ void GetColumnSizeSharedMemKernel(IterSpan<BatchIt> batch_iter,
                                             data::IsValidFunctor is_valid,
                                             Span<std::size_t> out_column_size) {
  extern __shared__ char smem[];

  auto smem_cs_ptr = reinterpret_cast<CounterT*>(smem);

  dh::BlockFill(smem_cs_ptr, out_column_size.size(), 0);

  cub::CTA_SYNC();

  auto n = batch_iter.size();

  for (auto idx : dh::GridStrideRange(static_cast<std::size_t>(0), n)) {
    auto e = batch_iter[idx];
    if (is_valid(e)) {
      atomicAdd(&smem_cs_ptr[e.column_idx], static_cast<CounterT>(1));
    }
  }

  cub::CTA_SYNC();

  auto out_global_ptr = out_column_size;
  for (auto i : dh::BlockStrideRange(static_cast<std::size_t>(0), out_column_size.size())) {
    atomicAdd(&out_global_ptr[i], static_cast<std::size_t>(smem_cs_ptr[i]));
  }
}

template <std::uint32_t kBlockThreads, typename Kernel>
std::uint32_t EstimateGridSize(std::int32_t device, Kernel kernel, std::size_t shared_mem) {
  int n_mps = 0;
  dh::safe_cuda(cudaDeviceGetAttribute(&n_mps, cudaDevAttrMultiProcessorCount, device));
  int n_blocks_per_mp = 0;
  dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n_blocks_per_mp, kernel,
                                                              kBlockThreads, shared_mem));
  std::uint32_t grid_size = n_blocks_per_mp * n_mps;
  return grid_size;
}

/**
 * \brief Get the size of each column. This is a histogram with additional handling of
 *        invalid values.
 *
 * \tparam BatchIt                 Type of input adapter batch.
 * \tparam force_use_global_memory Used for testing. Force global atomic add.
 * \tparam force_use_u64           Used for testing. For u64 as counter in shared memory.
 *
 * \param device     CUDA device ordinal.
 * \param batch_iter Iterator for input data from adapter batch.
 * \param is_valid   Whehter an element is considered as missing.
 * \param out_column_size Output buffer for the size of each column.
 */
template <typename BatchIt, bool force_use_global_memory = false, bool force_use_u64 = false>
void LaunchGetColumnSizeKernel(std::int32_t device, IterSpan<BatchIt> batch_iter,
                               data::IsValidFunctor is_valid, Span<std::size_t> out_column_size) {
  thrust::fill_n(thrust::device, dh::tbegin(out_column_size), out_column_size.size(), 0);

  std::size_t max_shared_memory = dh::MaxSharedMemory(device);
  // Not strictly correct as we should use number of samples to determine the type of
  // counter. However, the sample size is not known due to sliding window on number of
  // elements.
  std::size_t n = batch_iter.size();

  std::size_t required_shared_memory = 0;
  bool use_u32{false};
  if (!force_use_u64 && n < static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    required_shared_memory = out_column_size.size() * sizeof(std::uint32_t);
    use_u32 = true;
  } else {
    required_shared_memory = out_column_size.size() * sizeof(std::size_t);
    use_u32 = false;
  }
  bool use_shared = required_shared_memory <= max_shared_memory && required_shared_memory != 0;

  if (!force_use_global_memory && use_shared) {
    CHECK_NE(required_shared_memory, 0);
    std::uint32_t constexpr kBlockThreads = 512;
    if (use_u32) {
      CHECK(!force_use_u64);
      auto kernel = GetColumnSizeSharedMemKernel<kBlockThreads, std::uint32_t, BatchIt>;
      auto grid_size = EstimateGridSize<kBlockThreads>(device, kernel, required_shared_memory);
      dh::LaunchKernel{grid_size, kBlockThreads, required_shared_memory, dh::DefaultStream()}(
          kernel, batch_iter, is_valid, out_column_size);
    } else {
      auto kernel = GetColumnSizeSharedMemKernel<kBlockThreads, std::size_t, BatchIt>;
      auto grid_size = EstimateGridSize<kBlockThreads>(device, kernel, required_shared_memory);
      dh::LaunchKernel{grid_size, kBlockThreads, required_shared_memory, dh::DefaultStream()}(
          kernel, batch_iter, is_valid, out_column_size);
    }
  } else {
    auto d_out_column_size = out_column_size;
    dh::LaunchN(batch_iter.size(), [=] __device__(size_t idx) {
      auto e = batch_iter[idx];
      if (is_valid(e)) {
        atomicAdd(&d_out_column_size[e.column_idx], static_cast<size_t>(1));
      }
    });
  }
}

template <typename BatchIt>
void GetColumnSizesScan(int device, size_t num_columns, std::size_t num_cuts_per_feature,
                        IterSpan<BatchIt> batch_iter, data::IsValidFunctor is_valid,
                        HostDeviceVector<SketchContainer::OffsetT>* cuts_ptr,
                        dh::caching_device_vector<size_t>* column_sizes_scan) {
  column_sizes_scan->resize(num_columns + 1);
  cuts_ptr->SetDevice(device);
  cuts_ptr->Resize(num_columns + 1, 0);

  dh::XGBCachingDeviceAllocator<char> alloc;
  auto d_column_sizes_scan = dh::ToSpan(*column_sizes_scan);
  LaunchGetColumnSizeKernel(device, batch_iter, is_valid, d_column_sizes_scan);
  // Calculate cuts CSC pointer
  auto cut_ptr_it = dh::MakeTransformIterator<size_t>(
      column_sizes_scan->begin(), [=] __device__(size_t column_size) {
        return thrust::min(num_cuts_per_feature, column_size);
      });
  thrust::exclusive_scan(thrust::cuda::par(alloc), cut_ptr_it,
                         cut_ptr_it + column_sizes_scan->size(), cuts_ptr->DevicePointer());
  thrust::exclusive_scan(thrust::cuda::par(alloc), column_sizes_scan->begin(),
                         column_sizes_scan->end(), column_sizes_scan->begin());
}

/**
 * \brief Type for sorted index.
 */
using SortedIdxT = std::uint32_t;

/**
 * \brief Maximum number of elements for each batch, limited by the type of the sorted index.
 */
inline constexpr std::size_t kMaxNumEntrySort = std::numeric_limits<SortedIdxT>::max();

/**
 * \brief Return sorted index of input entries. KeyIt is an iterator that returns `xgboost::Entry`.
 */
template <typename KeyIt>
void ArgSortEntry(KeyIt key_it, dh::device_vector<SortedIdxT>* p_sorted_idx) {
  auto& sorted_idx = *p_sorted_idx;
  std::size_t n = sorted_idx.size();
  CHECK_LE(n, kMaxNumEntrySort);

  std::size_t bytes{0};
  std::byte* ptr{nullptr};
  cub_argsort::DeviceRadixSort<cub_argsort::EntryExtractor>::Argsort(ptr, bytes, key_it,
                                                                     sorted_idx.data().get(), n);
  dh::device_vector<std::byte> alloc(bytes);
  ptr = alloc.data().get();
  cub_argsort::DeviceRadixSort<cub_argsort::EntryExtractor>::Argsort(ptr, bytes, key_it,
                                                                     sorted_idx.data().get(), n);
}

/**
 * \brief Calcuate the length of sliding window. Returns `sketch_batch_num_elements`
 *        directly if it's not 0.
 */
std::size_t SketchBatchNumElements(size_t sketch_batch_num_elements, bst_row_t num_rows,
                                   bst_feature_t columns, size_t nnz, int device, size_t num_cuts,
                                   bool has_weight, bool d_can_read);

// Compute number of sample cuts needed on local node to maintain accuracy
// We take more cuts than needed and then reduce them later
size_t RequiredSampleCutsPerColumn(int max_bins, size_t num_rows);

/**
 * \brief Estimate required memory for each sliding window.
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
 * \param d_can_read   Whether the device alread has read access to the data.
 *
 * \return The estimated bytes
 */
std::size_t RequiredMemory(bst_row_t num_rows, bst_feature_t num_columns, std::size_t nnz,
                           bst_bin_t num_bins, bool with_weights, bool d_can_read);

/**
 * \brief Count the valid entries in each column and sort them.
 *
 * \param batch_iter       Iterator to data batch, with value_type as data::COOTuple.
 * \param range            Boundary of the current sliding window.
 * \param is_valid         Specify the missing value.
 * \param columns          Number of features.
 * \param cuts_per_feature Number of required cuts for each feature, which is estimated by
 *                         sketching container.
 * \param device           CUDA ordinal.
 * \param p_cut_sizes_scan Output cuts ptr.
 * \param p_column_sizes_scan Output feature ptr.
 * \param p_sorted_idx     Output sorted index of input data (batch_iter).
 */
template <typename BatchIter>
void MakeEntriesFromAdapter(BatchIter batch_iter, Range1d range, data::IsValidFunctor is_valid,
                            std::size_t columns, std::size_t cuts_per_feature, int device,
                            HostDeviceVector<SketchContainer::OffsetT>* p_cut_sizes_scan,
                            dh::caching_device_vector<size_t>* p_column_sizes_scan,
                            dh::device_vector<SortedIdxT>* p_sorted_idx) {
  auto n = range.end() - range.begin();
  auto span = IterSpan{batch_iter + range.begin(), n};
  // Work out how many valid entries we have in each column
  GetColumnSizesScan(device, columns, cuts_per_feature, span, is_valid, p_cut_sizes_scan,
                     p_column_sizes_scan);
  // Sort the current subset of valid elements.
  dh::device_vector<SortedIdxT>& sorted_idx = *p_sorted_idx;
  sorted_idx.resize(span.size());

  std::size_t n_valids = p_column_sizes_scan->back();

  auto key_it = dh::MakeTransformIterator<Entry>(
      span.data(), [=] XGBOOST_DEVICE(data::COOTuple const& tup) -> Entry {
        if (is_valid(tup)) {
          return {static_cast<bst_feature_t>(tup.column_idx), tup.value};
        }
        // Push invalid elements to the end
        return {std::numeric_limits<bst_feature_t>::max(), std::numeric_limits<float>::max()};
      });
  ArgSortEntry(key_it, &sorted_idx);

  sorted_idx.resize(n_valids);
}

template <typename BatchIter>
void RemoveDuplicatedCategories(int32_t device, MetaInfo const& info, Span<bst_row_t> d_cuts_ptr,
                                BatchIter batch_iter, dh::device_vector<SortedIdxT>* p_sorted_idx,
                                dh::caching_device_vector<size_t>* p_column_sizes_scan) {
  info.feature_types.SetDevice(device);
  auto d_feature_types = info.feature_types.ConstDeviceSpan();
  CHECK(!d_feature_types.empty());
  auto& column_sizes_scan = *p_column_sizes_scan;
  // Removing duplicated entries in categorical features.
  dh::caching_device_vector<size_t> new_column_scan(column_sizes_scan.size());
  auto d_sorted_idx = dh::ToSpan(*p_sorted_idx);
  dh::SegmentedUnique(
      column_sizes_scan.data().get(), column_sizes_scan.data().get() + column_sizes_scan.size(),
      dh::tcbegin(d_sorted_idx), dh::tcend(d_sorted_idx), new_column_scan.data().get(),
      dh::tbegin(d_sorted_idx), [=] __device__(SortedIdxT l, SortedIdxT r) {
        data::COOTuple const& le = batch_iter[l];
        data::COOTuple const& re = batch_iter[r];
        if (le.column_idx == re.column_idx) {
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
  thrust::exclusive_scan(new_cuts_size.cbegin(), new_cuts_size.cend(), d_cuts_ptr.data());
}
}  // namespace detail

// Compute sketch on DMatrix.
// sketch_batch_num_elements 0 means autodetect. Only modify this for testing.
HistogramCuts DeviceSketch(int device, DMatrix* dmat, int max_bins,
                           size_t sketch_batch_num_elements = 0);

// Quantile sketching on DMatrix. Exposed for tests.
void ProcessBatch(std::int32_t device, MetaInfo const& info, const SparsePage& page,
                  std::size_t begin, std::size_t end, SketchContainer* sketch_container,
                  bst_bin_t num_cuts_per_feature, std::size_t num_columns);

// Quantile sketching on DMatrix with weighted samples. Exposed for tests.
void ProcessWeightedBatch(int device, MetaInfo const& info, const SparsePage& page,
                          std::size_t begin, std::size_t end, SketchContainer* sketch_container,
                          bst_bin_t num_cuts_per_feature, bst_feature_t num_columns,
                          bool is_ranking, Span<bst_group_t const> d_group_ptr);

template <typename AdapterBatch>
void ProcessSlidingWindow(AdapterBatch const& batch, MetaInfo const& info, int device,
                          std::size_t columns, std::size_t begin, std::size_t end, float missing,
                          SketchContainer* sketch_container, int num_cuts) {
  dh::caching_device_vector<size_t> column_sizes_scan;
  auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(
      thrust::make_counting_iterator(0llu),
      [=] __device__(std::size_t idx) { return batch.GetElement(idx); });
  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;
  cuts_ptr.SetDevice(device);

  dh::device_vector<detail::SortedIdxT> sorted_idx;
  data::IsValidFunctor is_valid(missing);
  detail::MakeEntriesFromAdapter(batch_iter, {begin, end}, is_valid, columns, num_cuts, device,
                                 &cuts_ptr, &column_sizes_scan, &sorted_idx);

  if (sketch_container->HasCategorical()) {
    auto d_cuts_ptr = cuts_ptr.DeviceSpan();
    detail::RemoveDuplicatedCategories(device, info, d_cuts_ptr, batch_iter + begin, &sorted_idx,
                                       &column_sizes_scan);
  }
  auto entry_it = dh::MakeTransformIterator<Entry>(
      batch_iter + begin, [=] __device__(data::COOTuple const& tup) {
        return Entry{static_cast<bst_feature_t>(tup.column_idx), tup.value};
      });
  auto d_sorted_entry_it = thrust::make_permutation_iterator(entry_it, sorted_idx.cbegin());

  auto d_cuts_ptr = cuts_ptr.DeviceSpan();
  auto const& h_cuts_ptr = cuts_ptr.HostVector();
  // Extract the cuts from all columns concurrently
  sketch_container->Push(d_sorted_entry_it, dh::ToSpan(column_sizes_scan), d_cuts_ptr,
                         h_cuts_ptr.back());
}

template <typename Batch>
void ProcessWeightedSlidingWindow(Batch batch, MetaInfo const& info, int num_cuts_per_feature,
                                  bool is_ranking, float missing, int device, size_t columns,
                                  std::size_t begin, std::size_t end,
                                  SketchContainer* sketch_container) {
  dh::safe_cuda(cudaSetDevice(device));
  info.weights_.SetDevice(device);
  auto weights = info.weights_.ConstDeviceSpan();

  dh::XGBCachingDeviceAllocator<char> alloc;

  auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(
      thrust::make_counting_iterator(0llu),
      [=] __device__(std::size_t idx) { return batch.GetElement(idx); });
  dh::caching_device_vector<size_t> column_sizes_scan;
  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;

  dh::device_vector<detail::SortedIdxT> sorted_idx;
  data::IsValidFunctor is_valid(missing);
  detail::MakeEntriesFromAdapter(batch_iter, {begin, end}, is_valid, columns, num_cuts_per_feature,
                                 device, &cuts_ptr, &column_sizes_scan, &sorted_idx);

  // sorted_idx.size() represents the number of valid elements.
  dh::device_vector<float> temp_weights(sorted_idx.size());
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
    auto retit = thrust::copy_if(thrust::cuda::par(alloc), weight_iter + begin, weight_iter + end,
                                 batch_iter + begin,
                                 d_temp_weights.data(),  // output
                                 is_valid);
    CHECK_EQ(retit - d_temp_weights.data(), d_temp_weights.size());
  } else {
    CHECK_EQ(batch.NumRows(), weights.size());
    auto const weight_iter = dh::MakeTransformIterator<float>(
        thrust::make_counting_iterator(0lu),
        [=] __device__(size_t idx) -> float { return weights[batch.GetElement(idx).row_idx]; });
    auto retit = thrust::copy_if(thrust::cuda::par(alloc), weight_iter + begin, weight_iter + end,
                                 batch_iter + begin,
                                 d_temp_weights.data(),  // output
                                 is_valid);
    CHECK_EQ(retit - d_temp_weights.data(), d_temp_weights.size());
  }

  auto entry_it = dh::MakeTransformIterator<Entry>(
      batch_iter + begin, [=] __device__(data::COOTuple const& tup) {
        return Entry{static_cast<bst_feature_t>(tup.column_idx), tup.value};
      });
  auto d_sorted_entry_it = thrust::make_permutation_iterator(entry_it, sorted_idx.cbegin());
  auto d_sorted_weight_it =
      thrust::make_permutation_iterator(dh::tbegin(d_temp_weights), sorted_idx.cbegin());

  thrust::inclusive_scan_by_key(
      thrust::cuda::par(alloc), d_sorted_entry_it, d_sorted_entry_it + sorted_idx.size(),
      d_sorted_weight_it, d_sorted_weight_it,
      [=] __device__(const Entry& a, const Entry& b) { return a.index == b.index; });

  if (sketch_container->HasCategorical()) {
    auto d_cuts_ptr = cuts_ptr.DeviceSpan();
    detail::RemoveDuplicatedCategories(device, info, d_cuts_ptr, batch_iter + begin, &sorted_idx,
                                       &column_sizes_scan);
  }

  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();
  auto d_cuts_ptr = cuts_ptr.DeviceSpan();

  // Extract cuts
  sketch_container->Push(d_sorted_entry_it, dh::ToSpan(column_sizes_scan), d_cuts_ptr,
                         h_cuts_ptr.back(), IterSpan{d_sorted_weight_it, sorted_idx.size()});
}

/**
 * \brief Perform sketching on GPU in-place.
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
void AdapterDeviceSketch(Batch const& batch, bst_bin_t num_bins, MetaInfo const& info,
                         float missing, SketchContainer* sketch_container,
                         std::size_t sketch_batch_num_elements = 0) {
  size_t num_rows = batch.NumRows();
  size_t num_cols = batch.NumCols();
  std::size_t num_cuts_per_feature = detail::RequiredSampleCutsPerColumn(num_bins, num_rows);
  int32_t device = sketch_container->DeviceIdx();
  bool weighted = !info.weights_.Empty();

  if (weighted) {
    sketch_batch_num_elements = detail::SketchBatchNumElements(
        sketch_batch_num_elements, num_rows, num_cols, std::numeric_limits<size_t>::max(), device,
        num_cuts_per_feature, true, true);
    for (auto begin = 0ull; begin < batch.Size(); begin += sketch_batch_num_elements) {
      size_t end =
          std::min(batch.Size(), static_cast<std::size_t>(begin + sketch_batch_num_elements));
      ProcessWeightedSlidingWindow(batch, info, num_cuts_per_feature,
                                   HostSketchContainer::UseGroup(info), missing, device, num_cols,
                                   begin, end, sketch_container);
    }
  } else {
    sketch_batch_num_elements = detail::SketchBatchNumElements(
        sketch_batch_num_elements, num_rows, num_cols, std::numeric_limits<size_t>::max(), device,
        num_cuts_per_feature, false, true);
    for (auto begin = 0ull; begin < batch.Size(); begin += sketch_batch_num_elements) {
      size_t end =
          std::min(batch.Size(), static_cast<std::size_t>(begin + sketch_batch_num_elements));
      ProcessSlidingWindow(batch, info, device, num_cols, begin, end, missing, sketch_container,
                           num_cuts_per_feature);
    }
  }
}
}  // namespace xgboost::common

#endif  // COMMON_HIST_UTIL_CUH_
