/**
 * Copyright 2020-2024, XGBoost contributors
 *
 * \brief Front end and utilities for GPU based sketching.  Works on sliding window
 *        instead of stream.
 */
#ifndef COMMON_HIST_UTIL_CUH_
#define COMMON_HIST_UTIL_CUH_

#include <thrust/host_vector.h>
#include <thrust/sort.h>  // for sort

#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t
#include <limits>     // for numeric_limits

#include "../data/adapter.h"  // for IsValidFunctor
#include "algorithm.cuh"      // for CopyIf
#include "cuda_context.cuh"   // for CUDAContext
#include "device_helpers.cuh"
#include "hist_util.h"
#include "quantile.cuh"
#include "xgboost/span.h"  // for IterSpan

namespace xgboost::common {
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
std::uint32_t EstimateGridSize(DeviceOrd device, Kernel kernel, std::size_t shared_mem) {
  int n_mps = 0;
  dh::safe_cuda(cudaDeviceGetAttribute(&n_mps, cudaDevAttrMultiProcessorCount, device.ordinal));
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
void LaunchGetColumnSizeKernel(CUDAContext const* cuctx, DeviceOrd device,
                               IterSpan<BatchIt> batch_iter, data::IsValidFunctor is_valid,
                               Span<std::size_t> out_column_size) {
  thrust::fill_n(cuctx->CTP(), dh::tbegin(out_column_size), out_column_size.size(), 0);

  std::size_t max_shared_memory = dh::MaxSharedMemory(device.ordinal);
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
      dh::LaunchKernel{grid_size, kBlockThreads, required_shared_memory, cuctx->Stream()}(
          kernel, batch_iter, is_valid, out_column_size);
    } else {
      auto kernel = GetColumnSizeSharedMemKernel<kBlockThreads, std::size_t, BatchIt>;
      auto grid_size = EstimateGridSize<kBlockThreads>(device, kernel, required_shared_memory);
      dh::LaunchKernel{grid_size, kBlockThreads, required_shared_memory, cuctx->Stream()}(
          kernel, batch_iter, is_valid, out_column_size);
    }
  } else {
    auto d_out_column_size = out_column_size;
    dh::LaunchN(batch_iter.size(), cuctx->Stream(), [=] __device__(size_t idx) {
      auto e = batch_iter[idx];
      if (is_valid(e)) {
        atomicAdd(&d_out_column_size[e.column_idx], static_cast<size_t>(1));
      }
    });
  }
}

template <typename BatchIt>
void GetColumnSizesScan(CUDAContext const* cuctx, DeviceOrd device, size_t num_columns,
                        std::size_t num_cuts_per_feature, IterSpan<BatchIt> batch_iter,
                        data::IsValidFunctor is_valid,
                        HostDeviceVector<SketchContainer::OffsetT>* cuts_ptr,
                        dh::caching_device_vector<size_t>* column_sizes_scan) {
  column_sizes_scan->resize(num_columns + 1);
  cuts_ptr->SetDevice(device);
  cuts_ptr->Resize(num_columns + 1, 0);

  auto d_column_sizes_scan = dh::ToSpan(*column_sizes_scan);
  LaunchGetColumnSizeKernel(cuctx, device, batch_iter, is_valid, d_column_sizes_scan);
  // Calculate cuts CSC pointer
  auto cut_ptr_it = dh::MakeTransformIterator<size_t>(
      column_sizes_scan->begin(), [=] __device__(size_t column_size) {
        return thrust::min(num_cuts_per_feature, column_size);
      });
  thrust::exclusive_scan(cuctx->CTP(), cut_ptr_it,
                         cut_ptr_it + column_sizes_scan->size(), cuts_ptr->DevicePointer());
  thrust::exclusive_scan(cuctx->CTP(), column_sizes_scan->begin(), column_sizes_scan->end(),
                         column_sizes_scan->begin());
}

inline size_t constexpr BytesPerElement(bool has_weight) {
  // Double the memory usage for sorting.  We need to assign weight for each element, so
  // sizeof(float) is added to all elements.
  return (has_weight ? sizeof(Entry) + sizeof(float) : sizeof(Entry)) * 2;
}

struct SketchShape {
  bst_idx_t n_samples;
  bst_feature_t n_features;
  bst_idx_t nnz;

  template <typename F, std::enable_if_t<std::is_integral_v<F>>* = nullptr>
  SketchShape(bst_idx_t n_samples, F n_features, bst_idx_t nnz)
      : n_samples{n_samples}, n_features{static_cast<bst_feature_t>(n_features)}, nnz{nnz} {}

  [[nodiscard]] bst_idx_t Size() const { return n_samples * n_features; }
};

/**
 * @brief Calcuate the length of sliding window. Returns `sketch_batch_num_elements`
 *        directly if it's not 0.
 */
bst_idx_t SketchBatchNumElements(bst_idx_t sketch_batch_num_elements, SketchShape shape, int device,
                                 size_t num_cuts, bool has_weight, std::size_t container_bytes);

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
size_t RequiredMemory(bst_idx_t num_rows, bst_feature_t num_columns, size_t nnz,
                      size_t num_bins, bool with_weights);

// Count the valid entries in each column and copy them out.
template <typename AdapterBatch, typename BatchIter>
void MakeEntriesFromAdapter(CUDAContext const* cuctx, AdapterBatch const& batch,
                            BatchIter batch_iter, Range1d range, float missing, size_t columns,
                            size_t cuts_per_feature, DeviceOrd device,
                            HostDeviceVector<SketchContainer::OffsetT>* cut_sizes_scan,
                            dh::caching_device_vector<size_t>* column_sizes_scan,
                            dh::device_vector<Entry>* sorted_entries) {
  auto entry_iter = dh::MakeTransformIterator<Entry>(
      thrust::make_counting_iterator(0llu), [=] __device__(size_t idx) {
        return Entry(batch.GetElement(idx).column_idx, batch.GetElement(idx).value);
      });
  auto n = range.end() - range.begin();
  auto span = IterSpan{batch_iter + range.begin(), n};
  data::IsValidFunctor is_valid(missing);
  // Work out how many valid entries we have in each column
  GetColumnSizesScan(cuctx, device, columns, cuts_per_feature, span, is_valid, cut_sizes_scan,
                     column_sizes_scan);
  size_t num_valid = column_sizes_scan->back();
  // Copy current subset of valid elements into temporary storage and sort
  sorted_entries->resize(num_valid);
  CopyIf(cuctx, entry_iter + range.begin(), entry_iter + range.end(), sorted_entries->begin(),
         is_valid);
}

void SortByWeight(Context const* ctx, dh::device_vector<float>* weights,
                  dh::device_vector<Entry>* sorted_entries);

void RemoveDuplicatedCategories(Context const* ctx, MetaInfo const& info,
                                Span<bst_idx_t> d_cuts_ptr,
                                dh::device_vector<Entry>* p_sorted_entries,
                                dh::device_vector<float>* p_sorted_weights,
                                dh::caching_device_vector<size_t>* p_column_sizes_scan);

constexpr bst_idx_t UnknownSketchNumElements() { return 0; }
}  // namespace detail

/**
 * @brief Compute sketch on DMatrix with GPU and Hessian as weight.
 *
 * @param ctx     Runtime context
 * @param p_fmat  Training feature matrix
 * @param max_bin Maximum number of bins for each feature
 * @param hessian Hessian vector.
 * @param sketch_batch_num_elements 0 means autodetect. Only modify this for testing.
 *
 * @return Quantile cuts
 */
HistogramCuts DeviceSketchWithHessian(Context const* ctx, DMatrix* p_fmat, bst_bin_t max_bin,
                                      Span<float const> hessian,
                                      std::size_t sketch_batch_num_elements = detail::UnknownSketchNumElements());

/**
 * @brief Compute sketch on DMatrix with GPU.
 *
 * @param ctx     Runtime context
 * @param p_fmat  Training feature matrix
 * @param max_bin Maximum number of bins for each feature
 * @param sketch_batch_num_elements 0 means autodetect. Only modify this for testing.
 *
 * @return Quantile cuts
 */
inline HistogramCuts DeviceSketch(
    Context const* ctx, DMatrix* p_fmat, bst_bin_t max_bin,
    std::size_t sketch_batch_num_elements = detail::UnknownSketchNumElements()) {
  return DeviceSketchWithHessian(ctx, p_fmat, max_bin, {}, sketch_batch_num_elements);
}

template <typename AdapterBatch>
void ProcessSlidingWindow(Context const* ctx, AdapterBatch const& batch, MetaInfo const& info,
                          size_t n_features, size_t begin, size_t end, float missing,
                          SketchContainer* sketch_container, int num_cuts) {
  // Copy current subset of valid elements into temporary storage and sort
  dh::device_vector<Entry> sorted_entries;
  dh::caching_device_vector<size_t> column_sizes_scan;
  auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(
      thrust::make_counting_iterator(0llu),
      [=] __device__(size_t idx) { return batch.GetElement(idx); });
  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;
  cuts_ptr.SetDevice(ctx->Device());
  CUDAContext const* cuctx = ctx->CUDACtx();
  detail::MakeEntriesFromAdapter(cuctx, batch, batch_iter, {begin, end}, missing, n_features,
                                 num_cuts, ctx->Device(), &cuts_ptr, &column_sizes_scan,
                                 &sorted_entries);
  thrust::sort(cuctx->TP(), sorted_entries.begin(), sorted_entries.end(), detail::EntryCompareOp());

  if (sketch_container->HasCategorical()) {
    auto d_cuts_ptr = cuts_ptr.DeviceSpan();
    detail::RemoveDuplicatedCategories(ctx, info, d_cuts_ptr, &sorted_entries, nullptr,
                                       &column_sizes_scan);
  }

  auto d_cuts_ptr = cuts_ptr.DeviceSpan();
  auto const& h_cuts_ptr = cuts_ptr.HostVector();
  // Extract the cuts from all columns concurrently
  sketch_container->Push(ctx, dh::ToSpan(sorted_entries), dh::ToSpan(column_sizes_scan), d_cuts_ptr,
                         h_cuts_ptr.back());

  sorted_entries.clear();
  sorted_entries.shrink_to_fit();
}

template <typename Batch>
void ProcessWeightedSlidingWindow(Context const* ctx, Batch batch, MetaInfo const& info,
                                  int num_cuts_per_feature, bool is_ranking, float missing,
                                  size_t columns, size_t begin, size_t end,
                                  SketchContainer* sketch_container) {
  curt::SetDevice(ctx->Ordinal());
  info.weights_.SetDevice(ctx->Device());
  auto weights = info.weights_.ConstDeviceSpan();

  auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(
      thrust::make_counting_iterator(0llu),
      [=] __device__(size_t idx) { return batch.GetElement(idx); });
  auto cuctx = ctx->CUDACtx();
  dh::device_vector<Entry> sorted_entries;
  dh::caching_device_vector<size_t> column_sizes_scan;
  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;
  detail::MakeEntriesFromAdapter(cuctx, batch, batch_iter, {begin, end}, missing, columns,
                                 num_cuts_per_feature, ctx->Device(), &cuts_ptr, &column_sizes_scan,
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
    auto retit = thrust::copy_if(cuctx->CTP(),
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
    auto retit = thrust::copy_if(cuctx->CTP(),
                                 weight_iter + begin, weight_iter + end,
                                 batch_iter + begin,
                                 d_temp_weights.data(),  // output
                                 is_valid);
    CHECK_EQ(retit - d_temp_weights.data(), d_temp_weights.size());
  }

  detail::SortByWeight(ctx, &temp_weights, &sorted_entries);

  if (sketch_container->HasCategorical()) {
    auto d_cuts_ptr = cuts_ptr.DeviceSpan();
    detail::RemoveDuplicatedCategories(ctx, info, d_cuts_ptr, &sorted_entries, &temp_weights,
                                       &column_sizes_scan);
  }

  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();
  auto d_cuts_ptr = cuts_ptr.DeviceSpan();

  // Extract cuts
  sketch_container->Push(ctx, dh::ToSpan(sorted_entries), dh::ToSpan(column_sizes_scan), d_cuts_ptr,
                         h_cuts_ptr.back(), dh::ToSpan(temp_weights));
  sorted_entries.clear();
  sorted_entries.shrink_to_fit();
}

/**
 * @brief Perform sketching on GPU.
 *
 * @param batch            A batch from adapter.
 * @param num_bins         Bins per column.
 * @param info             Metainfo used for sketching.
 * @param missing          Floating point value that represents invalid value.
 * @param sketch_container Container for output sketch.
 * @param sketch_batch_num_elements Number of element per-sliding window, use it only for
 *                                  testing.
 */
template <typename Batch>
void AdapterDeviceSketch(Context const* ctx, Batch batch, bst_bin_t num_bins, MetaInfo const& info,
                         float missing, SketchContainer* sketch_container,
                         bst_idx_t sketch_batch_num_elements = detail::UnknownSketchNumElements()) {
  bst_idx_t num_rows = batch.NumRows();
  size_t num_cols = batch.NumCols();

  bool weighted = !info.weights_.Empty();

  bst_idx_t const kRemaining = batch.Size();
  bst_idx_t begin = 0;

  auto shape = detail::SketchShape{num_rows, num_cols, std::numeric_limits<bst_idx_t>::max()};

  while (begin < kRemaining) {
    // Use total number of samples to estimate the needed cuts first, this doesn't hurt
    // accuracy as total number of samples is larger.
    auto num_cuts_per_feature = detail::RequiredSampleCutsPerColumn(num_bins, num_rows);
    // Estimate the memory usage based on the current available memory.
    sketch_batch_num_elements = detail::SketchBatchNumElements(
        sketch_batch_num_elements, shape, ctx->Ordinal(), num_cuts_per_feature, weighted,
        sketch_container->MemCostBytes());
    // Re-estimate the needed number of cuts based on the size of the sub-batch.
    //
    // The estimation of `sketch_batch_num_elements` assumes dense input, so the
    // approximation here is reasonably accurate. It doesn't hurt accuracy since the
    // estimated n_samples must be greater or equal to the actual n_samples thanks to the
    // dense assumption.
    auto approx_n_samples = std::max(sketch_batch_num_elements / num_cols, bst_idx_t{1});
    num_cuts_per_feature = detail::RequiredSampleCutsPerColumn(num_bins, approx_n_samples);
    bst_idx_t end =
        std::min(batch.Size(), static_cast<std::size_t>(begin + sketch_batch_num_elements));

    if (weighted) {
      ProcessWeightedSlidingWindow(ctx, batch, info, num_cuts_per_feature,
                                   HostSketchContainer::UseGroup(info), missing, num_cols, begin,
                                   end, sketch_container);
    } else {
      ProcessSlidingWindow(ctx, batch, info, num_cols, begin, end, missing, sketch_container,
                           num_cuts_per_feature);
    }
    begin += sketch_batch_num_elements;
  }
}
}  // namespace xgboost::common
#endif  // COMMON_HIST_UTIL_CUH_
