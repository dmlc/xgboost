/**
 * Copyright 2022-2024, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_ALGORITHM_CUH_
#define XGBOOST_COMMON_ALGORITHM_CUH_

#include <thrust/copy.h>       // copy
#include <thrust/sort.h>       // stable_sort_by_key
#include <thrust/tuple.h>      // tuple,get

#include <cstddef>             // size_t
#include <cstdint>             // int32_t
#include <cub/cub.cuh>         // DispatchSegmentedRadixSort,NullType,DoubleBuffer
#include <iterator>            // distance
#include <limits>              // numeric_limits
#include <type_traits>         // conditional_t,remove_const_t

#include "common.h"            // safe_cuda
#include "cuda_context.cuh"    // CUDAContext
#include "device_helpers.cuh"  // TemporaryArray,SegmentId,LaunchN,Iota
#include "device_vector.cuh"   // for device_vector
#include "xgboost/base.h"      // XGBOOST_DEVICE
#include "xgboost/context.h"   // Context
#include "xgboost/logging.h"   // CHECK
#include "xgboost/span.h"      // Span,byte

namespace xgboost::common {
namespace detail {
// Wrapper around cub sort to define is_decending
template <bool IS_DESCENDING, typename KeyT, typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT>
static void DeviceSegmentedRadixSortKeys(CUDAContext const *ctx, void *d_temp_storage,
                                         std::size_t &temp_storage_bytes,  // NOLINT
                                         const KeyT *d_keys_in, KeyT *d_keys_out, int num_items,
                                         int num_segments, BeginOffsetIteratorT d_begin_offsets,
                                         EndOffsetIteratorT d_end_offsets, int begin_bit = 0,
                                         int end_bit = sizeof(KeyT) * 8,
                                         bool debug_synchronous = false) {
  using OffsetT = int;

  // Null value type
  cub::DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
  cub::DoubleBuffer<cub::NullType> d_values;

  dh::safe_cuda((cub::DispatchSegmentedRadixSort<
                 IS_DESCENDING, KeyT, cub::NullType, BeginOffsetIteratorT, EndOffsetIteratorT,
                 OffsetT>::Dispatch(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items,
                                    num_segments, d_begin_offsets, d_end_offsets, begin_bit,
                                    end_bit, false, ctx->Stream(), debug_synchronous)));
}

// Wrapper around cub sort for easier `descending` sort.
template <bool descending, typename KeyT, typename ValueT, typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT>
void DeviceSegmentedRadixSortPair(void *d_temp_storage,
                                  std::size_t &temp_storage_bytes,  // NOLINT
                                  const KeyT *d_keys_in, KeyT *d_keys_out,
                                  const ValueT *d_values_in, ValueT *d_values_out,
                                  std::size_t num_items, std::size_t num_segments,
                                  BeginOffsetIteratorT d_begin_offsets,
                                  EndOffsetIteratorT d_end_offsets, dh::CUDAStreamView stream,
                                  int begin_bit = 0, int end_bit = sizeof(KeyT) * 8) {
  cub::DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
  cub::DoubleBuffer<ValueT> d_values(const_cast<ValueT *>(d_values_in), d_values_out);
  // In old version of cub, num_items in dispatch is also int32_t, no way to change.
  using OffsetT = std::conditional_t<dh::BuildWithCUDACub() && dh::HasThrustMinorVer<13>(),
                                     std::size_t, std::int32_t>;
  CHECK_LE(num_items, std::numeric_limits<OffsetT>::max());
  // For Thrust >= 1.12 or CUDA >= 11.4, we require system cub installation

#if THRUST_MAJOR_VERSION >= 2
  dh::safe_cuda((cub::DispatchSegmentedRadixSort<
                 descending, KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT,
                 OffsetT>::Dispatch(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items,
                                    num_segments, d_begin_offsets, d_end_offsets, begin_bit,
                                    end_bit, false, stream)));
#elif (THRUST_MAJOR_VERSION == 1 && THRUST_MINOR_VERSION >= 13)
  dh::safe_cuda((cub::DispatchSegmentedRadixSort<
                 descending, KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT,
                 OffsetT>::Dispatch(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items,
                                    num_segments, d_begin_offsets, d_end_offsets, begin_bit,
                                    end_bit, false, stream, false)));
#else
  dh::safe_cuda(
      (cub::DispatchSegmentedRadixSort<descending, KeyT, ValueT, BeginOffsetIteratorT,
                                       OffsetT>::Dispatch(d_temp_storage, temp_storage_bytes,
                                                          d_keys, d_values, num_items, num_segments,
                                                          d_begin_offsets, d_end_offsets, begin_bit,
                                                          end_bit, false, stream, false)));
#endif
}
}  // namespace detail

template <typename U, typename V>
void SegmentedSequence(Context const *ctx, Span<U> d_offset_ptr, Span<V> out_sequence) {
  dh::LaunchN(out_sequence.size(), ctx->CUDACtx()->Stream(),
              [out_sequence, d_offset_ptr] __device__(std::size_t idx) {
                auto group = dh::SegmentId(d_offset_ptr, idx);
                out_sequence[idx] = idx - d_offset_ptr[group];
              });
}

template <bool descending, typename U, typename V>
inline void SegmentedSortKeys(Context const *ctx, Span<V const> group_ptr,
                              Span<U> out_sorted_values) {
  CHECK_GE(group_ptr.size(), 1ul);
  std::size_t n_groups = group_ptr.size() - 1;
  std::size_t bytes = 0;
  auto const *cuctx = ctx->CUDACtx();
  CHECK(cuctx);
  detail::DeviceSegmentedRadixSortKeys<descending>(
      cuctx, nullptr, bytes, out_sorted_values.data(), out_sorted_values.data(),
      out_sorted_values.size(), n_groups, group_ptr.data(), group_ptr.data() + 1);
  dh::TemporaryArray<byte> temp_storage(bytes);
  detail::DeviceSegmentedRadixSortKeys<descending>(
      cuctx, temp_storage.data().get(), bytes, out_sorted_values.data(), out_sorted_values.data(),
      out_sorted_values.size(), n_groups, group_ptr.data(), group_ptr.data() + 1);
}

/**
 * \brief Create sorted index for data with multiple segments.
 *
 * \tparam accending sorted in non-decreasing order.
 * \tparam per_seg_index Index starts from 0 for each segment if true, otherwise the
 *                       the index span the whole data.
 */
template <bool accending, bool per_seg_index, typename U, typename V, typename IdxT>
void SegmentedArgSort(Context const *ctx, Span<U> values, Span<V> group_ptr,
                      Span<IdxT> sorted_idx) {
  auto cuctx = ctx->CUDACtx();
  CHECK_GE(group_ptr.size(), 1ul);
  std::size_t n_groups = group_ptr.size() - 1;
  std::size_t bytes = 0;
  if (per_seg_index) {
    SegmentedSequence(ctx, group_ptr, sorted_idx);
  } else {
    dh::Iota(sorted_idx, cuctx->Stream());
  }
  dh::TemporaryArray<std::remove_const_t<U>> values_out(values.size());
  dh::TemporaryArray<std::remove_const_t<IdxT>> sorted_idx_out(sorted_idx.size());

  detail::DeviceSegmentedRadixSortPair<!accending>(
      nullptr, bytes, values.data(), values_out.data().get(), sorted_idx.data(),
      sorted_idx_out.data().get(), sorted_idx.size(), n_groups, group_ptr.data(),
      group_ptr.data() + 1, cuctx->Stream());
  dh::TemporaryArray<byte> temp_storage(bytes);
  detail::DeviceSegmentedRadixSortPair<!accending>(
      temp_storage.data().get(), bytes, values.data(), values_out.data().get(), sorted_idx.data(),
      sorted_idx_out.data().get(), sorted_idx.size(), n_groups, group_ptr.data(),
      group_ptr.data() + 1, cuctx->Stream());

  dh::safe_cuda(cudaMemcpyAsync(sorted_idx.data(), sorted_idx_out.data().get(),
                                sorted_idx.size_bytes(), cudaMemcpyDeviceToDevice,
                                cuctx->Stream()));
}

/**
 * \brief Different from the radix-sort-based argsort, this one can handle cases where
 *        segment doesn't start from 0, but as a result it uses comparison sort.
 */
template <typename SegIt, typename ValIt>
void SegmentedArgMergeSort(Context const *ctx, SegIt seg_begin, SegIt seg_end, ValIt val_begin,
                           ValIt val_end, dh::device_vector<std::size_t> *p_sorted_idx) {
  auto cuctx = ctx->CUDACtx();
  using Tup = thrust::tuple<std::int32_t, float>;
  auto &sorted_idx = *p_sorted_idx;
  std::size_t n = std::distance(val_begin, val_end);
  sorted_idx.resize(n);
  dh::Iota(dh::ToSpan(sorted_idx), cuctx->Stream());
  dh::device_vector<Tup> keys(sorted_idx.size());
  auto key_it = dh::MakeTransformIterator<Tup>(thrust::make_counting_iterator(0ul),
                                               [=] XGBOOST_DEVICE(std::size_t i) -> Tup {
                                                 std::int32_t seg_idx;
                                                 if (i < *seg_begin) {
                                                   seg_idx = -1;
                                                 } else {
                                                   seg_idx = dh::SegmentId(seg_begin, seg_end, i);
                                                 }
                                                 auto residue = val_begin[i];
                                                 return thrust::make_tuple(seg_idx, residue);
                                               });
  thrust::copy(ctx->CUDACtx()->CTP(), key_it, key_it + keys.size(), keys.begin());
  thrust::stable_sort_by_key(cuctx->TP(), keys.begin(), keys.end(), sorted_idx.begin(),
                             [=] XGBOOST_DEVICE(Tup const &l, Tup const &r) {
                               if (thrust::get<0>(l) != thrust::get<0>(r)) {
                                 return thrust::get<0>(l) < thrust::get<0>(r);  // segment index
                               }
                               return thrust::get<1>(l) < thrust::get<1>(r);    // residue
                             });
}

template <bool accending, typename IdxT, typename U>
void ArgSort(Context const *ctx, Span<U> keys, Span<IdxT> sorted_idx) {
  std::size_t bytes = 0;
  auto cuctx = ctx->CUDACtx();
  dh::Iota(sorted_idx, cuctx->Stream());

  using KeyT = typename decltype(keys)::value_type;
  using ValueT = std::remove_const_t<IdxT>;

  dh::TemporaryArray<KeyT> out(keys.size());
  cub::DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(keys.data()), out.data().get());
  dh::TemporaryArray<IdxT> sorted_idx_out(sorted_idx.size());
  cub::DoubleBuffer<ValueT> d_values(const_cast<ValueT *>(sorted_idx.data()),
                                     sorted_idx_out.data().get());

  // track https://github.com/NVIDIA/cub/pull/340 for 64bit length support
  using OffsetT = std::conditional_t<!dh::BuildWithCUDACub(), std::ptrdiff_t, int32_t>;
  CHECK_LE(sorted_idx.size(), std::numeric_limits<OffsetT>::max());
  if (accending) {
    void *d_temp_storage = nullptr;
#if THRUST_MAJOR_VERSION >= 2
    dh::safe_cuda((cub::DispatchRadixSort<false, KeyT, ValueT, OffsetT>::Dispatch(
        d_temp_storage, bytes, d_keys, d_values, sorted_idx.size(), 0, sizeof(KeyT) * 8, false,
        cuctx->Stream())));
#else
    dh::safe_cuda((cub::DispatchRadixSort<false, KeyT, ValueT, OffsetT>::Dispatch(
        d_temp_storage, bytes, d_keys, d_values, sorted_idx.size(), 0, sizeof(KeyT) * 8, false,
        nullptr, false)));
#endif
    dh::TemporaryArray<char> storage(bytes);
    d_temp_storage = storage.data().get();
#if THRUST_MAJOR_VERSION >= 2
    dh::safe_cuda((cub::DispatchRadixSort<false, KeyT, ValueT, OffsetT>::Dispatch(
        d_temp_storage, bytes, d_keys, d_values, sorted_idx.size(), 0, sizeof(KeyT) * 8, false,
        cuctx->Stream())));
#else
    dh::safe_cuda((cub::DispatchRadixSort<false, KeyT, ValueT, OffsetT>::Dispatch(
        d_temp_storage, bytes, d_keys, d_values, sorted_idx.size(), 0, sizeof(KeyT) * 8, false,
        nullptr, false)));
#endif
  } else {
    void *d_temp_storage = nullptr;
#if THRUST_MAJOR_VERSION >= 2
    dh::safe_cuda((cub::DispatchRadixSort<true, KeyT, ValueT, OffsetT>::Dispatch(
        d_temp_storage, bytes, d_keys, d_values, sorted_idx.size(), 0, sizeof(KeyT) * 8, false,
        cuctx->Stream())));
#else
    dh::safe_cuda((cub::DispatchRadixSort<true, KeyT, ValueT, OffsetT>::Dispatch(
        d_temp_storage, bytes, d_keys, d_values, sorted_idx.size(), 0, sizeof(KeyT) * 8, false,
        nullptr, false)));
#endif
    dh::TemporaryArray<char> storage(bytes);
    d_temp_storage = storage.data().get();
#if THRUST_MAJOR_VERSION >= 2
    dh::safe_cuda((cub::DispatchRadixSort<true, KeyT, ValueT, OffsetT>::Dispatch(
        d_temp_storage, bytes, d_keys, d_values, sorted_idx.size(), 0, sizeof(KeyT) * 8, false,
        cuctx->Stream())));
#else
    dh::safe_cuda((cub::DispatchRadixSort<true, KeyT, ValueT, OffsetT>::Dispatch(
        d_temp_storage, bytes, d_keys, d_values, sorted_idx.size(), 0, sizeof(KeyT) * 8, false,
        nullptr, false)));
#endif
  }

  dh::safe_cuda(cudaMemcpyAsync(sorted_idx.data(), sorted_idx_out.data().get(),
                                sorted_idx.size_bytes(), cudaMemcpyDeviceToDevice,
                                cuctx->Stream()));
}

template <typename InIt, typename OutIt, typename Predicate>
void CopyIf(CUDAContext const *cuctx, InIt in_first, InIt in_second, OutIt out_first,
            Predicate pred) {
  // We loop over batches because thrust::copy_if can't deal with sizes > 2^31
  // See thrust issue #1302, XGBoost #6822
  size_t constexpr kMaxCopySize = std::numeric_limits<int>::max() / 2;
  size_t length = std::distance(in_first, in_second);
  for (size_t offset = 0; offset < length; offset += kMaxCopySize) {
    auto begin_input = in_first + offset;
    auto end_input = in_first + std::min(offset + kMaxCopySize, length);
    out_first = thrust::copy_if(cuctx->CTP(), begin_input, end_input, out_first, pred);
  }
}

// Go one level down into cub::DeviceScan API to set OffsetT as 64 bit So we don't crash
// on n > 2^31.
template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename OffsetT>
void InclusiveScan(xgboost::Context const *ctx, InputIteratorT d_in, OutputIteratorT d_out,
                   ScanOpT scan_op, OffsetT num_items) {
  auto cuctx = ctx->CUDACtx();
  std::size_t bytes = 0;
#if THRUST_MAJOR_VERSION >= 2
  dh::safe_cuda((
      cub::DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, cub::NullType, OffsetT>::Dispatch(
          nullptr, bytes, d_in, d_out, scan_op, cub::NullType(), num_items, nullptr)));
#else
  safe_cuda((
      cub::DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, cub::NullType, OffsetT>::Dispatch(
          nullptr, bytes, d_in, d_out, scan_op, cub::NullType(), num_items, nullptr, false)));
#endif
  dh::TemporaryArray<char> storage(bytes);
#if THRUST_MAJOR_VERSION >= 2
  dh::safe_cuda((
      cub::DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, cub::NullType, OffsetT>::Dispatch(
          storage.data().get(), bytes, d_in, d_out, scan_op, cub::NullType(), num_items, nullptr)));
#else
  safe_cuda((
      cub::DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, cub::NullType, OffsetT>::Dispatch(
          storage.data().get(), bytes, d_in, d_out, scan_op, cub::NullType(), num_items, nullptr,
          false)));
#endif
}

template <typename InputIteratorT, typename OutputIteratorT, typename OffsetT>
void InclusiveSum(Context const *ctx, InputIteratorT d_in, OutputIteratorT d_out,
                  OffsetT num_items) {
  InclusiveScan(ctx, d_in, d_out, cub::Sum{}, num_items);
}
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_ALGORITHM_CUH_
