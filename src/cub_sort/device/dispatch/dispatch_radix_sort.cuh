/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * cub::DeviceRadixSort provides device-wide, parallel operations for computing a radix sort across
 * a sequence of data items residing within device-accessible memory.
 */

#pragma once

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cstdio>
#include <cub/config.cuh>
#include <cub/device/dispatch/dispatch_radix_sort.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <iterator>

#include "../../agent/agent_radix_sort_histogram.cuh"
#include "../../agent/agent_radix_sort_onesweep.cuh"
#include "../../util_type.cuh"
#include "xgboost/span.h"

// NOLINTBEGIN
namespace cub_argsort {
namespace detail {
CUB_RUNTIME_FUNCTION inline cudaError_t HasUVA(bool &has_uva) {
  has_uva = false;
  cudaError_t error = cudaSuccess;
  int device = -1;
  if (CubDebug(error = cudaGetDevice(&device)) != cudaSuccess) return error;
  int uva = 0;
  if (CubDebug(error = cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, device)) !=
      cudaSuccess) {
    return error;
  }
  has_uva = uva == 1;
  return error;
}

CUB_RUNTIME_FUNCTION inline cudaError_t DebugSyncStream(cudaStream_t) { return cudaSuccess; }
}  // namespace detail

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/
/**
 * Kernel for computing multiple histograms
 */

/**
 * Histogram kernel
 */
template <typename ChainedPolicyT, bool IS_DESCENDING, typename KeyIteratorT, typename OffsetT,
          typename DigitExtractorT>
__global__ void __launch_bounds__(ChainedPolicyT::ActivePolicy::HistogramPolicy::BLOCK_THREADS)
    DeviceRadixArgSortHistogramKernel(OffsetT *d_bins_out, const KeyIteratorT d_keys_in,
                                   OffsetT num_items, int start_bit, int end_bit) {
  typedef typename ChainedPolicyT::ActivePolicy::HistogramPolicy HistogramPolicyT;
  typedef AgentRadixSortHistogram<HistogramPolicyT, IS_DESCENDING, KeyIteratorT, OffsetT,
                                  DigitExtractorT>
      AgentT;
  __shared__ typename AgentT::TempStorage temp_storage;
  AgentT agent(temp_storage, d_bins_out, d_keys_in, num_items, start_bit, end_bit);
  agent.Process();
}

template <typename ChainedPolicyT, bool IS_DESCENDING, typename KeyIteratorT, typename ValueT,
          typename OffsetT, typename PortionOffsetT, typename DigitExtractor,
          typename AtomicOffsetT = PortionOffsetT>
__global__ void __launch_bounds__(ChainedPolicyT::ActivePolicy::OnesweepPolicy::BLOCK_THREADS)
    DeviceRadixSortOnesweepKernel(AtomicOffsetT *d_lookback, AtomicOffsetT *d_ctrs,
                                  OffsetT *d_bins_out, const OffsetT *d_bins_in,
                                  KeyIteratorT d_keys_in,
                                  xgboost::common::Span<std::uint32_t> d_idx_out,
                                  xgboost::common::Span<std::uint32_t const> d_idx_in,
                                  PortionOffsetT num_items, int current_bit, int num_bits) {
  typedef typename ChainedPolicyT::ActivePolicy::OnesweepPolicy OnesweepPolicyT;
  typedef AgentRadixSortOnesweep<OnesweepPolicyT, IS_DESCENDING, KeyIteratorT, std::uint32_t,
                                 OffsetT, PortionOffsetT, DigitExtractor>
      AgentT;
  __shared__ typename AgentT::TempStorage s;

  DigitExtractor de(current_bit, num_bits);  // fixme
  AgentT agent(s, d_lookback, d_ctrs, d_bins_out, d_bins_in, d_keys_in, d_idx_out, d_idx_in.data(),
               num_items, de);
  agent.Process();
}

/**
 * Exclusive sum kernel
 */
template <typename ChainedPolicyT, typename OffsetT>
__global__ void DeviceRadixSortExclusiveSumKernel(OffsetT *d_bins) {
  typedef typename ChainedPolicyT::ActivePolicy::ExclusiveSumPolicy ExclusiveSumPolicyT;
  const int RADIX_BITS = ExclusiveSumPolicyT::RADIX_BITS;
  const int RADIX_DIGITS = 1 << RADIX_BITS;
  const int BLOCK_THREADS = ExclusiveSumPolicyT::BLOCK_THREADS;
  const int BINS_PER_THREAD = (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS;
  typedef ::cub::BlockScan<OffsetT, BLOCK_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // load the bins
  OffsetT bins[BINS_PER_THREAD];
  int bin_start = blockIdx.x * RADIX_DIGITS;
#pragma unroll
  for (int u = 0; u < BINS_PER_THREAD; ++u) {
    int bin = threadIdx.x * BINS_PER_THREAD + u;
    if (bin >= RADIX_DIGITS) break;
    bins[u] = d_bins[bin_start + bin];
  }

  // compute offsets
  BlockScan(temp_storage).ExclusiveSum(bins, bins);

// store the offsets
#pragma unroll
  for (int u = 0; u < BINS_PER_THREAD; ++u) {
    int bin = threadIdx.x * BINS_PER_THREAD + u;
    if (bin >= RADIX_DIGITS) break;
    d_bins[bin_start + bin] = bins[u];
  }
}

template <typename KeyIt>
struct SortedKeyOp {
  using KeyT = std::remove_reference_t<typename std::iterator_traits<KeyIt>::value_type>;

  KeyIt d_keys;
  std::uint32_t *s_idx_in;

  __device__ KeyT operator()(std::size_t i) const {
    auto idx = s_idx_in[i];
    return d_keys[idx];
  }
};

/**
 * Utility class for dispatching the appropriately-tuned kernels for device-wide radix sort
 */
template <bool IS_DESCENDING,     ///< Whether or not the sorted-order is high-to-low
          typename KeyIteratorT,  ///< Key type
          typename ValueT,        ///< Value type
          typename OffsetT,       ///< Signed integer type for global offsets
          typename SortedIdxT,    ///< Type for sorted index, must be integer
          typename DigitExtractorT,
          typename SelectedPolicy = ::cub::DeviceRadixSortPolicy<
              typename std::iterator_traits<KeyIteratorT>::value_type, ValueT, OffsetT> >
struct DispatchRadixArgSort : SelectedPolicy {
  //------------------------------------------------------------------------------
  // Problem state
  //------------------------------------------------------------------------------

  void *d_temp_storage;  ///< [in] Device-accessible allocation of temporary storage.  When NULL,
                         ///< the required allocation size is written to \p temp_storage_bytes and
                         ///< no work is done.
  size_t
      &temp_storage_bytes;  ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
  using KeyT = std::remove_reference_t<typename std::iterator_traits<KeyIteratorT>::value_type>;
  KeyIteratorT d_keys;
  OffsetT num_items;  ///< [in] Number of items to sort
  int begin_bit;  ///< [in] The beginning (least-significant) bit index needed for key comparison
  int end_bit;    ///< [in] The past-the-end (most-significant) bit index needed for key comparison
  cudaStream_t
      stream;       ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
  int ptx_version;  ///< [in] PTX version
  SortedIdxT *d_idx;

  //------------------------------------------------------------------------------
  // Constructor
  //------------------------------------------------------------------------------

  CUB_RUNTIME_FUNCTION __forceinline__ DispatchRadixArgSort(
      void *d_temp_storage, size_t &temp_storage_bytes, KeyIteratorT d_keys, OffsetT num_items,
      int begin_bit, int end_bit, cudaStream_t stream, int ptx_version, SortedIdxT *d_idx_out)
      : d_temp_storage(d_temp_storage),
        temp_storage_bytes(temp_storage_bytes),
        d_keys(d_keys),
        num_items(num_items),
        begin_bit(begin_bit),
        end_bit(end_bit),
        stream(stream),
        ptx_version(ptx_version),
        d_idx{d_idx_out} {}

  //------------------------------------------------------------------------------
  // Normal problem size invocation
  //------------------------------------------------------------------------------
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t InvokeOnesweep() {
    typedef typename DispatchRadixArgSort::MaxPolicy MaxPolicyT;
    // PortionOffsetT is used for offsets within a portion, and must be signed.
    typedef int PortionOffsetT;
    typedef PortionOffsetT AtomicOffsetT;

    // compute temporary storage size
    const int RADIX_BITS = ActivePolicyT::ONESWEEP_RADIX_BITS;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int ONESWEEP_ITEMS_PER_THREAD = ActivePolicyT::OnesweepPolicy::ITEMS_PER_THREAD;
    const int ONESWEEP_BLOCK_THREADS = ActivePolicyT::OnesweepPolicy::BLOCK_THREADS;
    const int ONESWEEP_TILE_ITEMS = ONESWEEP_ITEMS_PER_THREAD * ONESWEEP_BLOCK_THREADS;
    // portions handle inputs with >=2**30 elements, due to the way lookback works
    // for testing purposes, one portion is <= 2**28 elements
    const PortionOffsetT PORTION_SIZE = ((1 << 28) - 1) / ONESWEEP_TILE_ITEMS * ONESWEEP_TILE_ITEMS;
    int num_passes = ::cub::DivideAndRoundUp(end_bit - begin_bit, RADIX_BITS);
    OffsetT num_portions = static_cast<OffsetT>(::cub::DivideAndRoundUp(num_items, PORTION_SIZE));

    PortionOffsetT max_num_blocks = ::cub::DivideAndRoundUp(
        static_cast<int>(CUB_MIN(num_items, static_cast<OffsetT>(PORTION_SIZE))),
        ONESWEEP_TILE_ITEMS);

    std::size_t allocation_sizes[] = {
        // bins
        num_portions * num_passes * RADIX_DIGITS * sizeof(OffsetT),
        // lookback
        max_num_blocks * RADIX_DIGITS * sizeof(AtomicOffsetT),
        // counters
        num_portions * num_passes * sizeof(AtomicOffsetT),
        // index
        num_items * sizeof(SortedIdxT),
    };
    const int NUM_ALLOCATIONS = sizeof(allocation_sizes) / sizeof(allocation_sizes[0]);
    void *allocations[NUM_ALLOCATIONS] = {};
    ::cub::AliasTemporaries<NUM_ALLOCATIONS>(d_temp_storage, temp_storage_bytes, allocations,
                                             allocation_sizes);

    // just return if no temporary storage is provided
    cudaError_t error = cudaSuccess;
    if (d_temp_storage == nullptr) return error;

    OffsetT *d_bins = static_cast<OffsetT *>(allocations[0]);
    AtomicOffsetT *d_lookback = static_cast<AtomicOffsetT *>(allocations[1]);
    AtomicOffsetT *d_ctrs = static_cast<AtomicOffsetT *>(allocations[2]);
    SortedIdxT *d_idx_tmp = static_cast<SortedIdxT *>(allocations[3]);

    thrust::sequence(thrust::cuda::par.on(stream), d_idx, d_idx + num_items);
    ::cub::DoubleBuffer<SortedIdxT> d_idx_out{d_idx, d_idx_tmp};

    do {
      // initialization
      if (CubDebug(error = cudaMemsetAsync(
                       d_ctrs, 0, num_portions * num_passes * sizeof(AtomicOffsetT), stream))) {
        break;
      }

      // compute num_passes histograms with RADIX_DIGITS bins each
      if (CubDebug(error = cudaMemsetAsync(d_bins, 0, num_passes * RADIX_DIGITS * sizeof(OffsetT),
                                           stream))) {
        break;
      }
      int device = -1;
      int num_sms = 0;
      if (CubDebug(error = cudaGetDevice(&device))) break;
      if (CubDebug(error =
                       cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device))) {
        break;
      }

      auto s_idx_out = xgboost::common::Span<SortedIdxT>(d_idx_out.Current(), num_items);
      const int HISTO_BLOCK_THREADS = ActivePolicyT::HistogramPolicy::BLOCK_THREADS;
      int histo_blocks_per_sm = 1;
      auto histogram_kernel =
          DeviceRadixArgSortHistogramKernel<MaxPolicyT, IS_DESCENDING, KeyIteratorT, OffsetT,
                                            DigitExtractorT>;
      if (CubDebug(error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &histo_blocks_per_sm, histogram_kernel, HISTO_BLOCK_THREADS, 0))) {
        break;
      }
      error = THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                  histo_blocks_per_sm * num_sms, HISTO_BLOCK_THREADS, 0, stream)
                  .doit(histogram_kernel, d_bins, d_keys, num_items, begin_bit, end_bit);
      if (CubDebug(error)) {
        break;
      }

      error = detail::DebugSyncStream(stream);
      if (CubDebug(error)) {
        break;
      }

      // exclusive sums to determine starts
      const int SCAN_BLOCK_THREADS = ActivePolicyT::ExclusiveSumPolicy::BLOCK_THREADS;
      error = THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(num_passes,
                                                                      SCAN_BLOCK_THREADS, 0, stream)
                  .doit(DeviceRadixSortExclusiveSumKernel<MaxPolicyT, OffsetT>, d_bins);
      if (CubDebug(error)) {
        break;
      }

      error = detail::DebugSyncStream(stream);
      if (CubDebug(error)) {
        break;
      }

      auto d_keys = this->d_keys;
      static_assert(RADIX_BITS == 8);
      for (int current_bit = begin_bit, pass = 0; current_bit < end_bit;
           current_bit += RADIX_BITS, ++pass) {
        int num_bits = CUB_MIN(end_bit - current_bit, RADIX_BITS);
        for (OffsetT portion = 0; portion < num_portions; ++portion) {
          PortionOffsetT portion_num_items = static_cast<PortionOffsetT>(
              CUB_MIN(num_items - portion * PORTION_SIZE, static_cast<OffsetT>(PORTION_SIZE)));
          PortionOffsetT num_blocks =
              ::cub::DivideAndRoundUp(portion_num_items, ONESWEEP_TILE_ITEMS);
          if (CubDebug(error = cudaMemsetAsync(d_lookback, 0,
                                               num_blocks * RADIX_DIGITS * sizeof(AtomicOffsetT),
                                               stream))) {
            break;
          }

          auto s_idx_in = d_idx_out.Current();
          // fixme: this doesn't work well with idx when iterating through portion.
          auto key_in = MakeTransformIterator<KeyT>(thrust::make_counting_iterator(0ul),
                                                    SortedKeyOp<KeyIteratorT>{d_keys, s_idx_in});

          auto onesweep_kernel =
              DeviceRadixSortOnesweepKernel<MaxPolicyT, IS_DESCENDING, decltype(key_in), ValueT,
                                            OffsetT, PortionOffsetT, DigitExtractorT>;
          error =
              THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                  num_blocks, ONESWEEP_BLOCK_THREADS, 0, stream)
                  .doit(onesweep_kernel, d_lookback, d_ctrs + portion * num_passes + pass,
                        portion < num_portions - 1
                            ? d_bins + ((portion + 1) * num_passes + pass) * RADIX_DIGITS
                            : nullptr,
                        d_bins + (portion * num_passes + pass) * RADIX_DIGITS,
                        key_in + portion * PORTION_SIZE,
                        xgboost::common::Span<SortedIdxT>{d_idx_out.Alternate(), num_items},
                        xgboost::common::Span<SortedIdxT>{d_idx_out.Current(), num_items}.subspan(
                            portion * PORTION_SIZE),
                        portion_num_items, current_bit, num_bits);
          if (CubDebug(error)) {
            break;
          }

          error = detail::DebugSyncStream(stream);
          if (CubDebug(error)) {
            break;
          }
        }

        if (error != cudaSuccess) {
          break;
        }

        d_idx_out.selector ^= 1;
      }
    } while (false);

    return error;
  }

  //------------------------------------------------------------------------------
  // Chained policy invocation
  //------------------------------------------------------------------------------
  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t Invoke() {
    typedef typename DispatchRadixArgSort::MaxPolicy MaxPolicyT;
    // Return if empty problem, or if no bits to sort and double-buffering is used
    if (num_items == 0 || (begin_bit == end_bit)) {
      if (d_temp_storage == nullptr) {
        temp_storage_bytes = 1;
      }
      return cudaSuccess;
    }
    return InvokeOnesweep<ActivePolicyT>();
  }

  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t Dispatch(
      void *d_temp_storage, size_t &temp_storage_bytes, KeyIteratorT d_keys, OffsetT num_items,
      int begin_bit, int end_bit, cudaStream_t stream, SortedIdxT *d_idx_out) {
    typedef typename DispatchRadixArgSort::MaxPolicy MaxPolicyT;

    cudaError_t error;
    do {
      // Get PTX version
      int ptx_version = 0;
      if (CubDebug(error = ::cub::PtxVersion(ptx_version))) break;

      // Create dispatch functor
      DispatchRadixArgSort dispatch{d_temp_storage, temp_storage_bytes, d_keys,
                                    num_items,      begin_bit,          end_bit,
                                    stream,         ptx_version,        d_idx_out};

      // Dispatch to chained policy
      if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch))) {
        break;
      }
    } while (false);

    return error;
  }
};
}  // namespace cub_argsort
// NOLINTEND
