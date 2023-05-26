/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
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
 * agent_radix_sort_histogram.cuh implements a stateful abstraction of CUDA
 * thread blocks for participating in the device histogram kernel used for
 * one-sweep radix sorting.
 */

#pragma once

#include <thrust/iterator/transform_iterator.h>

#include <cub/block/block_load.cuh>
#include <cub/config.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/util_math.cuh>
#include <memory>  // for memcpy

#include "../block/radix_rank_sort_operations.cuh"
#include "../util_type.cuh"

// NOLINTBEGIN
namespace cub_argsort {
template <
  int _BLOCK_THREADS,
  int _ITEMS_PER_THREAD,
  int NOMINAL_4B_NUM_PARTS,
  typename ComputeT,
  int _RADIX_BITS>
struct AgentRadixSortHistogramPolicy
{
    enum
    {
        BLOCK_THREADS = _BLOCK_THREADS,
        ITEMS_PER_THREAD = _ITEMS_PER_THREAD,
        /** NUM_PARTS is the number of private histograms (parts) each histogram is split
         * into. Each warp lane is assigned to a specific part based on the lane
         * ID. However, lanes with the same ID in different warp use the same private
         * histogram. This arrangement helps reduce the degree of conflicts in atomic
         * operations. */
        NUM_PARTS = CUB_MAX(1, NOMINAL_4B_NUM_PARTS * 4 / CUB_MAX(sizeof(ComputeT), 4)),
        RADIX_BITS = _RADIX_BITS,
    };
};

template <
    int _BLOCK_THREADS,
    int _RADIX_BITS>
struct AgentRadixSortExclusiveSumPolicy
{
    enum
    {
        BLOCK_THREADS = _BLOCK_THREADS,
        RADIX_BITS = _RADIX_BITS,
    };
};

template <typename AgentRadixSortHistogramPolicy, bool IS_DESCENDING, typename KeyIteratorT,
          typename OffsetT, typename DigitExtractorT>
struct AgentRadixSortHistogram {
    using KeyT = typename std::iterator_traits<KeyIteratorT>::value_type;
    // constants
    enum {
      ITEMS_PER_THREAD = AgentRadixSortHistogramPolicy::ITEMS_PER_THREAD,
      BLOCK_THREADS = AgentRadixSortHistogramPolicy::BLOCK_THREADS,
      TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
      RADIX_BITS = AgentRadixSortHistogramPolicy::RADIX_BITS,
      RADIX_DIGITS = 1 << RADIX_BITS,
      MAX_NUM_PASSES = (sizeof(KeyT) * 8 + RADIX_BITS - 1) / RADIX_BITS,
      NUM_PARTS = AgentRadixSortHistogramPolicy::NUM_PARTS,
    };

    using Twiddle = cub_argsort::RadixSortTwiddle<IS_DESCENDING, KeyT>;
    using ShmemCounterT = std::uint32_t;
    using ShmemAtomicCounterT = ShmemCounterT;
    using UnsignedBits = typename cub_argsort::MyTraits<
        typename std::iterator_traits<KeyIteratorT>::value_type>::UnsignedBits;

    struct _TempStorage {
      ShmemAtomicCounterT bins[MAX_NUM_PASSES][RADIX_DIGITS][NUM_PARTS];
    };

    struct TempStorage : ::cub::Uninitialized<_TempStorage> {};

    // thread fields
    // shared memory storage
    _TempStorage& s;

    // bins for the histogram
    OffsetT* d_bins_out;

    // data to compute the histogram
    KeyIteratorT d_keys_in;

    // number of data items
    OffsetT num_items;

    // begin and end bits for sorting
    int begin_bit, end_bit;

    // number of sorting passes
    int num_passes;

    __device__ __forceinline__ AgentRadixSortHistogram(TempStorage& temp_storage,
                                                       OffsetT* d_bins_out, KeyIteratorT d_keys_in,
                                                       OffsetT num_items, int begin_bit,
                                                       int end_bit)
        : s(temp_storage.Alias()),
          d_bins_out(d_bins_out),
          d_keys_in{d_keys_in},
          num_items(num_items),
          begin_bit(begin_bit),
          end_bit(end_bit),
          num_passes((end_bit - begin_bit + RADIX_BITS - 1) / RADIX_BITS) {}

    __device__ __forceinline__ void Init() {
// Initialize bins to 0.
#pragma unroll
      for (int bin = threadIdx.x; bin < RADIX_DIGITS; bin += BLOCK_THREADS) {
#pragma unroll
        for (int pass = 0; pass < num_passes; ++pass) {
#pragma unroll
          for (int part = 0; part < NUM_PARTS; ++part) {
            s.bins[pass][bin][part] = 0;
          }
        }
      }
      ::cub::CTA_SYNC();
    }

    __device__ __forceinline__ void LoadTileKeys(OffsetT tile_offset,
                                                 UnsignedBits (&keys)[ITEMS_PER_THREAD]) {
      // tile_offset < num_items always, hence the line below works
      bool full_tile = num_items - tile_offset >= TILE_ITEMS;
      auto it = thrust::make_transform_iterator(d_keys_in, [] __device__(auto const& v) {
        static_assert(sizeof(std::remove_reference_t<decltype(v)>) == sizeof(UnsignedBits));
        UnsignedBits dst;
        std::memcpy(&dst, &v, sizeof(v));
        return dst;
      });
      if (full_tile) {
        ::cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, it + tile_offset, keys);
      } else {
        ::cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, it + tile_offset, keys,
                                                num_items - tile_offset, Twiddle::DefaultKey());
      }

#pragma unroll
      for (int u = 0; u < ITEMS_PER_THREAD; ++u) {
        keys[u] = Twiddle::In(keys[u]);
      }
    }

    __device__ __forceinline__ void AccumulateSharedHistograms(
        OffsetT tile_offset, UnsignedBits (&keys)[ITEMS_PER_THREAD]) {
      int part = ::cub::LaneId() % NUM_PARTS;
#pragma unroll
      for (int current_bit = begin_bit, pass = 0; current_bit < end_bit;
           current_bit += RADIX_BITS, ++pass) {
        int num_bits = CUB_MIN(RADIX_BITS, end_bit - current_bit);
        DigitExtractorT digit_extractor(current_bit, num_bits);
#pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u) {
          int bin = digit_extractor.Digit(keys[u]);
          // Using cuda::atomic<> results in lower performance on GP100,
          // so atomicAdd() is used instead.
          atomicAdd(&s.bins[pass][bin][part], 1);
        }
      }
    }

    __device__ __forceinline__ void AccumulateGlobalHistograms() {
#pragma unroll
      for (int bin = threadIdx.x; bin < RADIX_DIGITS; bin += BLOCK_THREADS) {
#pragma unroll
        for (int pass = 0; pass < num_passes; ++pass) {
          OffsetT count = ::cub::internal::ThreadReduce(s.bins[pass][bin], ::cub::Sum());
          if (count > 0) {
            // Using cuda::atomic<> here would also require using it in
            // other kernels. However, other kernels of onesweep sorting
            // (ExclusiveSum, Onesweep) don't need atomic
            // access. Therefore, atomicAdd() is used, until
            // cuda::atomic_ref<> becomes available.
            atomicAdd(&d_bins_out[pass * RADIX_DIGITS + bin], count);
          }
        }
      }
    }

    __device__ __forceinline__ void Process() {
      // Within a portion, avoid overflowing (u)int32 counters.
      // Between portions, accumulate results in global memory.
      const OffsetT MAX_PORTION_SIZE = 1 << 30;
      OffsetT num_portions = ::cub::DivideAndRoundUp(num_items, MAX_PORTION_SIZE);
      for (OffsetT portion = 0; portion < num_portions; ++portion) {
        // Reset the counters.
        Init();
        ::cub::CTA_SYNC();

        // Process the tiles.
        OffsetT portion_offset = portion * MAX_PORTION_SIZE;
        OffsetT portion_size = CUB_MIN(MAX_PORTION_SIZE, num_items - portion_offset);
        for (OffsetT offset = blockIdx.x * TILE_ITEMS; offset < portion_size;
             offset += TILE_ITEMS * gridDim.x) {
          OffsetT tile_offset = portion_offset + offset;
          UnsignedBits keys[ITEMS_PER_THREAD];
          LoadTileKeys(tile_offset, keys);
          AccumulateSharedHistograms(tile_offset, keys);
        }
        ::cub::CTA_SYNC();

        // Accumulate the result in global memory.
        AccumulateGlobalHistograms();
        ::cub::CTA_SYNC();
      }
    }
};
}  // namespace cub_argsort
// NOLINTEND
