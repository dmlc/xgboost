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
 * agent_radix_sort_onesweep.cuh implements a stateful abstraction of CUDA
 * thread blocks for participating in the device one-sweep radix sort kernel.
 */

#pragma once

#include <thrust/iterator/transform_iterator.h>
#include <xgboost/span.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_rank.cuh>
#include <cub/block/block_store.cuh>
#include <cub/config.cuh>
#include <cub/util_ptx.cuh>

#include "../block/radix_rank_sort_operations.cuh"
#include "../util_type.cuh"

// NOLINTBEGIN
namespace cub_argsort {
template <typename ReturnT, typename IterT, typename FuncT>
XGBOOST_DEVICE thrust::transform_iterator<FuncT, IterT, ReturnT> MakeTransformIterator(IterT iter,
                                                                                       FuncT func) {
  return thrust::transform_iterator<FuncT, IterT, ReturnT>(iter, func);
}

struct EntryExtractor {
  static_assert(sizeof(xgboost::Entry) == 8);
  std::uint32_t bit_start{0};
  // Need special handling for floating point.
  ::cub::ShiftDigitExtractor<float> shift_fv;
  ::cub::ShiftDigitExtractor<xgboost::bst_feature_t> shift_idx;

  using UnsignedBits = typename MyTraits<xgboost::Entry>::UnsignedBits;

  explicit XGBOOST_DEVICE EntryExtractor(std::uint32_t bit_start = 0, std::uint32_t num_bits = 0)
      : bit_start{bit_start},
        shift_fv{bit_start >= 8 * sizeof(float)
                     ? bit_start - static_cast<std::uint32_t>(8 * sizeof(float))
                     : bit_start,
                 num_bits},
        shift_idx{bit_start >= 8 * sizeof(float)
                      ? bit_start - static_cast<std::uint32_t>(8 * sizeof(float))
                      : bit_start,
                  num_bits} {}

  __device__ __forceinline__ std::uint32_t Digit(UnsignedBits key) {
    static_assert(sizeof(UnsignedBits) == sizeof(xgboost::Entry));
    auto* ptr = reinterpret_cast<std::uint32_t*>(&key);
    std::uint32_t f;

    if (bit_start < sizeof(float) * 8) {
      auto v = ptr[0];  // fvalue
      std::memcpy(&f, &v, sizeof(f));
      static_assert(sizeof(f) == sizeof(v));
      return shift_fv.Digit(f);
    } else {
      auto v = ptr[1];  // findex
      std::memcpy(&f, &v, sizeof(f));
      return shift_idx.Digit(f);
    }
  }
};

/** \brief cub::RadixSortStoreAlgorithm enumerates different algorithms to write
 * partitioned elements (keys, values) stored in shared memory into global
 * memory. Currently applies only to writing 4B keys in full tiles; in all other cases,
 * RADIX_SORT_STORE_DIRECT is used.
 */
enum RadixSortStoreAlgorithm
{
    /** \brief Elements are statically distributed among block threads, which write them
     * into the appropriate partition in global memory. This results in fewer instructions
     * and more writes in flight at a given moment, but may generate more transactions. */
    RADIX_SORT_STORE_DIRECT,
    /** \brief Elements are distributed among warps in a block distribution. Each warp
     * goes through its elements and tries to write them while minimizing the number of
     * memory transactions. This results in fewer memory transactions, but more
     * instructions and less writes in flight at a given moment. */
    RADIX_SORT_STORE_ALIGNED
};

template <
    int NOMINAL_BLOCK_THREADS_4B,
    int NOMINAL_ITEMS_PER_THREAD_4B,
    typename ComputeT,
    /** \brief Number of private histograms to use in the ranker;
        ignored if the ranking algorithm is not one of RADIX_RANK_MATCH_EARLY_COUNTS_* */
    int _RANK_NUM_PARTS,
    /** \brief Ranking algorithm used in the onesweep kernel. Only algorithms that
      support warp-strided key arrangement and count callbacks are supported. */
    ::cub::RadixRankAlgorithm _RANK_ALGORITHM,
    ::cub::BlockScanAlgorithm _SCAN_ALGORITHM,
    RadixSortStoreAlgorithm _STORE_ALGORITHM,
    int _RADIX_BITS,
    typename ScalingType = ::cub::RegBoundScaling<
        NOMINAL_BLOCK_THREADS_4B, NOMINAL_ITEMS_PER_THREAD_4B, ComputeT> >
struct AgentRadixSortOnesweepPolicy : ScalingType
{
    enum
    {
        RANK_NUM_PARTS = _RANK_NUM_PARTS,
        RADIX_BITS = _RADIX_BITS,
    };
  static const ::cub::RadixRankAlgorithm RANK_ALGORITHM = _RANK_ALGORITHM;
  static const ::cub::BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
  static const RadixSortStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;
};

template <typename AgentRadixSortOnesweepPolicy, bool IS_DESCENDING, typename KeyIteratorT,
          typename ValueT, typename OffsetT, typename PortionOffsetT, typename DigitExtractor>
struct AgentRadixSortOnesweep {
  // constants
  enum {
    ITEMS_PER_THREAD = AgentRadixSortOnesweepPolicy::ITEMS_PER_THREAD,
    KEYS_ONLY = std::is_same<ValueT, ::cub::NullType>::value,
    BLOCK_THREADS = AgentRadixSortOnesweepPolicy::BLOCK_THREADS,
    RANK_NUM_PARTS = AgentRadixSortOnesweepPolicy::RANK_NUM_PARTS,
    TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
    RADIX_BITS = AgentRadixSortOnesweepPolicy::RADIX_BITS,
    RADIX_DIGITS = 1 << RADIX_BITS,
    BINS_PER_THREAD = (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS,
    FULL_BINS = BINS_PER_THREAD * BLOCK_THREADS == RADIX_DIGITS,
    WARP_THREADS = CUB_PTX_WARP_THREADS,
    BLOCK_WARPS = BLOCK_THREADS / WARP_THREADS,
    WARP_MASK = ~0,
    LOOKBACK_PARTIAL_MASK = 1 << (PortionOffsetT(sizeof(PortionOffsetT)) * 8 - 2),
    LOOKBACK_GLOBAL_MASK = 1 << (PortionOffsetT(sizeof(PortionOffsetT)) * 8 - 1),
    LOOKBACK_KIND_MASK = LOOKBACK_PARTIAL_MASK | LOOKBACK_GLOBAL_MASK,
    LOOKBACK_VALUE_MASK = ~LOOKBACK_KIND_MASK,
  };

  using KeyT = typename std::iterator_traits<KeyIteratorT>::value_type;

  using UnsignedBits = typename MyTraits<KeyT>::UnsignedBits;
  using AtomicOffsetT = PortionOffsetT;

  static const ::cub::RadixRankAlgorithm RANK_ALGORITHM =
      AgentRadixSortOnesweepPolicy::RANK_ALGORITHM;
  static const ::cub::BlockScanAlgorithm SCAN_ALGORITHM =
      AgentRadixSortOnesweepPolicy::SCAN_ALGORITHM;
  static const RadixSortStoreAlgorithm STORE_ALGORITHM =
      sizeof(UnsignedBits) == sizeof(uint32_t) ? AgentRadixSortOnesweepPolicy::STORE_ALGORITHM
                                               : RADIX_SORT_STORE_DIRECT;
  using Twiddle = RadixSortTwiddle<IS_DESCENDING, KeyT>;

  static_assert(RANK_ALGORITHM == ::cub::RADIX_RANK_MATCH ||
                    RANK_ALGORITHM == ::cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY ||
                    RANK_ALGORITHM == ::cub::RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR,
                "for onesweep agent, the ranking algorithm must warp-strided key arrangement");

  using BlockRadixRankT = std::conditional_t<
      RANK_ALGORITHM == ::cub::RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR,
      ::cub::BlockRadixRankMatchEarlyCounts<BLOCK_THREADS, RADIX_BITS, false, SCAN_ALGORITHM,
                                            ::cub::WARP_MATCH_ATOMIC_OR, RANK_NUM_PARTS>,
      std::conditional_t<
          RANK_ALGORITHM == ::cub::RADIX_RANK_MATCH,
          ::cub::BlockRadixRankMatch<BLOCK_THREADS, RADIX_BITS, false, SCAN_ALGORITHM>,
          ::cub::BlockRadixRankMatchEarlyCounts<BLOCK_THREADS, RADIX_BITS, false, SCAN_ALGORITHM,
                                                ::cub::WARP_MATCH_ANY, RANK_NUM_PARTS>>>;

  // temporary storage
  struct TempStorage_ {
    union {
      UnsignedBits keys_out[TILE_ITEMS];
      ValueT values_out[TILE_ITEMS];
      typename BlockRadixRankT::TempStorage rank_temp_storage;
    };
    union {
      OffsetT global_offsets[RADIX_DIGITS];
      PortionOffsetT block_idx;
    };
    };

    using TempStorage = ::cub::Uninitialized<TempStorage_>;

    // thread variables
    TempStorage_& s;

    // kernel parameters
    AtomicOffsetT* d_lookback;
    AtomicOffsetT* d_ctrs;
    OffsetT* d_bins_out;
    const OffsetT*  d_bins_in;
  // const UnsignedBits
    KeyIteratorT d_keys_in;
    xgboost::common::Span<ValueT> d_values_out;
    const ValueT* d_values_in;
    // common::Span<std::uint32_t> d_idx_out;
    PortionOffsetT num_items;
    DigitExtractor digit_extractor;

    // other thread variables
    int warp;
    int lane;
    PortionOffsetT block_idx;
    bool full_block;

    // helper methods
    __device__ __forceinline__ int Digit(UnsignedBits key)
    {
        return digit_extractor.Digit(key);
    }

    __device__ __forceinline__ int ThreadBin(int u)
    {
        return threadIdx.x * BINS_PER_THREAD + u;
    }

    __device__ __forceinline__ void LookbackPartial(int (&bins)[BINS_PER_THREAD])
    {
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u)
        {
            int bin = ThreadBin(u);
            if (FULL_BINS || bin < RADIX_DIGITS)
            {
                // write the local sum into the bin
                AtomicOffsetT& loc = d_lookback[block_idx * RADIX_DIGITS + bin];
                PortionOffsetT value = bins[u] | LOOKBACK_PARTIAL_MASK;
                ::cub::ThreadStore<::cub::STORE_VOLATILE>(&loc, value);
            }
        }
    }

    struct CountsCallback {
        using AgentT =
            AgentRadixSortOnesweep<AgentRadixSortOnesweepPolicy, IS_DESCENDING, KeyIteratorT,
                                   ValueT, OffsetT, PortionOffsetT, DigitExtractor>;
        AgentT& agent;
        int (&bins)[BINS_PER_THREAD];
        UnsignedBits (&keys)[ITEMS_PER_THREAD];
        static const bool EMPTY = false;

        __device__ __forceinline__ CountsCallback(AgentT& agent, int (&bins)[BINS_PER_THREAD],
                                                  UnsignedBits (&keys)[ITEMS_PER_THREAD])
            : agent(agent), bins(bins), keys(keys) {}
        __device__ __forceinline__ void operator()(int (&other_bins)[BINS_PER_THREAD]) {
#pragma unroll
            for (int u = 0; u < BINS_PER_THREAD; ++u) {
                bins[u] = other_bins[u];
            }
            agent.LookbackPartial(bins);

            // agent.TryShortCircuit(keys, bins);
        }
    };

    __device__ __forceinline__ void LookbackGlobal(int (&bins)[BINS_PER_THREAD])
    {
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u)
        {
            int bin = ThreadBin(u);
            if (FULL_BINS || bin < RADIX_DIGITS)
            {
                PortionOffsetT inc_sum = bins[u];
                int want_mask = ~0;
                // backtrack as long as necessary
                for (PortionOffsetT block_jdx = block_idx - 1; block_jdx >= 0; --block_jdx)
                {
                    // wait for some value to appear
                    PortionOffsetT value_j = 0;
                    AtomicOffsetT& loc_j = d_lookback[block_jdx * RADIX_DIGITS + bin];
                    do {
                        __threadfence_block(); // prevent hoisting loads from loop
                        value_j = ::cub::ThreadLoad<::cub::LOAD_VOLATILE>(&loc_j);
                    } while (value_j == 0);

                    inc_sum += value_j & LOOKBACK_VALUE_MASK;
                    want_mask = ::cub::WARP_BALLOT((value_j & LOOKBACK_GLOBAL_MASK) == 0, want_mask);
                    if (value_j & LOOKBACK_GLOBAL_MASK) break;
                }
                AtomicOffsetT& loc_i = d_lookback[block_idx * RADIX_DIGITS + bin];
                PortionOffsetT value_i = inc_sum | LOOKBACK_GLOBAL_MASK;
                ::cub::ThreadStore<::cub::STORE_VOLATILE>(&loc_i, value_i);
                s.global_offsets[bin] += inc_sum - bins[u];
            }
        }
    }

    __device__ __forceinline__ void LoadKeys(OffsetT tile_offset,
                                             UnsignedBits (&keys)[ITEMS_PER_THREAD]) {
        auto it = MakeTransformIterator<UnsignedBits>(d_keys_in, [] __device__(auto const& v) {
          static_assert(sizeof(std::remove_reference_t<decltype(v)>) == sizeof(UnsignedBits));
          UnsignedBits dst;
          std::memcpy(&dst, &v, sizeof(v));
          return dst;
        });

        if (full_block) {
            ::cub::LoadDirectWarpStriped(threadIdx.x, it + tile_offset, keys);
        } else {
            ::cub::LoadDirectWarpStriped(threadIdx.x, it + tile_offset, keys,
                                         num_items - tile_offset, Twiddle::DefaultKey());
        }

#pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u) {
            keys[u] = Twiddle::In(keys[u]);
        }
    }

    __device__ __forceinline__ void LoadValues(OffsetT tile_offset,
                                               ValueT (&values)[ITEMS_PER_THREAD]) {
        if (full_block) {
            ::cub::LoadDirectWarpStriped(threadIdx.x, d_values_in + tile_offset, values);
        } else {
            int tile_items = num_items - tile_offset;
            ::cub::LoadDirectWarpStriped(threadIdx.x, d_values_in + tile_offset, values,
                                         tile_items);
        }
    }

    __device__ __forceinline__
    void ScatterKeysShared(UnsignedBits (&keys)[ITEMS_PER_THREAD], int (&ranks)[ITEMS_PER_THREAD])
    {
        // write to shared memory
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            s.keys_out[ranks[u]] = keys[u];
        }
    }

    __device__ __forceinline__
    void ScatterValuesShared(ValueT (&values)[ITEMS_PER_THREAD], int (&ranks)[ITEMS_PER_THREAD])
    {
        // write to shared memory
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            s.values_out[ranks[u]] = values[u];
        }
    }

    __device__ __forceinline__ void LoadBinsToOffsetsGlobal(int (&offsets)[BINS_PER_THREAD])
    {
        // global offset - global part
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u)
        {
            int bin = ThreadBin(u);
            if (FULL_BINS || bin < RADIX_DIGITS)
            {
                s.global_offsets[bin] = d_bins_in[bin] - offsets[u];
            }
        }
    }

    __device__ __forceinline__ void UpdateBinsGlobal(int (&bins)[BINS_PER_THREAD],
                                                     int (&offsets)[BINS_PER_THREAD])
    {
        bool last_block = (block_idx + 1) * TILE_ITEMS >= num_items;
        if (d_bins_out != NULL && last_block)
        {
            #pragma unroll
            for (int u = 0; u < BINS_PER_THREAD; ++u)
            {
                int bin = ThreadBin(u);
                if (FULL_BINS || bin < RADIX_DIGITS)
                {
                    d_bins_out[bin] = s.global_offsets[bin] + offsets[u] + bins[u];
                }
            }
        }
    }

    template <bool FULL_TILE>
    __device__ __forceinline__ void ScatterValuesGlobalDirect(int (&digits)[ITEMS_PER_THREAD])
    {
        int tile_items = FULL_TILE ? TILE_ITEMS : num_items - block_idx * TILE_ITEMS;
#pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            int idx = threadIdx.x + u * BLOCK_THREADS;
            ValueT value = s.values_out[idx];
            OffsetT global_idx = idx + s.global_offsets[digits[u]];
            if (FULL_TILE || idx < tile_items) {
                d_values_out[global_idx] = value;
            }
            ::cub::WARP_SYNC(WARP_MASK);
        }
    }

    __device__ __forceinline__ void ScatterValuesGlobal(int (&digits)[ITEMS_PER_THREAD]) {
        // write block data to global memory
        if (full_block) {
            ScatterValuesGlobalDirect<true>(digits);
        } else {
            ScatterValuesGlobalDirect<false>(digits);
        }
    }

    __device__ __forceinline__ void ComputeKeyDigits(int (&digits)[ITEMS_PER_THREAD])
    {
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            int idx = threadIdx.x + u * BLOCK_THREADS;
            digits[u] = Digit(s.keys_out[idx]);
        }
    }

    __device__ __forceinline__ void GatherScatterValues(
        int (&ranks)[ITEMS_PER_THREAD], ::cub::Int2Type<false> keys_only)
    {
        // compute digits corresponding to the keys
        int digits[ITEMS_PER_THREAD];
        ComputeKeyDigits(digits);

        // load values
        ValueT values[ITEMS_PER_THREAD];
        LoadValues(block_idx * TILE_ITEMS, values);

        // scatter values
        cub::CTA_SYNC();
        ScatterValuesShared(values, ranks);

        cub::CTA_SYNC();
        ScatterValuesGlobal(digits);
    }

    __device__ __forceinline__ void Process()
    {
        // load keys
        // if warp1 < warp2, all elements of warp1 occur before those of warp2
        // in the source array
        UnsignedBits keys[ITEMS_PER_THREAD];
        LoadKeys(block_idx * TILE_ITEMS, keys);

        // rank keys
        int ranks[ITEMS_PER_THREAD];
        int exclusive_digit_prefix[BINS_PER_THREAD];
        int bins[BINS_PER_THREAD];
        BlockRadixRankT(s.rank_temp_storage).RankKeys(
            keys, ranks, digit_extractor, exclusive_digit_prefix,
            CountsCallback(*this, bins, keys));

        // scatter keys in shared memory
        ::cub::CTA_SYNC();
        ScatterKeysShared(keys, ranks);

        // compute global offsets
        LoadBinsToOffsetsGlobal(exclusive_digit_prefix);
        LookbackGlobal(bins);
        UpdateBinsGlobal(bins, exclusive_digit_prefix);

        // scatter keys in global memory
        ::cub::CTA_SYNC();

        // scatter values if necessary
        GatherScatterValues(ranks, ::cub::Int2Type<KEYS_ONLY>());
    }

    __device__ __forceinline__ //
    AgentRadixSortOnesweep(TempStorage &temp_storage,
                           AtomicOffsetT *d_lookback,
                           AtomicOffsetT *d_ctrs,
                           OffsetT *d_bins_out,
                           const OffsetT *d_bins_in,
                           KeyIteratorT d_keys_in,
                           xgboost::common::Span<ValueT> d_values_out,
                           const ValueT *d_values_in,
                           PortionOffsetT num_items,
                           DigitExtractor de)
        : s(temp_storage.Alias())
        , d_lookback(d_lookback)
        , d_ctrs(d_ctrs)
        , d_bins_out(d_bins_out)
        , d_bins_in(d_bins_in)
        , d_keys_in{d_keys_in}
        , d_values_out(d_values_out)
        , d_values_in(d_values_in)
        // , d_idx_out{d_idx_out}
        , num_items(num_items)
        , digit_extractor{de}
        , warp(threadIdx.x / WARP_THREADS)
        , lane(::cub::LaneId())
    {
        // initialization
        if (threadIdx.x == 0)
        {
            s.block_idx = atomicAdd(d_ctrs, 1);
        }
        ::cub::CTA_SYNC();
        block_idx = s.block_idx;
        full_block = (block_idx + 1) * TILE_ITEMS <= num_items;
    }
};
}  // namespace cub_argsort
// NOLINTEND
