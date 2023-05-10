
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * @file cub::DeviceRadixSort provides device-wide, parallel operations for
 *       computing a radix sort across a sequence of data items residing within
 *       device-accessible memory.
 */

#pragma once

#include <cub/config.cuh>
#include <cub/util_deprecated.cuh>

#include "dispatch/dispatch_radix_sort.cuh"
#include "xgboost/span.h"

namespace cub_argsort {
namespace detail {

/**
 * ChooseOffsetT checks NumItemsT, the type of the num_items parameter, and
 * selects the offset type based on it.
 */
template <typename NumItemsT>
struct ChooseOffsetT {
  // NumItemsT must be an integral type (but not bool).
  static_assert(std::is_integral<NumItemsT>::value &&
                    !std::is_same<typename std::remove_cv<NumItemsT>::type, bool>::value,
                "NumItemsT must be an integral type, but not bool");

  // Unsigned integer type for global offsets.
  using Type =
      typename std::conditional<sizeof(NumItemsT) <= 4, std::uint32_t, unsigned long long>::type;
};
}  // namespace detail

template <typename DigitExtractor>
struct DeviceRadixSort {
  template <typename KeyIteratorT, typename NumItemsT, typename SortedIdxT>
  CUB_RUNTIME_FUNCTION static cudaError_t Argsort(void *d_temp_storage,
                                                  std::size_t &temp_storage_bytes,
                                                  KeyIteratorT d_keys_in, SortedIdxT *d_idx_out,
                                                  NumItemsT num_items,
                                                  cudaStream_t stream = nullptr) {
    // Unsigned integer type for global offsets.
    using OffsetT = typename detail::ChooseOffsetT<NumItemsT>::Type;
    using KeyT = typename std::iterator_traits<KeyIteratorT>::value_type;

    int constexpr kBeginBit = 0;
    int constexpr kEndBit = sizeof(KeyT) * 8;

    return DispatchRadixArgSort<false, KeyIteratorT, ::cub::NullType, OffsetT, SortedIdxT,
                                DigitExtractor>::Dispatch(d_temp_storage, temp_storage_bytes,
                                                          d_keys_in,
                                                          static_cast<OffsetT>(num_items),
                                                          kBeginBit, kEndBit, stream, d_idx_out);
  }
};
}  // namespace cub_argsort
