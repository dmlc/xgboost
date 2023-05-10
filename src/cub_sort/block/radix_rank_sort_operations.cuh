#pragma once

#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/config.cuh>
#include <cub/util_ptx.cuh>

#include "../util_type.cuh"

namespace cub_argsort {
/** \brief Twiddling keys for radix sort. */
template <bool IS_DESCENDING, typename KeyT>
struct RadixSortTwiddle {
  using TraitsT = MyTraits<KeyT>;
  using UnsignedBits = typename TraitsT::UnsignedBits;
  static __host__ __device__ __forceinline__ UnsignedBits In(UnsignedBits key) {
    key = TraitsT::TwiddleIn(key);
    if (IS_DESCENDING) key = ~key;
    return key;
  }
  static __host__ __device__ __forceinline__ UnsignedBits Out(UnsignedBits key) {
    if (IS_DESCENDING) key = ~key;
    key = TraitsT::TwiddleOut(key);
    return key;
  }
  static __host__ __device__ __forceinline__ UnsignedBits DefaultKey() {
    return Out(~UnsignedBits(0));
  }
};
}  // namespace cub_argsort
