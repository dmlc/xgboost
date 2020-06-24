#ifndef SEGMENTED_UNIQUES_H_
#define SEGMENTED_UNIQUES_H_

#include <thrust/unique.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>

#include "xgboost/span.h"
#include "device_helpers.cuh"

template <typename Key, typename KeyOutIt>
struct OutputOp{
  KeyOutIt key_out;

  Key const& __device__ operator()(Key const& key) const {
    atomicAdd(&(*(key_out + key.first)), 1);
    return key;
  }
};

template <typename KeyInIt, typename KeyOutIt, typename ValInIt,
          typename ValOutIt, typename Comp>
size_t
SegmentedUnique(KeyInIt key_first, KeyInIt key_last, ValInIt val_first,
                ValInIt val_last, KeyOutIt key_out, ValOutIt val_out,
                Comp comp) {
  using Key = thrust::pair<size_t, typename thrust::iterator_traits<ValInIt>::value_type>;
  dh::XGBCachingDeviceAllocator<char> alloc;
  auto unique_key_it = dh::MakeTransformIterator<Key>(
      thrust::make_counting_iterator(static_cast<size_t>(0)),
      [=] __device__(size_t i) {
        size_t column =
            thrust::upper_bound(thrust::seq, key_first, key_last, i) - 1 -
            key_first;
        return thrust::make_pair(column, *(val_first + i));
      });
  size_t n_segments = key_last - key_first;
  thrust::fill(thrust::device, key_out, key_out + n_segments, 0);
  size_t n_inputs = std::distance(val_first, val_last);

  auto out = thrust::make_transform_output_iterator(thrust::make_discard_iterator(), OutputOp<Key, KeyOutIt>{key_out});
  auto uniques_ret = thrust::unique_by_key_copy(
      thrust::cuda::par(alloc), unique_key_it, unique_key_it + n_inputs,
      val_first, out, val_out,
      [=] __device__(Key const &l, Key const &r) {
        if (l.first == r.first) {
          return comp(l.second, r.second);
        }
        return false;
      });
  auto n_uniques = uniques_ret.second - val_out;
  CHECK_LE(n_uniques, n_inputs);
  thrust::exclusive_scan(thrust::cuda::par(alloc), key_out, key_out + n_segments, key_out,
                         0);
  return n_uniques;
}
#endif  // SEGMENTED_UNIQUES_H_