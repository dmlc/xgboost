#ifndef SEGMENTED_UNIQUES_H_
#define SEGMENTED_UNIQUES_H_

#include <thrust/unique.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>
#include "device_helpers.cuh"

template <typename KeyInIt, typename KeyOutIt, typename ValInIt,
          typename ValOutIt, typename Comp>
size_t
SegmentedUnique(KeyInIt key_first, KeyInIt key_last, ValInIt val_first,
                ValInIt val_last, KeyOutIt key_out, ValOutIt val_out,
                Comp comp) {
  using Key = thrust::pair<size_t, typename thrust::iterator_traits<ValInIt>::value_type>;
  auto unique_key_it = dh::MakeTransformIterator<Key>(
      thrust::make_counting_iterator(static_cast<size_t>(0)),
      [=] __device__(size_t i) {
        size_t column =
            thrust::upper_bound(thrust::seq, key_first, key_last, i) - 1 -
            key_first;
        return thrust::make_pair(column, *(val_first + i));
      });
  size_t n_segments = key_last - key_first;
  size_t n_inputs = std::distance(val_first, val_last);
  dh::caching_device_vector<Key> keys(n_inputs);

  auto uniques_ret = thrust::unique_by_key_copy(
      thrust::device, unique_key_it, unique_key_it + n_inputs,
      val_first, keys.begin(), val_out,
      [=] __device__(Key const &l, Key const &r) {
        if (l.first == r.first) {
          return comp(l.second, r.second);
        }
        return false;
      });
  auto n_uniques = uniques_ret.first - keys.begin();
  auto encode_it = dh::MakeTransformIterator<size_t>(
      keys.begin(), [] __device__(Key k) { return k.first; });
  auto new_end = thrust::reduce_by_key(
      thrust::device, encode_it, encode_it + n_uniques,
      thrust::make_constant_iterator(1ul), thrust::make_discard_iterator(),
      key_out);

  thrust::exclusive_scan(thrust::device, key_out, key_out + n_segments, key_out,
                         0);
  return n_uniques;
}

#endif  // SEGMENTED_UNIQUES_H_