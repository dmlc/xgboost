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
struct SegmentedUniqueReduceOp {
  KeyOutIt key_out;

  Key const& __device__ operator()(Key const& key) const {
    auto constexpr kOne = static_cast<std::remove_reference_t<decltype(*(key_out + key.first))>>(1);
    atomicAdd(&(*(key_out + key.first)), kOne);
    return key;
  }
};

/* \brief Segmented unique function.  Keys are pointers to segments with key_last -
 *        key_first = n_segments + 1.
 *
 * \pre   Input segment and output segment must not overlap.
 * \pre   Segment type must be compatible with `atomicAdd`.
 *
 * \param key_first Beginning iterator of segments.
 * \param key_last  End iterator of segments.
 * \param val_first Beginning iterator of values.
 * \param val_last  End iterator of values.
 * \param key_out   Output iterator of segments.
 * \param val_out   Output iterator of values.
 *
 * \return Number of unique values in total.
 */
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
        size_t seg = dh::SegmentId(key_first, key_last, i);
        return thrust::make_pair(seg, *(val_first + i));
      });
  size_t segments_len = key_last - key_first;
  thrust::fill(thrust::device, key_out, key_out + segments_len, 0);
  size_t n_inputs = std::distance(val_first, val_last);
  // Reduce the number of uniques elements per segment, avoid creating an intermediate
  // array for `reduce_by_key`.  It's limited by the types that atomicAdd supports.  For
  // example, size_t is not supported as of CUDA 10.2.
  auto reduce_it = thrust::make_transform_output_iterator(
      thrust::make_discard_iterator(), SegmentedUniqueReduceOp<Key, KeyOutIt>{key_out});
  auto uniques_ret = thrust::unique_by_key_copy(
      thrust::cuda::par(alloc), unique_key_it, unique_key_it + n_inputs,
      val_first, reduce_it, val_out,
      [=] __device__(Key const &l, Key const &r) {
        if (l.first == r.first) {
          // In the same segment.
          return comp(l.second, r.second);
        }
        return false;
      });
  auto n_uniques = uniques_ret.second - val_out;
  CHECK_LE(n_uniques, n_inputs);
  thrust::exclusive_scan(thrust::cuda::par(alloc), key_out, key_out + segments_len,
                         key_out, 0);
  return n_uniques;
}
#endif  // SEGMENTED_UNIQUES_H_