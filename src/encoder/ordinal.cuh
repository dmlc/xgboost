/**
 * Copyright 2025, XGBoost contributors
 */
#pragma once

#include <thrust/binary_search.h>                // for lower_bound
#include <thrust/copy.h>                         // for copy
#include <thrust/device_vector.h>                // for device_vector
#include <thrust/find.h>                         // for find_if
#include <thrust/for_each.h>                     // for for_each_n
#include <thrust/iterator/counting_iterator.h>   // for make_counting_iterator
#include <thrust/iterator/transform_iterator.h>  // for make_transform_iterator
#include <thrust/sort.h>                         // for sort

#include <cstddef>           // for size_t
#include <cstdint>           // for int32_t, int8_t
#include <cuda/functional>   // for proclaim_return_type
#include <cuda/std/utility>  // for make_pair, pair
#include <cuda/std/variant>  // for get
#include <sstream>           // for stringstream

#include "../common/device_helpers.cuh"
#include "ordinal.h"
#include "types.h"  // for Overloaded

namespace enc {
namespace cuda_impl {
struct SegmentedSearchSortedStrOp {
  DeviceColumnsView haystack_v;             // The training set
  Span<std::int32_t const> ref_sorted_idx;  // Sorted index for the training set
  DeviceColumnsView needles_v;              // Keys
  std::size_t f_idx;                        // Feature (segment) index

  [[nodiscard]] __device__ std::int32_t operator()(std::int32_t i) const {
    using detail::SearchKey;
    auto haystack = cuda::std::get<CatStrArrayView>(haystack_v.columns[f_idx]);
    auto needles = cuda::std::get<CatStrArrayView>(needles_v.columns[f_idx]);
    // Get the search key
    auto idx = i - needles_v.feature_segments[f_idx];  // index local to the feature
    auto begin = needles.offsets[idx];
    auto end = needles.offsets[idx + 1];
    auto needle = needles.values.subspan(begin, end - begin);

    // Search the key from the training set
    auto it = thrust::make_counting_iterator(0);
    auto f_sorted_idx = ref_sorted_idx.subspan(
        haystack_v.feature_segments[f_idx],
        haystack_v.feature_segments[f_idx + 1] - haystack_v.feature_segments[f_idx]);
    auto end_it = it + f_sorted_idx.size();
    auto ret_it = thrust::lower_bound(thrust::seq, it, end_it, SearchKey(), [&](auto l, auto r) {
      Span<std::int8_t const> l_str;
      if (l == SearchKey()) {
        l_str = needle;
      } else {
        auto l_idx = f_sorted_idx[l];
        auto l_beg = haystack.offsets[l_idx];
        auto l_end = haystack.offsets[l_idx + 1];
        l_str = haystack.values.subspan(l_beg, l_end - l_beg);
      }

      Span<std::int8_t const> r_str;
      if (r == SearchKey()) {
        r_str = needle;
      } else {
        auto r_idx = f_sorted_idx[r];
        auto r_beg = haystack.offsets[r_idx];
        auto r_end = haystack.offsets[r_idx + 1];
        r_str = haystack.values.subspan(r_beg, r_end - r_beg);
      }

      return l_str < r_str;
    });
    if (ret_it == it + f_sorted_idx.size()) {
      return detail::NotFound();
    }
    return *ret_it;
  }
};

template <typename T>
struct SegmentedSearchSortedNumOp {
  DeviceColumnsView haystack_v;             // The training set
  Span<std::int32_t const> ref_sorted_idx;  // Sorted index for the training set
  DeviceColumnsView needles_v;              // Keys
  std::size_t f_idx;                        // Feature (segment) index

  [[nodiscard]] __device__ std::int32_t operator()(std::int32_t i) const {
    using detail::SearchKey;
    auto haystack = cuda::std::get<Span<T const>>(haystack_v.columns[f_idx]);
    auto needles = cuda::std::get<Span<T const>>(needles_v.columns[f_idx]);
    // Get the search key
    auto idx = i - needles_v.feature_segments[f_idx];  // index local to the feature
    auto needle = needles[idx];
    // Search the key from the training set
    auto it = thrust::make_counting_iterator(0);
    auto f_sorted_idx = ref_sorted_idx.subspan(
        haystack_v.feature_segments[f_idx],
        haystack_v.feature_segments[f_idx + 1] - haystack_v.feature_segments[f_idx]);
    auto end_it = it + f_sorted_idx.size();
    auto ret_it = thrust::lower_bound(thrust::seq, it, end_it, SearchKey(), [&](auto l, auto r) {
      T l_value = l == SearchKey() ? needle : haystack[f_sorted_idx[l]];
      T r_value = r == SearchKey() ? needle : haystack[f_sorted_idx[r]];
      return l_value < r_value;
    });
    if (ret_it == it + f_sorted_idx.size()) {
      return detail::NotFound();
    }
    return *ret_it;
  }
};

template <typename ThrustExec, typename U, typename V>
void SegmentedIota(ThrustExec const& policy, Span<U> d_offset_ptr, Span<V> out_sequence) {
  thrust::for_each_n(policy, thrust::make_counting_iterator(0ul), out_sequence.size(),
                     [out_sequence, d_offset_ptr] __device__(std::size_t idx) {
                       auto group = dh::SegmentId(d_offset_ptr, idx);
                       out_sequence[idx] = idx - d_offset_ptr[group];
                     });
}

struct DftThrustPolicy {
  template <typename T>
  using ThrustAllocator = thrust::device_allocator<T>;

  [[nodiscard]] auto ThrustPolicy() const { return thrust::cuda::par_nosync; }
  [[nodiscard]] auto Stream() const { return cudaStreamPerThread; }
};
}  // namespace cuda_impl

/**
 * @brief Default exection policy for the device implementation. Users are expected to
 *        customize it.
 */
using DftDevicePolicy = Policy<cuda_impl::DftThrustPolicy, detail::DftErrorHandler>;

/**
 * @brief Sort the categories for the training set. Returns a list of sorted index.
 *
 * @tparam ExecPolicy The @ref Policy class, accepts an error policy and a thrust exec policy.
 *
 * @param policy     The execution policy.
 * @param orig_enc   The encoding scheme of the training set.
 * @param sorted_idx The output sorted index.
 */
template <typename ExecPolicy>
void SortNames(ExecPolicy const& policy, DeviceColumnsView orig_enc,
               Span<std::int32_t> sorted_idx) {
  typename ExecPolicy::template ThrustAllocator<char> alloc;
  auto exec = thrust::cuda::par_nosync(alloc).on(policy.Stream());

  auto n_total_cats = orig_enc.n_total_cats;
  if (static_cast<std::int32_t>(sorted_idx.size()) != orig_enc.n_total_cats) {
    policy.Error("`sorted_idx` should have the same size as `n_total_cats`.");
  }
  auto d_sorted_idx = dh::ToSpan(sorted_idx);
  cuda_impl::SegmentedIota(exec, orig_enc.feature_segments, d_sorted_idx);

  // <fidx, sorted_idx>
  using Pair = cuda::std::pair<std::int32_t, std::int32_t>;
  using Alloc = typename ExecPolicy::template ThrustAllocator<Pair>;
  thrust::device_vector<Pair, Alloc> keys(n_total_cats);
  auto key_it = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      cuda::proclaim_return_type<Pair>([=] __device__(std::int32_t i) {
        auto seg = dh::SegmentId(orig_enc.feature_segments, i);
        auto idx = d_sorted_idx[i];
        return cuda::std::make_pair(static_cast<std::int32_t>(seg), idx);
      }));
  thrust::copy(exec, key_it, key_it + n_total_cats, keys.begin());

  thrust::sort(exec, keys.begin(), keys.end(),
               cuda::proclaim_return_type<bool>([=] __device__(Pair const& l, Pair const& r) {
                 if (l.first == r.first) {  // same feature
                   auto const& col = orig_enc.columns[l.first];
                   return cuda::std::visit(
                       Overloaded{[&l, &r](CatStrArrayView const& str) -> bool {
                                    auto l_beg = str.offsets[l.second];
                                    auto l_end = str.offsets[l.second + 1];
                                    auto l_str = str.values.subspan(l_beg, l_end - l_beg);

                                    auto r_beg = str.offsets[r.second];
                                    auto r_end = str.offsets[r.second + 1];
                                    auto r_str = str.values.subspan(r_beg, r_end - r_beg);
                                    return l_str < r_str;
                                  },
                                  [&](auto&& values) {
                                    return values[l.second] < values[r.second];
                                  }},
                       col);
                 }
                 return l.first < r.first;
               }));

  // Extract the sorted index out from sorted keys.
  auto s_keys = dh::ToSpan(keys);
  auto it = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      cuda::proclaim_return_type<decltype(Pair{}.second)>(
          [=] __device__(std::int32_t i) { return s_keys[i].second; }));
  thrust::copy(exec, it, it + sorted_idx.size(), dh::tbegin(sorted_idx));
}

/**
 * @brief Calculate a mapping for recoding the data given old and new encoding.
 *
 * @tparam ExecPolicy The @ref Policy class, accepts an error policy and a thrust exec policy
 *
 * @param policy     The execution policy.
 * @param orig_enc   The encoding scheme of the training set.
 * @param sorted_idx The sorted index of the training set encoding scheme, produced by
 *                   @ref SortNames .
 * @param new_enc    The scheme that needs to be recoded.
 * @param mapping    The output mapping.
 */
template <typename ExecPolicy>
void Recode(ExecPolicy const& policy, DeviceColumnsView orig_enc,
            Span<std::int32_t const> sorted_idx, DeviceColumnsView new_enc,
            Span<std::int32_t> mapping) {
  typename ExecPolicy::template ThrustAllocator<char> alloc;
  auto exec = thrust::cuda::par_nosync(alloc).on(policy.Stream());
  detail::BasicChecks(policy, orig_enc, sorted_idx, new_enc, mapping);
  /**
   * Check consistency.
   */
  auto check_it = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0ul),
      cuda::proclaim_return_type<bool>([=] __device__(std::size_t i) {
        auto const& l_f = orig_enc.columns[i];
        auto const& r_f = new_enc.columns[i];
        if (l_f.index() != r_f.index()) {
          return false;
        }
        auto l_is_empty = cuda::std::visit([](auto&& arg) { return arg.empty(); }, l_f);
        auto r_is_empty = cuda::std::visit([](auto&& arg) { return arg.empty(); }, r_f);
        return l_is_empty == r_is_empty;
      }));
  bool valid = thrust::reduce(
      exec, check_it, check_it + new_enc.Size(), true,
      cuda::proclaim_return_type<bool>([=] __device__(bool l, bool r) { return l && r; }));
  if (!valid) {
    policy.Error(
        "Invalid new DataFrame. "
        "The data type doesn't match the one used in the training dataset. "
        "Both should be either numeric or categorical. "
        "For a categorical feature, the index type must match between the training and test set.");
  }

  /**
   * search the index for the new encoding
   */
  thrust::for_each_n(
      exec, thrust::make_counting_iterator(0), new_enc.n_total_cats,
      [=] __device__(std::int32_t i) {
        auto f_idx = dh::SegmentId(new_enc.feature_segments, i);
        std::int32_t searched_idx{detail::NotFound()};
        auto const& col = orig_enc.columns[f_idx];
        cuda::std::visit(Overloaded{[&](CatStrArrayView const&) {
                                      auto op = cuda_impl::SegmentedSearchSortedStrOp{
                                          orig_enc, sorted_idx, new_enc, f_idx};
                                      searched_idx = op(i);
                                    },
                                    [&](auto&& values) {
                                      using T = typename std::decay_t<decltype(values)>::value_type;
                                      auto op = cuda_impl::SegmentedSearchSortedNumOp<T>{
                                          orig_enc, sorted_idx, new_enc, f_idx};
                                      searched_idx = op(i);
                                    }},
                         col);

        auto f_sorted_idx = sorted_idx.subspan(
            orig_enc.feature_segments[f_idx],
            orig_enc.feature_segments[f_idx + 1] - orig_enc.feature_segments[f_idx]);

        std::int32_t idx = -1;
        if (searched_idx != detail::NotFound()) {
          idx = f_sorted_idx[searched_idx];
        }

        auto f_beg = new_enc.feature_segments[f_idx];
        auto f_end = new_enc.feature_segments[f_idx + 1];
        auto f_mapping = mapping.subspan(f_beg, f_end - f_beg);
        f_mapping[i - f_beg] = idx;
      });

  auto err_it = thrust::find_if(exec, dh::tcbegin(mapping), dh::tcend(mapping),
                                cuda::proclaim_return_type<bool>([=] __device__(std::int32_t v) {
                                  return v == detail::NotFound();
                                }));

  if (err_it != dh::tcend(mapping)) {
    // Report missing cat.
    std::vector<decltype(mapping)::value_type> h_mapping(mapping.size());
    thrust::copy_n(dh::tcbegin(mapping), mapping.size(), h_mapping.begin());
    std::vector<decltype(new_enc.feature_segments)::value_type> h_feature_segments(
        new_enc.feature_segments.size());
    thrust::copy(dh::tcbegin(new_enc.feature_segments), dh::tcend(new_enc.feature_segments),
                 h_feature_segments.begin());
    auto h_idx = std::distance(dh::tcbegin(mapping), err_it);
    auto f_idx = dh::SegmentId(Span<std::int32_t const>{h_feature_segments}, h_idx);
    auto f_beg = h_feature_segments[f_idx];
    auto f_local_idx = h_idx - f_beg;

    std::vector<DeviceColumnsView::VariantT> h_columns(new_enc.columns.size());
    thrust::copy_n(dh::tcbegin(new_enc.columns), new_enc.columns.size(), h_columns.begin());

    std::stringstream name;
    auto const& col = h_columns[f_idx];
    cuda::std::visit(
        Overloaded{[&](CatStrArrayView const& str) {
                     std::vector<CatCharT> values(str.values.size());
                     std::vector<std::int32_t> offsets(str.offsets.size());
                     thrust::copy_n(dh::tcbegin(str.values), str.values.size(), values.data());
                     thrust::copy_n(dh::tcbegin(str.offsets), str.offsets.size(), offsets.data());

                     auto cat = Span{values}.subspan(
                         offsets[f_local_idx], offsets[f_local_idx + 1] - offsets[f_local_idx]);
                     for (auto v : cat) {
                       name.put(v);
                     }
                   },
                   [&](auto&& values) {
                     using T = typename std::decay_t<decltype(values)>::value_type;
                     std::vector<std::remove_cv_t<T>> h_values(values.size());
                     thrust::copy_n(dh::tcbegin(values), values.size(), h_values.data());
                     auto cat = h_values[f_local_idx];
                     name << cat;
                   }},
        col);

    detail::ReportMissing(policy, name.str(), f_idx);
  }
}
}  // namespace enc
