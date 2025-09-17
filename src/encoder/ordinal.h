/**
 * Copyright 2025, XGBoost contributors
 *
 * @brief Orindal re-coder for categorical features.
 *
 * For training with dataframes, we use the default encoding provided by the dataframe
 * implementation. However, we need a way to ensure the encoding is consistent at test
 * time, which is often not the case. This module re-codes the test data given the train
 * time encoding (mapping between categories to dense discrete integers starting from 0).
 *
 * The algorithm proceeds as follow:
 *
 * Given the categories used for training [c, b, d, a], the ordering of this list is the
 * encoding, c maps to 0, b maps to 1, so on and so forth. At test time, we recieve an
 * encoding [c, a, b], which differs from the encoding used for training and we need to
 * re-code the data.
 *
 * First, we perform an `argsort` on the training categories in the increasing order,
 * obtaining a list of index: [3, 1, 0, 2], which corresponds to [a, b, c, d] as a sorted
 * list. Then we perform binary search for each category in the test time encoding [c, a,
 * b] with the training encoding as the sorted haystack. Since c is the third item of
 * sorted training encoding, we have an index 2 (0-based) for c, index 0 for a, and index
 * 1 for b. After the bianry search, we obtain a new list of index [2, 0, 1]. Using this
 * index list, we can recover the training encoding for the test dataset [0, 3, 1]. This
 * has O(NlogN) complexity with N as the number of categories (assuming the length of the
 * strings as constant). Originally, the encoding for test data set is [0, 1, 2] for [c,
 * a, b], now we have a mapping {0 -> 0, 1 -> 3, 2 -> 1} for re-coding the data.
 *
 * This module exposes 2 functions and an execution policy:
 * - @ref Recode
 * - @ref SortNames
 * Each of them has a device counterpart.
 */

#pragma once
#include <algorithm>    // for stable_sort, lower_bound
#include <cstddef>      // for size_t
#include <cstdint>      // for int32_t, int8_t
#include <iterator>     // for iterator_traits, distance
#include <numeric>      // for accumulate, iota
#include <sstream>      // for stringstream
#include <stdexcept>    // for logic_error
#include <string>       // for string
#include <tuple>        // for tuple
#include <type_traits>  // for decay_t
#include <utility>      // for forward
#include <variant>      // for variant, visit
#include <vector>       // for vector

#include "../common/transform_iterator.h"  // for MakeIndexTransformIter
#include "types.h"                         // for Overloaded, TupToVarT
#include "xgboost/span.h"                  // for Span

namespace enc {
using xgboost::common::MakeIndexTransformIter;
using xgboost::common::Span;

using CatCharT = std::int8_t;

/**
 * @brief String names of categorical data. Represented in the arrow StringArray format.
 */
struct CatStrArrayView {
  Span<std::int32_t const> offsets;
  Span<CatCharT const> values;

  [[nodiscard]] ENC_DEVICE bool empty() const { return offsets.empty(); }  // NOLINT
  [[nodiscard]] ENC_DEVICE std::size_t size() const {                      // NOLINT
    return this->empty() ? 0 : this->offsets.size() - 1;
  }

  [[nodiscard]] std::size_t SizeBytes() const {
    return this->offsets.size_bytes() + values.size_bytes();
  }
};

// We keep a single type list here for supported types and use various transformations to
// add specializations. This way we can modify the type list with ease.

/**
 * @brief All the primitive types supported by the encoder.
 */
using CatPrimIndexTypes =
    std::tuple<std::uint8_t, std::int8_t, std::uint16_t, std::int16_t, std::uint32_t, std::int32_t,
               std::uint64_t, std::int64_t, float, double>;

/**
 * @brief All the column types supported by the encoder.
 */
using CatIndexViewTypes =
    decltype(std::tuple_cat(std::tuple<CatStrArrayView>{}, PrimToSpan<CatPrimIndexTypes>::Type{}));

/**
 * @brief Host categories view for a single column.
 */
using HostCatIndexView = cpu_impl::TupToVarT<CatIndexViewTypes>;

#if defined(XGBOOST_USE_CUDA)
/**
 * @brief Device categories view for a single column.
 */
using DeviceCatIndexView = cuda_impl::TupToVarT<CatIndexViewTypes>;
#endif  // defined(XGBOOST_USE_CUDA)

/**
 * @brief Container for the execution policies used by the encoder.
 *
 * Accepted policies:
 *
 * - A class with a `ThrustPolicy` method that returns a thrust execution policy, along with a
 *   `ThrustAllocator` template type. In addition, a `Stream` method that returns a CUDA stream.
 *   This is only used for the GPU implementation.
 *
 * - An error handling policy that exposes a single `Error` method, which takes a single
 *   string parameter for error message.
 */
template <typename... Derived>
struct Policy : public Derived... {};

namespace detail {
constexpr std::int32_t SearchKey() { return -1; }
constexpr std::int32_t NotFound() { return -1; }

template <typename Variant>
struct ColumnsViewImpl {
  using VariantT = Variant;

  Span<Variant const> columns;

  // Segment pointer for features, each segment represents the number of categories in a feature.
  Span<std::int32_t const> feature_segments;
  // The total number of cats in all features, equals feature_segments.back()
  std::int32_t n_total_cats{0};

  [[nodiscard]] std::size_t Size() const { return columns.size(); }
  [[nodiscard]] bool Empty() const { return this->Size() == 0; }
  [[nodiscard]] auto operator[](std::size_t i) const { return columns[i]; }
  [[nodiscard]] auto HasCategorical() const { return n_total_cats != 0; }
};

struct DftErrorHandler {
  void Error(std::string &&msg) const { throw std::logic_error{std::forward<std::string>(msg)}; }
};

template <typename ExecPolicy>
void ReportMissing(ExecPolicy const &policy, std::string const &name, std::size_t f_idx) {
  std::stringstream ss;
  ss << "Found a category not in the training set for the " << f_idx << "th (0-based) column: `"
     << name << "`";
  policy.Error(ss.str());
}
}  // namespace detail

/**
 * @brief Host view of the encoding scheme for all columns.
 */
using HostColumnsView = detail::ColumnsViewImpl<HostCatIndexView>;
#if defined(XGBOOST_USE_CUDA)
/**
 * @brief Device view of the encoding scheme for all columns.
 */
using DeviceColumnsView = detail::ColumnsViewImpl<DeviceCatIndexView>;
#endif  // defined(XGBOOST_USE_CUDA)

namespace detail {
template <typename ExecPolicy, typename IndexType>
void BasicChecks(ExecPolicy const &policy, detail::ColumnsViewImpl<IndexType> orig_enc,
                 Span<std::int32_t const> sorted_idx, detail::ColumnsViewImpl<IndexType> new_enc,
                 Span<std::int32_t> mapping) {
  if (orig_enc.Size() != new_enc.Size()) {
    policy.Error("New and old encoding should have the same number of columns.");
  }
  if (static_cast<std::int32_t>(mapping.size()) != new_enc.n_total_cats) {
    policy.Error("`mapping` should have the same size as `new_enc.n_total_cats`.");
  }
  if (static_cast<std::int32_t>(sorted_idx.size()) != orig_enc.n_total_cats) {
    policy.Error("`sorted_idx` should have the same size as `orig_enc.n_total_cats`.");
  }
  if (orig_enc.feature_segments.size() != orig_enc.columns.size() + 1) {
    policy.Error("Invalid original encoding.");
  }
  if (new_enc.feature_segments.size() != new_enc.columns.size() + 1) {
    policy.Error("Invalid new encoding.");
  }
}
}  // namespace detail

/**
 * @brief The result encoding. User needs to construct it from the offsets from the new
 *        dictionary along with the mapping returned by the recode function.
 */
struct MappingView {
  Span<std::int32_t const> offsets;
  Span<std::int32_t const> mapping;

  /**
   * @brief Get the encoding for a specific feature.
   */
  [[nodiscard]] ENC_DEVICE auto operator[](std::size_t f_idx) const {
    return mapping.subspan(offsets[f_idx], offsets[f_idx + 1] - offsets[f_idx]);
  }
  [[nodiscard]] ENC_DEVICE bool Empty() const { return offsets.empty(); }
};

namespace cpu_impl {
template <typename InIt, typename OutIt, typename Comp>
void ArgSort(InIt in_first, InIt in_last, OutIt out_first, Comp comp = std::less{}) {
  auto n = std::distance(in_first, in_last);
  using Idx = typename std::iterator_traits<OutIt>::value_type;

  auto out_last = out_first + n;
  std::iota(out_first, out_last, 0);
  auto op = [&](Idx const &l, Idx const &r) {
    return comp(in_first[l], in_first[r]);
  };
  std::stable_sort(out_first, out_last, op);
}

[[nodiscard]] inline std::int32_t SearchSorted(CatStrArrayView haystack,
                                               Span<std::int32_t const> ref_sorted_idx,
                                               Span<std::int8_t const> needle) {
  auto it = MakeIndexTransformIter([](auto i) { return static_cast<std::int32_t>(i); });
  auto const h_off = haystack.offsets;
  auto const h_data = haystack.values;
  using detail::SearchKey;
  auto ret_it = std::lower_bound(it, it + haystack.size(), SearchKey(), [&](auto l, auto r) {
    Span<std::int8_t const> l_str;
    if (l == SearchKey()) {
      l_str = needle;
    } else {
      auto l_idx = ref_sorted_idx[l];
      auto l_beg = h_off[l_idx];
      auto l_end = h_off[l_idx + 1];
      l_str = h_data.subspan(l_beg, l_end - l_beg);
    }

    Span<std::int8_t const> r_str;
    if (r == SearchKey()) {
      r_str = needle;
    } else {
      auto r_idx = ref_sorted_idx[r];
      auto r_beg = h_off[r_idx];
      auto r_end = h_off[r_idx + 1];
      r_str = h_data.subspan(r_beg, r_end - r_beg);
    }

    return l_str < r_str;
  });
  if (ret_it == it + haystack.size()) {
    return detail::NotFound();
  }
  return *ret_it;
}

template <typename T>
[[nodiscard]] std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>, std::int32_t>
SearchSorted(Span<T const> haystack, Span<std::int32_t const> ref_sorted_idx, T needle) {
  using detail::SearchKey;
  auto it = MakeIndexTransformIter([](auto i) { return static_cast<std::int32_t>(i); });
  auto ret_it = std::lower_bound(it, it + haystack.size(), SearchKey(), [&](auto l, auto r) {
    T l_value = l == SearchKey() ? needle : haystack[ref_sorted_idx[l]];
    T r_value = r == SearchKey() ? needle : haystack[ref_sorted_idx[r]];
    return l_value < r_value;
  });
  if (ret_it == it + haystack.size()) {
    return detail::NotFound();
  }
  return *ret_it;
}

template <typename ExecPolicy>
void SortNames(ExecPolicy const &policy, HostCatIndexView const &cats,
               Span<std::int32_t> sorted_idx) {
  auto it = MakeIndexTransformIter([](auto i) { return i; });
  using T = typename std::iterator_traits<decltype(it)>::value_type;
  auto n_categories = std::visit([](auto &&arg) { return arg.size(); }, cats);
  if (sorted_idx.size() != n_categories) {
    policy.Error("Invalid size of sorted index.");
  }
  std::visit(Overloaded{[&](CatStrArrayView const &str) {
                          cpu_impl::ArgSort(it, it + str.size(), sorted_idx.begin(), [&](T l, T r) {
                            auto l_beg = str.offsets[l];
                            auto l_str = str.values.subspan(l_beg, str.offsets[l + 1] - l_beg);

                            auto r_beg = str.offsets[r];
                            auto r_str = str.values.subspan(r_beg, str.offsets[r + 1] - r_beg);

                            return l_str < r_str;
                          });
                        },
                        [&](auto &&values) {
                          cpu_impl::ArgSort(it, it + values.size(), sorted_idx.begin(),
                                            [&](T l, T r) { return values[l] < values[r]; });
                        }},
             cats);
}
}  // namespace cpu_impl

/**
 * @brief Sort the categories for the training set. Returns a list of sorted index.
 *
 * @tparam ExecPolicy The @ref Policy class, only an error policy is needed for the CPU
 *                    implementation.
 *
 * @param policy     The execution policy.
 * @param orig_enc   The encoding scheme of the training set.
 * @param sorted_idx The output sorted index.
 */
template <typename ExecPolicy>
void SortNames(ExecPolicy const &policy, HostColumnsView orig_enc, Span<std::int32_t> sorted_idx) {
  if (static_cast<std::int32_t>(sorted_idx.size()) != orig_enc.n_total_cats) {
    policy.Error("`sorted_idx` should have the same size as `n_total_cats`.");
  }
  for (std::size_t f_idx = 0, n = orig_enc.Size(); f_idx < n; ++f_idx) {
    auto beg = orig_enc.feature_segments[f_idx];
    auto f_sorted_idx = sorted_idx.subspan(beg, orig_enc.feature_segments[f_idx + 1] - beg);
    cpu_impl::SortNames(policy, orig_enc.columns[f_idx], f_sorted_idx);
  }
}

/**
 * @brief Default exection policy for the host implementation. Users are expected to
 *        customize it.
 */
using DftHostPolicy = Policy<detail::DftErrorHandler>;

/**
 * @brief Calculate a mapping for recoding the data given old and new encoding.
 *
 * @tparam ExecPolicy The @ref Policy class, only an error policy is needed for the CPU
 *                    implementation.
 *
 * @param policy     The execution policy.
 * @param orig_enc   The encoding scheme of the training set.
 * @param sorted_idx The sorted index of the training set encoding scheme, produced by
 *                   @ref SortNames .
 * @param new_enc    The scheme that needs to be recoded.
 * @param mapping    The output mapping.
 */
template <typename ExecPolicy>
void Recode(ExecPolicy const &policy, HostColumnsView orig_enc, Span<std::int32_t const> sorted_idx,
            HostColumnsView new_enc, Span<std::int32_t> mapping) {
  detail::BasicChecks(policy, orig_enc, sorted_idx, new_enc, mapping);

  std::size_t out_idx = 0;
  for (std::size_t f_idx = 0, n_features = orig_enc.Size(); f_idx < n_features; f_idx++) {
    auto const& l_f = orig_enc.columns[f_idx];
    auto const& r_f = new_enc.columns[f_idx];
    auto report = [&] {
      std::stringstream ss;
      ss << "Invalid new DataFrame input for the: " << f_idx << "th feature (0-based). "
         << "The data type doesn't match the one used in the training dataset. "
         << "Both should be either numeric or categorical. For a categorical feature, the index "
            "type must match between the training and test set.";
      policy.Error(ss.str());
    };
    if (l_f.index() != r_f.index()) {
      report();
    }
    bool is_empty = std::visit([](auto &&arg) { return arg.empty(); }, l_f);
    bool new_is_empty = std::visit([](auto &&arg) { return arg.empty(); }, r_f);
    if (is_empty != new_is_empty) {
      report();
    }
    if (is_empty) {
      continue;
    }

    auto f_beg = orig_enc.feature_segments[f_idx];
    auto ref_sorted_idx = sorted_idx.subspan(f_beg, orig_enc.feature_segments[f_idx + 1] - f_beg);

    auto n_new_categories =
        std::visit([](auto &&arg) { return arg.size(); }, new_enc.columns[f_idx]);
    std::vector<std::int32_t> searched_idx(n_new_categories, -1);
    auto const &col = new_enc.columns[f_idx];
    std::visit(Overloaded{[&](CatStrArrayView const &str) {
                            for (std::size_t j = 1, m = n_new_categories + 1; j < m; ++j) {
                              auto begin = str.offsets[j - 1];
                              auto end = str.offsets[j];
                              auto needle = str.values.subspan(begin, end - begin);
                              searched_idx[j - 1] = cpu_impl::SearchSorted(
                                  std::get<CatStrArrayView>(orig_enc.columns[f_idx]),
                                  ref_sorted_idx, needle);
                              if (searched_idx[j - 1] == detail::NotFound()) {
                                std::stringstream ss;
                                for (auto c : needle) {
                                  ss.put(c);
                                }
                                detail::ReportMissing(policy, ss.str(), f_idx);
                              }
                            }
                          },
                          [&](auto &&values) {
                            using T = typename std::decay_t<decltype(values)>::value_type;
                            for (std::size_t j = 0; j < n_new_categories; ++j) {
                              auto needle = values[j];
                              searched_idx[j] = cpu_impl::SearchSorted(
                                  std::get<Span<std::add_const_t<T>>>(orig_enc.columns[f_idx]),
                                  ref_sorted_idx, needle);
                              if (searched_idx[j] == detail::NotFound()) {
                                std::stringstream ss;
                                ss << needle;
                                detail::ReportMissing(policy, ss.str(), f_idx);
                              }
                            }
                          }},
               col);

    for (auto i : searched_idx) {
      auto idx = ref_sorted_idx[i];
      mapping[out_idx++] = idx;
    }
  }
}

inline std::ostream &operator<<(std::ostream &os, CatStrArrayView const &strings) {
  auto const &offset = strings.offsets;
  auto const &data = strings.values;
  os << "[";
  for (std::size_t i = 1, n = offset.size(); i < n; ++i) {
    auto begin = offset[i - 1];
    auto end = offset[i];
    auto str = data.subspan(begin, end - begin);
    for (auto v : str) {
      os.put(v);
    }
    if (i != n - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

inline std::ostream &operator<<(std::ostream &os, HostColumnsView const &h_enc) {
  for (std::size_t i = 0; i < h_enc.columns.size(); ++i) {
    auto const &col = h_enc.columns[i];
    os << "f" << i << ": ";
    std::visit(enc::Overloaded{[&](enc::CatStrArrayView const &str) { os << str; },
                               [&](auto &&values) {
                                 os << "[";
                                 for (std::size_t j = 0, n = values.size(); j < n; ++j) {
                                   os << values[j];
                                   if (j != n - 1) {
                                     os << ", ";
                                   }
                                 }
                                 os << "]";
                               }},
               col);
    os << std::endl;
  }
  return os;
}
}  // namespace enc
