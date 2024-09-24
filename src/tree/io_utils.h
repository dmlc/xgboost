/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_IO_UTILS_H_
#define XGBOOST_TREE_IO_UTILS_H_
#include <string>          // for string
#include <type_traits>     // for enable_if_t, is_same_v, conditional_t
#include <vector>          // for vector

#include "xgboost/json.h"  // for Json

namespace xgboost {
template <bool typed>
using FloatArrayT = std::conditional_t<typed, F32Array const, Array const>;
template <bool typed>
using U8ArrayT = std::conditional_t<typed, U8Array const, Array const>;
template <bool typed>
using I32ArrayT = std::conditional_t<typed, I32Array const, Array const>;
template <bool typed>
using I64ArrayT = std::conditional_t<typed, I64Array const, Array const>;
template <bool typed, bool feature_is_64>
using IndexArrayT = std::conditional_t<feature_is_64, I64ArrayT<typed>, I32ArrayT<typed>>;

// typed array, not boolean
template <typename JT, typename T>
std::enable_if_t<!std::is_same_v<T, Json> && !std::is_same_v<JT, Boolean>, T> GetElem(
    std::vector<T> const& arr, size_t i) {
  return arr[i];
}
// typed array boolean
template <typename JT, typename T>
std::enable_if_t<
    !std::is_same_v<T, Json> && std::is_same_v<T, uint8_t> && std::is_same_v<JT, Boolean>, bool>
GetElem(std::vector<T> const& arr, size_t i) {
  return arr[i] == 1;
}
// json array
template <typename JT, typename T>
std::enable_if_t<std::is_same_v<T, Json>,
                 std::conditional_t<std::is_same_v<JT, Integer>, int64_t,
                                    std::conditional_t<std::is_same_v<Boolean, JT>, bool, float>>>
GetElem(std::vector<T> const& arr, size_t i) {
  if (std::is_same_v<JT, Boolean> && !IsA<Boolean>(arr[i])) {
    return get<Integer const>(arr[i]) == 1;
  }
  return get<JT const>(arr[i]);
}

namespace tree_field {
inline std::string const kLossChg{"loss_changes"};
inline std::string const kSumHess{"sum_hessian"};
inline std::string const kBaseWeight{"base_weights"};

inline std::string const kSplitIdx{"split_indices"};
inline std::string const kSplitCond{"split_conditions"};
inline std::string const kDftLeft{"default_left"};

inline std::string const kParent{"parents"};
inline std::string const kLeft{"left_children"};
inline std::string const kRight{"right_children"};
}  // namespace tree_field
}  // namespace xgboost
#endif  // XGBOOST_TREE_IO_UTILS_H_
