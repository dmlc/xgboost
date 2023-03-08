/**
 * Copyright 2023 by XGBoost Contributors
 */
#ifndef XGBOOST_TREE_IO_UTILS_H_
#define XGBOOST_TREE_IO_UTILS_H_
#include <string>          // for string
#include <type_traits>     // for enable_if_t, is_same, conditional_t
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
std::enable_if_t<!std::is_same<T, Json>::value && !std::is_same<JT, Boolean>::value, T> GetElem(
    std::vector<T> const& arr, size_t i) {
  return arr[i];
}
// typed array boolean
template <typename JT, typename T>
std::enable_if_t<!std::is_same<T, Json>::value && std::is_same<T, uint8_t>::value &&
                     std::is_same<JT, Boolean>::value,
                 bool>
GetElem(std::vector<T> const& arr, size_t i) {
  return arr[i] == 1;
}
// json array
template <typename JT, typename T>
std::enable_if_t<
    std::is_same<T, Json>::value,
    std::conditional_t<std::is_same<JT, Integer>::value, int64_t,
                       std::conditional_t<std::is_same<Boolean, JT>::value, bool, float>>>
GetElem(std::vector<T> const& arr, size_t i) {
  if (std::is_same<JT, Boolean>::value && !IsA<Boolean>(arr[i])) {
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
