/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include <cstdint>  // for int32_t, int8_t
#include <tuple>    // for tuple
#include <vector>   // for vector

#include "../encoder/ordinal.h"  // for DictionaryView

namespace xgboost {
namespace cpu_impl {
struct CatStrArray {
  std::vector<std::int32_t> offsets;
  std::vector<enc::CatCharT> values;

  [[nodiscard]] explicit operator enc::CatStrArrayView() const { return {offsets, values}; }
  [[nodiscard]] std::size_t size() const {  // NOLINT
    return enc::CatStrArrayView(*this).size();
  }
};

template <typename T>
struct ViewToStorageImpl;

template <>
struct ViewToStorageImpl<enc::CatStrArrayView> {
  using Type = CatStrArray;
};

template <typename T>
struct ViewToStorageImpl<common::Span<T const>> {
  using Type = std::vector<T>;
};

template <typename... Ts>
struct ViewToStorage;

template <typename... Ts>
struct ViewToStorage<std::tuple<Ts...>> {
  using Type = std::tuple<typename ViewToStorageImpl<Ts>::Type...>;
};

using CatIndexTypes = ViewToStorage<enc::CatIndexViewTypes>::Type;
using ColumnType = enc::cpu_impl::TupToVarT<CatIndexTypes>;
}  // namespace cpu_impl
}  // namespace xgboost
