/**
 * Copyright 2024, XGBoost Contributors
 */
#pragma once
#include "../common/device_helpers.cuh"  // for ToSpan
#include "../common/device_vector.cuh"   // for device_vector, XGBDeviceAllocator
#include "../encoder/ordinal.h"          // for CatCharT

namespace xgboost::cuda_impl {
struct CatStrArray {
  dh::device_vector<std::int32_t> offsets;
  dh::device_vector<enc::CatCharT> values;

  CatStrArray() = default;
  CatStrArray(CatStrArray const& that) = delete;
  CatStrArray& operator=(CatStrArray const& that) = delete;

  CatStrArray(CatStrArray&& that) = default;
  CatStrArray& operator=(CatStrArray&& that) = default;

  [[nodiscard]] explicit operator enc::CatStrArrayView() const {
    return {dh::ToSpan(offsets), dh::ToSpan(values)};
  }
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
  using Type = dh::device_vector<T>;
};

template <typename... Ts>
struct ViewToStorage;

template <typename... Ts>
struct ViewToStorage<std::tuple<Ts...>> {
  using Type = std::tuple<typename ViewToStorageImpl<Ts>::Type...>;
};

using CatIndexTypes = ViewToStorage<enc::CatIndexViewTypes>::Type;
}  // namespace xgboost::cuda_impl
