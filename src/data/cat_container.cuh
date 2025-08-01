/**
 * Copyright 2024-2025, XGBoost Contributors
 */
#pragma once
#include "../common/device_helpers.cuh"  // for ToSpan
#include "../common/device_vector.cuh"   // for device_vector, XGBDeviceAllocator
#include "../encoder/ordinal.cuh"        // for Recode
#include "../encoder/ordinal.h"          // for CatCharT
#include "cat_container.h"               // for EncErrorPolicy

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

  void Copy(CatStrArray const& that) {
    this->offsets = that.offsets;
    this->values = that.values;
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
using ColumnType = enc::cpu_impl::TupToVarT<CatIndexTypes>;

struct EncThrustPolicy {
  template <typename T>
  using ThrustAllocator = dh::XGBDeviceAllocator<T>;

  [[nodiscard]] auto ThrustPolicy() const {
    dh::XGBCachingDeviceAllocator<char> alloc;
    auto exec = thrust::cuda::par_nosync(alloc).on(dh::DefaultStream());
    return exec;
  }
  [[nodiscard]] auto Stream() const { return dh::DefaultStream(); }
};

using EncPolicyT = enc::Policy<EncErrorPolicy, EncThrustPolicy>;

inline EncPolicyT EncPolicy = EncPolicyT{};

inline auto MakeCatAccessor(Context const* ctx, enc::DeviceColumnsView const& new_enc,
                            CatContainer const* orig_cats) {
  dh::DeviceUVector<std::int32_t> mapping(new_enc.n_total_cats);
  auto d_sorted_idx = orig_cats->RefSortedIndex(ctx);
  auto orig_enc = orig_cats->DeviceView(ctx);
  enc::Recode(EncPolicy, orig_enc, d_sorted_idx, new_enc, dh::ToSpan(mapping));
  CHECK_EQ(new_enc.feature_segments.size(), orig_enc.feature_segments.size());
  auto cats_mapping = enc::MappingView{new_enc.feature_segments, dh::ToSpan(mapping)};
  auto acc = CatAccessor{cats_mapping};
  return std::tuple{acc, std::move(mapping)};
}
}  // namespace xgboost::cuda_impl
