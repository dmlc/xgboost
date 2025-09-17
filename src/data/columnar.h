/**
 *  Copyright 2025, XGBoost Contributors
 *
 * @brief Helpers for handling columnar data with adapters.
 */
#pragma once

#include <algorithm>    // for max
#include <cstddef>      // for size_t
#include <cstdint>      // for int32_t
#include <type_traits>  // for is_floating_point_v
#include <vector>       // for vector

#include "../common/error_msg.h"  // for NoFloatCat
#include "../encoder/ordinal.h"   // for CatStrArrayView
#include "array_interface.h"      // for ArrayInterfaceHandler
#include "xgboost/context.h"      // for DeviceOrd
#include "xgboost/json.h"         // for Json, Object
#include "xgboost/span.h"         // for Span

#if !defined(XGBOOST_USE_CUDA)
#include "../common/common.h"  // for AssertGPUSupport
#else
#include <cuda_runtime_api.h>  // for cudaMemcpy
#endif

namespace xgboost::data {
/**
 * @brief Get string-based category index from arrow.
 *
 * @return The extracted category index
 */
template <typename CategoricalIndex>
auto GetArrowNames(Object::Map const& jnames, std::vector<CategoricalIndex>* p_cat_columns) {
  auto& cat_columns = *p_cat_columns;
  // There are 3 buffers for a StringArray, validity mask, offset, and data. Mask
  // and data are represented by a single masked array.
  auto const& joffset = get<Object const>(jnames.at("offsets"));
  auto offset = ArrayInterface<1>{joffset};
  auto const& jstr = get<Object const>(jnames.at("values"));
  auto strbuf = ArrayInterface<1>(jstr);

  // Obtain the size of the string buffer using the offset
  CHECK_GE(offset.n, 2);
  auto offset_last_idx = offset.n - 1;
  if (ArrayInterfaceHandler::IsCudaPtr(offset.data)) {
    CHECK_EQ(strbuf.n, 0);  // Unknown
#if defined(XGBOOST_USE_CUDA)
    DispatchDType(offset.type, [&](auto t) {
      using T = decltype(t);
      if (!std::is_same_v<T, std::int32_t>) {
        LOG(FATAL) << "Invalid type for the string offset from category index.";
      }
#if defined(__CUDACC__)
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20208  // long double is treated as double in device code
#endif  // defined(__CUDACC__)
      T back{0};
      dh::safe_cuda(cudaMemcpy(&back, static_cast<T const*>(offset.data) + offset_last_idx,
                               sizeof(T), cudaMemcpyDeviceToHost));
      strbuf.n = back;
#if defined(__CUDACC__)
#pragma nv_diagnostic pop
#endif  // defined(__CUDACC__)
    });
#else
    common::AssertGPUSupport();
#endif
  } else {
    DispatchDType(offset.type, [&](auto t) {
      using T = decltype(t);
      if (!std::is_same_v<T, std::int32_t>) {
        LOG(FATAL) << "Invalid type for the string offset from category index.";
      }
      auto back = offset(offset_last_idx);
      strbuf.n = back;
    });
  }

  CHECK_EQ(strbuf.type, ArrayInterfaceHandler::kI1);
  CHECK_EQ(offset.type, ArrayInterfaceHandler::kI4);
  auto names = enc::CatStrArrayView{
      common::Span{static_cast<std::int32_t const*>(offset.data), offset.Shape<0>()},
      common::Span<std::int8_t const>{reinterpret_cast<std::int8_t const*>(strbuf.data), strbuf.n}};
  cat_columns.emplace_back(names);
  return names;
}

/**
 * @brief Get string names and codes for categorical features.
 *
 * @return The number of categories for the current column.
 */
template <typename CategoricalIndex, bool allow_mask>
[[nodiscard]] std::size_t GetArrowDictionary(Json const& jcol,
                                             std::vector<CategoricalIndex>* p_cat_columns,
                                             std::vector<ArrayInterface<1, allow_mask>>* p_columns,
                                             std::size_t* p_n_bytes, bst_idx_t* p_n_samples) {
  auto const& tup = get<Array const>(jcol);
  CHECK_EQ(tup.size(), 2);

  auto names = GetArrowNames(get<Object const>(tup[0]), p_cat_columns);

  // arrow Integer array for encoded categories
  auto const& jcodes = get<Object const>(tup[1]);
  auto codes = ArrayInterface<1>{jcodes};
  p_columns->push_back(codes);

  auto& n_bytes = *p_n_bytes;
  n_bytes += codes.ElementSize() * codes.Shape<0>();
  n_bytes += names.SizeBytes();

  *p_n_samples = std::max(*p_n_samples, static_cast<bst_idx_t>(codes.Shape<0>()));
  return names.size();
}

/**
 * @brief Get numeric-based category index from arrow.
 *
 * @return The extracted category index
 */
template <typename CategoricalIndex>
[[nodiscard]] std::size_t GetArrowNumericNames(DeviceOrd device, Object::Map const& jnames,
                                               std::vector<CategoricalIndex>* p_cat_columns,
                                               std::size_t* p_n_bytes) {
  auto names = ArrayInterface<1>{jnames};
  auto& n_bytes = *p_n_bytes;
  DispatchDType(names, device, [&](auto t) {
    using T = typename decltype(t)::value_type;
    constexpr bool kKnownType = enc::MemberOf<std::remove_cv_t<T>, enc::CatPrimIndexTypes>::value;
    CHECK(kKnownType) << "Unsupported categorical index type: `"
                      << ArrayInterfaceHandler::TypeStr(names.type) << "`.";
    if constexpr (std::is_floating_point_v<T>) {
      LOG(FATAL) << error::NoFloatCat();
    }
    auto span = common::Span{t.Values().data(), t.Size()};
    if constexpr (kKnownType) {
      p_cat_columns->emplace_back(span);
      n_bytes += span.size_bytes();
    }
  });
  return names.n;
}

/**
 * @brief Get numeric names and codes for categorical features.
 *
 * @return The number of categories for the current column.
 */
template <typename CategoricalIndex, bool allow_mask>
[[nodiscard]] std::size_t GetArrowNumericIndex(
    DeviceOrd device, Json jcol, std::vector<CategoricalIndex>* p_cat_columns,
    std::vector<ArrayInterface<1, allow_mask>>* p_columns, std::size_t* p_n_bytes,
    bst_idx_t* p_n_samples) {
  auto const& first = get<Object const>(jcol[0]);
  auto n_cats = GetArrowNumericNames(device, first, p_cat_columns, p_n_bytes);
  auto& n_bytes = *p_n_bytes;
  auto const& jcodes = get<Object const>(jcol[1]);
  auto codes = ArrayInterface<1>{jcodes};
  p_columns->push_back(codes);

  n_bytes += codes.ElementSize() * codes.Shape<0>();
  *p_n_samples = std::max(*p_n_samples, static_cast<bst_idx_t>(codes.Shape<0>()));

  return n_cats;
}
}  // namespace xgboost::data
