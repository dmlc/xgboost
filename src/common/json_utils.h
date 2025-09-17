/**
 * Copyright 2023-2025, XGBoost Contributors
 *
 * @brief Utils tailored for XGBoost.
 */
#pragma once

#include <algorithm>    // for transform, copy
#include <string>       // for string
#include <type_traits>  // for enable_if_t, remove_const_t
#include <vector>       // for vector

#include "xgboost/json.h"
#include "xgboost/string_view.h"  // for StringView

namespace xgboost {
namespace detail {
template <typename Head>
bool TypeCheckImpl(Json const &value) {
  return IsA<Head>(value);
}

template <typename Head, typename... JT>
std::enable_if_t<sizeof...(JT) != 0, bool> TypeCheckImpl(Json const &value) {
  return IsA<Head>(value) || TypeCheckImpl<JT...>(value);
}

template <typename Head>
std::string TypeCheckError() {
  return "`" + Head{}.TypeStr() + "`";
}

template <typename Head, typename... JT>
std::enable_if_t<sizeof...(JT) != 0, std::string> TypeCheckError() {
  return "`" + Head{}.TypeStr() + "`, " + TypeCheckError<JT...>();
}
}  // namespace detail

/**
 * @brief Type check for JSON-based parameters
 *
 * @tparam JT    Expected JSON types.
 * @param  value Value to be checked.
 */
template <typename... JT>
void TypeCheck(Json const &value, StringView name) {
  if (!detail::TypeCheckImpl<JT...>(value)) {
    LOG(FATAL) << "Invalid type for: `" << name << "`, expecting one of the: {"
               << detail::TypeCheckError<JT...>() << "}, got: `" << value.GetValue().TypeStr()
               << "`";
  }
}

template <typename JT>
auto const &RequiredArg(Json const &in, StringView key, StringView func) {
  auto const &obj = get<Object const>(in);
  auto it = obj.find(key);
  if (it == obj.cend() || IsA<Null>(it->second)) {
    LOG(FATAL) << "Argument `" << key << "` is required for `" << func << "`.";
  }
  TypeCheck<JT>(it->second, StringView{key});
  return get<std::remove_const_t<JT> const>(it->second);
}

template <typename JT, typename T>
auto const &OptionalArg(Json const &in, StringView key, T const &dft) {
  auto const &obj = get<Object const>(in);
  auto it = obj.find(key);
  if (it != obj.cend() && !IsA<Null>(it->second)) {
    TypeCheck<JT>(it->second, key);

    return get<std::remove_const_t<JT> const>(it->second);
  }
  return dft;
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>> * = nullptr>
void SaveVector(std::vector<T> const &in, Json *p_out) {
  auto &out = *p_out;
  if (IsA<F32Array>(out)) {
    auto &out_array = get<F32Array>(out);
    out_array.resize(in.size());
    std::copy(in.cbegin(), in.cend(), out_array.begin());
  } else if (IsA<F64Array>(out)) {
    auto &out_array = get<F64Array>(out);
    out_array.resize(in.size());
    std::copy(in.cbegin(), in.cend(), out_array.begin());
  } else {
    LOG(FATAL) << "Invalid array type.";
  }
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>> * = nullptr>
void LoadVector(Json const &in, std::vector<T> *out) {
  if (IsA<F32Array>(in)) {
    // JSON
    auto const &array = get<F32Array const>(in);
    out->resize(array.size());
    std::copy(array.cbegin(), array.cend(), out->begin());
  } else if (IsA<F64Array>(in)) {
    auto const &array = get<F64Array const>(in);
    out->resize(array.size());
    std::copy(array.cbegin(), array.cend(), out->begin());
  } else {
    // UBJSON
    auto const &array = get<Array const>(in);
    out->resize(array.size());
    std::transform(array.cbegin(), array.cend(), out->begin(),
                   [](Json const &v) { return get<Number const>(v); });
  }
}
}  // namespace xgboost
