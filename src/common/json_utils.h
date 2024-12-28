/**
 * Copyright 2023, XGBoost Contributors
 *
 * @brief Utils tailored for XGBoost.
 */
#pragma once

#include <string>       // for string
#include <type_traits>  // for enable_if_t, remove_const_t

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
}  // namespace xgboost
