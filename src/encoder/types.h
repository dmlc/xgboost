/**
 * Copyright 2024, XGBoost contributors
 */
#pragma once

#if defined(__CUDA__) || defined(__NVCC__)
#define ENC_DEVICE __host__ __device__
#else
#define ENC_DEVICE
#endif  // defined (__CUDA__) || defined(__NVCC__)

#include <tuple>    // for tuple
#include <variant>  // for variant

#if defined(XGBOOST_USE_CUDA)

#include <cuda/std/variant>  // for variant

#endif  // defined(XGBOOST_USE_CUDA)

namespace enc {
template <typename... Ts>
struct Overloaded : Ts... {
  using Ts::operator()...;
};

template <typename... Ts>
ENC_DEVICE Overloaded(Ts...) -> Overloaded<Ts...>;

namespace cpu_impl {
template <typename... Ts>
struct TupToVar;

template <typename... Ts>
struct TupToVar<std::tuple<Ts...>> {
  using Type = std::variant<Ts...>;
};

template <typename... Ts>
using TupToVarT = typename TupToVar<Ts...>::Type;
}  // namespace cpu_impl

#if defined(XGBOOST_USE_CUDA)
namespace cuda_impl {
template <typename... Ts>
struct TupToVar {};

template <typename... Ts>
struct TupToVar<std::tuple<Ts...>> {
  using Type = cuda::std::variant<Ts...>;
};

template <typename... Ts>
using TupToVarT = typename TupToVar<Ts...>::Type;
}  // namespace cuda_impl
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace enc
