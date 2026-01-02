/**
 * Copyright 2022-2026, XGBoost Contributors
 */
#pragma once

#include <dmlc/endian.h>  // for ByteSwap
#include <xgboost/base.h>
#include <xgboost/windefs.h>

#include <cstdint>

#if defined(xgboost_IS_WIN)

#include <cstdlib>  // for _byteswap_uint64, _byteswap_ulong, _byteswap_ushort

#endif  // defined(xgboost_IS_WIN)

namespace xgboost {
#if defined(__CUDA_ARCH__)
// CUDA kernel version
template <typename T>
[[nodiscard]] __device__ T ByteSwap(T v);

template <>
inline __device__ std::uint16_t ByteSwap(std::uint16_t v) {
  return __nv_bswap16(v);
}

template <>
inline __device__ std::uint32_t ByteSwap(std::uint32_t v) {
  return __nv_bswap32(v);
}

template <>
inline __device__ std::uint64_t ByteSwap(std::uint64_t v) {
  return __nv_bswap64(v);
}

#elif defined(__GLIBC__)
// Host gcc/clang
template <typename T>
T ByteSwap(T v);

template <>
inline std::uint16_t ByteSwap(std::uint16_t v) {
  return __builtin_bswap16(v);
}

template <>
inline std::uint32_t ByteSwap(std::uint32_t v) {
  return __builtin_bswap32(v);
}

template <>
inline std::uint64_t ByteSwap(std::uint64_t v) {
  return __builtin_bswap64(v);
}

#elif defined(xgboost_IS_WIN) && !defined(__MINGW32__)
// MSVC
template <typename T>
T ByteSwap(T v);

template <>
inline std::uint16_t ByteSwap(std::uint16_t v) {
  return _byteswap_ushort(v);
}

template <>
inline std::uint32_t ByteSwap(std::uint32_t v) {
  return _byteswap_ulong(v);
}

template <>
inline std::uint64_t ByteSwap(std::uint64_t v) {
  return _byteswap_uint64(v);
}

#else

template <typename T>
T ByteSwap(T v) {
  dmlc::ByteSwap(&v, sizeof(v), 1);
  return v;
}

#endif  //  defined(__CUDA_ARCH__)
}  // namespace xgboost
