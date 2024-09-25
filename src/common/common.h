/**
 * Copyright 2015-2024, XGBoost Contributors
 * \file common.h
 * \brief Common utilities
 */
#ifndef XGBOOST_COMMON_COMMON_H_
#define XGBOOST_COMMON_COMMON_H_

#include <array>      // for array
#include <cmath>      // for ceil
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t, int64_t
#include <sstream>    // for basic_istream, operator<<, istringstream
#include <string>     // for string, basic_string, getline, char_traits
#include <tuple>      // for make_tuple
#include <utility>    // for forward, index_sequence, make_index_sequence
#include <vector>     // for vector

#include "xgboost/base.h"     // for XGBOOST_DEVICE
#include "xgboost/logging.h"  // for LOG, LOG_FATAL, LogMessageFatal

// magic to define functions based on the compiler.
#if defined(__CUDACC__)

#define WITH_CUDA() true

#else

#define WITH_CUDA() false

#endif  // defined(__CUDACC__)

#if defined(XGBOOST_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace dh {
#if defined(XGBOOST_USE_CUDA)
/*
 * Error handling functions
 */
void ThrowOnCudaError(cudaError_t code, const char *file, int line);

#define safe_cuda(ans) ThrowOnCudaError((ans), __FILE__, __LINE__)

#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace dh

namespace xgboost::common {
/*!
 * \brief Split a string by delimiter
 * \param s String to be split.
 * \param delim The delimiter.
 */
inline std::vector<std::string> Split(const std::string& s, char delim) {
  std::string item;
  std::istringstream is(s);
  std::vector<std::string> ret;
  while (std::getline(is, item, delim)) {
    ret.push_back(item);
  }
  return ret;
}

/**
 * @brief Add escapes for a UTF-8 string.
 */
void EscapeU8(std::string const &string, std::string *p_buffer);

/**
 * @brief Add escapes for a UTF-8 string with newly created buffer as return.
 */
inline std::string EscapeU8(std::string const &str) {
  std::string buffer;
  EscapeU8(str, &buffer);
  return buffer;
}

template <typename T>
XGBOOST_DEVICE T Max(T a, T b) {
  return a < b ? b : a;
}

template <typename T1, typename T2>
XGBOOST_DEVICE T1 DivRoundUp(const T1 a, const T2 b) {
  return static_cast<T1>(std::ceil(static_cast<double>(a) / b));
}

namespace detail {
template <class T, std::size_t N, std::size_t... Idx>
constexpr auto UnpackArr(std::array<T, N> &&arr, std::index_sequence<Idx...>) {
  return std::make_tuple(std::forward<std::array<T, N>>(arr)[Idx]...);
}
}  // namespace detail

template <class T, std::size_t N>
constexpr auto UnpackArr(std::array<T, N> &&arr) {
  return detail::UnpackArr(std::forward<std::array<T, N>>(arr),
                           std::make_index_sequence<N>{});
}

/*
 * Range iterator
 */
class Range {
 public:
  using DifferenceType = int64_t;

  class Iterator {
    friend class Range;

   public:
    XGBOOST_DEVICE DifferenceType operator*() const { return i_; }
    XGBOOST_DEVICE const Iterator &operator++() {
      i_ += step_;
      return *this;
    }
    XGBOOST_DEVICE Iterator operator++(int) {
      Iterator res {*this};
      i_ += step_;
      return res;
    }

    XGBOOST_DEVICE bool operator==(const Iterator &other) const {
      return i_ >= other.i_;
    }
    XGBOOST_DEVICE bool operator!=(const Iterator &other) const {
      return i_ < other.i_;
    }

    XGBOOST_DEVICE void Step(DifferenceType s) { step_ = s; }

   protected:
    XGBOOST_DEVICE explicit Iterator(DifferenceType start) : i_(start) {}
    XGBOOST_DEVICE explicit Iterator(DifferenceType start, DifferenceType step) :
        i_{start}, step_{step} {}

   private:
    int64_t i_;
    DifferenceType step_ = 1;
  };

  XGBOOST_DEVICE Iterator begin() const { return begin_; }  // NOLINT
  XGBOOST_DEVICE Iterator end() const { return end_; }      // NOLINT

  XGBOOST_DEVICE Range(DifferenceType begin, DifferenceType end)
      : begin_(begin), end_(end) {}
  XGBOOST_DEVICE Range(DifferenceType begin, DifferenceType end,
                       DifferenceType step)
      : begin_(begin, step), end_(end) {}

  XGBOOST_DEVICE bool operator==(const Range& other) const {
    return *begin_ == *other.begin_ && *end_ == *other.end_;
  }
  XGBOOST_DEVICE bool operator!=(const Range& other) const {
    return !(*this == other);
  }

  XGBOOST_DEVICE void Step(DifferenceType s) { begin_.Step(s); }

 private:
  Iterator begin_;
  Iterator end_;
};

inline void AssertGPUSupport() {
#ifndef XGBOOST_USE_CUDA
    LOG(FATAL) << "XGBoost version not compiled with GPU support.";
#endif  // XGBOOST_USE_CUDA
}

inline void AssertNCCLSupport() {
#if !defined(XGBOOST_USE_NCCL)
    LOG(FATAL) << "XGBoost version not compiled with NCCL support.";
#endif  // !defined(XGBOOST_USE_NCCL)
}

inline void AssertSYCLSupport() {
#ifndef XGBOOST_USE_SYCL
    LOG(FATAL) << "XGBoost version not compiled with SYCL support.";
#endif  // XGBOOST_USE_SYCL
}

/**
 * @brief Last index of a group in a CSR style of index pointer.
 */
template <typename Indexable>
XGBOOST_DEVICE size_t LastOf(size_t group, Indexable const &indptr) {
  return indptr[group + 1] - 1;
}

// Convert the number of bytes to a human readable unit.
std::string HumanMemUnit(std::size_t n_bytes);
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_COMMON_H_
