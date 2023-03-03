/**
 * Copyright 2015-2023 by XGBoost Contributors
 * \file common.h
 * \brief Common utilities
 */
#ifndef XGBOOST_COMMON_COMMON_H_
#define XGBOOST_COMMON_COMMON_H_

#include <xgboost/base.h>
#include <xgboost/logging.h>
#include <xgboost/span.h>

#include <algorithm>
#include <exception>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__CUDACC__)
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#define WITH_CUDA() true

#else

#define WITH_CUDA() false

#endif  // defined(__CUDACC__)

namespace dh {
#if defined(__CUDACC__)
/*
 * Error handling  functions
 */
#define safe_cuda(ans) ThrowOnCudaError((ans), __FILE__, __LINE__)

inline cudaError_t ThrowOnCudaError(cudaError_t code, const char *file,
                                    int line) {
  if (code != cudaSuccess) {
    LOG(FATAL) << thrust::system_error(code, thrust::cuda_category(),
                                       std::string{file} + ": " +  // NOLINT
                                       std::to_string(line)).what();
  }
  return code;
}
#endif  // defined(__CUDACC__)
}  // namespace dh

namespace xgboost {
namespace common {
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

template <typename T>
XGBOOST_DEVICE T Max(T a, T b) {
  return a < b ? b : a;
}

// simple routine to convert any data to string
template<typename T>
inline std::string ToString(const T& data) {
  std::ostringstream os;
  os << data;
  return os.str();
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

int AllVisibleGPUs();

inline void AssertGPUSupport() {
#ifndef XGBOOST_USE_CUDA
    LOG(FATAL) << "XGBoost version not compiled with GPU support.";
#endif  // XGBOOST_USE_CUDA
}

inline void AssertOneAPISupport() {
#ifndef XGBOOST_USE_ONEAPI
    LOG(FATAL) << "XGBoost version not compiled with OneAPI support.";
#endif  // XGBOOST_USE_ONEAPI
}

void SetDevice(std::int32_t device);

#if !defined(XGBOOST_USE_CUDA)
inline void SetDevice(std::int32_t device) {
  if (device >= 0) {
    AssertGPUSupport();
  }
}
#endif

/**
 * Last index of a group in a CSR style of index pointer.
 */
template <typename Indexable>
XGBOOST_DEVICE size_t LastOf(size_t group, Indexable const &indptr) {
  return indptr[group + 1] - 1;
}
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_COMMON_H_
