/*!
 * Copyright 2015-2018 by Contributors
 * \file common.h
 * \brief Common utilities
 */
#ifndef XGBOOST_COMMON_COMMON_H_
#define XGBOOST_COMMON_COMMON_H_

#include <xgboost/base.h>
#include <xgboost/logging.h>

#include <exception>
#include <limits>
#include <type_traits>
#include <vector>
#include <string>
#include <sstream>

#if defined(__CUDACC__)
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#define WITH_CUDA() true

#else

#define WITH_CUDA() false

#endif

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
#endif
}  // namespace dh

namespace xgboost {
namespace common {
/*!
 * \brief Split a string by delimiter
 * \param s String to be splitted.
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

// simple routine to convert any data to string
template<typename T>
inline std::string ToString(const T& data) {
  std::ostringstream os;
  os << data;
  return os.str();
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

   public:
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

}  // namespace common
struct AllVisibleImpl {
  static int AllVisible();
};
/* \brief set of devices across which HostDeviceVector can be distributed.
 *
 * Currently implemented as a range, but can be changed later to something else,
 *   e.g. a bitset
 */
class GPUSet {
 public:
  using GpuIdType = int;
  static constexpr GpuIdType kAll = -1;

 private:
  common::Range devices_;
  static GPUSet singleton_;
  static bool initialized_;

 public:
  explicit GPUSet(int start = 0, int ndevices = 0)
      : devices_(start, start + ndevices) {}

  static GPUSet Init(GpuIdType gpu_id, GpuIdType n_gpus);
  /*! \brief Return a GPUSet with zero device. */
  static GPUSet Empty() { return GPUSet(); }
  /*! \brief Return a range of devices. */
  static GPUSet Range(GpuIdType start, GpuIdType n_gpus);
  /*! \brief All devices found by calling CUDA API. */
  static GPUSet AllVisible();
  /*! \brief Obtain global devices set. */
  static GPUSet Global();

  /*! \brief Number of GPUs in GPUSet. */
  size_t Size() const;

  /*
   * By default, we have two configurations of identifying device, one
   * is the device id obtained from `cudaGetDevice'.  But we sometimes
   * store objects that allocated one for each device in a list, which
   * requires a zero-based index.
   *
   * Hence, `DeviceId' converts a zero-based index to actual device id,
   * `Index' converts a device id to a zero-based index.
   */
  GpuIdType DeviceId(size_t index) const;
  size_t Index(GpuIdType device) const;

  bool IsEmpty() const { return Size() == 0; }

  bool Contains(GpuIdType device) const {
    return *devices_.begin() <= device && device < *devices_.end();
  }

  common::Range::Iterator begin() const { return devices_.begin(); }  // NOLINT
  common::Range::Iterator end() const { return devices_.end(); }      // NOLINT

  friend bool operator==(const GPUSet& lhs, const GPUSet& rhs) {
    return lhs.devices_ == rhs.devices_;
  }
  friend bool operator!=(const GPUSet& lhs, const GPUSet& rhs) {
    return !(lhs == rhs);
  }
};

}  // namespace xgboost
#endif  // XGBOOST_COMMON_COMMON_H_
