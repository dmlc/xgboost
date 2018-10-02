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
  explicit GPUSet(int start = 0, int ndevices = 0)
      : devices_(start, start + ndevices) {}

  static GPUSet Empty() { return GPUSet(); }

  static GPUSet Range(int start, int ndevices) {
    return ndevices <= 0 ? Empty() : GPUSet{start, ndevices};
  }
  /*! \brief ndevices and num_rows both are upper bounds. */
  static GPUSet All(int ndevices, int num_rows = std::numeric_limits<int>::max()) {
    int n_devices_visible = AllVisible().Size();
    if (ndevices < 0 || ndevices >  n_devices_visible) {
      ndevices = n_devices_visible;
    }
    // fix-up device number to be limited by number of rows
    ndevices = ndevices > num_rows ? num_rows : ndevices;
    return Range(0, ndevices);
  }
  static GPUSet AllVisible() {
    int n =  AllVisibleImpl::AllVisible();
    return Range(0, n);
  }
  /*! \brief Ensure gpu_id is correct, so not dependent upon user knowing details */
  static int GetDeviceIdx(int gpu_id) {
    auto devices = AllVisible();
    CHECK(!devices.IsEmpty()) << "Empty device.";
    return (std::abs(gpu_id) + 0) % devices.Size();
  }
  /*! \brief Counting from gpu_id */
  GPUSet Normalised(int gpu_id) const {
    return Range(gpu_id, Size());
  }
  /*! \brief Counting from 0 */
  GPUSet Unnormalised() const {
    return Range(0, Size());
  }

  int Size() const {
    int res = *devices_.end() - *devices_.begin();
    return res < 0 ? 0 : res;
  }
  /*! \brief Get normalised device id. */
  int operator[](int index) const {
    CHECK(index >= 0 && index < Size());
    return *devices_.begin() + index;
  }

  bool IsEmpty() const { return Size() == 0; }
  /*! \brief Get un-normalised index. */
  int Index(int device) const {
    CHECK(Contains(device));
    return device - *devices_.begin();
  }

  bool Contains(int device) const {
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

 private:
  common::Range devices_;
};

}  // namespace xgboost
#endif  // XGBOOST_COMMON_COMMON_H_
