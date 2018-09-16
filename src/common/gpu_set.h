/*!
 * Copyright 2018 XGBoost contributors
 */
#ifndef XGBOOST_COMMON_GPU_SET_H_
#define XGBOOST_COMMON_GPU_SET_H_

#include <xgboost/base.h>
#include <xgboost/logging.h>

#include <limits>
#include <string>

#include "common.h"
#include "span.h"

#if defined(__CUDACC__)
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
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
    throw thrust::system_error(code, thrust::cuda_category(),
                               std::string{file} + "(" +  // NOLINT
                               std::to_string(line) + ")");
  }
  return code;
}
#endif
}  // namespace dh

namespace xgboost {

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
  /* \brief ndevices and num_rows both are upper bounds. */
  static GPUSet All(int ndevices, int num_rows = std::numeric_limits<int>::max()) {
    int n_devices_visible = AllVisible().Size();
    ndevices = ndevices < 0 ? n_devices_visible : ndevices;
    // fix-up device number to be limited by number of rows
    ndevices = ndevices > num_rows ? num_rows : ndevices;
    return Range(0, ndevices);
  }

  static GPUSet AllVisible() {
    int n_visgpus = 0;
#if defined(__CUDACC__)
    dh::safe_cuda(cudaGetDeviceCount(&n_visgpus));
#endif
    return Range(0, n_visgpus);
  }
  /* \brief Ensure gpu_id is correct, so not dependent upon user knowing details */
  static int GetDeviceIdx(int gpu_id) {
    return (std::abs(gpu_id) + 0) % AllVisible().Size();
  }
  /* \brief Counting from gpu_id */
  GPUSet Normalised(int gpu_id) const {
    return Range(gpu_id, Size());
  }
  /* \brief Counting from 0 */
  GPUSet Unnormalised() const {
    return Range(0, Size());
  }

  int Size() const {
    int res = *devices_.end() - *devices_.begin();
    return res < 0 ? 0 : res;
  }

  int operator[](int index) const {
    CHECK(index >= 0 && index < *(devices_.end()));
    return *devices_.begin() + index;
  }

  bool IsEmpty() const { return Size() == 0; }  // NOLINT

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

#endif  // XGBOOST_COMMON_GPU_SET_H_
