/*!
 * Copyright 2015-2018 by Contributors
 * \file common.cc
 * \brief Enable all kinds of global variables in common.
 */
#include <dmlc/thread_local.h>

#include "common.h"
#include "./random.h"

namespace xgboost {
namespace common {
/*! \brief thread local entry for random. */
struct RandomThreadLocalEntry {
  /*! \brief the random engine instance. */
  GlobalRandomEngine engine;
};

using RandomThreadLocalStore = dmlc::ThreadLocalStore<RandomThreadLocalEntry>;

GlobalRandomEngine& GlobalRandom() {
  return RandomThreadLocalStore::Get()->engine;
}
}  // namespace common

#if !defined(XGBOOST_USE_CUDA)
int AllVisibleImpl::AllVisible() {
  return 0;
}
#endif
GPUSet GPUSet::singleton_ = GPUSet::Range(0, 1);
bool GPUSet::initialized_ = false;

GPUSet GPUSet::Init(GpuIdType gpu_id, GpuIdType n_gpus) {
  CHECK_GE(gpu_id, 0) << "gpu_id must be >= 0.";
  CHECK_GE(n_gpus, -1) << "n_gpus must be >= -1.";

  GpuIdType const n_devices_visible = AllVisible().Size();
  if (n_devices_visible == 0 || n_gpus == 0) {
    singleton_ = Empty();
    initialized_ = true;
    return singleton_;
  }

  GpuIdType const n_available_devices = n_devices_visible - gpu_id;

  if (n_gpus == kAll) {  // Use all devices starting from `gpu_id'.
    CHECK(gpu_id < n_devices_visible)
        << "\ngpu_id should be less than number of visible devices.\ngpu_id: "
        << gpu_id
        << ", number of visible devices: "
        << n_devices_visible;
    singleton_ = Range(gpu_id, n_available_devices);
  } else {  // Use devices in ( gpu_id, gpu_id + n_gpus ).
    CHECK_LE(n_gpus, n_available_devices)
        << "Starting from gpu id: " << gpu_id << ", there are only "
        << n_available_devices << " available devices, while n_gpus is set to: "
        << n_gpus;
    singleton_ = Range(gpu_id, n_gpus);
  }

  initialized_ = true;
  return singleton_;
}

GPUSet GPUSet::Global() {
  if (!initialized_) {
    Init(0, 1);
  }
  return singleton_;
}

GPUSet GPUSet::Range(GpuIdType start, GpuIdType n_gpus) {
  return n_gpus <= 0 ? Empty() : GPUSet{start, n_gpus};
}

GPUSet GPUSet::AllVisible() {
  GpuIdType n =  AllVisibleImpl::AllVisible();
  return Range(0, n);
}

size_t GPUSet::Size() const {
  GpuIdType size = *devices_.end() - *devices_.begin();
  GpuIdType res = size < 0 ? 0 : size;
  return static_cast<size_t>(res);
}

GPUSet::GpuIdType GPUSet::DeviceId(size_t index) const {
  GpuIdType result = *devices_.begin() + static_cast<GpuIdType>(index);
  CHECK(Contains(result)) << "\nDevice " << result << " is not in GPUSet."
                          << "\nIndex: " << index
                          << "\nGPUSet: (" << *begin() << ", " << *end() << ")"
                          << std::endl;
  return result;
}

size_t GPUSet::Index(GPUSet::GpuIdType device) const {
  CHECK(Contains(device)) << "\nDevice " << device << " is not in GPUSet."
                          << "\nGPUSet: (" << *begin() << ", " << *end() << ")"
                          << std::endl;
  size_t result = static_cast<size_t>(device - *devices_.begin());
  return result;
}

}  // namespace xgboost
