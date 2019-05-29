/*!
 * Copyright 2015-2019 by Contributors
 * \file common.cc
 * \brief Enable all kinds of global variables in common.
 */
#include <dmlc/thread_local.h>
#include <xgboost/logging.h>

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
#endif  // !defined(XGBOOST_USE_CUDA)

constexpr GPUSet::GpuIdType GPUSet::kAll;

GPUSet GPUSet::All(GpuIdType gpu_id, GpuIdType n_gpus, int32_t n_rows) {
  CHECK_GE(gpu_id, 0) << "gpu_id must be >= 0.";
  CHECK_GE(n_gpus, -1) << "n_gpus must be >= -1.";

  GpuIdType const n_devices_visible = AllVisible().Size();
  if (n_devices_visible == 0 || n_gpus == 0 || n_rows == 0) {
    LOG(DEBUG) << "Runing on CPU.";
    return Empty();
  }

  GpuIdType const n_available_devices = n_devices_visible - gpu_id;

  if (n_gpus == kAll) {  // Use all devices starting from `gpu_id'.
    CHECK(gpu_id < n_devices_visible)
        << "\ngpu_id should be less than number of visible devices.\ngpu_id: "
        << gpu_id
        << ", number of visible devices: "
        << n_devices_visible;
    GpuIdType n_devices =
        n_available_devices < n_rows ? n_available_devices : n_rows;
    LOG(DEBUG) << "GPU ID: " << gpu_id << ", Number of GPUs: " << n_devices;
    return Range(gpu_id, n_devices);
  } else {  // Use devices in ( gpu_id, gpu_id + n_gpus ).
    CHECK_LE(n_gpus, n_available_devices)
        << "Starting from gpu id: " << gpu_id << ", there are only "
        << n_available_devices << " available devices, while n_gpus is set to: "
        << n_gpus;
    GpuIdType n_devices = n_gpus < n_rows ? n_gpus : n_rows;
    LOG(DEBUG) << "GPU ID: " << gpu_id << ", Number of GPUs: " << n_devices;
    return Range(gpu_id, n_devices);
  }
}

}  // namespace xgboost
