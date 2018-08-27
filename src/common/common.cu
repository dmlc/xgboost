/*!
 * Copyright 2018 by Contributors
 * \file common.h
 * \brief Common utilities for CUDA.
 */
#include "common.h"

namespace xgboost {

GPUSet GPUSet::All(int ndevices, int num_rows) {
  int n_devices_visible = AllVisible().Size();
  ndevices = ndevices < 0 ? n_devices_visible : ndevices;
  // fix-up device number to be limited by number of rows
  ndevices = ndevices > num_rows ? num_rows : ndevices;
  return Range(0, ndevices);
}

GPUSet GPUSet::AllVisible() {
  int n_visgpus = 0;
  dh::safe_cuda(cudaGetDeviceCount(&n_visgpus));
  return Range(0, n_visgpus);
}

int GPUSet::GetDeviceIdx(int gpu_id) {
  auto devices = AllVisible();
  CHECK(!devices.IsEmpty()) << "Empty device.";
  return (std::abs(gpu_id) + 0) % devices.Size();
}

}  // namespace xgboost
