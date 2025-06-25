/*!
 * Copyright 2017-2025 by Contributors
 * \file context_helper.cc
 */

#include <sycl/sycl.hpp>


#include "device_manager.h"
#include "context_helper.h"

namespace xgboost {
namespace sycl {

DeviceOrd DeviceFP64(const DeviceOrd& device) {
  DeviceManager device_manager;
  bool support_fp64 = device_manager.GetQueue(device)->get_device().has(::sycl::aspect::fp64);
  if (support_fp64) {
    return device;
  } else {
    LOG(WARNING) << "Current device doesn't support fp64";
    return DeviceOrd::CPU();
  }
}
}  // namespace sycl
}  // namespace xgboost
