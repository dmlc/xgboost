/*!
 * Copyright 2017-2023 by Contributors
 * \file device_manager.h
 */
#ifndef PLUGIN_SYCL_DEVICE_MANAGER_H_
#define PLUGIN_SYCL_DEVICE_MANAGER_H_

#include <vector>
#include <mutex>
#include <string>
#include <unordered_map>

#include <sycl/sycl.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include "xgboost/context.h"
#pragma GCC diagnostic pop

namespace xgboost {
namespace sycl {

class DeviceManager {
 public:
  ::sycl::queue* GetQueue(const DeviceOrd& device_spec) const;

 private:
  constexpr static int kDefaultOrdinal = -1;

  struct DeviceRegister {
    std::vector<::sycl::queue> queues;
    std::unordered_map<::sycl::device, size_t> devices;
    std::vector<size_t> cpu_devices_idxes;
    std::vector<size_t> gpu_devices_idxes;
  };

  DeviceRegister& GetDevicesRegister() const;

  mutable std::mutex device_registering_mutex;
};

}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_DEVICE_MANAGER_H_
