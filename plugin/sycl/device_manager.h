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
  ::sycl::queue GetQueue(const DeviceOrd& device_spec) const;

  ::sycl::device GetDevice(const DeviceOrd& device_spec) const;

 private:
  using QueueRegister_t = std::unordered_map<std::string, ::sycl::queue>;
  constexpr static int kDefaultOrdinal = -1;

  struct DeviceRegister {
    std::vector<::sycl::device> devices;
    std::vector<::sycl::device> cpu_devices;
    std::vector<::sycl::device> gpu_devices;
  };

  QueueRegister_t& GetQueueRegister() const;

  DeviceRegister& GetDevicesRegister() const;

  mutable std::mutex queue_registering_mutex;
  mutable std::mutex device_registering_mutex;
};

}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_DEVICE_MANAGER_H_
