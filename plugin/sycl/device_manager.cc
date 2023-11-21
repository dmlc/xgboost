/*!
 * Copyright 2017-2022 by Contributors
 * \file device_manager.cc
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <rabit/rabit.h>
#pragma GCC diagnostic pop

#include "../sycl/device_manager.h"

namespace xgboost {
namespace sycl {

::sycl::device DeviceManager::GetDevice(const DeviceOrd& device_spec) const {
    if (!device_spec.IsSycl()) {
        LOG(WARNING) << "Sycl kernel is executed with non-sycl context: "
                     << device_spec.Name() << ". "
                     << "Default sycl device_selector will be used.";
    }

    bool not_use_default_selector = (device_spec.ordinal != kDefaultOrdinal) ||
                                    (rabit::IsDistributed());
    if (not_use_default_selector) {
      DeviceRegister& device_register = GetDevicesRegister();
      const int device_idx = rabit::IsDistributed() ? rabit::GetRank() : device_spec.ordinal;
      if (device_spec.IsSyclDefault()) {
          auto& devices = device_register.devices;
          CHECK_LT(device_idx, devices.size());
          return devices[device_idx];
      } else if (device_spec.IsSyclCPU()) {
          auto& cpu_devices = device_register.cpu_devices;
          CHECK_LT(device_idx, cpu_devices.size());
          return cpu_devices[device_idx];
      } else {
          auto& gpu_devices = device_register.gpu_devices;
          CHECK_LT(device_idx, gpu_devices.size());
          return gpu_devices[device_idx];
      }   
    } else {
        if(device_spec.IsSyclCPU()) {
            return ::sycl::device(::sycl::cpu_selector_v);
        } else if(device_spec.IsSyclGPU()) {
            return ::sycl::device(::sycl::gpu_selector_v);
        } else {
            return ::sycl::device(::sycl::default_selector_v);
        }
    }
}

::sycl::queue DeviceManager::GetQueue(const DeviceOrd& device_spec) const {
    if (!device_spec.IsSycl()) {
        LOG(WARNING) << "Sycl kernel is executed with non-sycl context: "
                     << device_spec.Name() << ". "
                     << "Default sycl device_selector will be used.";
    }

    QueueRegister_t& queue_register = GetQueueRegister();
    if (queue_register.count(device_spec.Name()) > 0) {
        return queue_register.at(device_spec.Name());
    }

    bool not_use_default_selector = (device_spec.ordinal != kDefaultOrdinal) ||
                                    (rabit::IsDistributed());
    std::lock_guard<std::mutex> guard(queue_registering_mutex);
    if (not_use_default_selector) {
      DeviceRegister& device_register = GetDevicesRegister();
      const int device_idx = rabit::IsDistributed() ? rabit::GetRank() : device_spec.ordinal;
      if (device_spec.IsSyclDefault()) {
          auto& devices = device_register.devices;
          CHECK_LT(device_idx, devices.size());
          queue_register[device_spec.Name()] = ::sycl::queue(devices[device_idx]);
      } else if (device_spec.IsSyclCPU()) {
          auto& cpu_devices = device_register.cpu_devices;
          CHECK_LT(device_idx, cpu_devices.size());
          queue_register[device_spec.Name()] = ::sycl::queue(cpu_devices[device_idx]);;
      } else if (device_spec.IsSyclGPU()) {
          auto& gpu_devices = device_register.gpu_devices;
          CHECK_LT(device_idx, gpu_devices.size());
          queue_register[device_spec.Name()] = ::sycl::queue(gpu_devices[device_idx]);
      }
    } else {
        if (device_spec.IsSyclCPU()) {
            queue_register[device_spec.Name()] = ::sycl::queue(::sycl::cpu_selector_v);
        } else if (device_spec.IsSyclGPU()) {
            queue_register[device_spec.Name()] = ::sycl::queue(::sycl::gpu_selector_v);
        } else {
            queue_register[device_spec.Name()] = ::sycl::queue(::sycl::default_selector_v);
        }
    }
    return queue_register.at(device_spec.Name());
}  

DeviceManager::DeviceRegister& DeviceManager::GetDevicesRegister() const {
    static DeviceRegister device_register;

    if (device_register.devices.size() == 0) {
        std::lock_guard<std::mutex> guard(device_registering_mutex);
        std::vector<::sycl::device> devices = ::sycl::device::get_devices();
        for (size_t i = 0; i < devices.size(); i++) {
            LOG(INFO) << "device_index = " << i << ", name = " << devices[i].get_info<::sycl::info::device::name>();
        }

        for (size_t i = 0; i < devices.size(); i++) {
            device_register.devices.push_back(devices[i]);
            if (devices[i].is_cpu()) {
                device_register.cpu_devices.push_back(devices[i]);
            } else if (devices[i].is_gpu()) {
                device_register.gpu_devices.push_back(devices[i]);
            }
        }
    }
    return device_register;
}    

DeviceManager::QueueRegister_t& DeviceManager::GetQueueRegister() const {
    static QueueRegister_t queue_register;
    return queue_register;
}

}  // namespace sycl
}  // namespace xgboost