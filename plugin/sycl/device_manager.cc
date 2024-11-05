/*!
 * Copyright 2017-2023 by Contributors
 * \file device_manager.cc
 */
#include "../sycl/device_manager.h"

#include "../../src/collective/communicator-inl.h"

namespace xgboost {
namespace sycl {

::sycl::queue* DeviceManager::GetQueue(const DeviceOrd& device_spec) const {
    if (!device_spec.IsSycl()) {
        LOG(WARNING) << "Sycl kernel is executed with non-sycl context: "
                     << device_spec.Name() << ". "
                     << "Default sycl device_selector will be used.";
    }

    size_t queue_idx;
    bool not_use_default_selector = (device_spec.ordinal != kDefaultOrdinal) ||
                                    (collective::IsDistributed());
    DeviceRegister& device_register = GetDevicesRegister();
    if (not_use_default_selector) {
        const int device_idx =
            collective::IsDistributed() ? collective::GetRank() : device_spec.ordinal;
        if (device_spec.IsSyclDefault()) {
            auto& devices = device_register.devices;
            CHECK_LT(device_idx, devices.size());
            queue_idx = device_idx;
        } else if (device_spec.IsSyclCPU()) {
            auto& cpu_devices_idxes = device_register.cpu_devices_idxes;
            CHECK_LT(device_idx, cpu_devices_idxes.size());
            queue_idx = cpu_devices_idxes[device_idx];
        } else if (device_spec.IsSyclGPU()) {
            auto& gpu_devices_idxes = device_register.gpu_devices_idxes;
            CHECK_LT(device_idx, gpu_devices_idxes.size());
            queue_idx = gpu_devices_idxes[device_idx];
        } else {
            LOG(WARNING) << device_spec << " is not sycl, sycl:cpu or sycl:gpu";
            auto device = ::sycl::queue(::sycl::default_selector_v).get_device();
            queue_idx = device_register.devices.at(device);
        }
    } else {
        if (device_spec.IsSyclCPU()) {
            auto device = ::sycl::queue(::sycl::cpu_selector_v).get_device();
            queue_idx = device_register.devices.at(device);
        } else if (device_spec.IsSyclGPU()) {
            auto device = ::sycl::queue(::sycl::gpu_selector_v).get_device();
            queue_idx = device_register.devices.at(device);
        } else {
            auto device = ::sycl::queue(::sycl::default_selector_v).get_device();
            queue_idx = device_register.devices.at(device);
        }
    }
    return &(device_register.queues[queue_idx]);
}

DeviceManager::DeviceRegister& DeviceManager::GetDevicesRegister() const {
    static DeviceRegister device_register;

    if (device_register.devices.size() == 0) {
        std::lock_guard<std::mutex> guard(device_registering_mutex);
        std::vector<::sycl::device> devices = ::sycl::device::get_devices();
        for (size_t i = 0; i < devices.size(); i++) {
            LOG(INFO) << "device_index = " << i << ", name = "
                      << devices[i].get_info<::sycl::info::device::name>();
        }

        for (size_t i = 0; i < devices.size(); i++) {
            device_register.devices[devices[i]] = i;
            device_register.queues.push_back(::sycl::queue(devices[i]));
            if (devices[i].is_cpu()) {
                device_register.cpu_devices_idxes.push_back(i);
            } else if (devices[i].is_gpu()) {
                device_register.gpu_devices_idxes.push_back(i);
            }
        }
    }
    return device_register;
}

}  // namespace sycl
}  // namespace xgboost
