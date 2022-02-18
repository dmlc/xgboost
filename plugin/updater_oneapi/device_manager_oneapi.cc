/*!
 * Copyright 2017-2022 by Contributors
 * \file device_manager_oneapi.cc
 */
#include "./device_manager_oneapi.h"

namespace xgboost {

sycl::device DeviceManagerOneAPI::GetDevice(const DeviceId::Specification& device_spec) const {   
    if (device_spec.Index() == DeviceId::kDefaultIndex) {
        if (device_spec.Type() == DeviceType::kOneAPI_CPU) {
            return sycl::device(sycl::cpu_selector());
        } else {
            return sycl::device(sycl::gpu_selector());
        }
    } else {
        DeviceRegister& device_register = GetDevicesRegister();
        if (device_spec.Type() == DeviceType::kOneAPI_CPU) {
            auto& cpu_devices = device_register.cpu_devices;
            CHECK_LT(device_spec.Index(), cpu_devices.size());
            return cpu_devices[device_spec.Index()];
        } else {
            auto& gpu_devices = device_register.gpu_devices;
            CHECK_LT(device_spec.Index(), gpu_devices.size());
            return gpu_devices[device_spec.Index()];
        }
    }
}

sycl::queue DeviceManagerOneAPI::GetQueue(const DeviceId::Specification& device_spec) const {
    QueueRegister_t& queue_register = GetQueueRegister();
    if (queue_register.count(device_spec.Name()) > 0) {
        return queue_register.at(device_spec.Name());
    }
    
    std::lock_guard<std::mutex> guard(queue_registering_mutex);
    if (device_spec.Index() != DeviceId::kDefaultIndex) {
        DeviceRegister& device_register = GetDevicesRegister();
        if (device_spec.Type() == DeviceType::kOneAPI_CPU) {
            auto& cpu_devices = device_register.cpu_devices;
            queue_register[device_spec.Name()] = sycl::queue(cpu_devices[device_spec.Index()]);
        } else if (device_spec.Type() == DeviceType::kOneAPI_GPU) {
            auto& gpu_devices = device_register.gpu_devices;
            queue_register[device_spec.Name()] = sycl::queue(gpu_devices[device_spec.Index()]);
        }
    } else {
        if (device_spec.Type() == DeviceType::kOneAPI_CPU) {
            sycl::cpu_selector selector;
            queue_register[device_spec.Name()] = sycl::queue(selector);
        } else if (device_spec.Type() == DeviceType::kOneAPI_GPU) {
            sycl::gpu_selector selector;
            queue_register[device_spec.Name()] = sycl::queue(selector);
        }
    }
    return queue_register.at(device_spec.Name());
}  

DeviceManagerOneAPI::DeviceRegister& DeviceManagerOneAPI::GetDevicesRegister() const {
    static DeviceRegister device_register;

    if (device_register.cpu_devices.size() == 0) {
        std::lock_guard<std::mutex> guard(device_registering_mutex);
        std::vector<sycl::device> devices = sycl::device::get_devices();
        for (size_t i = 0; i < devices.size(); i++) {
            LOG(INFO) << "device_index = " << i << ", name = " << devices[i].get_info<sycl::info::device::name>();
        }

        for (size_t i = 0; i < devices.size(); i++) {
            if (devices[i].is_cpu()) {
                device_register.cpu_devices.push_back(devices[i]);
            } else if (devices[i].is_gpu()) {
                device_register.gpu_devices.push_back(devices[i]);
            }
        }
    }
    return device_register;
}    

DeviceManagerOneAPI::QueueRegister_t& DeviceManagerOneAPI::GetQueueRegister() const {
    static QueueRegister_t queue_register;
    return queue_register;
}

}  // namespace xgboost
