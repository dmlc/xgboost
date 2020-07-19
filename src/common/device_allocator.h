/*!
 * Copyright 2020 by XGBoost Contributors
 * \file device_allocator.h
 * \brief Store callback functions for allocating and de-allocating memory on GPU devices.
 */
#ifndef XGBOOST_COMMON_DEVICE_ALLOCATOR_H_
#define XGBOOST_COMMON_DEVICE_ALLOCATOR_H_

#include <mutex>

namespace dh {
namespace detail {

using DeviceAllocateFunc = void *(*)(size_t);
using DeviceDeallocateFunc = void (*)(void *, size_t);

struct DeviceMemoryResource {
  DeviceAllocateFunc allocate;
  DeviceDeallocateFunc deallocate;
};

extern DeviceMemoryResource DeviceMemoryResourceSingleton;
extern std::mutex DeviceMemoryResourceSingletonMutex;

inline DeviceMemoryResource GetDeviceMemoryResource() {
  std::lock_guard<std::mutex> guard(DeviceMemoryResourceSingletonMutex);
  return DeviceMemoryResourceSingleton;
}

inline void RegisterDeviceAllocatorCallback(DeviceAllocateFunc allocate,
                                            DeviceDeallocateFunc deallocate) {
  std::lock_guard<std::mutex> guard(DeviceMemoryResourceSingletonMutex);
  DeviceMemoryResourceSingleton = {allocate, deallocate};
}

}  // namespace detail
}  // namespace dh

#endif  // XGBOOST_COMMON_DEVICE_ALLOCATOR_H_
