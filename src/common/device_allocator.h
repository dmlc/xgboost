/*!
 * Copyright 2020 by XGBoost Contributors
 * \file device_allocator.h
 * \brief Store callback functions for allocating and de-allocating memory on GPU devices.
 */
#ifndef XGBOOST_COMMON_DEVICE_ALLOCATOR_H_
#define XGBOOST_COMMON_DEVICE_ALLOCATOR_H_

#include <mutex>
#include <xgboost/logging.h>

namespace dh {
namespace detail {

using DeviceAllocateFunc = void* (*)(size_t);
using DeviceDeallocateFunc = void (*)(void*, size_t);
using LibraryHandle = void*;
using FunctionHandle = void*;

struct DeviceMemoryResource {
  DeviceAllocateFunc allocate;
  DeviceDeallocateFunc deallocate;
};

extern DeviceMemoryResource DeviceMemoryResourceSingleton;
extern std::mutex DeviceMemoryResourceSingletonMutex;

LibraryHandle OpenLibrary(const char* libpath);
void CloseLibrary(LibraryHandle handle);
FunctionHandle LoadFunction(LibraryHandle lib_handle, const char* name);

inline DeviceMemoryResource GetDeviceMemoryResource() {
  std::lock_guard<std::mutex> guard(DeviceMemoryResourceSingletonMutex);
  return DeviceMemoryResourceSingleton;
}

inline void RegisterGPUDeviceAllocator(const char* libpath) {
  LibraryHandle lib = OpenLibrary(libpath);
  CHECK(lib) << "Failed to load dynamic shared library `" << libpath << "'";
  auto allocate = reinterpret_cast<DeviceAllocateFunc>(LoadFunction(lib, "allocate"));
  auto deallocate = reinterpret_cast<DeviceDeallocateFunc>(LoadFunction(lib, "deallocate"));
  CHECK(allocate) << "Could not load function void* allocate(size_t)";
  CHECK(deallocate) << "Could not load function void deallocate(void*, size_t)";
  std::lock_guard<std::mutex> guard(DeviceMemoryResourceSingletonMutex);
  DeviceMemoryResourceSingleton = {allocate, deallocate};
}

}  // namespace detail
}  // namespace dh

#endif  // XGBOOST_COMMON_DEVICE_ALLOCATOR_H_
