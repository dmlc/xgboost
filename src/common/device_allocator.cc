/*!
 * Copyright 2020 by XGBoost Contributors
 * \file device_allocator.cc
 * \brief Store callback functions for allocating and de-allocating memory on GPU devices.
 */
#include <mutex>
#include "device_allocator.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace dh {
namespace detail {

DeviceMemoryResource DeviceMemoryResourceSingleton{nullptr, nullptr};
LibraryHandle LibraryHandleSingleton{nullptr};
std::mutex DeviceMemoryResourceSingletonMutex;

LibraryHandle OpenLibrary(const char* libpath) {
#ifdef _WIN32
  HMODULE handle = LoadLibraryA(libpath);
#else
  void* handle = dlopen(libpath, RTLD_LAZY | RTLD_LOCAL);
#endif
  return static_cast<LibraryHandle>(handle);
}

void CloseLibrary(LibraryHandle handle) {
#ifdef _WIN32
  FreeLibrary(static_cast<HMODULE>(handle));
#else
  dlclose(static_cast<void*>(handle));
#endif
}

FunctionHandle LoadFunction(LibraryHandle lib_handle, const char* name) {
#ifdef _WIN32
  FARPROC func_handle = GetProcAddress(static_cast<HMODULE>(lib_handle), name);
#else
  void* func_handle = dlsym(static_cast<void*>(lib_handle), name);
#endif
  return static_cast<FunctionHandle>(func_handle);
}

}  // namespace detail
}  // namespace dh
