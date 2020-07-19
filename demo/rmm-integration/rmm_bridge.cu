#include <vector>
#include <string>
#include <cstdio>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/thread_safe_resource_adaptor.hpp>

#include "rmm_bridge.h"

using RMMCUDAMemoryResource = rmm::mr::cuda_memory_resource;
using RMMPoolMemoryResource = rmm::mr::pool_memory_resource<RMMCUDAMemoryResource>;
using RMMThreadSafePoolMemoryResource = rmm::mr::thread_safe_resource_adaptor<RMMPoolMemoryResource>;
void default_log(const char* msg) {}

RMMCUDAMemoryResource cuda_mr;
RMMPoolMemoryResource pool_mr{&cuda_mr};
RMMThreadSafePoolMemoryResource thread_safe_mr{&pool_mr};
void (*write_log)(const char*) = default_log;

void* allocate(size_t nbyte) {
  std::string msg = "Allocating " + std::to_string(nbyte) + " bytes using the RMM pool allocator";
  write_log(msg.c_str());
  return thread_safe_mr.allocate(nbyte);
}

void deallocate(void* ptr, size_t nbyte) {
  std::string msg = "Freeing " + std::to_string(nbyte) + " bytes using the RMM pool allocator";
  write_log(msg.c_str());
  return thread_safe_mr.deallocate(ptr, nbyte);
}

void set_log_callback(void (*callback) (const char*)) {
  write_log = callback;
}
