#include <vector>

#include "rmm/mr/device/cuda_memory_resource.hpp"
#include "rmm/mr/device/pool_memory_resource.hpp"

#include "rmm_bridge.h"

using RMMCUDAMemoryResource = rmm::mr::cuda_memory_resource;
using RMMPoolMemoryResource = rmm::mr::pool_memory_resource<RMMCUDAMemoryResource>;

RMMCUDAMemoryResource cuda_mr;
RMMPoolMemoryResource pool_mr{&cuda_mr};

void* allocate(size_t nbyte) {
  std::cerr << "Allocating " << nbyte << " bytes using the RMM pool allocator" << std::endl;
  return pool_mr.allocate(nbyte);
}

void deallocate(void* ptr, size_t nbyte) {
  std::cerr << "Freeing " << nbyte << " bytes from the RMM pool allocator" << std::endl;
  return pool_mr.deallocate(ptr, nbyte);
}
