#include <iostream>
#include "device_helpers.cuh"

size_t PeakCudaMemory(std::string name, bool start) {
  if (start) {
    std::cout << "S:";
  } else {
    std::cout << "E:";
  }
  std::cout << name << ": " << dh::GlobalMemoryLogger().PeakMemory() << std::endl;
  return dh::GlobalMemoryLogger().PeakMemory();
}