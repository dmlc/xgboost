/**
 * Copyright 2018-2024, XGBoost contributors
 */
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#include "common.h"

namespace dh {
void ThrowOnCudaError(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::string f;
    if (file != nullptr) {
      f = file;
    }
    LOG(FATAL) << thrust::system_error(code, thrust::cuda_category(),
                                       f + ": " + std::to_string(line))
                      .what();
  }
}
}  // namespace dh
