/**
 * Copyright 2018-2024, XGBoost contributors
 */
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <unistd.h>  // getpid

#include "../collective/communicator-inl.h"
#include "common.h"

namespace dh {
void ThrowOnCudaError(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::string f;
    if (file != nullptr) {
      f = file;
    }
    std::string error =
        thrust::system_error(code, thrust::cuda_category(), f + ": " + std::to_string(line)).what();
    auto rank = xgboost::collective::GetRank();
    LOG(CONSOLE) << "CUDA error:" << error << "\nrank:" << rank << " pid:" << getpid() << "\n"
                 << dmlc::StackTrace(1, 32) << std::endl;
    LOG(FATAL) << error;
  }
}
}  // namespace dh
