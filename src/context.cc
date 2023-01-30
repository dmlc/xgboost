/**
 * Copyright 2014-2023 by XGBoost Contributors
 *
 * \brief Context object used for controlling runtime parameters.
 */
#include <xgboost/context.h>

#include "common/common.h"  // AssertGPUSupport
#include "common/threading_utils.h"

namespace xgboost {

DMLC_REGISTER_PARAMETER(Context);

std::int32_t constexpr Context::kCpuId;
std::int64_t constexpr Context::kDefaultSeed;

Context::Context() : cfs_cpu_count_{common::GetCfsCPUCount()} {}

void Context::ConfigureGpuId(bool require_gpu) {
#if defined(XGBOOST_USE_CUDA)
  if (gpu_id == kCpuId) {  // 0. User didn't specify the `gpu_id'
    if (require_gpu) {     // 1. `tree_method' or `predictor' or both are using
                           // GPU.
      // 2. Use device 0 as default.
      this->UpdateAllowUnknown(Args{{"gpu_id", "0"}});
    }
  }

  // 3. When booster is loaded from a memory image (Python pickle or R
  // raw model), number of available GPUs could be different.  Wrap around it.
  int32_t n_gpus = common::AllVisibleGPUs();
  if (n_gpus == 0) {
    if (gpu_id != kCpuId) {
      LOG(WARNING) << "No visible GPU is found, setting `gpu_id` to -1";
    }
    this->UpdateAllowUnknown(Args{{"gpu_id", std::to_string(kCpuId)}});
  } else if (fail_on_invalid_gpu_id) {
    CHECK(gpu_id == kCpuId || gpu_id < n_gpus)
        << "Only " << n_gpus << " GPUs are visible, gpu_id " << gpu_id << " is invalid.";
  } else if (gpu_id != kCpuId && gpu_id >= n_gpus) {
    LOG(WARNING) << "Only " << n_gpus << " GPUs are visible, setting `gpu_id` to "
                 << gpu_id % n_gpus;
    this->UpdateAllowUnknown(Args{{"gpu_id", std::to_string(gpu_id % n_gpus)}});
  }
#else
  // Just set it to CPU, don't think about it.
  this->UpdateAllowUnknown(Args{{"gpu_id", std::to_string(kCpuId)}});
  (void)(require_gpu);
#endif  // defined(XGBOOST_USE_CUDA)

  common::SetDevice(this->gpu_id);
}

std::int32_t Context::Threads() const {
  auto n_threads = common::OmpGetNumThreads(nthread);
  if (cfs_cpu_count_ > 0) {
    n_threads = std::min(n_threads, cfs_cpu_count_);
  }
  return n_threads;
}

#if !defined(XGBOOST_USE_CUDA)
CUDAContext const* Context::CUDACtx() const {
  common::AssertGPUSupport();
  return nullptr;
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
