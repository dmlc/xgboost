#include <cctype>  // std::tolower
#include <string>

#include "common/common.h"
#include "common/threading_utils.h"
#include "xgboost/generic_parameters.h"
#include "xgboost/global_config.h"
#include "xgboost/string_view.h"

namespace xgboost {
int32_t constexpr Context::kCpuId;
int64_t constexpr Context::kDefaultSeed;

Context::Context() noexcept(false) : cfs_cpu_count_{common::GetCfsCPUCount()} {
  StringView msg{R"(Invalid argument for `device`. Expected to be one of the following:
- CPU
- CUDA
- CUDA:<device ordinal>
)"};
  auto original = GlobalConfigThreadLocalStore::Get()->device;
  std::string device = original;
  std::transform(device.cbegin(), device.cend(), device.begin(),
                 [](auto c) { return std::tolower(c); });
  auto split_it = std::find(device.cbegin(), device.cend(), ':');
  gpu_id = -2;  // mark it invalid for check.
  if (split_it == device.cend()) {
    // no ordinal.
    if (device == "cpu") {
      gpu_id = kCpuId;
    } else if (device == "cuda") {
      gpu_id = 0;  // use 0 as default;
    } else {
      LOG(FATAL) << msg << "Got: " << original;
    }
  } else {
    // must be CUDA when ordinal is specifed.
    auto splited = common::Split(device, ':');
    CHECK_EQ(splited.size(), 2) << msg;
    device = splited[0];
    CHECK_EQ(device, "cuda") << msg << "Got: " << original;

    // boost::lexical_cast should be used instead, but for now some basic checks will do
    auto ordinal = splited[1];
    CHECK_GE(ordinal.size(), 1) << msg << "Got: " << original;
    CHECK(std::isdigit(ordinal.front())) << msg << "Got: " << original;
    try {
      gpu_id = std::stoi(splited[1]);
    } catch (std::exception const& e) {
      LOG(FATAL) << msg << "Got: " << original;
    }
  }
  CHECK_GE(gpu_id, kCpuId) << msg;
}

void Context::ConfigureGpuId() {
#if defined(XGBOOST_USE_CUDA)
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
#endif  // defined(XGBOOST_USE_CUDA)
}

int32_t Context::Threads() const {
  auto n_threads = common::OmpGetNumThreads(nthread);
  if (cfs_cpu_count_ > 0) {
    n_threads = std::min(n_threads, cfs_cpu_count_);
  }
  return n_threads;
}

DMLC_REGISTER_PARAMETER(Context);
}  // namespace xgboost
