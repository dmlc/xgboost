/**
 * Copyright 2014-2023 by XGBoost Contributors
 *
 * \brief Context object used for controlling runtime parameters.
 */
#include "xgboost/context.h"

#include <algorithm>  // for find_if
#include <optional>   // for optional
#include <regex>      // for regex_replace, regex_match

#include "common/common.h"     // AssertGPUSupport
#include "common/error_msg.h"  // WarnDeprecatedGPUId
#include "common/threading_utils.h"
#include "xgboost/string_view.h"

namespace xgboost {

DMLC_REGISTER_PARAMETER(Context);

bst_d_ordinal_t constexpr Context::kCpuId;
std::int64_t constexpr Context::kDefaultSeed;

Context::Context() : cfs_cpu_count_{common::GetCfsCPUCount()} {}

namespace {
inline constexpr char const* kDevice = "device";

#if !defined(XGBOOST_USE_CUDA)
DeviceOrd CUDAOrdinal(DeviceOrd device, bool) {
  device = DeviceOrd::CPU();
  return device;
}
#else
// Check CUDA on the current device, wrap the ordinal if necessary.
DeviceOrd CUDAOrdinal(DeviceOrd device, bool fail_on_invalid) {
  // When booster is loaded from a memory image (Python pickle or R raw model), number of
  // available GPUs could be different.  Wrap around it.
  std::int32_t n_visible = common::AllVisibleGPUs();
  if (n_visible == 0) {
    if (device.IsCUDA()) {
      LOG(WARNING) << "No visible GPU is found, setting device to CPU.";
    }
    device = DeviceOrd::CPU();
  } else if (fail_on_invalid) {
    CHECK(device.IsCPU() || device.ordinal < n_visible)
        << "Only " << n_visible << " GPUs are visible, ordinal " << device.ordinal
        << " is invalid.";
  } else if (device.IsCUDA() && device.ordinal >= n_visible) {
    device.ordinal = device.ordinal % n_visible;
    LOG(WARNING) << "Only " << n_visible << " GPUs are visible, setting device ordinal to "
                 << device.ordinal;
  }

  if (device.IsCUDA()) {
    common::SetDevice(device.ordinal);
  }
  return device;
}
#endif  //  !defined(XGBOOST_USE_CUDA)

std::optional<std::int32_t> ParseInt(std::string const& ordinal) {
  // boost::lexical_cast should be used instead, but for now some basic checks will do
  if (ordinal.empty()) {
    return std::nullopt;
  }

  std::size_t offset{0};
  if (ordinal.front() == '-') {
    offset = 1;
  }
  if (ordinal.size() <= offset) {
    return std::nullopt;
  }

  bool valid = std::all_of(ordinal.cbegin() + offset, ordinal.cend(),
                           [](auto c) { return std::isdigit(c); });
  if (!valid) {
    return std::nullopt;
  }

  std::int32_t parsed_id{Context::kCpuId};
  try {
    parsed_id = std::stoi(ordinal);
  } catch (std::exception const& e) {
    return std::nullopt;
  }

  return parsed_id;
}

DeviceOrd MakeDeviceOrd(std::string const& input, bool fail_on_invalid_gpu_id) {
  StringView msg{R"(Invalid argument for `device`. Expected to be one of the following:
- cpu
- cuda
- cuda:<device ordinal>  # e.g. cuda:0
- gpu
- gpu:<device ordinal>   # e.g. gpu:0
)"};

  std::regex pattern{"gpu(:[0-9]+)?|cuda(:[0-9]+)?|cpu"};
  if (!std::regex_match(input, pattern)) {
    LOG(FATAL) << msg << "Got:" << input;
  }
  // handle alias
  std::string device_str = std::regex_replace(input, std::regex{"gpu"}, "cuda");

  auto split_it = std::find(device_str.cbegin(), device_str.cend(), ':');

  DeviceOrd device;
  device.ordinal = -2;  // mark it invalid for check.
  if (split_it == device_str.cend()) {
    // no ordinal.
    if (device_str == "cpu") {
      device = DeviceOrd::CPU();
    } else if (device_str == "cuda") {
      device = DeviceOrd::CUDA(0);  // use 0 as default;
    } else {
      LOG(FATAL) << msg << "Got: " << input;
    }
  } else {
    // must be CUDA when ordinal is specifed.
    auto splited = common::Split(device_str, ':');
    device_str = splited[0];
    auto ordinal_str = splited[1];
    auto opt_id = ParseInt(ordinal_str);
    if (!opt_id.has_value()) {
      LOG(FATAL) << msg << "Got: " << input;
    }
    CHECK_LE(opt_id.value(), std::numeric_limits<bst_d_ordinal_t>::max())
        << "Ordinal value too large.";
    device = DeviceOrd::CUDA(opt_id.value());
  }
  CHECK_GE(device.ordinal, Context::kCpuId) << msg;

  device = CUDAOrdinal(device, fail_on_invalid_gpu_id);

  return device;
}
}  // namespace

void Context::ConfigureGpuId(bool require_gpu) {
  if (this->IsCPU() && require_gpu) {
    this->UpdateAllowUnknown(Args{{kDevice, "cuda"}});
  }
}

void Context::SetDeviceOrdinal(Args const& kwargs) {
  auto gpu_id_it = std::find_if(kwargs.cbegin(), kwargs.cend(),
                                [](auto const& p) { return p.first == "gpu_id"; });
  auto has_gpu_id = gpu_id_it != kwargs.cend();
  auto device_it = std::find_if(kwargs.cbegin(), kwargs.cend(),
                                [](auto const& p) { return p.first == kDevice; });
  auto has_device = device_it != kwargs.cend();
  if (has_device && has_gpu_id) {
    LOG(FATAL) << "Both `device` and `gpu_id` are specified. Use `device` instead.";
  }

  if (has_gpu_id) {
    // Compatible with XGBoost < 2.0.0
    error::WarnDeprecatedGPUId();
    auto opt_id = ParseInt(gpu_id_it->second);
    CHECK(opt_id.has_value()) << "Invalid value for `gpu_id`. Got:" << gpu_id_it->second;
    if (opt_id.value() > Context::kCpuId) {
      this->UpdateAllowUnknown(Args{{kDevice, DeviceOrd::CUDA(opt_id.value()).Name()}});
    } else {
      this->UpdateAllowUnknown(Args{{kDevice, DeviceOrd::CPU().Name()}});
    }
    return;
  }

  auto new_d = MakeDeviceOrd(this->device, this->fail_on_invalid_gpu_id);

  if (!has_device) {
    CHECK_EQ(new_d.ordinal, this->device_.ordinal);  // unchanged
  }
  this->SetDevice(new_d);

  if (this->IsCPU()) {
    CHECK_EQ(this->device_.ordinal, kCpuId);
  } else {
    CHECK_GT(this->device_.ordinal, kCpuId);
  }
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
