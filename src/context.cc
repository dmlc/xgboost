/**
 * Copyright 2014-2023 by XGBoost Contributors
 *
 * \brief Context object used for controlling runtime parameters.
 */
#include "xgboost/context.h"

#include <algorithm>  // for find_if
#include <charconv>   // for from_chars
#include <iterator>   // for distance
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
[[nodiscard]] DeviceOrd CUDAOrdinal(DeviceOrd device, bool fail_on_invalid) {
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

[[nodiscard]] std::optional<std::int32_t> ParseInt(StringView ordinal) {
  // Some basic checks to ensure valid `gpu_id` and device ordinal instead of directly parsing and
  // letting go of unknown characters.
  if (ordinal.empty()) {
    return std::nullopt;
  }

  std::size_t offset{0};
  if (ordinal[0] == '-') {
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
  auto res = std::from_chars(ordinal.c_str(), ordinal.c_str() + ordinal.size(), parsed_id);
  if (res.ec != std::errc()) {
    return std::nullopt;
  }

  return parsed_id;
}

[[nodiscard]] DeviceOrd MakeDeviceOrd(std::string const& input, bool fail_on_invalid_gpu_id) {
  StringView msg{R"(Invalid argument for `device`. Expected to be one of the following:
- cpu
- cuda
- cuda:<device ordinal>  # e.g. cuda:0
- gpu
- gpu:<device ordinal>   # e.g. gpu:0
)"};
  auto fatal = [&] { LOG(FATAL) << msg << "Got: `" << input << "`."; };

#if defined(__MINGW32__)
  // mingw hangs on regex using rtools 430. Basic checks only.
  CHECK_GE(input.size(), 3) << msg;
  auto substr = input.substr(0, 3);
  bool valid = substr == "cpu" || substr == "cud" || substr == "gpu";
  CHECK(valid) << msg;
#else
  std::regex pattern{"gpu(:[0-9]+)?|cuda(:[0-9]+)?|cpu"};
  if (!std::regex_match(input, pattern)) {
    fatal();
  }
#endif  // defined(__MINGW32__)

  // handle alias
  std::string s_device = std::regex_replace(input, std::regex{"gpu"}, DeviceSym::CUDA());

  auto split_it = std::find(s_device.cbegin(), s_device.cend(), ':');
  DeviceOrd device;
  device.ordinal = Context::InvalidOrdinal();  // mark it invalid for check.
  if (split_it == s_device.cend()) {
    // no ordinal.
    if (s_device == DeviceSym::CPU()) {
      device = DeviceOrd::CPU();
    } else if (s_device == DeviceSym::CUDA()) {
      device = DeviceOrd::CUDA(0);  // use 0 as default;
    } else {
      fatal();
    }
  } else {
    // must be CUDA when ordinal is specifed.
    // +1 for colon
    std::size_t offset = std::distance(s_device.cbegin(), split_it) + 1;
    // substr
    StringView s_ordinal = {s_device.data() + offset, s_device.size() - offset};
    if (s_ordinal.empty()) {
      fatal();
    }
    auto opt_id = ParseInt(s_ordinal);
    if (!opt_id.has_value()) {
      fatal();
    }
    CHECK_LE(opt_id.value(), std::numeric_limits<bst_d_ordinal_t>::max())
        << "Ordinal value too large.";
    device = DeviceOrd::CUDA(opt_id.value());
  }

  if (device.ordinal < Context::kCpuId) {
    fatal();
  }
  device = CUDAOrdinal(device, fail_on_invalid_gpu_id);

  return device;
}
}  // namespace

void Context::ConfigureGpuId(bool require_gpu) {
  if (this->IsCPU() && require_gpu) {
    this->UpdateAllowUnknown(Args{{kDevice, DeviceSym::CUDA()}});
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
    auto opt_id = ParseInt(StringView{gpu_id_it->second});
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
