/**
 * Copyright 2024-2025, XGBoost contributors
 */
#if defined(XGBOOST_USE_CUDA)
#include "cuda_dr_utils.h"

#include <algorithm>  // for max
#include <cstdint>    // for int32_t
#include <cstring>    // for memset
#include <memory>     // for make_unique
#include <mutex>      // for call_once
#include <sstream>    // for stringstream
#include <string>     // for string, stoi

#include "common.h"               // for safe_cuda, TrimFirst, Split
#include "cuda_rt_utils.h"        // for CurrentDevice
#include "io.h"                   // for CmdOutput
#include "xgboost/string_view.h"  // for StringView

namespace xgboost::cudr {
CuDriverApi::CuDriverApi() {
  // similar to dlopen, but without the need to release a handle.
  auto safe_load = [](xgboost::StringView name, auto **fnptr) {
    cudaDriverEntryPointQueryResult status;
#if (CUDA_VERSION / 1000) >= 13
    dh::safe_cuda(cudaGetDriverEntryPointByVersion(name.c_str(), reinterpret_cast<void **>(fnptr),
                                                   12080, cudaEnablePerThreadDefaultStream,
                                                   &status));
#else
    dh::safe_cuda(cudaGetDriverEntryPoint(name.c_str(), reinterpret_cast<void **>(fnptr),
                                          cudaEnablePerThreadDefaultStream, &status));
#endif  // (CUDA_VERSION / 1000) >= 13
    CHECK(status == cudaDriverEntryPointSuccess) << name;
    CHECK(*fnptr);
  };

  safe_load("cuMemGetAllocationGranularity", &this->cuMemGetAllocationGranularity);
  safe_load("cuMemCreate", &this->cuMemCreate);
  safe_load("cuMemMap", &this->cuMemMap);
  safe_load("cuMemAddressReserve", &this->cuMemAddressReserve);
  safe_load("cuMemSetAccess", &this->cuMemSetAccess);
  safe_load("cuMemUnmap", &this->cuMemUnmap);
  safe_load("cuMemRelease", &this->cuMemRelease);
  safe_load("cuMemAddressFree", &this->cuMemAddressFree);
  safe_load("cuGetErrorString", &this->cuGetErrorString);
  safe_load("cuGetErrorName", &this->cuGetErrorName);
  safe_load("cuDeviceGetAttribute", &this->cuDeviceGetAttribute);
  safe_load("cuDeviceGet", &this->cuDeviceGet);

  CHECK(this->cuMemGetAllocationGranularity);
}

void CuDriverApi::ThrowIfError(CUresult status, StringView fn, std::int32_t line,
                               char const *file) const {
  if (status == CUDA_SUCCESS) {
    return;
  }
  std::string cuerr{"CUDA driver error:"};

  char const *name{nullptr};
  auto err0 = this->cuGetErrorName(status, &name);
  if (err0 != CUDA_SUCCESS) {
    LOG(WARNING) << cuerr << status << ". Then we failed to get error name:" << err0;
  }
  char const *msg{nullptr};
  auto err1 = this->cuGetErrorString(status, &msg);
  if (err1 != CUDA_SUCCESS) {
    LOG(WARNING) << cuerr << status << ". Then we failed to get error string:" << err1;
  }

  std::stringstream ss;
  ss << fn << "[" << file << ":" << line << "]:";
  if (name != nullptr && err0 == CUDA_SUCCESS) {
    ss << cuerr << " " << name << ".";
  }
  if (msg != nullptr && err1 == CUDA_SUCCESS) {
    ss << " " << msg << "\n";
  }
  LOG(FATAL) << ss.str();
}

[[nodiscard]] CuDriverApi &GetGlobalCuDriverApi() {
  static std::once_flag flag;
  static std::unique_ptr<CuDriverApi> cu;
  std::call_once(flag, [&] { cu = std::make_unique<CuDriverApi>(); });
  return *cu;
}

void MakeCuMemLocation(CUmemLocationType type, CUmemLocation *loc) {
  auto ordinal = curt::CurrentDevice();
  loc->type = type;

  if (type == CU_MEM_LOCATION_TYPE_DEVICE) {
    loc->id = ordinal;
  } else {
    std::int32_t numa_id = -1;
    CUdevice device;
    safe_cu(GetGlobalCuDriverApi().cuDeviceGet(&device, ordinal));
    safe_cu(GetGlobalCuDriverApi().cuDeviceGetAttribute(&numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                                                        device));
    numa_id = std::max(numa_id, 0);

    loc->id = numa_id;
  }
}

[[nodiscard]] CUmemAllocationProp MakeAllocProp(CUmemLocationType type) {
  CUmemAllocationProp prop;
  std::memset(&prop, '\0', sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  MakeCuMemLocation(type, &prop.location);
  return prop;
}

[[nodiscard]] bool GetVersionFromSmi(std::int32_t *p_major, std::int32_t *p_minor) {
  using ::xgboost::common::Split;
  using ::xgboost::common::TrimFirst;
  // `nvidia-smi --version` is not available for older versions, as a result, we can't query the
  // cuda driver version unless we want to parse the table output.

  // Example output on a 2-GPU system:
  //
  // $ nvidia-smi --query-gpu=driver_version --format=csv
  //
  // driver_version
  // 570.124.06
  // 570.124.06
  //
  auto cmd = "nvidia-smi --query-gpu=driver_version --format=csv";
  auto smi_out_str = common::CmdOutput(StringView{cmd});

  auto Invalid = [=] {
    *p_major = *p_minor = -1;
    return false;
  };
  if (smi_out_str.empty()) {
    return Invalid();
  }

  auto smi_split = Split(smi_out_str, '\n');
  if (smi_split.size() < 2) {
    return Invalid();
  }

  // Use the first GPU
  auto smi_ver = Split(TrimFirst(smi_split[1]), '.');
  // 570.124.06
  // On WSL2, you can have driver version with two components, e.g. 573.24
  if (smi_ver.size() != 2 && smi_ver.size() != 3) {
    return Invalid();
  }
  try {
    *p_major = std::stoi(smi_ver[0]);
    *p_minor = std::stoi(smi_ver[1]);
    LOG(INFO) << "Driver version: `" << *p_major << "." << *p_minor << "`";
    return true;
  } catch (std::exception const &) {
  }

  return Invalid();
}
}  // namespace xgboost::cudr
#endif
