/**
 * Copyright 2017-2024, XGBoost contributors
 */
#include "../collective/communicator-inl.h"  // for GetRank
#include "common.h"                          // for HumanMemUnit
#include "cuda_dr_utils.h"
#include "device_helpers.cuh"  // for CurrentDevice
#include "device_vector.cuh"

namespace dh {
namespace detail {
void ThrowOOMError(std::string const &err, std::size_t bytes) {
  auto device = CurrentDevice();
  auto rank = xgboost::collective::GetRank();
  using xgboost::common::HumanMemUnit;
  std::stringstream ss;
  ss << "Memory allocation error on worker " << rank << ": " << err << "\n"
     << "- Free memory: " << HumanMemUnit(dh::AvailableMemory(device)) << "\n"
     << "- Requested memory: " << HumanMemUnit(bytes) << std::endl;
  LOG(FATAL) << ss.str();
}

GrowOnlyVirtualMemVec::GrowOnlyVirtualMemVec(CUmemLocationType type)
    : prop_{xgboost::cudr::MakeAllocProp(type)} {
  CHECK(type == CU_MEM_LOCATION_TYPE_DEVICE || type == CU_MEM_LOCATION_TYPE_HOST_NUMA);
  // Get the allocation granularity.
  this->granularity_ = xgboost::cudr::GetAllocGranularity(&this->prop_);
  auto ordinal = CurrentDevice();

  // Assign the access descriptor
  CUmemAccessDesc dacc;
  dacc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  dacc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  dacc.location.id = ordinal;
  this->access_desc_.push_back(dacc);

  if (type == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
    CUmemAccessDesc hacc;
    hacc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    xgboost::cudr::GetCuLocation(CU_MEM_LOCATION_TYPE_HOST_NUMA, &hacc.location);
    this->access_desc_.push_back(hacc);
  }
}
}  // namespace detail

#if defined(XGBOOST_USE_RMM)
LoggingResource *GlobalLoggingResource() {
  static auto mr{std::make_unique<LoggingResource>()};
  return mr.get();
}
#endif  // defined(XGBOOST_USE_RMM)
}  // namespace dh
