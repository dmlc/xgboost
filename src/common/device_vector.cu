/**
 * Copyright 2017-2024, XGBoost contributors
 */
#include <numeric>  // for accumulate

#include "../collective/communicator-inl.h"  // for GetRank
#include "common.h"                          // for HumanMemUnit
#include "cuda_dr_utils.h"
#include "device_helpers.cuh"  // for CurrentDevice
#include "device_vector.cuh"
#include "transform_iterator.h"  // for MakeIndexTransformIter

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

[[nodiscard]] std::size_t GrowOnlyVirtualMemVec::PhyCapacity() const {
  auto it = xgboost::common::MakeIndexTransformIter(
      [&](std::size_t i) { return this->handles_[i]->size; });
  return std::accumulate(it, it + this->handles_.size(), static_cast<std::size_t>(0));
}

void GrowOnlyVirtualMemVec::Reserve(std::size_t new_size) {
  auto va_capacity = this->Capacity();
  if (new_size < va_capacity) {
    return;
  }

  // Try to reserve new virtual address.
  auto const aligned_size = RoundUp(new_size, this->granularity_);
  auto const new_reserve_size = aligned_size - va_capacity;
  CUresult status = CUDA_SUCCESS;
  auto hint = this->DevPtr() + va_capacity;

  bool failed{false};
  auto range = std::make_unique<VaRange>(new_reserve_size, hint, &status, &failed);
  if (failed) {
    // Failed to reserve the requested address.
    // Slow path, try to reserve a new address with full size.
    range = std::make_unique<VaRange>(aligned_size, 0ULL, &status, &failed);
    safe_cu(status);
    CHECK(!failed);

    // New allocation is successful. Map the pyhsical address to the virtual address.
    // First unmap the existing ptr.
    if (this->DevPtr() != 0) {
      // Unmap the existing ptr.
      safe_cu(cu_.cuMemUnmap(this->DevPtr(), this->PhyCapacity()));

      // Then remap all the existing physical addresses to the new ptr.
      CUdeviceptr ptr = range->DevPtr();
      for (auto const &hdl : this->handles_) {
        this->MapBlock(ptr, hdl);
        ptr += hdl->size;
      }

      // Release the existing ptr.
      va_ranges_.clear();
    }
  }

  va_ranges_.emplace_back(std::move(range));
}

GrowOnlyVirtualMemVec::GrowOnlyVirtualMemVec(CUmemLocationType type)
    : prop_{xgboost::cudr::MakeAllocProp(type)},
      granularity_{xgboost::cudr::GetAllocGranularity(&this->prop_)} {
  CHECK(type == CU_MEM_LOCATION_TYPE_DEVICE || type == CU_MEM_LOCATION_TYPE_HOST_NUMA);
  // Assign the access descriptor
  CUmemAccessDesc dacc;
  dacc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  xgboost::cudr::MakeCuMemLocation(CU_MEM_LOCATION_TYPE_DEVICE, &dacc.location);
  this->access_desc_.push_back(dacc);

  if (type == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
    CUmemAccessDesc hacc;
    hacc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    xgboost::cudr::MakeCuMemLocation(type, &hacc.location);
    this->access_desc_.push_back(hacc);
  }
}

[[nodiscard]] std::size_t GrowOnlyVirtualMemVec::Capacity() const {
  auto it = xgboost::common::MakeIndexTransformIter(
      [&](std::size_t i) { return this->va_ranges_[i]->Size(); });
  return std::accumulate(it, it + this->va_ranges_.size(), static_cast<std::size_t>(0));
}
}  // namespace detail

#if defined(XGBOOST_USE_RMM)
LoggingResource *GlobalLoggingResource() {
  static auto mr{std::make_unique<LoggingResource>()};
  return mr.get();
}
#endif  // defined(XGBOOST_USE_RMM)
}  // namespace dh
