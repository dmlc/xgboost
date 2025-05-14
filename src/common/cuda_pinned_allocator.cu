/**
 * Copyright 2025, XGBoost Contributors
 */
#include "cuda_pinned_allocator.h"

#if defined(XGBOOST_USE_CUDA)

#include <cuda_runtime_api.h>  // for cudaMemPoolCreate, cudaMemPoolDestroy

#include <array>    // for array
#include <cstring>  // for memset
#include <memory>   // for unique_ptr

#endif  // defined(XGBOOST_USE_CUDA)

#include "common.h"
#include "cuda_dr_utils.h"  // for CUDA_HW_DECOM_AVAILABLE
#include "cuda_rt_utils.h"  // for CurrentDevice

namespace xgboost::common::cuda_impl {
[[nodiscard]] MemPoolHdl CreateHostMemPool() {
  auto mem_pool = std::unique_ptr<cudaMemPool_t, void (*)(cudaMemPool_t*)>{
      [] {
        cudaMemPoolProps h_props;
        std::memset(&h_props, '\0', sizeof(h_props));
        auto numa_id = curt::GetNumaId();
        h_props.location.id = numa_id;
        h_props.location.type = cudaMemLocationTypeHostNuma;
        h_props.allocType = cudaMemAllocationTypePinned;
#if defined(CUDA_HW_DECOM_AVAILABLE)
        h_props.usage = cudaMemPoolCreateUsageHwDecompress;
#endif  // defined(CUDA_HW_DECOM_AVAILABLE)
        h_props.handleTypes = cudaMemHandleTypeNone;

        cudaMemPoolProps d_props;
        std::memset(&d_props, '\0', sizeof(d_props));
        auto device_idx = curt::CurrentDevice();
        d_props.location.id = device_idx;
        d_props.location.type = cudaMemLocationTypeDevice;
        d_props.allocType = cudaMemAllocationTypePinned;
#if defined(CUDA_HW_DECOM_AVAILABLE)
        d_props.usage = cudaMemPoolCreateUsageHwDecompress;
#endif  // defined(CUDA_HW_DECOM_AVAILABLE)
        d_props.handleTypes = cudaMemHandleTypeNone;

        std::array<cudaMemPoolProps, 2> vprops{h_props, d_props};

        cudaMemPool_t* mem_pool = new cudaMemPool_t;
        dh::safe_cuda(cudaMemPoolCreate(mem_pool, vprops.data()));

        cudaMemAccessDesc h_desc;
        h_desc.location = h_props.location;
        h_desc.flags = cudaMemAccessFlagsProtReadWrite;

        cudaMemAccessDesc d_desc;
        d_desc.location = d_props.location;
        d_desc.flags = cudaMemAccessFlagsProtReadWrite;

        std::array<cudaMemAccessDesc, 2> descs{h_desc, d_desc};
        dh::safe_cuda(cudaMemPoolSetAccess(*mem_pool, descs.data(), descs.size()));
        return mem_pool;
      }(),
      [](cudaMemPool_t* mem_pool) {
        if (mem_pool) {
          dh::safe_cuda(cudaMemPoolDestroy(*mem_pool));
        }
      }};
  return mem_pool;
}
}  // namespace xgboost::common::cuda_impl
