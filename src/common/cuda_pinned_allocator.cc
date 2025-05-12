/**
 * Copyright 2025, XGBoost Contributors
 */
#include "cuda_pinned_allocator.h"

#include "common.h"
#include "cuda_rt_utils.h"

#if CUDART_VERSION >= 12080
#define CUDA_HW_DECOM_AVAILABLE 1
#endif

namespace xgboost::common::cuda_impl {
[[nodiscard]] MemPoolHdl CreateHostMemPool() {
  auto mem_pool = std::unique_ptr<cudaMemPool_t, void (*)(cudaMemPool_t*)>{
      [] {
        cudaMemPoolProps props;
        std::memset(&props, '\0', sizeof(props));
        props.location.type = cudaMemLocationTypeHostNuma;
        props.location.id = curt::GetNumaId();
        props.allocType = cudaMemAllocationTypePinned;
#if defined(CUDA_HW_DECOM_AVAILABLE)
        props.usage = cudaMemPoolCreateUsageHwDecompress;
#endif  // defined(CUDA_HW_DECOM_AVAILABLE)
        props.handleTypes = cudaMemHandleTypeNone;

        cudaMemPoolProps dprops;
        std::memset(&dprops, '\0', sizeof(dprops));
        dprops.location.type = cudaMemLocationTypeDevice;
        dprops.location.id = curt::CurrentDevice();
        dprops.allocType = cudaMemAllocationTypePinned;
#if defined(CUDA_HW_DECOM_AVAILABLE)
        dprops.usage = cudaMemPoolCreateUsageHwDecompress;
#endif  // defined(CUDA_HW_DECOM_AVAILABLE)
        dprops.handleTypes = cudaMemHandleTypeNone;

        std::vector<cudaMemPoolProps> vprops{props, dprops};

        cudaMemPool_t* mem_pool = new cudaMemPool_t;
        dh::safe_cuda(cudaMemPoolCreate(mem_pool, vprops.data()));

        cudaMemAccessDesc h_desc;
        h_desc.location.type = cudaMemLocationTypeHostNuma;
        h_desc.location.id = 0;
        h_desc.flags = cudaMemAccessFlagsProtReadWrite;

        cudaMemAccessDesc d_desc;
        d_desc.location.type = cudaMemLocationTypeDevice;
        d_desc.location.id = 0;
        d_desc.flags = cudaMemAccessFlagsProtReadWrite;

        std::vector<cudaMemAccessDesc> descs{h_desc, d_desc};
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
