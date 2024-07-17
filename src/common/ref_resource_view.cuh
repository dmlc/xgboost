/**
 * Copyright 2024, XGBoost Contributors
 */
#pragma once

#include <cstddef>  // for size_t
#include <memory>   // for make_shared

#include "cuda_context.cuh"     // for CUDAContext
#include "ref_resource_view.h"  // for RefResourceView
#include "resource.cuh"         // for CudaAllocResource
#include "xgboost/context.h"    // for Context

namespace xgboost::common {
/**
 * @brief Make a fixed size `RefResourceView` with cudaMalloc resource.
 */
template <typename T>
[[nodiscard]] RefResourceView<T> MakeFixedVecWithCudaMalloc(Context const* ctx,
                                                            std::size_t n_elements, T const& init) {
  auto resource = std::make_shared<common::CudaMallocResource>(n_elements * sizeof(T));
  auto ref = RefResourceView{resource->DataAs<T>(), n_elements, resource};
  thrust::fill_n(ctx->CUDACtx()->CTP(), ref.data(), ref.size(), init);
  return ref;
}
}  // namespace xgboost::common
