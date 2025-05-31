/**
 * Copyright 2024-2025, XGBoost Contributors
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
[[nodiscard]] RefResourceView<T> MakeFixedVecWithCudaMalloc(std::size_t n_elements) {
  auto resource = std::make_shared<common::CudaMallocResource>(n_elements * sizeof(T));
  auto ref = RefResourceView{resource->DataAs<T>(), n_elements, resource};
  return ref;
}

template <typename T>
[[nodiscard]] RefResourceView<T> MakeCudaGrowOnly(std::size_t n_elements) {
  auto resource = std::make_shared<common::CudaGrowOnlyResource>(n_elements * sizeof(T));
  auto ref = RefResourceView{resource->DataAs<T>(), n_elements, resource};
  return ref;
}

/**
 * @brief Initialize the data in addition to allocation.
 */
template <typename T>
[[nodiscard]] RefResourceView<T> MakeFixedVecWithCudaMalloc(Context const* ctx,
                                                            std::size_t n_elements, T const& init) {
  auto ref = MakeFixedVecWithCudaMalloc<T>(n_elements);
  thrust::fill_n(ctx->CUDACtx()->CTP(), ref.data(), ref.size(), init);
  return ref;
}

template <typename T>
[[nodiscard]] RefResourceView<T> MakeFixedVecWithPinnedMalloc(std::size_t n_elements) {
  auto resource = std::make_shared<common::CudaPinnedResource>(n_elements * sizeof(T));
  auto ref = RefResourceView{resource->DataAs<T>(), n_elements, resource};
  return ref;
}

/**
 * @brief Create a fixed size resource view from a shared pinned memory pool.
 */
template <typename T>
[[nodiscard]] RefResourceView<T> MakeFixedVecWithPinnedMemPool(
    std::shared_ptr<cuda_impl::HostPinnedMemPool> pool, std::size_t n_elements,
    dh::CUDAStreamView stream) {
  auto resource = std::make_shared<common::HostPinnedMemPoolResource>(
      std::move(pool), n_elements * sizeof(T), stream);
  auto ref = RefResourceView{resource->DataAs<T>(), n_elements, resource};
  return ref;
}
}  // namespace xgboost::common
