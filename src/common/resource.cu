/**
 * Copyright 2024, XGBoost Contributors
 */
#include "device_helpers.cuh"  // for CurrentDevice
#include "resource.cuh"
#include "xgboost/string_view.h"  // for StringView

namespace xgboost::common {
CudaMmapResource::CudaMmapResource(StringView path, std::size_t offset, std::size_t length)
    : ResourceHandler{kCudaMmap},
      handle_{detail::OpenMmap(std::string{path}, offset, length),
              [](MMAPFile* handle) {
                // Don't close the mmap while CUDA kernel is running.
                if (handle) {
                  dh::DefaultStream().Sync();
                }
                detail::CloseMmap(handle);
              }},
      n_{length} {
  auto device = dh::CurrentDevice();
  dh::safe_cuda(
      cudaMemAdvise(handle_->base_ptr, handle_->base_size, cudaMemAdviseSetReadMostly, device));
  dh::safe_cuda(cudaMemAdvise(handle_->base_ptr, handle_->base_size,
                              cudaMemAdviseSetPreferredLocation, device));
  dh::safe_cuda(
      cudaMemAdvise(handle_->base_ptr, handle_->base_size, cudaMemAdviseSetAccessedBy, device));
  dh::safe_cuda(
      cudaMemPrefetchAsync(handle_->base_ptr, handle_->base_size, device, dh::DefaultStream()));
}

[[nodiscard]] void* CudaMmapResource::Data() {
  if (!handle_) {
    return nullptr;
  }
  return this->handle_->Data();
}

[[nodiscard]] std::size_t CudaMmapResource::Size() const { return n_; }

CudaMmapResource::~CudaMmapResource() noexcept(false) = default;

PrivateCudaMmapConstStream::~PrivateCudaMmapConstStream() noexcept(false) = default;
}  // namespace xgboost::common
