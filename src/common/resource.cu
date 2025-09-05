/**
 * Copyright 2024-2025, XGBoost Contributors
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
  auto ptr = handle_->BasePtr();
#if (CUDA_VERSION / 1000) >= 13
  cudaMemLocation loc;
  loc.type = cudaMemLocationTypeDevice;
  loc.id = device;
#else
  auto loc = device;
#endif  // (CUDA_VERSION / 1000) >= 13
  dh::safe_cuda(cudaMemAdvise(ptr.data(), ptr.size(), cudaMemAdviseSetReadMostly, loc));
  dh::safe_cuda(cudaMemAdvise(ptr.data(), ptr.size(), cudaMemAdviseSetPreferredLocation, loc));
  dh::safe_cuda(cudaMemAdvise(ptr.data(), ptr.size(), cudaMemAdviseSetAccessedBy, loc));
#if (CUDA_VERSION / 1000) >= 13
  dh::safe_cuda(cudaMemPrefetchAsync(ptr.data(), ptr.size(), loc, 0, dh::DefaultStream()));
#else
  dh::safe_cuda(cudaMemPrefetchAsync(ptr.data(), ptr.size(), device, dh::DefaultStream()));
#endif  // (CUDA_VERSION / 1000) >= 13
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
