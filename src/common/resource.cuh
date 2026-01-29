/**
 * Copyright 2024-2025, XGBoost Contributors
 */
#pragma once
#include <cstddef>     // for size_t
#include <functional>  // for function
#include <utility>     // for move

#include "cuda_pinned_allocator.h"  // for SamAllocator, HostPinnedMemPool
#include "cuda_stream.h"            // for StreamRef
#include "device_vector.cuh"        // for DeviceUVector, GrowOnlyVirtualMemVec
#include "io.h"                     // for ResourceHandler, MMAPFile
#include "xgboost/string_view.h"    // for StringView

namespace xgboost::common {
/**
 * @brief Resource backed by `cudaMalloc`.
 */
class CudaMallocResource : public ResourceHandler {
  dh::DeviceUVector<std::byte> storage_;

  void Clear() noexcept(true) { this->Resize(0); }

 public:
  explicit CudaMallocResource(std::size_t n_bytes) : ResourceHandler{kCudaMalloc} {
    this->Resize(n_bytes);
  }
  ~CudaMallocResource() noexcept(true) override { this->Clear(); }

  [[nodiscard]] void* Data() override { return storage_.data(); }
  [[nodiscard]] std::size_t Size() const override { return storage_.size(); }
  void Resize(std::size_t n_bytes) { this->storage_.resize(n_bytes); }
};

/**
 * @brief Device resource that only grows in size.
 */
class CudaGrowOnlyResource : public ResourceHandler {
  static auto MakeNew() {
    return std::make_unique<dh::detail::GrowOnlyVirtualMemVec>(CU_MEM_LOCATION_TYPE_DEVICE);
  }

  std::unique_ptr<dh::detail::GrowOnlyVirtualMemVec> alloc_;
  std::size_t n_bytes_{0};

 public:
  explicit CudaGrowOnlyResource(std::size_t n_bytes)
      : ResourceHandler{kCudaGrowOnly}, alloc_{MakeNew()} {
    this->Resize(n_bytes);
  }
  void Resize(std::size_t n_bytes) {
    this->alloc_->GrowTo(n_bytes);
    this->n_bytes_ = n_bytes;
  }
  void Clear() {
    this->alloc_.reset();
    this->alloc_ = MakeNew();
    this->n_bytes_ = 0;
  }
  [[nodiscard]] void* Data() final { return this->alloc_->data(); }
  [[nodiscard]] std::size_t Size() const final { return this->n_bytes_; }
};

class CudaPinnedResource : public ResourceHandler {
  std::vector<std::byte, cuda_impl::SamAllocator<std::byte>> storage_;

  void Clear() noexcept(true) { this->Resize(0); }

 public:
  explicit CudaPinnedResource(std::size_t n_bytes) : ResourceHandler{kCudaHostCache} {
    this->Resize(n_bytes);
  }
  ~CudaPinnedResource() noexcept(true) override { this->Clear(); }

  [[nodiscard]] void* Data() override { return storage_.data(); }
  [[nodiscard]] std::size_t Size() const override { return storage_.size(); }
  void Resize(std::size_t n_bytes) { this->storage_.resize(n_bytes); }
};

/**
 * @brief Resource for fixed-size memory allocated by @ref HostPinnedMemPool.
 *
 * This container shares the pool but owns the memory.
 */
class HostPinnedMemPoolResource : public ResourceHandler {
  std::shared_ptr<cuda_impl::HostPinnedMemPool> pool_;
  std::size_t n_bytes_;
  curt::StreamRef stream_;
  void* ptr_;

 public:
  explicit HostPinnedMemPoolResource(std::shared_ptr<cuda_impl::HostPinnedMemPool> pool,
                                     std::size_t n_bytes, curt::StreamRef stream)
      : ResourceHandler{kCudaPinnedMemPool},
        pool_{std::move(pool)},
        n_bytes_{n_bytes},
        stream_{stream},
        ptr_{this->pool_->AllocateAsync(n_bytes, stream)} {}
  ~HostPinnedMemPoolResource() override { this->pool_->DeallocateAsync(this->ptr_, this->stream_); }
  [[nodiscard]] std::size_t Size() const override { return this->n_bytes_; }
  [[nodiscard]] void* Data() override { return this->ptr_; }
};

class CudaMmapResource : public ResourceHandler {
  std::unique_ptr<MMAPFile, std::function<void(MMAPFile*)>> handle_;
  std::size_t n_;

 public:
  CudaMmapResource() : ResourceHandler{kCudaMmap} {}
  CudaMmapResource(StringView path, std::size_t offset, std::size_t length);
  ~CudaMmapResource() noexcept(false) override;

  [[nodiscard]] void* Data() override;
  [[nodiscard]] std::size_t Size() const override;
};

class PrivateCudaMmapConstStream : public AlignedResourceReadStream {
 public:
  explicit PrivateCudaMmapConstStream(StringView path, std::size_t offset, std::size_t length)
      : AlignedResourceReadStream{
            std::shared_ptr<CudaMmapResource>{new CudaMmapResource{path, offset, length}}} {}
  ~PrivateCudaMmapConstStream() noexcept(false) override;
};
}  // namespace xgboost::common
