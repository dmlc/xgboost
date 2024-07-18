/**
 * Copyright 2024, XGBoost Contributors
 */
#pragma once
#include <cstddef>     // for size_t
#include <functional>  // for function

#include "device_vector.cuh"      // for DeviceUVector
#include "io.h"                   // for ResourceHandler, MMAPFile
#include "xgboost/string_view.h"  // for StringView

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
