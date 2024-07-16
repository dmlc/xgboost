/**
 * Copyright 2024, XGBoost Contributors
 */
#pragma once
#include <cstddef>  // for size_t

#include "device_vector.cuh"
#include "io.h"

namespace xgboost::common {
class CudaAllocResource : public ResourceHandler {
  dh::DeviceUVector<std::byte> storage_;

  void Clear() noexcept(true) { this->Resize(0); }

 public:
  explicit CudaAllocResource(std::size_t n_bytes) : ResourceHandler{kCudaMalloc} {
    this->Resize(n_bytes);
  }
  ~CudaAllocResource() noexcept(true) override { this->Clear(); }

  void* Data() override { return thrust::raw_pointer_cast(storage_.data()); }
  [[nodiscard]] std::size_t Size() const override { return storage_.size(); }
  void Resize(std::size_t n_bytes, std::byte init = std::byte{0}) {
    this->storage_.resize(n_bytes, init);
  }
};
}  // namespace xgboost::common
