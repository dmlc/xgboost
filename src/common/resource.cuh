#pragma once
#include <cstddef>  // for size_t

#include "device_vector.cuh"
#include "io.h"

namespace xgboost::common {
class CudaAllocResource : public ResourceHandler {
  dh::DeviceUVector<std::byte> storage_;

  void Clear() noexcept(true) {}

 public:
  explicit CudaAllocResource(std::size_t n_bytes) : ResourceHandler{kMalloc} {
    this->Resize(n_bytes);
  }
  ~CudaAllocResource() noexcept(true) override { this->Clear(); }

  void* Data() override { return thrust::raw_pointer_cast(storage_.data()); }
  [[nodiscard]] std::size_t Size() const override { return storage_.size(); }
  void Resize(std::size_t n_bytes, std::byte init = std::byte{0}) {
    this->storage_.Resize(n_bytes, init);
  }
};
}  // namespace xgboost::common
