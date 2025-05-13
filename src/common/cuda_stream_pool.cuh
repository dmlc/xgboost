/**
 * Copyright 2025, XGBoost contributors
 */
#pragma once
#include <atomic>   // for atomic
#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "device_helpers.cuh"  // for CUDAStreamView, CUDAStream

namespace xgboost::curt {
// rmm cuda_stream_pool
class StreamPool {
  mutable std::atomic<std::size_t> next_{0};
  std::vector<dh::CUDAStream> stream_;

 public:
  explicit StreamPool(std::size_t n) : stream_(n) {}
  ~StreamPool() = default;
  StreamPool(StreamPool const& that) = delete;
  StreamPool& operator=(StreamPool const& that) = delete;

  [[nodiscard]] dh::CUDAStreamView operator[](std::size_t i) const { return stream_[i].View(); }
  [[nodiscard]] dh::CUDAStreamView Next() const {
    return stream_[(next_++) % stream_.size()].View();
  }
  [[nodiscard]] std::size_t Size() const { return stream_.size(); }
};
}  // namespace xgboost::curt
