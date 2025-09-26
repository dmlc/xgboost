/**
 * Copyright 2025, XGBoost contributors
 */
#pragma once
#include <atomic>   // for atomic
#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "cuda_stream.h"       // for StreamRef, Stream

namespace xgboost::curt {
// rmm cuda_stream_pool
class StreamPool {
  mutable std::atomic<std::size_t> next_{0};
  std::vector<curt::Stream> stream_;

 public:
  explicit StreamPool(std::size_t n) : stream_(n) {}
  ~StreamPool() = default;
  StreamPool(StreamPool const& that) = delete;
  StreamPool& operator=(StreamPool const& that) = delete;

  [[nodiscard]] curt::StreamRef operator[](std::size_t i) const { return stream_[i].View(); }
  [[nodiscard]] curt::StreamRef Next() const { return stream_[(next_++) % stream_.size()].View(); }
  [[nodiscard]] std::size_t Size() const { return stream_.size(); }
};
}  // namespace xgboost::curt
