/**
 * Copyright 2024, XGBoost Contributors
 *
 * @brief RAII guard, simplified version of absl::Cleanup
 */
#pragma once
#include <functional>  // for function
#include <utility>     // for forward

namespace xgboost::common {
class Cleanup {
  std::function<void()> cb_;

 public:
  template <typename Callback>
  explicit Cleanup(Callback&& cb) : cb_{std::forward<Callback>(cb)} {}

  ~Cleanup() { this->cb_(); }
};

template <typename Callback>
auto MakeCleanup(Callback&& cb) {
  return Cleanup{std::forward<Callback>(cb)};
}
}  // namespace xgboost::common
