/**
 * Copyright 2024-2025, XGBoost Contributors
 */
#pragma once
#include <functional>  // for function
#include <utility>     // for forward

#include "xgboost/base.h"

namespace xgboost::common {
/** @brief RAII guard, simplified version of absl::Cleanup . */
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

template <typename R>
struct NoOp {
  R val;

  explicit NoOp(R&& v) : val{std::forward<R>(v)} {}

  template <typename... Args>
  XGBOOST_DEVICE R operator()(Args&&...) const {
    return val;
  }
};

template <>
struct NoOp<void> {
  template <typename... Args>
  XGBOOST_DEVICE void operator()(Args&&...) const {}
};
}  // namespace xgboost::common
