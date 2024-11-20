/**
 * Copyright 2021-2024, XGBoost Contributors
 * \file transform.h
 */
#ifndef PLUGIN_SYCL_COMMON_TRANSFORM_H_
#define PLUGIN_SYCL_COMMON_TRANSFORM_H_

#include "../device_manager.h"

#include <sycl/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {

template <typename Functor, typename... SpanType>
void LaunchSyclKernel(DeviceOrd device, Functor&& _func, xgboost::common::Range _range,
                      SpanType... _spans) {
  sycl::DeviceManager device_manager;
  auto* qu = device_manager.GetQueue(device);

  size_t size = *(_range.end());
  qu->submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(size),
                       [=](::sycl::id<1> pid) {
      const size_t idx = pid[0];
      const_cast<Functor&&>(_func)(idx, _spans...);
    });
  }).wait();
}

}  // namespace common
}  // namespace sycl
}  // namespace xgboost
#endif  // PLUGIN_SYCL_COMMON_TRANSFORM_H_
