/*!
 * Copyright by Contributors 2017-2026
 */

#include "../../../src/common/stats.h"

#include <sycl/sycl.hpp>

#include "../device_manager.h"

namespace xgboost::common::sycl_impl {
void Mean(Context const* ctx, linalg::VectorView<float const> v, linalg::VectorView<float> out) {
  sycl::DeviceManager device_manager;
  auto* qu = device_manager.GetQueue(ctx->Device());

  qu->submit([&](::sycl::handler& cgh) {
      auto reduction = ::sycl::reduction(&(out(0)), 0.0f, ::sycl::plus<float>(),
                                         ::sycl::property::reduction::initialize_to_identity());
      cgh.parallel_for<>(::sycl::range<1>(v.Size()), reduction, [=](::sycl::id<1> pid, auto& sum) {
        size_t i = pid[0];
        sum += v(i);
      });
    }).wait_and_throw();
}
}  // namespace xgboost::common::sycl_impl
