/*!
 * Copyright by Contributors 2017-2025
 */
#include <sycl/sycl.hpp>

#include "../../../src/common/optional_weight.h"

#include "../device_manager.h"

namespace xgboost::common::sycl_impl {
double SumOptionalWeights(Context const* ctx, OptionalWeights const& weights) {
  sycl::DeviceManager device_manager;
  auto* qu = device_manager.GetQueue(ctx->Device());

  const auto* data = weights.Data();
  double result = 0;
  {
    ::sycl::buffer<double> buff(&result, 1);
    qu->submit([&](::sycl::handler& cgh) {
      auto reduction = ::sycl::reduction(buff, cgh, ::sycl::plus<>());
      cgh.parallel_for<>(::sycl::range<1>(weights.Size()), reduction,
                        [=](::sycl::id<1> pid, auto& sum) {
        size_t i = pid[0];
        sum += data[i];
      });
    }).wait_and_throw();
  }

  return result;
}
}  // namespace xgboost::common::sycl_impl
