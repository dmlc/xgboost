/**
 * Copyright 2022-2024, XGBoost Contributors
 */
#include "numeric.h"

#include <type_traits>  // std::is_same_v

#include "xgboost/context.h"             // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector

namespace xgboost {
namespace common {
double Reduce(Context const* ctx, HostDeviceVector<float> const& values) {
  if (ctx->IsCUDA()) {
    return cuda_impl::Reduce(ctx, values);
  } else {
    auto const& h_values = values.ConstHostVector();
    auto result = cpu_impl::Reduce(ctx, h_values.cbegin(), h_values.cend(), 0.0);
    static_assert(std::is_same_v<decltype(result), double>);
    return result;
  }
}
}  // namespace common
}  // namespace xgboost
