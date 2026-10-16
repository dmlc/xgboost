/**
 * Copyright 2022-2026, XGBoost Contributors
 */
#include "numeric.h"

#include <cstddef>  // for size_t

#include "xgboost/context.h"             // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector

namespace xgboost {
namespace common {
double Reduce(Context const* ctx, HostDeviceVector<float> const& values) {
  if (ctx->IsCUDA()) {
    return cuda_impl::Reduce(ctx, values);
  } else {
    auto const& h_values = values.ConstHostVector();
    return TransformReduce(h_values.size(), ctx->Threads(), 0.0,
                           [&](std::size_t i) { return h_values[i]; });
  }
}
}  // namespace common
}  // namespace xgboost
