/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include "numeric.h"

#include <numeric>      // std::accumulate
#include <type_traits>  // std::is_same

#include "threading_utils.h"             // MemStackAllocator, ParallelFor, DefaultMaxThreads
#include "xgboost/generic_parameters.h"  // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector

namespace xgboost {
namespace common {
double Reduce(Context const* ctx, HostDeviceVector<float> const& values) {
  if (ctx->IsCPU()) {
    auto const& h_values = values.ConstHostVector();
    MemStackAllocator<double, DefaultMaxThreads()> result_tloc(ctx->Threads(), 0);
    ParallelFor(h_values.size(), ctx->Threads(),
                [&](auto i) { result_tloc[omp_get_thread_num()] += h_values[i]; });
    auto result = std::accumulate(result_tloc.cbegin(), result_tloc.cend(), 0.0);
    static_assert(std::is_same<decltype(result), double>::value, "");
    return result;
  }
  return cuda::Reduce(ctx, values);
}
}  // namespace common
}  // namespace xgboost
