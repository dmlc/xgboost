/**
 * Copyright 2022 by XGBoost Contributors
 *
 * \brief Utilities for estimating initial score.
 */

#if !defined(NOMINMAX) && defined(_WIN32)
#define NOMINMAX
#endif  // !defined(NOMINMAX)
#include "init_estimation.h"

#include <algorithm>  // std::max

#include "../collective/communicator-inl.h"
#include "../common/numeric.h"             // cpu_impl::Reduce
#include "../common/transform_iterator.h"  // MakeIndexTransformIter

namespace xgboost {
namespace obj {
namespace cpu_impl {
double FitStump(Context const* ctx, HostDeviceVector<GradientPair> const& gpair) {
  auto const& h_gpair = gpair.ConstHostVector();
  auto it = common::MakeIndexTransformIter([&](auto i) {
    auto const& g = h_gpair[i];
    return GradientPairPrecise{g};
  });
  auto sum = common::cpu_impl::Reduce(ctx, it, it + gpair.Size(), GradientPairPrecise{});
  static_assert(sizeof(sum) == sizeof(double) * 2, "");
  collective::Allreduce<collective::Operation::kSum>(reinterpret_cast<double*>(&sum), 2);
  return -sum.GetGrad() / std::max(sum.GetHess(), 1e-6);
}
}  // namespace cpu_impl

double FitStump(Context const* ctx, HostDeviceVector<GradientPair> const& gpair) {
  return ctx->IsCPU() ? cpu_impl::FitStump(ctx, gpair) : cuda_impl::FitStump(ctx, gpair);
}
}  // namespace obj
}  // namespace xgboost
