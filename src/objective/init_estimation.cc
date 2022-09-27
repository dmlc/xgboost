/**
 * Copyright 2022 by XGBoost Contributors
 *
 * \brief Utilities for estimating initial score.
 */

#include "init_estimation.h"

#include "../common/common.h"              // OptionalWeights
#include "../common/numeric.h"             // cpu_impl::Reduce
#include "../common/transform_iterator.h"  // MakeIndexTransformIter
#include "rabit/rabit.h"
#include "xgboost/linalg.h"  // UnravelIndex

namespace xgboost {
namespace obj {
namespace cpu_impl {
double WeightedMean(Context const* ctx, MetaInfo const& info) {
  std::uint64_t n_samples = info.num_row_;
  rabit::Allreduce<rabit::op::Sum>(&n_samples, 1);
  auto y = info.labels.HostView();
  auto w = common::OptionalWeights{info.weights_.ConstHostSpan()};
  auto it = common::MakeIndexTransformIter([&](size_t i) -> double {
    size_t r, c;
    std::tie(r, c) = linalg::UnravelIndex(i, y.Shape());
    return y(r, c) * w[r] / static_cast<double>(n_samples);
  });
  auto res = common::cpu_impl::Reduce(ctx, it, it + y.Size(), 0.0);
  rabit::Allreduce<rabit::op::Sum>(&res, 1);
  return res;
}

double FitStump(Context const* ctx, HostDeviceVector<GradientPair> const& gpair) {
  auto const& h_gpair = gpair.ConstHostVector();
  auto it = common::MakeIndexTransformIter([&](auto i) {
    auto const& g = h_gpair[i];
    return GradientPairPrecise{g};
  });
  auto sum = common::cpu_impl::Reduce(ctx, it, it + gpair.Size(), GradientPairPrecise{});
  return sum.GetGrad() / std::max(sum.GetHess(), 1e-6);
}
}  // namespace cpu_impl

double FitStump(Context const* ctx, HostDeviceVector<GradientPair> const& gpair) {
  return ctx->IsCPU() ? cpu_impl::FitStump(ctx, gpair) : cuda_impl::FitStump(ctx, gpair);
}
}  // namespace obj
}  // namespace xgboost
