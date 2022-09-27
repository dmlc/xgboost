/**
 * Copyright 2022 by XGBoost Contributors
 *
 * \brief Utilities for estimating initial score.
 */
#include <thrust/iterator/counting_iterator.h>  // thrust::make_counting_iterator

#include <cinttypes>  // std::uint64_t
#include <cstddef>    // std::size_t

#include "../common/device_helpers.cuh"  // dh::MakeTransformIterator
#include "../common/numeric.cuh"         // Reduce
#include "init_estimation.h"
#include "rabit/rabit.h"
#include "xgboost/data.h"                // MetaInfo
#include "xgboost/generic_parameters.h"  // Context
#include "xgboost/linalg.h"              // UnravelIndex

namespace xgboost {
namespace obj {
namespace cuda_impl {
double WeightedMean(Context const* ctx, MetaInfo const& info) {
  std::uint64_t n_samples = info.num_row_;
  rabit::Allreduce<rabit::op::Sum>(&n_samples, 1);
  auto y = info.labels.View(ctx->gpu_id);
  auto w = common::OptionalWeights{info.weights_.ConstHostSpan()};
  auto it = dh::MakeTransformIterator<double>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) -> double {
        auto idx = linalg::UnravelIndex(i, y.Shape());
        std::size_t r{std::get<0>(idx)}, c{std::get<1>(idx)};
        return y(r, c) * w[r] / static_cast<double>(n_samples);
      });
  return common::cuda_impl::Reduce(ctx, it, it + y.Size(), 0.0);
}

double FitStump(Context const* ctx, HostDeviceVector<GradientPair> const& gpair) {
  gpair.SetDevice(ctx->gpu_id);
  auto const& d_gpair = gpair.ConstDeviceSpan();
  auto it = dh::MakeTransformIterator<GradientPairPrecise>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(std::size_t i) -> GradientPairPrecise {
        return GradientPairPrecise{d_gpair[i]};
      });
  auto sum = common::cuda_impl::Reduce(ctx, it, it + d_gpair.size(), GradientPairPrecise{});
  return sum.GetGrad() / std::max(sum.GetHess(), 1e-6);
}
}  // namespace cuda_impl
}  // namespace obj
}  // namespace xgboost
