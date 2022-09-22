/**
 * \brief Utilities for estimating initial score.
 */

#ifndef XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_
#define XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_

#include "../common/common.h"   // OptionalWeights, MakeIndexTransformIter
#include "../common/numeric.h"  // cpu_impl::Reduce
#include "rabit/rabit.h"
#include "xgboost/data.h"  // MetaInfo

namespace xgboost {
namespace obj {
namespace cpu_impl {
inline double WeightedMean(Context const* ctx, MetaInfo const& info) {
  std::uint64_t n_samples = info.num_row_;
  auto y = info.labels.HostView();
  auto w = common::OptionalWeights{info.weights_.ConstHostSpan()};
  auto it = common::MakeIndexTransformIter([=] XGBOOST_DEVICE(size_t i) -> double {
    size_t r, c;
    std::tie(r, c) = linalg::UnravelIndex(i, y.Shape());
    return y(r, c) * w[r] / static_cast<double>(n_samples);
  });
  auto res = common::cpu_impl::Reduce(ctx, it, it + y.Size(), 0.0);
  rabit::Allreduce<rabit::op::Sum>(&res, 1);
  return res;
}
}  // namespace cpu_impl

namespace cuda_impl {
double WeightedMean(Context const* ctx, MetaInfo const& info);
}  // namespace cuda_impl

double WeightedMean(Context const* ctx, MetaInfo const& info) {
  return ctx->IsCPU() ? cpu_impl::WeightedMean(ctx, info) : cuda_impl::WeightedMean(ctx, info);
}
}  // namespace obj
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_
