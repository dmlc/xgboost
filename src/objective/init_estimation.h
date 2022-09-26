/**
 * \brief Utilities for estimating initial score.
 */

#ifndef XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_
#define XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_

#include "../common/common.h"              // OptionalWeights
#include "../common/numeric.h"             // cpu_impl::Reduce
#include "../common/transform_iterator.h"  // MakeIndexTransformIter
#include "rabit/rabit.h"
#include "xgboost/data.h"    // MetaInfo
#include "xgboost/linalg.h"  // UnravelIndex

namespace xgboost {
namespace obj {
namespace cpu_impl {
inline double WeightedMean(Context const* ctx, MetaInfo const& info) {
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
}  // namespace cpu_impl

namespace cuda_impl {
double WeightedMean(Context const* ctx, MetaInfo const& info);
}  // namespace cuda_impl

/**
 * \brief Weighted mean for distributed env. Not a general implementation since we have
 *        2-dim label with 1-dim weight.
 */
inline double WeightedMean(Context const* ctx, MetaInfo const& info) {
  return ctx->IsCPU() ? cpu_impl::WeightedMean(ctx, info) : cuda_impl::WeightedMean(ctx, info);
}
}  // namespace obj
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_INIT_ESTIMATION_H_
