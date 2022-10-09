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
#include <cinttypes>  // std::uint64_t

#include "../collective/communicator-inl.h"
#include "../common/linalg_op.h"           // cbegin, cend
#include "../common/math.h"                // CloseTo
#include "../common/numeric.h"             // cpu_impl::Reduce
#include "../common/transform_iterator.h"  // MakeIndexTransformIter
#include "xgboost/linalg.h"                // TensorView
#include "xgboost/objective.h"             // ObjFunction

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
  return -sum.GetGrad() / std::max(sum.GetHess(), 1e-6);
}

double WeightedMean(Context const* ctx, MetaInfo const& info) {
  std::uint64_t n_samples = info.num_row_;
  collective::Allreduce<collective::Operation::kSum>(&n_samples, 1);
  auto y = info.labels.HostView();
  auto w = common::OptionalWeights{info.weights_.ConstHostSpan()};
  auto it = common::MakeIndexTransformIter([&](size_t i) -> double {
    size_t r, c;
    std::tie(r, c) = linalg::UnravelIndex(i, y.Shape());
    return y(r, c) * w[r] / static_cast<double>(n_samples);
  });
  auto res = common::cpu_impl::Reduce(ctx, it, it + y.Size(), 0.0);
  collective::Allreduce<collective::Operation::kSum>(&res, 1);
  return res;
}
}  // namespace cpu_impl

double FitStump(Context const* ctx, HostDeviceVector<GradientPair> const& gpair) {
  return ctx->IsCPU() ? cpu_impl::FitStump(ctx, gpair) : cuda_impl::FitStump(ctx, gpair);
}

void NormalizeBaseScore(double w, linalg::TensorView<float, 1> in_out) {
  // Weighted average base score across all workers
  collective::Allreduce<collective::Operation::kSum>(in_out.Values().data(),
                                                     in_out.Values().size());
  collective::Allreduce<collective::Operation::kSum>(&w, 1);

  if (common::CloseTo(w, 0.0)) {
    // Mostly for handling empty dataset test.
    LOG(WARNING) << "Sum of weights is close to 0.0, skipping base score estimation.";
    in_out(0) = ObjFunction::DefaultBaseScore();
    return;
  }
  std::transform(linalg::cbegin(in_out), linalg::cend(in_out), linalg::begin(in_out),
                 [w](float v) { return v / w; });
}
}  // namespace obj
}  // namespace xgboost
