/**
 * Copyright 2022 by XGBoost Contributors
 *
 * \brief Utilities for estimating initial score.
 */
#include "fit_stump.h"

#include <cinttypes>  // std::int32_t
#include <cstddef>    // std::size_t

#include "../collective/aggregator.h"
#include "../collective/communicator-inl.h"
#include "../common/common.h"              // AssertGPUSupport
#include "../common/numeric.h"             // cpu_impl::Reduce
#include "../common/threading_utils.h"     // ParallelFor
#include "../common/transform_iterator.h"  // MakeIndexTransformIter
#include "xgboost/base.h"                  // bst_target_t, GradientPairPrecise
#include "xgboost/context.h"               // Context
#include "xgboost/linalg.h"                // TensorView, Tensor, Constant
#include "xgboost/logging.h"               // CHECK_EQ

namespace xgboost {
namespace tree {
namespace cpu_impl {
void FitStump(Context const* ctx, MetaInfo const& info,
              linalg::TensorView<GradientPair const, 2> gpair,
              linalg::VectorView<float> out) {
  auto n_targets = out.Size();
  CHECK_EQ(n_targets, gpair.Shape(1));
  linalg::Tensor<GradientPairPrecise, 2> sum_tloc =
      linalg::Constant(ctx, GradientPairPrecise{}, ctx->Threads(), n_targets);
  auto h_sum_tloc = sum_tloc.HostView();
  // first dim for gpair is samples, second dim is target.
  // Reduce by column, parallel by samples
  common::ParallelFor(gpair.Shape(0), ctx->Threads(), [&](auto i) {
    for (bst_target_t t = 0; t < n_targets; ++t) {
      h_sum_tloc(omp_get_thread_num(), t) += GradientPairPrecise{gpair(i, t)};
    }
  });
  // Aggregate to the first row.
  auto h_sum = h_sum_tloc.Slice(0, linalg::All());
  for (std::int32_t i = 1; i < ctx->Threads(); ++i) {
    for (bst_target_t j = 0; j < n_targets; ++j) {
      h_sum(j) += h_sum_tloc(i, j);
    }
  }
  CHECK(h_sum.CContiguous());

  collective::GlobalSum(info, reinterpret_cast<double*>(h_sum.Values().data()), h_sum.Size() * 2);

  for (std::size_t i = 0; i < h_sum.Size(); ++i) {
    out(i) = static_cast<float>(CalcUnregularizedWeight(h_sum(i).GetGrad(), h_sum(i).GetHess()));
  }
}
}  // namespace cpu_impl

namespace cuda_impl {
void FitStump(Context const* ctx, linalg::TensorView<GradientPair const, 2> gpair,
              linalg::VectorView<float> out);

#if !defined(XGBOOST_USE_CUDA)
inline void FitStump(Context const*, linalg::TensorView<GradientPair const, 2>,
                     linalg::VectorView<float>) {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace cuda_impl

void FitStump(Context const* ctx, MetaInfo const& info, HostDeviceVector<GradientPair> const& gpair,
              bst_target_t n_targets, linalg::Vector<float>* out) {
  out->SetDevice(ctx->gpu_id);
  out->Reshape(n_targets);
  auto n_samples = gpair.Size() / n_targets;

  gpair.SetDevice(ctx->gpu_id);
  auto gpair_t = linalg::MakeTensorView(ctx, &gpair, n_samples, n_targets);
  ctx->IsCPU() ? cpu_impl::FitStump(ctx, info, gpair_t, out->HostView())
               : cuda_impl::FitStump(ctx, gpair_t, out->View(ctx->gpu_id));
}
}  // namespace tree
}  // namespace xgboost
