/**
 * Copyright 2022-2024, XGBoost Contributors
 *
 * @brief Utilities for estimating initial score.
 */
#include "fit_stump.h"

#include <cstddef>  // std::size_t
#include <cstdint>  // for int32_t

#include "../collective/aggregator.h"   // for GlobalSum
#include "../common/threading_utils.h"  // ParallelFor
#include "xgboost/base.h"               // bst_target_t, GradientPairPrecise
#include "xgboost/context.h"            // Context
#include "xgboost/linalg.h"             // TensorView, Tensor, Constant
#include "xgboost/logging.h"            // CHECK_EQ

#if !defined(XGBOOST_USE_CUDA)
#include "../common/common.h"  // AssertGPUSupport
#endif

namespace xgboost::tree {
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
  auto as_double = linalg::MakeTensorView(
      ctx, common::Span{reinterpret_cast<double*>(h_sum.Values().data()), h_sum.Size() * 2},
      h_sum.Size() * 2);
  auto rc = collective::GlobalSum(ctx, info, as_double);
  collective::SafeColl(rc);

  for (std::size_t i = 0; i < h_sum.Size(); ++i) {
    out(i) = static_cast<float>(CalcUnregularizedWeight(h_sum(i).GetGrad(), h_sum(i).GetHess()));
  }
}
}  // namespace cpu_impl

namespace cuda_impl {
void FitStump(Context const* ctx, MetaInfo const& info,
              linalg::TensorView<GradientPair const, 2> gpair, linalg::VectorView<float> out);

#if !defined(XGBOOST_USE_CUDA)
inline void FitStump(Context const*, MetaInfo const&, linalg::TensorView<GradientPair const, 2>,
                     linalg::VectorView<float>) {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace cuda_impl

void FitStump(Context const* ctx, MetaInfo const& info, linalg::Matrix<GradientPair> const& gpair,
              bst_target_t n_targets, linalg::Vector<float>* out) {
  out->SetDevice(ctx->Device());
  out->Reshape(n_targets);

  gpair.SetDevice(ctx->Device());
  auto gpair_t = gpair.View(ctx->Device().IsSycl() ? DeviceOrd::CPU() : ctx->Device());
  ctx->IsCUDA() ? cuda_impl::FitStump(ctx, info, gpair_t, out->View(ctx->Device()))
                : cpu_impl::FitStump(ctx, info, gpair_t, out->HostView());
}
}  // namespace xgboost::tree
