/**
 * Copyright 2022 by XGBoost Contributors
 *
 * \brief Utilities for estimating initial score.
 */

#if !defined(NOMINMAX) && defined(_WIN32)
#define NOMINMAX
#endif  // !defined(NOMINMAX)
#include "fit_stump.h"

#include <algorithm>  // std::max
#include <cstddef>    // std::size_t

#include "../collective/communicator-inl.h"
#include "../common/numeric.h"             // cpu_impl::Reduce
#include "../common/transform_iterator.h"  // MakeIndexTransformIter
#include "xgboost/linalg.h"                // TensorView

namespace xgboost {
namespace obj {
namespace cpu_impl {
void FitStump(Context const* ctx, linalg::TensorView<GradientPair const, 2> gpair,
              linalg::VectorView<float> out) {
  auto n_targets = out.Size();
  CHECK_EQ(n_targets, gpair.Shape(1));
  linalg::Vector<GradientPairPrecise> sum = linalg::Constant(ctx, GradientPairPrecise{}, n_targets);
  auto h_sum = sum.HostView();
  // first dim for gpair is samples, second dim is target.
  // Reduce by column
  common::ParallelFor(gpair.Shape(1), 1, [&](auto j) {
    for (std::size_t i = 0; i < gpair.Shape(0); ++i) {
      h_sum(j) += GradientPairPrecise{gpair(i, j)};
    }
  });
  collective::Allreduce<collective::Operation::kSum>(
      reinterpret_cast<double*>(h_sum.Values().data()), h_sum.Size() * 2);

  for (std::size_t i = 0; i < h_sum.Size(); ++i) {
    out(i) = static_cast<float>(CalcUnregulatedWeight(h_sum(i).GetGrad(), h_sum(i).GetHess()));
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
  return 0.0;
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace cuda_impl

void FitStump(Context const* ctx, HostDeviceVector<GradientPair> const& gpair,
              bst_target_t n_targets, linalg::Vector<float>* out) {
  out->SetDevice(ctx->gpu_id);
  out->Reshape(n_targets);
  // column-major
  auto n_samples = gpair.Size() / n_targets;
  std::size_t shape[2]{n_samples, n_targets};
  std::size_t strides[2];
  linalg::detail::CalcStride<2, true>(shape, strides);

  gpair.SetDevice(ctx->gpu_id);
  linalg::TensorView<GradientPair const, 2> gpair_t{
      ctx->IsCPU() ? gpair.ConstHostSpan() : gpair.ConstDeviceSpan(), shape, strides, ctx->gpu_id};
  ctx->IsCPU() ? cpu_impl::FitStump(ctx, gpair_t, out->HostView())
               : cuda_impl::FitStump(ctx, gpair_t, out->View(ctx->gpu_id));
}
}  // namespace obj
}  // namespace xgboost
