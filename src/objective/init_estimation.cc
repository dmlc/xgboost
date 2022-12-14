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
  // 2 rows, first one is gradient, sencond one is hessian. Number of columns equal to
  // number of targets.
  auto n_targets = out.Size();
  CHECK_EQ(n_targets, gpair.Shape(1));
  linalg::Tensor<double, 2> sum = linalg::Zeros<double>(ctx, 2, n_targets);
  CHECK(sum.HostView().CContiguous());
  auto sum_grad = sum.HostView().Slice(0, linalg::All());
  auto sum_hess = sum.HostView().Slice(1, linalg::All());

  // first dim for gpair is samples, second dim is target.
  // Reduce by column
  common::ParallelFor(gpair.Shape(1), 1, [&](auto j) {
    for (std::size_t i = 0; i < gpair.Shape(0); ++i) {
      sum_grad(j) += gpair(i, j).GetGrad();
      sum_hess(j) += gpair(i, j).GetHess();
    }
  });
  CHECK(sum_grad.CContiguous());
  collective::Allreduce<collective::Operation::kSum>(sum_grad.Values().data(), sum_grad.Size());
  CHECK(sum_hess.CContiguous());
  collective::Allreduce<collective::Operation::kSum>(sum_hess.Values().data(), sum_hess.Size());

  for (std::size_t i = 0; i < sum_hess.Size(); ++i) {
    out(i) = static_cast<float>(CalcUnregulatedWeight(sum_grad(i), sum_hess(i)));
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
