/**
 * Copyright 2025, XGBoost Contributors
 */
#include "../common/cuda_context.cuh"
#include "array_interface.h"
#include "xgboost/gradient.h"

namespace xgboost::cuda_impl {
void CopyGrad(Context const *ctx, ArrayInterface<2, false> const &i_grad,
              ArrayInterface<2, false> const &i_hess, linalg::Matrix<GradientPair> *out_gpair) {
  auto grad_dev = dh::CudaGetPointerDevice(i_grad.data);
  auto hess_dev = dh::CudaGetPointerDevice(i_hess.data);
  CHECK_EQ(grad_dev, hess_dev) << "gradient and hessian should be on the same device.";
  auto &gpair = *out_gpair;
  gpair.SetDevice(DeviceOrd::CUDA(grad_dev));
  gpair.Reshape(i_grad.Shape<0>(), i_grad.Shape<1>());
  auto d_gpair = gpair.View(DeviceOrd::CUDA(grad_dev));
  auto cuctx = ctx->CUDACtx();

  DispatchDType(i_grad, DeviceOrd::CUDA(grad_dev), [&](auto &&t_grad) {
    DispatchDType(i_hess, DeviceOrd::CUDA(hess_dev), [&](auto &&t_hess) {
      CHECK_EQ(t_grad.Size(), t_hess.Size());
      thrust::for_each_n(cuctx->CTP(), thrust::make_counting_iterator(0ul), t_grad.Size(),
                         CustomGradHessOp{t_grad, t_hess, d_gpair});
    });
  });
}
}  // namespace xgboost::cuda_impl
