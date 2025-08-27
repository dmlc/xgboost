/**
 * Copyright 2025, XGBoost Contributors
 */
#include "linalg_op.cuh"

namespace xgboost::linalg::cuda_impl {
void VecScaMul(Context const* ctx, linalg::VectorView<float> x, double mul) {
  thrust::for_each_n(ctx->CUDACtx()->CTP(), thrust::make_counting_iterator(0ul), x.Size(),
                     [=] XGBOOST_DEVICE(std::size_t i) mutable { x(i) = x(i) * mul; });
}
}  // namespace xgboost::linalg::cuda_impl
