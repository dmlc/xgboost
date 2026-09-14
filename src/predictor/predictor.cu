/**
 * Copyright 2025, XGBoost Contributors
 */
#include <thrust/for_each.h>  // for for_each_n

#include "../common/cuda_compat.cuh"  // for CUDA compatibility
#include "../common/cuda_context.cuh"
#include "xgboost/linalg.h"  // for UnravelIndex

namespace xgboost::cuda_impl {
void InitOutPredictions(Context const* ctx, linalg::VectorView<float const> base_score,
                        linalg::MatrixView<float> predt) {
  thrust::for_each_n(ctx->CUDACtx()->CTP(), dh::make_counting_iterator(0ul), predt.Size(),
                     [=] XGBOOST_DEVICE(std::size_t k) mutable {
                       auto [i, j] = linalg::UnravelIndex(k, predt.Shape());
                       predt(i, j) = base_score(j);
                     });
}
}  // namespace xgboost::cuda_impl
