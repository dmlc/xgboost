/**
 * Copyright 2025, XGBoost Contributors
 */
#include <dmlc/registry.h>
#include <thrust/for_each.h>  // for for_each_n

#include "../common/cuda_compat.cuh"  // for CUDA compatibility
#include "../common/cuda_context.cuh"
#include "../common/cuda_rt_utils.h"
#include "../common/kernel.h"
#include "prediction_kernel.h"
#include "xgboost/linalg.h"  // for UnravelIndex

namespace xgboost::predictor {
DMLC_REGISTRY_FILE_TAG(prediction_cuda);
namespace {
void InitBaseScoreCUDA(Context const* ctx, linalg::VectorView<float const> base_score,
                       linalg::MatrixView<float> predt) {
  curt::SetDevice(ctx->Ordinal());
  thrust::for_each_n(ctx->CUDACtx()->CTP(), dh::make_counting_iterator(0ul), predt.Size(),
                     [=] XGBOOST_DEVICE(std::size_t k) mutable {
                       auto [i, j] = linalg::UnravelIndex(k, predt.Shape());
                       predt(i, j) = base_score(j);
                     });
}
common::KernelRegistration<InitBaseScoreKernel> const kInitBaseScoreCUDA{DeviceOrd::kCUDA,
                                                                         &InitBaseScoreCUDA};
}  // namespace
}  // namespace xgboost::predictor
