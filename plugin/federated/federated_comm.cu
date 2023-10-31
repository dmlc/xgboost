/**
 * Copyright 2023, XGBoost Contributors
 */
#include <memory>  // for shared_ptr

#include "../../src/common/cuda_context.cuh"
#include "federated_comm.cuh"
#include "xgboost/context.h"  // for Context

namespace xgboost::collective {
CUDAFederatedComm::CUDAFederatedComm(Context const* ctx, std::shared_ptr<FederatedComm const> impl)
    : FederatedComm{impl}, stream_{ctx->CUDACtx()->Stream()} {
  CHECK(impl);
}

Comm* FederatedComm::MakeCUDAVar(Context const* ctx, std::shared_ptr<Coll>) const {
  return new CUDAFederatedComm{
      ctx, std::dynamic_pointer_cast<FederatedComm const>(this->shared_from_this())};
}
}  // namespace xgboost::collective
