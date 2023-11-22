/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once

#include <memory>  // for shared_ptr

#include "../../src/collective/coll.h"          // for Coll
#include "../../src/common/device_helpers.cuh"  // for CUDAStreamView
#include "federated_comm.h"                     // for FederatedComm
#include "xgboost/context.h"                    // for Context
#include "xgboost/logging.h"

namespace xgboost::collective {
class CUDAFederatedComm : public FederatedComm {
  dh::CUDAStreamView stream_;

 public:
  explicit CUDAFederatedComm(Context const* ctx, std::shared_ptr<FederatedComm const> impl);
  [[nodiscard]] auto Stream() const { return stream_; }
  Comm* MakeCUDAVar(Context const*, std::shared_ptr<Coll>) const override {
    LOG(FATAL) << "[Internal Error]: Invalid request for CUDA variant.";
    return nullptr;
  }
};
}  // namespace xgboost::collective
