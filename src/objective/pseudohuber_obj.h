/**
 * Copyright 2026, XGBoost Contributors
 * \file pseudohuber_obj.h
 * \brief Shared declarations for the pseudo-Huber objective.
 */
#ifndef XGBOOST_OBJECTIVE_PSEUDOHUBER_OBJ_H_
#define XGBOOST_OBJECTIVE_PSEUDOHUBER_OBJ_H_

#include <cmath>  // for sqrtf

#include "xgboost/base.h"                // for GradientPair, bst_target_t
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix

namespace xgboost::obj {
struct PseudoHuberLoss {
  XGBOOST_DEVICE static GradientPair Gradient(float predt, float label, float weight, float slope) {
    auto z = predt - label;
    auto slope_sq = slope * slope;
    auto z_sq = z * z;
    auto scale_sqrt = sqrtf(1.0f + z_sq / slope_sq);
    auto grad = z / scale_sqrt;
    auto scale = slope_sq + z_sq;
    auto hess = slope_sq / (scale * scale_sqrt);
    return {grad * weight, hess * weight};
  }
};

struct PseudoHuberGradientKernel {
  using Signature = void(Context const*, HostDeviceVector<float> const&, MetaInfo const&,
                         bst_target_t, float, linalg::Matrix<GradientPair>*);
};
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_PSEUDOHUBER_OBJ_H_
