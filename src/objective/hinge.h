/**
 * Copyright 2018-2026, XGBoost Contributors
 * \file hinge.h
 * \brief Shared declarations for the hinge loss objective.
 */
#ifndef XGBOOST_OBJECTIVE_HINGE_H_
#define XGBOOST_OBJECTIVE_HINGE_H_

#include <limits>  // for numeric_limits

#include "xgboost/base.h"                // for GradientPair, bst_target_t
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix

namespace xgboost::obj {
struct HingeLoss {
  XGBOOST_DEVICE static GradientPair Gradient(float margin, float label, float weight) {
    auto y = label * 2.0f - 1.0f;
    if (margin * y < 1.0f) {
      return GradientPair{-y * weight, weight};
    }
    return GradientPair{0.0f, std::numeric_limits<float>::min()};
  }

  XGBOOST_DEVICE static float PredTransform(float margin) { return margin > 0.0f ? 1.0f : 0.0f; }
};

struct HingeGradientKernel {
  using Signature = void(Context const*, HostDeviceVector<float> const&, MetaInfo const&,
                         bst_target_t, linalg::Matrix<GradientPair>*);
};

struct HingePredTransformKernel {
  using Signature = void(Context const*, HostDeviceVector<float>*);
};

}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_HINGE_H_
