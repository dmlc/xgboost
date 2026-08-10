/**
 * Copyright 2026, XGBoost Contributors
 * \file multiclass_obj.h
 * \brief Shared declarations for multiclass objectives.
 */
#ifndef XGBOOST_OBJECTIVE_MULTICLASS_OBJ_H_
#define XGBOOST_OBJECTIVE_MULTICLASS_OBJ_H_

#include <cmath>    // for floor
#include <cstdint>  // for int64_t

#include "elementwise_objective.h"       // for TransformKernel, ValidationKernel
#include "xgboost/base.h"                // for GradientPair
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix

namespace xgboost::obj {
struct MulticlassLabelCheck {
  std::int64_t n_classes;
  XGBOOST_DEVICE bool operator()(float value) const {
    return value >= 0.0f && value < n_classes && std::floor(value) == value;
  }
};

struct MulticlassCenter {
  float mean;
  XGBOOST_DEVICE float operator()(float value) const { return value - mean; }
};

struct MulticlassGradientKernel {
  using Signature = void(Context const*, HostDeviceVector<float> const&, MetaInfo const&,
                         std::int64_t, linalg::Matrix<GradientPair>*);
};

struct MulticlassInitEstimationKernel {
  using Signature = void(Context const*, MetaInfo const&, std::int64_t, linalg::Vector<float>*);
};

struct MulticlassTransformKernel {
  using Signature = void(Context const*, HostDeviceVector<float>*, std::int32_t, bool);
};

using MulticlassValidationKernel = elementwise::ValidationKernel<MulticlassLabelCheck>;
using MulticlassCenterKernel = elementwise::TransformKernel<MulticlassCenter>;
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_MULTICLASS_OBJ_H_
