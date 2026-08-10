/**
 * Copyright 2026, XGBoost Contributors
 * \file absolute_error_obj.h
 * \brief Shared declarations for the absolute-error objective.
 */
#ifndef XGBOOST_OBJECTIVE_ABSOLUTE_ERROR_OBJ_H_
#define XGBOOST_OBJECTIVE_ABSOLUTE_ERROR_OBJ_H_

#include "xgboost/base.h"                // for GradientPair, bst_target_t
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix

namespace xgboost::obj {
struct AbsoluteErrorGradientKernel {
  using Signature = void(Context const*, HostDeviceVector<float> const&, MetaInfo const&,
                         bst_target_t, linalg::Matrix<GradientPair>*);
};

struct AbsoluteErrorInitEstimationKernel {
  using Signature = void(Context const*, MetaInfo const&, bst_target_t, linalg::Vector<float>*);
};
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_ABSOLUTE_ERROR_OBJ_H_
