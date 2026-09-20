/**
 * Copyright 2026, XGBoost Contributors
 * \file aft_obj.h
 * \brief Shared declarations for the AFT objective.
 */
#ifndef XGBOOST_OBJECTIVE_AFT_OBJ_H_
#define XGBOOST_OBJECTIVE_AFT_OBJ_H_

#include "../common/survival_util.h"     // for ProbabilityDistributionType
#include "xgboost/base.h"                // for GradientPair
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix, Vector

namespace xgboost::obj {
struct AFTGradientKernel {
  using Signature = void(Context const*, HostDeviceVector<float> const&, MetaInfo const&,
                         common::ProbabilityDistributionType, float, linalg::Matrix<GradientPair>*);
};

struct AFTPredTransformKernel {
  using Signature = void(Context const*, HostDeviceVector<float>*);
};

struct AFTProbToMarginKernel {
  using Signature = void(Context const*, linalg::Vector<float>*);
};
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_AFT_OBJ_H_
