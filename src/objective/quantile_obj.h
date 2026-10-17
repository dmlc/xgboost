/**
 * Copyright 2026, XGBoost Contributors
 * \file quantile_obj.h
 * \brief Shared declarations for the quantile objective.
 */
#ifndef XGBOOST_OBJECTIVE_QUANTILE_OBJ_H_
#define XGBOOST_OBJECTIVE_QUANTILE_OBJ_H_

#include <cstddef>  // for size_t

#include "xgboost/base.h"                // for GradientPair, bst_target_t
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix
namespace xgboost::obj {
struct QuantileGradientKernel {
  using Signature = void(Context const*, HostDeviceVector<float> const&, MetaInfo const&,
                         bst_target_t, HostDeviceVector<float> const&,
                         linalg::Matrix<GradientPair>*);
};

struct QuantileTransformKernel {
  using Signature = void(Context const*, HostDeviceVector<float>*, std::size_t);
};
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_QUANTILE_OBJ_H_
