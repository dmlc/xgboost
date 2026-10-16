/**
 * Copyright 2026, XGBoost Contributors
 * \file expectile_obj.h
 * \brief Shared declarations for the expectile objective.
 */
#ifndef XGBOOST_OBJECTIVE_EXPECTILE_OBJ_H_
#define XGBOOST_OBJECTIVE_EXPECTILE_OBJ_H_

#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/linalg.h"

namespace xgboost::obj {
struct ExpectileGradientKernel {
  using Signature = void(Context const*, HostDeviceVector<float> const&, MetaInfo const&,
                         HostDeviceVector<float> const&, bst_target_t,
                         linalg::Matrix<GradientPair>*);
};
struct ExpectileInitEstimationKernel {
  using Signature = void(Context const*, MetaInfo const&, HostDeviceVector<float> const&,
                         bst_target_t, linalg::Vector<float>*);
};
struct ExpectilePredTransformKernel {
  using Signature = void(Context const*, HostDeviceVector<float>*, std::size_t);
};
}  // namespace xgboost::obj

#endif  // XGBOOST_OBJECTIVE_EXPECTILE_OBJ_H_
