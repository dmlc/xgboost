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
#include "xgboost/span.h"                // for Span

namespace xgboost::obj {
XGBOOST_DEVICE inline void SortQuantilePredictions(common::Span<float> predictions) {
  for (std::size_t i{1}; i < predictions.size(); ++i) {
    auto value = predictions[i];
    auto pos = i;
    while (pos > 0 && predictions[pos - 1] > value) {
      predictions[pos] = predictions[pos - 1];
      --pos;
    }
    predictions[pos] = value;
  }
}

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
