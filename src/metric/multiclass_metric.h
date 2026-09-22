/**
 * Copyright 2026, XGBoost Contributors
 * \file multiclass_metric.h
 * \brief Shared policies and typed kernels for multiclass metrics.
 */
#ifndef XGBOOST_METRIC_MULTICLASS_METRIC_H_
#define XGBOOST_METRIC_MULTICLASS_METRIC_H_

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "../common/math.h"
#include "metric_common.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"

namespace xgboost::metric {
/*! \brief match error */
struct EvalMatchError {
  static const char* Name() { return "merror"; }
  XGBOOST_DEVICE static bst_float EvalRow(int label, const bst_float* pred, size_t nclass) {
    return common::FindMaxIndex(pred, pred + nclass) != pred + static_cast<int>(label);
  }
};

/*! \brief match error */
struct EvalMultiLogLoss {
  static const char* Name() { return "mlogloss"; }
  XGBOOST_DEVICE static bst_float EvalRow(int label, const bst_float* pred, size_t /*nclass*/) {
    const bst_float eps = 1e-16f;
    auto k = static_cast<size_t>(label);
    if (pred[k] > eps) {
      return -std::log(pred[k]);
    } else {
      return -std::log(eps);
    }
  }
};

template <typename EvalFn>
struct MultiClassEvalKernel {
  using Signature = PackedReduceResult(Context const*, HostDeviceVector<float> const&,
                                       MetaInfo const&, std::size_t,
                                       HostDeviceVector<std::int32_t>*);
};
using MultiClassErrorEvalKernel = MultiClassEvalKernel<EvalMatchError>;
using MultiClassLogLossEvalKernel = MultiClassEvalKernel<EvalMultiLogLoss>;

inline void CheckMultiClassLabel(std::int32_t label_error, std::size_t n_class) {
  CHECK(label_error >= 0 && label_error < static_cast<std::int32_t>(n_class))
      << "MultiClassEvaluation: label must be in [0, num_class),"
      << " num_class=" << n_class << " but found " << label_error << " in label";
}
}  // namespace xgboost::metric

#endif  // XGBOOST_METRIC_MULTICLASS_METRIC_H_
