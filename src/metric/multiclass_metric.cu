/*!
 * Copyright 2015-2019 by Contributors
 * \file multiclass_metric.cc
 * \brief evaluation metrics for multiclass classification.
 * \author Kailong Chen, Tianqi Chen
 */
#include <rabit/rabit.h>
#include <xgboost/metric/metric.h>
#include <xgboost/metric/metric_common.h>
#include <cmath>

#include "../common/math.h"
#include "../common/common.h"

#if defined(XGBOOST_USE_CUDA)
#include <thrust/execution_policy.h>  // thrust::cuda::par
#include <thrust/functional.h>        // thrust::plus<>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

#include "../common/device_helpers.cuh"
#endif  // XGBOOST_USE_CUDA

namespace xgboost {
namespace metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(multiclass_metric);

/*! \brief match error */
struct EvalMatchError : public EvalMClassBase<EvalMatchError> {
  const char* Name() const override {
    return "merror";
  }
  XGBOOST_DEVICE static bst_float EvalRow(int label,
                                          const bst_float *pred,
                                          size_t nclass) {
    return common::FindMaxIndex(pred, pred + nclass) != pred + static_cast<int>(label);
  }
};

/*! \brief match error */
struct EvalMultiLogLoss : public EvalMClassBase<EvalMultiLogLoss> {
  const char* Name() const override {
    return "mlogloss";
  }
  XGBOOST_DEVICE static bst_float EvalRow(int label,
                                          const bst_float *pred,
                                          size_t nclass) {
    const bst_float eps = 1e-16f;
    auto k = static_cast<size_t>(label);
    if (pred[k] > eps) {
      return -std::log(pred[k]);
    } else {
      return -std::log(eps);
    }
  }
};

XGBOOST_REGISTER_METRIC(MatchError, "merror")
.describe("Multiclass classification error.")
.set_body([](const char* param) { return new EvalMatchError(); });

XGBOOST_REGISTER_METRIC(MultiLogLoss, "mlogloss")
.describe("Multiclass negative loglikelihood.")
.set_body([](const char* param) { return new EvalMultiLogLoss(); });
}  // namespace metric
}  // namespace xgboost
