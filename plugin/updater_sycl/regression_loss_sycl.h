/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#ifndef XGBOOST_OBJECTIVE_REGRESSION_LOSS_SYCL_H_
#define XGBOOST_OBJECTIVE_REGRESSION_LOSS_SYCL_H_

#include <dmlc/omp.h>
#include <xgboost/logging.h>
#include <algorithm>
#include "../../src/common/math.h"

namespace xgboost {
namespace obj {

// common regressions
// linear regression
struct LinearSquareLossSycl {
  static bst_float PredTransform(bst_float x) { return x; }
  static bool CheckLabel(bst_float x) { return true; }
  static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    return predt - label;
  }
  static bst_float SecondOrderGradient(bst_float predt, bst_float label) {
    return 1.0f;
  }
  static bst_float ProbToMargin(bst_float base_score) { return base_score; }
  static const char* LabelErrorMsg() { return ""; }
  static const char* DefaultEvalMetric() { return "rmse"; }

  static const char* Name() { return "reg:squarederror_sycl"; }
};

// TODO: DPC++ does not support std math inside offloaded kernels
struct SquaredLogErrorSycl {
  static bst_float PredTransform(bst_float x) { return x; }
  static bool CheckLabel(bst_float label) {
    return label > -1;
  }
  static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    predt = std::max(predt, (bst_float)(-1 + 1e-6));  // ensure correct value for log1p
    return (std::log1p(predt) - std::log1p(label)) / (predt + 1);
  }
  static bst_float SecondOrderGradient(bst_float predt, bst_float label) {
    predt = std::max(predt, (bst_float)(-1 + 1e-6));
    float res = (-std::log1p(predt) + std::log1p(label) + 1) /
                std::pow(predt + 1, 2);
    res = std::max(res, (bst_float)1e-6f);
    return res;
  }
  static bst_float ProbToMargin(bst_float base_score) { return base_score; }
  static const char* LabelErrorMsg() {
    return "label must be greater than -1 for rmsle so that log(label + 1) can be valid.";
  }
  static const char* DefaultEvalMetric() { return "rmsle"; }

  static const char* Name() { return "reg:squaredlogerror_sycl"; }
};

// logistic loss for probability regression task
struct LogisticRegressionSycl {
  // duplication is necessary, as __device__ specifier
  // cannot be made conditional on template parameter
  static bst_float PredTransform(bst_float x) { return common::Sigmoid(x); }
  static bool CheckLabel(bst_float x) { return x >= 0.0f && x <= 1.0f; }
  static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    return predt - label;
  }
  static bst_float SecondOrderGradient(bst_float predt, bst_float label) {
    const bst_float eps = 1e-16f;
    return std::max(predt * (1.0f - predt), eps);
  }
  template <typename T>
  static T PredTransform(T x) { return common::Sigmoid(x); }
  template <typename T>
  static T FirstOrderGradient(T predt, T label) { return predt - label; }
  template <typename T>
  static T SecondOrderGradient(T predt, T label) {
    const T eps = T(1e-16f);
    return std::max(predt * (T(1.0f) - predt), eps);
  }
  static bst_float ProbToMargin(bst_float base_score) {
    CHECK(base_score > 0.0f && base_score < 1.0f)
        << "base_score must be in (0,1) for logistic loss, got: " << base_score;
    return -logf(1.0f / base_score - 1.0f);
  }
  static const char* LabelErrorMsg() {
    return "label must be in [0,1] for logistic regression";
  }
  static const char* DefaultEvalMetric() { return "rmse"; }

  static const char* Name() { return "reg:logistic_sycl"; }
};

// logistic loss for binary classification task
struct LogisticClassificationSycl : public LogisticRegressionSycl {
  static const char* DefaultEvalMetric() { return "error"; }
  static const char* Name() { return "binary:logistic_sycl"; }
};

// logistic loss, but predict un-transformed margin
struct LogisticRawSycl : public LogisticRegressionSycl {
  // duplication is necessary, as __device__ specifier
  // cannot be made conditional on template parameter
  static bst_float PredTransform(bst_float x) { return x; }
  static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    predt = common::Sigmoid(predt);
    return predt - label;
  }
  static bst_float SecondOrderGradient(bst_float predt, bst_float label) {
    const bst_float eps = 1e-16f;
    predt = common::Sigmoid(predt);
    return std::max(predt * (1.0f - predt), eps);
  }
  template <typename T>
    static T PredTransform(T x) { return x; }
  template <typename T>
    static T FirstOrderGradient(T predt, T label) {
    predt = common::Sigmoid(predt);
    return predt - label;
  }
  template <typename T>
    static T SecondOrderGradient(T predt, T label) {
    const T eps = T(1e-16f);
    predt = common::Sigmoid(predt);
    return std::max(predt * (T(1.0f) - predt), eps);
  }
  static const char* DefaultEvalMetric() { return "auc"; }

  static const char* Name() { return "binary:logitraw_sycl"; }
};

}  // namespace obj
}  // namespace xgboost

#endif  // XGBOOST_OBJECTIVE_REGRESSION_LOSS_SYCL_H_
