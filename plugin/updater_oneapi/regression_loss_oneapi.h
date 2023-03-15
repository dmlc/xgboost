/*!
 * Copyright 2017-2020 XGBoost contributors
 */
#ifndef XGBOOST_OBJECTIVE_REGRESSION_LOSS_ONEAPI_H_
#define XGBOOST_OBJECTIVE_REGRESSION_LOSS_ONEAPI_H_

#include <dmlc/omp.h>
#include <xgboost/logging.h>
#include <algorithm>

#include "CL/sycl.hpp"

namespace xgboost {
namespace obj {

/*!
 * \brief calculate the sigmoid of the input.
 * \param x input parameter
 * \return the transformed value.
 */
inline float SigmoidOneAPI(float x) {
  return 1.0f / (1.0f + cl::sycl::exp(-x));
}

// common regressions
// linear regression
struct LinearSquareLossOneAPI {
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

  static const char* Name() { return "reg:squarederror_oneapi"; }
};

// TODO: DPC++ does not fully support std math inside offloaded kernels
struct SquaredLogErrorOneAPI {
  static bst_float PredTransform(bst_float x) { return x; }
  static bool CheckLabel(bst_float label) {
    return label > -1;
  }
  static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    predt = std::max(predt, (bst_float)(-1 + 1e-6));  // ensure correct value for log1p
    return (cl::sycl::log1p(predt) - cl::sycl::log1p(label)) / (predt + 1);
  }
  static bst_float SecondOrderGradient(bst_float predt, bst_float label) {
    predt = std::max(predt, (bst_float)(-1 + 1e-6));
    float res = (-cl::sycl::log1p(predt) + cl::sycl::log1p(label) + 1) /
                cl::sycl::pow(predt + 1, (bst_float)2);
    res = std::max(res, (bst_float)1e-6f);
    return res;
  }
  static bst_float ProbToMargin(bst_float base_score) { return base_score; }
  static const char* LabelErrorMsg() {
    return "label must be greater than -1 for rmsle so that log(label + 1) can be valid.";
  }
  static const char* DefaultEvalMetric() { return "rmsle"; }

  static const char* Name() { return "reg:squaredlogerror_oneapi"; }
};

// logistic loss for probability regression task
struct LogisticRegressionOneAPI {
  // duplication is necessary, as __device__ specifier
  // cannot be made conditional on template parameter
  static bst_float PredTransform(bst_float x) { return SigmoidOneAPI(x); }
  static bool CheckLabel(bst_float x) { return x >= 0.0f && x <= 1.0f; }
  static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    return predt - label;
  }
  static bst_float SecondOrderGradient(bst_float predt, bst_float label) {
    const bst_float eps = 1e-16f;
    return std::max(predt * (1.0f - predt), eps);
  }
  template <typename T>
  static T PredTransform(T x) { return SigmoidOneAPI(x); }
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

  static const char* Name() { return "reg:logistic_oneapi"; }
};

// logistic loss for binary classification task
struct LogisticClassificationOneAPI : public LogisticRegressionOneAPI {
  static const char* DefaultEvalMetric() { return "logloss"; }
  static const char* Name() { return "binary:logistic_oneapi"; }
};

// logistic loss, but predict un-transformed margin
struct LogisticRawOneAPI : public LogisticRegressionOneAPI {
  // duplication is necessary, as __device__ specifier
  // cannot be made conditional on template parameter
  static bst_float PredTransform(bst_float x) { return x; }
  static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    predt = SigmoidOneAPI(predt);
    return predt - label;
  }
  static bst_float SecondOrderGradient(bst_float predt, bst_float label) {
    const bst_float eps = 1e-16f;
    predt = SigmoidOneAPI(predt);
    return std::max(predt * (1.0f - predt), eps);
  }
  template <typename T>
    static T PredTransform(T x) { return x; }
  template <typename T>
    static T FirstOrderGradient(T predt, T label) {
    predt = SigmoidOneAPI(predt);
    return predt - label;
  }
  template <typename T>
    static T SecondOrderGradient(T predt, T label) {
    const T eps = T(1e-16f);
    predt = SigmoidOneAPI(predt);
    return std::max(predt * (T(1.0f) - predt), eps);
  }
  static const char* DefaultEvalMetric() { return "logloss"; }

  static const char* Name() { return "binary:logitraw_oneapi"; }
};

}  // namespace obj
}  // namespace xgboost

#endif  // XGBOOST_OBJECTIVE_REGRESSION_LOSS_ONEAPI_H_
