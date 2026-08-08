/**
 * Copyright 2017-2026, XGBoost contributors
 */
#ifndef XGBOOST_OBJECTIVE_REGRESSION_LOSS_H_
#define XGBOOST_OBJECTIVE_REGRESSION_LOSS_H_

#include <cmath>

#include "../common/common.h"  // Min, Max
#include "../common/math.h"
#include "xgboost/string_view.h"
#include "xgboost/task.h"  // ObjInfo

namespace xgboost::obj {
// linear regression
struct LinearSquareLoss {
  XGBOOST_DEVICE static bst_float PredTransform(bst_float x) { return x; }
  XGBOOST_DEVICE static bool CheckLabel(bst_float) { return true; }
  XGBOOST_DEVICE static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    return predt - label;
  }
  XGBOOST_DEVICE static bst_float SecondOrderGradient(bst_float, bst_float) { return 1.0f; }

  XGBOOST_DEVICE static float ProbToMargin(float base_score) { return base_score; }
  constexpr static StringView InterceptErrorMsg() { return ""; }
  XGBOOST_DEVICE static bool CheckIntercept(float) { return true; }

  static const char* LabelErrorMsg() { return ""; }
  static const char* DefaultEvalMetric() { return "rmse"; }

  static const char* Name() { return "reg:squarederror"; }
  static ObjInfo Info() { return {ObjInfo::kRegression, true}; }
};

// logistic loss for probability regression task
struct LogisticRegression {
  XGBOOST_DEVICE static bst_float PredTransform(bst_float x) { return common::Sigmoid(x); }
  XGBOOST_DEVICE static bool CheckLabel(bst_float x) { return x >= 0.0f && x <= 1.0f; }
  XGBOOST_DEVICE static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    return predt - label;
  }
  XGBOOST_DEVICE static bst_float SecondOrderGradient(bst_float predt, bst_float) {
    const float eps = 1e-16f;
    return fmaxf(predt * (1.0f - predt), eps);
  }
  XGBOOST_DEVICE static float ProbToMargin(float base_score) {
    // Bound the base score
    base_score = common::Min(common::Max(base_score, kRtEps), 1.0f - kRtEps);
    return common::Logit(base_score);
  }
  constexpr static StringView InterceptErrorMsg() {
    return "base_score must be in (0,1) for the logistic loss.";
  }
  XGBOOST_DEVICE static bool CheckIntercept(float base_score) {
    // We accept equality for degenerate cases where all label is the same.
    // https://github.com/dmlc/xgboost/issues/11499
    return base_score >= 0.0f && base_score <= 1.0f;
  }

  static const char* LabelErrorMsg() { return "label must be in (0, 1) for logistic regression"; }
  static const char* DefaultEvalMetric() { return "rmse"; }

  static const char* Name() { return "reg:logistic"; }

  static ObjInfo Info() { return ObjInfo::kRegression; }
};

// logistic loss for binary classification task
struct LogisticClassification : public LogisticRegression {
  static const char* DefaultEvalMetric() { return "logloss"; }
  static const char* Name() { return "binary:logistic"; }
  static ObjInfo Info() { return ObjInfo::kBinary; }
};

// logistic loss, but predict un-transformed margin
struct LogisticRaw : public LogisticRegression {
  XGBOOST_DEVICE static bst_float PredTransform(bst_float x) { return x; }
  XGBOOST_DEVICE static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    predt = common::Sigmoid(predt);
    return predt - label;
  }
  XGBOOST_DEVICE static bst_float SecondOrderGradient(bst_float predt, bst_float) {
    const float eps = 1e-16f;
    predt = common::Sigmoid(predt);
    return fmaxf(predt * (1.0f - predt), eps);
  }

  XGBOOST_DEVICE static float ProbToMargin(float base_score) { return base_score; }
  constexpr static StringView InterceptErrorMsg() { return ""; }
  XGBOOST_DEVICE static bool CheckIntercept(float) { return true; }

  static const char* DefaultEvalMetric() { return "logloss"; }

  static const char* Name() { return "binary:logitraw"; }

  static ObjInfo Info() { return ObjInfo::kRegression; }
};

// Label validation for Poisson regression (labels must be non-negative)
struct PoissonLabel {
  XGBOOST_DEVICE static bool CheckLabel(float x) { return x >= 0.0f; }
  static const char* LabelErrorMsg() {
    return "label must be non-negative for Poisson/Tweedie regression.";
  }
};

// Label validation for Tweedie regression (labels must be non-negative)
using TweedieLabel = PoissonLabel;
}  // namespace xgboost::obj
#endif  // XGBOOST_OBJECTIVE_REGRESSION_LOSS_H_
