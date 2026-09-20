/**
 * Copyright 2026, XGBoost Contributors
 * \file error_metric.h
 * \brief Shared declarations for the classification error metric.
 */
#ifndef XGBOOST_METRIC_ERROR_METRIC_H_
#define XGBOOST_METRIC_ERROR_METRIC_H_

#include <cstdio>   // for sscanf
#include <sstream>  // for ostringstream
#include <string>   // for string

#include "elementwise_metric.h"  // for elementwise kernels
#include "xgboost/base.h"        // for bst_float

namespace xgboost::metric {
struct EvalError {
  explicit EvalError(const char* param) {
    if (param != nullptr) {
      CHECK_EQ(sscanf(param, "%f", &threshold_), 1)
          << "unable to parse the threshold value for the error metric";
      has_param_ = true;
    } else {
      threshold_ = 0.5f;
      has_param_ = false;
    }
  }
  [[nodiscard]] const char* Name() const {
    static thread_local std::string name;
    if (has_param_) {
      std::ostringstream os;
      os << "error";
      if (threshold_ != 0.5f) os << '@' << threshold_;
      name = os.str();
      return name.c_str();
    } else {
      return "error";
    }
  }

  [[nodiscard]] XGBOOST_DEVICE bst_float operator()(bst_float label, bst_float pred) const {
    // assume label is in [0,1]
    return pred > threshold_ ? 1.0f - label : label;
  }

  static double GetFinal(double esum, double wsum) { return wsum == 0 ? esum : esum / wsum; }

 private:
  bst_float threshold_;
  bool has_param_;
};

using ErrorEvalKernel = elementwise::EvalKernel<EvalError>;
}  // namespace xgboost::metric

#endif  // XGBOOST_METRIC_ERROR_METRIC_H_
