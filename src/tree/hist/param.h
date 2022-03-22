/*!
 * Copyright 2021 XGBoost contributors
 */
#ifndef XGBOOST_TREE_HIST_PARAM_H_
#define XGBOOST_TREE_HIST_PARAM_H_
#include "xgboost/parameter.h"

namespace xgboost {
namespace tree {
// training parameters specific to this algorithm
struct CPUHistMakerTrainParam
    : public XGBoostParameter<CPUHistMakerTrainParam> {
  bool single_precision_histogram;
  // declare parameters
  DMLC_DECLARE_PARAMETER(CPUHistMakerTrainParam) {
    DMLC_DECLARE_FIELD(single_precision_histogram).set_default(false).describe(
        "Use single precision to build histograms.");
  }
};
}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_HIST_PARAM_H_
