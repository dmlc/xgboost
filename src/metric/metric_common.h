/*!
 * Copyright 2018 by Contributors
 * \file metric_param.cc
 */
#ifndef XGBOOST_METRIC_METRIC_PARAM_H_
#define XGBOOST_METRIC_METRIC_PARAM_H_

#include <dmlc/parameter.h>
#include "../common/common.h"

namespace xgboost {
namespace metric {

// Created exclusively for GPU.
struct MetricParam : public dmlc::Parameter<MetricParam> {
  int n_gpus;
  int gpu_id;
  DMLC_DECLARE_PARAMETER(MetricParam) {
    DMLC_DECLARE_FIELD(n_gpus).set_default(1).set_lower_bound(GPUSet::kAll)
        .describe("Number of GPUs to use for multi-gpu algorithms.");
    DMLC_DECLARE_FIELD(gpu_id)
        .set_lower_bound(0)
        .set_default(0)
        .describe("gpu to use for objective function evaluation");
  };
};

class PackedReduceResult {
  double residue_sum_;
  double weights_sum_;

public:
  XGBOOST_DEVICE PackedReduceResult() : residue_sum_{0}, weights_sum_{0} {}
  XGBOOST_DEVICE PackedReduceResult(double residue, double weight)
      : residue_sum_{residue}, weights_sum_{weight} {}

  XGBOOST_DEVICE
  PackedReduceResult operator+(PackedReduceResult const &other) const {
    return PackedReduceResult{residue_sum_ + other.residue_sum_,
                              weights_sum_ + other.weights_sum_};
  }
  PackedReduceResult &operator+=(PackedReduceResult const &other) {
    this->residue_sum_ += other.residue_sum_;
    this->weights_sum_ += other.weights_sum_;
    return *this;
  }
  double Residue() const { return residue_sum_; }
  double Weights() const { return weights_sum_; }
};

}  // namespace metric
}  // namespace xgboost

#endif  // XGBOOST_METRIC_METRIC_PARAM_H_
