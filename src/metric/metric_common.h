/*!
 * Copyright 2018-2019 by Contributors
 * \file metric_param.cc
 */
#ifndef XGBOOST_METRIC_METRIC_COMMON_H_
#define XGBOOST_METRIC_METRIC_COMMON_H_

#include "../common/common.h"

namespace xgboost {
namespace metric {

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

#endif  // XGBOOST_METRIC_METRIC_COMMON_H_
