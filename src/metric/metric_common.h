/**
 * Copyright 2018-2026, XGBoost Contributors
 */
#ifndef XGBOOST_METRIC_METRIC_COMMON_H_
#define XGBOOST_METRIC_METRIC_COMMON_H_

#include <limits>
#include <memory>  // shared_ptr
#include <string>

#include "../collective/aggregator.h"
#include "xgboost/logging.h"
#include "xgboost/metric.h"

namespace xgboost {
struct Context;
// Metric that doesn't need to cache anything based on input data.
class MetricNoCache : public Metric {
 public:
  virtual double Eval(HostDeviceVector<float> const &predts, MetaInfo const &info) = 0;

  double Evaluate(HostDeviceVector<float> const &predts, std::shared_ptr<DMatrix> p_fmat) final {
    double result{0.0};
    auto const &info = p_fmat->Info();
    collective::ApplyWithLabels(ctx_, info, &result, sizeof(double),
                                [&] { result = this->Eval(predts, info); });
    return result;
  }
};

namespace metric {
inline void CheckRowWeights(MetaInfo const &info) {
  if (info.weights_.Empty()) {
    return;
  }
  CHECK(info.group_ptr_.empty()) << "Row-wise metric does not support query group weights.";
  CHECK_EQ(info.weights_.Size(), info.num_row_)
      << "Number of weights should be equal to the number of data points.";
}

// Ranking config to be used on device and host
struct EvalRankConfig {
 public:
  // Parsed from metric name, the top-n number of instances within a group after
  // ranking to use for evaluation.
  unsigned topn{std::numeric_limits<unsigned>::max()};
  std::string name;
  bool minus{false};
};

class PackedReduceResult {
  double residue_sum_{0};
  double weights_sum_{0};

 public:
  XGBOOST_DEVICE PackedReduceResult() {}  // NOLINT
  XGBOOST_DEVICE PackedReduceResult(double residue, double weight)
      : residue_sum_{residue}, weights_sum_{weight} {}

  XGBOOST_DEVICE
  PackedReduceResult operator+(PackedReduceResult const &other) const {
    return PackedReduceResult{residue_sum_ + other.residue_sum_, weights_sum_ + other.weights_sum_};
  }
  PackedReduceResult &operator+=(PackedReduceResult const &other) {
    this->residue_sum_ += other.residue_sum_;
    this->weights_sum_ += other.weights_sum_;
    return *this;
  }
  [[nodiscard]] double Residue() const { return residue_sum_; }
  [[nodiscard]] double Weights() const { return weights_sum_; }
};

}  // namespace metric
}  // namespace xgboost

#endif  // XGBOOST_METRIC_METRIC_COMMON_H_
