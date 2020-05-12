/*!
 * Copyright 2018-2020 by Contributors
 * \file metric_common.h
 */
#ifndef XGBOOST_METRIC_METRIC_COMMON_H_
#define XGBOOST_METRIC_METRIC_COMMON_H_

#include <limits>
#include <string>

#include "../common/common.h"

namespace xgboost {

// This creates a GPU metric instance dynamically and adds it to the GPU metric registry, if not
// present already. This is created when there is a device ordinal present and if xgboost
// is compiled with CUDA support
struct GPUMetric : Metric {
  static Metric *CreateGPUMetric(const std::string& name, GenericParameter const* tparam);
};

/*!
 * \brief Internal registry entries for GPU Metric factory functions.
 *  The additional parameter const char* param gives the value after @, can be null.
 *  For example, metric map@3, then: param == "3".
 */
struct MetricGPUReg
  : public dmlc::FunctionRegEntryBase<MetricGPUReg,
                                      std::function<Metric * (const char*)> > {
};

/*!
 * \brief Macro to register metric computed on GPU.
 *
 * \code
 * // example of registering a objective ndcg@k
 * XGBOOST_REGISTER_GPU_METRIC(NDCG_GPU, "ndcg")
 * .describe("NDCG metric computer on GPU.")
 * .set_body([](const char* param) {
 *     int at_k = atoi(param);
 *     return new NDCG(at_k);
 *   });
 * \endcode
 */

// Note: Metric names registered in the GPU registry should follow this convention:
// - GPU metric types should be registered with the same name as the non GPU metric types
#define XGBOOST_REGISTER_GPU_METRIC(UniqueId, Name)                         \
  ::xgboost::MetricGPUReg&  __make_ ## MetricGPUReg ## _ ## UniqueId ## __ =  \
      ::dmlc::Registry< ::xgboost::MetricGPUReg>::Get()->__REGISTER__(Name)

namespace metric {

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
  double residue_sum_ { 0 };
  double weights_sum_ { 0 };

 public:
  XGBOOST_DEVICE PackedReduceResult() {}  // NOLINT
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
