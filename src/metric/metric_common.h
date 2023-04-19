/*!
 * Copyright 2018-2022 by Contributors
 * \file metric_common.h
 */
#ifndef XGBOOST_METRIC_METRIC_COMMON_H_
#define XGBOOST_METRIC_METRIC_COMMON_H_

#include <limits>
#include <memory>  // shared_ptr
#include <string>

#include "../collective/aggregator.h"
#include "../collective/communicator-inl.h"
#include "../common/common.h"
#include "xgboost/metric.h"

namespace xgboost {
struct Context;
// Metric that doesn't need to cache anything based on input data.
class MetricNoCache : public Metric {
 public:
  virtual double Eval(HostDeviceVector<float> const &predts, MetaInfo const &info) = 0;

  double Evaluate(HostDeviceVector<float> const &predts, std::shared_ptr<DMatrix> p_fmat) final {
    double result{0.0};
    auto const& info = p_fmat->Info();
    collective::ApplyWithLabels(info, &result, sizeof(double), [&] {
      result = this->Eval(predts, info);
    });
    return result;
  }
};

// This creates a GPU metric instance dynamically and adds it to the GPU metric registry, if not
// present already. This is created when there is a device ordinal present and if xgboost
// is compiled with CUDA support
struct GPUMetric : public MetricNoCache {
  static GPUMetric *CreateGPUMetric(const std::string &name, Context const *tparam);
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
