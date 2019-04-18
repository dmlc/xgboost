/*
 * Copyright 2015-2019 by Contributors
 */

#ifndef XGBOOST_METRIC_ELEMENTWISE_METRIC_H_
#define XGBOOST_METRIC_ELEMENTWISE_METRIC_H_

#include <xgboost/metric/metric.h>
#include <xgboost/metric/metric_common.h>

#include <functional>
#include <utility>
#include <string>
#include <vector>

#include "../../../src/common/common.h"

#if defined(XGBOOST_USE_CUDA)
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>  // thrust::plus<>

#include "../src/common/device_helpers.cuh"
#endif  // XGBOOST_USE_CUDA

/*!
 * \brief base class of element-wise evaluation
 * \tparam Derived the name of subclass
 */
namespace xgboost {
namespace metric {

template<typename EvalRow>
class MetricsReduction {
 public:
  class PackedReduceResult {
    double residue_sum_;
    double weights_sum_;
    friend MetricsReduction;

   public:
    XGBOOST_DEVICE PackedReduceResult() : residue_sum_{0}, weights_sum_{0} {}

    XGBOOST_DEVICE PackedReduceResult(double residue, double weight) :
            residue_sum_{residue}, weights_sum_{weight} {}

    XGBOOST_DEVICE
    PackedReduceResult operator+(PackedReduceResult const &other) const {
      return PackedReduceResult{residue_sum_ + other.residue_sum_,
                                weights_sum_ + other.weights_sum_};
    }

    double Residue() const { return residue_sum_; }

    double Weights() const { return weights_sum_; }
  };

 public:
  explicit MetricsReduction(EvalRow policy) :
          policy_(std::move(policy)) {}

  PackedReduceResult CpuReduceMetrics(
          const HostDeviceVector <bst_float> &weights,
          const HostDeviceVector <bst_float> &labels,
          const HostDeviceVector <bst_float> &preds) const {
    size_t ndata = labels.Size();

    const auto &h_labels = labels.HostVector();
    const auto &h_weights = weights.HostVector();
    const auto &h_preds = preds.HostVector();

    bst_float residue_sum = 0;
    bst_float weights_sum = 0;

#pragma omp parallel for reduction(+: residue_sum, weights_sum) schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) {
     const bst_float wt = h_weights.size() > 0 ? h_weights[i] : 1.0f;
     residue_sum += policy_.EvalRow(h_labels[i], h_preds[i]) * wt;
     weights_sum += wt;
    }
    PackedReduceResult res{residue_sum, weights_sum};
    return res;
  }

#if defined(XGBOOST_USE_CUDA)

  PackedReduceResult DeviceReduceMetrics(
     GPUSet::GpuIdType device_id,
     size_t device_index,
     const HostDeviceVector<bst_float>& weights,
     const HostDeviceVector<bst_float>& labels,
     const HostDeviceVector<bst_float>& preds) {
    size_t n_data = preds.DeviceSize(device_id);

    thrust::counting_iterator<size_t> begin(0);
    thrust::counting_iterator<size_t> end = begin + n_data;

    auto s_label = labels.DeviceSpan(device_id);
    auto s_preds = preds.DeviceSpan(device_id);
    auto s_weights = weights.DeviceSpan(device_id);

    bool const is_null_weight = weights.Size() == 0;

    auto d_policy = policy_;

    PackedReduceResult result = thrust::transform_reduce(
    thrust::cuda::par(allocators_.at(device_index)),
    begin, end,
    [=] XGBOOST_DEVICE(size_t idx) {
    bst_float weight = is_null_weight ? 1.0f : s_weights[idx];

    bst_float residue = d_policy.EvalRow(s_label[idx], s_preds[idx]);
    residue *= weight;
    return PackedReduceResult{ residue, weight };
    },
    PackedReduceResult(),
    thrust::plus<PackedReduceResult>());

    return result;
  }

#endif  // XGBOOST_USE_CUDA

  PackedReduceResult Reduce(
      GPUSet devices,
      const HostDeviceVector <bst_float> &weights,
      const HostDeviceVector <bst_float> &labels,
      const HostDeviceVector <bst_float> &preds) {
    PackedReduceResult result;

    if (devices.IsEmpty()) {
     result = CpuReduceMetrics(weights, labels, preds);
    }
#if defined(XGBOOST_USE_CUDA)
    else {  // NOLINT
      if (allocators_.size() != devices.Size()) {
       allocators_.clear();
       allocators_.resize(devices.Size());
      }
      preds.Reshard(devices);
      labels.Reshard(devices);
      weights.Reshard(devices);
      std::vector<PackedReduceResult> res_per_device(devices.Size());

#pragma omp parallel for schedule(static, 1) if (devices.Size() > 1)
      for (GPUSet::GpuIdType id = *devices.begin(); id < *devices.end(); ++id) {
        dh::safe_cuda(cudaSetDevice(id));
        size_t index = devices.Index(id);
        res_per_device.at(index) = DeviceReduceMetrics(id, index, weights, labels, preds);
      }

      for (size_t i = 0; i < devices.Size(); ++i) {
       result.residue_sum_ += res_per_device[i].residue_sum_;
       result.weights_sum_ += res_per_device[i].weights_sum_;
      }
    }
#endif  // defined(XGBOOST_USE_CUDA)
    return result;
  }

 private:
  EvalRow policy_;
#if defined(XGBOOST_USE_CUDA)
  std::vector<dh::CubMemory> allocators_;
#endif  // defined(XGBOOST_USE_CUDA)
};

template<typename Policy>
struct EvalEWiseBase : public Metric {
  EvalEWiseBase() : policy_{}, reducer_{policy_} {}

  explicit EvalEWiseBase(Policy &policy) : policy_{policy}, reducer_{policy_} {}

  explicit EvalEWiseBase(char const *policy_param) :
          policy_{policy_param}, reducer_{policy_} {}

  void Configure(
          const std::vector<std::pair<std::string, std::string>> &args) override {
    param_.InitAllowUnknown(args);
  }

  bst_float Eval(const HostDeviceVector <bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size())
      << "label and prediction size not match, "
      << "hint: use merror or mlogloss for multi-class classification";
    const auto ndata = static_cast<omp_ulong>(info.labels_.Size());
    // Dealing with ndata < n_gpus.
    GPUSet devices = GPUSet::All(param_.gpu_id, param_.n_gpus, ndata);

    auto result =
            reducer_.Reduce(devices, info.weights_, info.labels_, preds);

    double dat[2]{result.Residue(), result.Weights()};
    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    return Policy::GetFinal(dat[0], dat[1]);
  }

  const char *Name() const override {
    return policy_.Name();
  }

 private:
  Policy policy_;

  MetricParam param_;

  MetricsReduction<Policy> reducer_;
};

}  // namespace metric
}  // namespace xgboost
#endif  // XGBOOST_METRIC_ELEMENTWISE_METRIC_H_
