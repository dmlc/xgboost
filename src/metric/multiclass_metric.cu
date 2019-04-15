/*!
 * Copyright 2015-2019 by Contributors
 * \file multiclass_metric.cc
 * \brief evaluation metrics for multiclass classification.
 * \author Kailong Chen, Tianqi Chen
 */
#include <rabit/rabit.h>
#include <xgboost/metric.h>
#include <cmath>

#include "metric_common.h"
#include "../common/math.h"
#include "../common/common.h"

#if defined(XGBOOST_USE_CUDA)
#include <thrust/execution_policy.h>  // thrust::cuda::par
#include <thrust/functional.h>        // thrust::plus<>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

#include "../common/device_helpers.cuh"
#endif  // XGBOOST_USE_CUDA

namespace xgboost {
namespace metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(multiclass_metric);

template <typename EvalRowPolicy>
class MultiClassMetricsReduction {
 public:
  MultiClassMetricsReduction() = default;

  PackedReduceResult CpuReduceMetrics(
      const HostDeviceVector<bst_float>& weights,
      const HostDeviceVector<bst_float>& labels,
      const HostDeviceVector<bst_float>& preds,
      const size_t n_class) const {
    size_t ndata = labels.Size();

    const auto& h_labels = labels.HostVector();
    const auto& h_weights = weights.HostVector();
    const auto& h_preds = preds.HostVector();

    bst_float residue_sum = 0;
    bst_float weights_sum = 0;
    int label_error = 0;
    bool const is_null_weight = weights.Size() == 0;

#pragma omp parallel for reduction(+: residue_sum, weights_sum) schedule(static)
    for (omp_ulong idx = 0; idx < ndata; ++idx) {
      bst_float weight = is_null_weight ? 1.0f : h_weights[idx];
      auto label = static_cast<int>(h_labels[idx]);
      if (label >= 0 && label < static_cast<int>(n_class)) {
        residue_sum += EvalRowPolicy::EvalRow(
            label, h_preds.data() + idx * n_class, n_class) * weight;
        weights_sum += weight;
      } else {
        label_error = label;
      }
    }
    CHECK(label_error >= 0 && label_error < static_cast<int>(n_class))
        << "MultiClassEvaluation: label must be in [0, num_class),"
        << " num_class=" << n_class << " but found " << label_error << " in label";
    PackedReduceResult res { residue_sum, weights_sum };
    return res;
  }

#if defined(XGBOOST_USE_CUDA)

  PackedReduceResult DeviceReduceMetrics(
      GPUSet::GpuIdType device_id,
      size_t device_index,
      const HostDeviceVector<bst_float>& weights,
      const HostDeviceVector<bst_float>& labels,
      const HostDeviceVector<bst_float>& preds,
      const size_t n_class) {
    size_t n_data = labels.DeviceSize(device_id);

    thrust::counting_iterator<size_t> begin(0);
    thrust::counting_iterator<size_t> end = begin + n_data;

    auto s_labels = labels.DeviceSpan(device_id);
    auto s_preds = preds.DeviceSpan(device_id);
    auto s_weights = weights.DeviceSpan(device_id);

    bool const is_null_weight = weights.Size() == 0;

    PackedReduceResult result = thrust::transform_reduce(
        thrust::cuda::par(allocators_.at(device_index)),
        begin, end,
        [=] XGBOOST_DEVICE(size_t idx) {
          bst_float weight = is_null_weight ? 1.0f : s_weights[idx];
          bst_float residue = 0;
          auto label = static_cast<int>(s_labels[idx]);
          if (label >= 0 && label < static_cast<int>(n_class)) {
            residue = EvalRowPolicy::EvalRow(
                label, &s_preds[idx * n_class], n_class) * weight;
          }
          return PackedReduceResult{ residue, weight };
        },
        PackedReduceResult(),
        thrust::plus<PackedReduceResult>());

    return result;
  }

#endif  // XGBOOST_USE_CUDA

  PackedReduceResult Reduce(
      GPUSet devices,
      size_t n_class,
      const HostDeviceVector<bst_float>& weights,
      const HostDeviceVector<bst_float>& labels,
      const HostDeviceVector<bst_float>& preds) {
    PackedReduceResult result;

    if (devices.IsEmpty()) {
      result = CpuReduceMetrics(weights, labels, preds, n_class);
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
        res_per_device.at(index) =
            DeviceReduceMetrics(id, index, weights, labels, preds, n_class);
      }

      for (auto const& res : res_per_device) {
        result += res;
      }
    }
#endif  // defined(XGBOOST_USE_CUDA)
    return result;
  }

 private:
#if defined(XGBOOST_USE_CUDA)
  std::vector<dh::CubMemory> allocators_;
#endif  // defined(XGBOOST_USE_CUDA)
};

/*!
 * \brief base class of multi-class evaluation
 * \tparam Derived the name of subclass
 */
template<typename Derived>
struct EvalMClassBase : public Metric {
  void Configure(
      const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }

  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK(preds.Size() % info.labels_.Size() == 0)
        << "label and prediction size not match";
    const size_t nclass = preds.Size() / info.labels_.Size();
    CHECK_GE(nclass, 1U)
        << "mlogloss and merror are only used for multi-class classification,"
        << " use logloss for binary classification";
    const auto ndata = static_cast<bst_omp_uint>(info.labels_.Size());

    GPUSet devices = GPUSet::All(param_.gpu_id, param_.n_gpus, ndata);
    auto result = reducer_.Reduce(devices, nclass, info.weights_, info.labels_, preds);
    double dat[2] { result.Residue(), result.Weights() };

    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    return Derived::GetFinal(dat[0], dat[1]);
  }
  /*!
   * \brief to be implemented by subclass,
   *   get evaluation result from one row
   * \param label label of current instance
   * \param pred prediction value of current instance
   * \param nclass number of class in the prediction
   */
  XGBOOST_DEVICE static bst_float EvalRow(int label,
                                          const bst_float *pred,
                                          size_t nclass);
  /*!
   * \brief to be overridden by subclass, final transformation
   * \param esum the sum statistics returned by EvalRow
   * \param wsum sum of weight
   */
  inline static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return esum / wsum;
  }

 private:
  MultiClassMetricsReduction<Derived> reducer_;
  MetricParam param_;
  // used to store error message
  const char *error_msg_;
};

/*! \brief match error */
struct EvalMatchError : public EvalMClassBase<EvalMatchError> {
  const char* Name() const override {
    return "merror";
  }
  XGBOOST_DEVICE static bst_float EvalRow(int label,
                                          const bst_float *pred,
                                          size_t nclass) {
    return common::FindMaxIndex(pred, pred + nclass) != pred + static_cast<int>(label);
  }
};

/*! \brief match error */
struct EvalMultiLogLoss : public EvalMClassBase<EvalMultiLogLoss> {
  const char* Name() const override {
    return "mlogloss";
  }
  XGBOOST_DEVICE static bst_float EvalRow(int label,
                                          const bst_float *pred,
                                          size_t nclass) {
    const bst_float eps = 1e-16f;
    auto k = static_cast<size_t>(label);
    if (pred[k] > eps) {
      return -std::log(pred[k]);
    } else {
      return -std::log(eps);
    }
  }
};

XGBOOST_REGISTER_METRIC(MatchError, "merror")
.describe("Multiclass classification error.")
.set_body([](const char* param) { return new EvalMatchError(); });

XGBOOST_REGISTER_METRIC(MultiLogLoss, "mlogloss")
.describe("Multiclass negative loglikelihood.")
.set_body([](const char* param) { return new EvalMultiLogLoss(); });
}  // namespace metric
}  // namespace xgboost
