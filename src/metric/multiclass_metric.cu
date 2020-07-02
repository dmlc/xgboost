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
  void CheckLabelError(int32_t label_error, size_t n_class) const {
    CHECK(label_error >= 0 && label_error < static_cast<int32_t>(n_class))
        << "MultiClassEvaluation: label must be in [0, num_class),"
        << " num_class=" << n_class << " but found " << label_error << " in label";
  }

 public:
  MultiClassMetricsReduction() = default;

  PackedReduceResult CpuReduceMetrics(
      const HostDeviceVector<bst_float>& weights,
      const HostDeviceVector<bst_float>& labels,
      const HostDeviceVector<bst_float>& preds,
      size_t ndata,
      const size_t n_class) const {
    const auto h_labels = labels.HostSpan();
    const auto h_weights = weights.HostSpan();
    const auto h_preds = preds.HostSpan();

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
            h_labels, h_preds, idx, n_class) * weight;
        weights_sum += weight;
      } else {
        label_error = label;
      }
    }
    CheckLabelError(label_error, n_class);
    PackedReduceResult res { residue_sum, weights_sum };

    return res;
  }

#if defined(XGBOOST_USE_CUDA)

  PackedReduceResult DeviceReduceMetrics(
      const HostDeviceVector<bst_float>& weights,
      const HostDeviceVector<bst_float>& labels,
      const HostDeviceVector<bst_float>& preds,
      size_t const n_data,
      const size_t n_class) {
    thrust::counting_iterator<size_t> begin(0);
    thrust::counting_iterator<size_t> end = begin + n_data;

    auto s_labels = labels.DeviceSpan();
    auto s_preds = preds.DeviceSpan();
    auto s_weights = weights.DeviceSpan();

    bool const is_null_weight = weights.Size() == 0;
    auto s_label_error = label_error_.GetSpan<int32_t>(1);
    s_label_error[0] = 0;

    dh::XGBCachingDeviceAllocator<char> alloc;
    PackedReduceResult result = thrust::transform_reduce(
        thrust::cuda::par(alloc),
        begin, end,
        [=] XGBOOST_DEVICE(size_t idx) {
          bst_float weight = is_null_weight ? 1.0f : s_weights[idx];
          bst_float residue = 0;
          auto label = static_cast<int>(s_labels[idx]);
          if (label >= 0 && label < static_cast<int32_t>(n_class)) {
            residue = EvalRowPolicy::EvalRow(
                s_labels, s_preds, idx, n_class) * weight;
          } else {
            s_label_error[0] = label;
          }
          return PackedReduceResult{ residue, weight };
        },
        PackedReduceResult(),
        thrust::plus<PackedReduceResult>());
    CheckLabelError(s_label_error[0], n_class);

    return result;
  }

#endif  // XGBOOST_USE_CUDA

  PackedReduceResult Reduce(
      const GenericParameter &tparam,
      int device,
      size_t n_class,
      MetaInfo const& info,
      const HostDeviceVector<bst_float>& preds) {
    PackedReduceResult result;
    auto const& labels = info.labels_;
    auto const& weights = info.weights_;

    if (device < 0) {
      result = CpuReduceMetrics(weights, labels, preds, info.num_row_, n_class);
    }
#if defined(XGBOOST_USE_CUDA)
    else {  // NOLINT
      device_ = tparam.gpu_id;
      preds.SetDevice(device_);
      labels.SetDevice(device_);
      weights.SetDevice(device_);

      dh::safe_cuda(cudaSetDevice(device_));
      result = DeviceReduceMetrics(weights, labels, preds, info.num_row_, n_class);
    }
#endif  // defined(XGBOOST_USE_CUDA)
    return result;
  }

 private:
#if defined(XGBOOST_USE_CUDA)
  dh::PinnedMemory label_error_;
  int device_{-1};
#endif  // defined(XGBOOST_USE_CUDA)
};

/*!
 * \brief base class of multi-class evaluation
 * \tparam Derived the name of subclass
 */
template<typename Derived>
struct EvalMClassBase : public Metric {
  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK(preds.Size() % info.labels_.Size() == 0)
        << "label and prediction size not match";
    const size_t nclass = preds.Size() / info.num_row_;
    CHECK_GE(nclass, 1U)
        << "mlogloss and merror are only used for multi-class classification,"
        << " use logloss for binary classification";

    int device = tparam_->gpu_id;
    auto result = reducer_.Reduce(*tparam_, device, nclass, info, preds);
    double dat[2] { result.Residue(), result.Weights() };

    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    return Derived::GetFinal(dat[0], dat[1]);
  }
  /*!
   * \brief to be implemented by subclass,
   *   get evaluation result from one row
   * \param s_labels label of current instance
   * \param s_predt prediction value of current instance
   * \param idx index of current instance
   * \param nclass number of class in the prediction
   */
  XGBOOST_DEVICE static bst_float EvalRow(common::Span<float const> s_label,
                                          common::Span<float const> s_predt,
                                          size_t idx,
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
  // used to store error message
  const char *error_msg_;
};

/*! \brief match error */
struct EvalMatchError : public EvalMClassBase<EvalMatchError> {
  const char* Name() const override {
    return "merror";
  }
  XGBOOST_DEVICE static bst_float EvalRow(common::Span<float const> s_label,
                                          common::Span<float const> s_predt,
                                          size_t idx,
                                          size_t nclass) {
    auto pred = s_predt.subspan(idx * nclass, nclass);
    auto label = static_cast<int32_t>(s_label[idx]);
    return
        common::FindMaxIndex(pred.begin(), pred.begin() + nclass) !=
        pred.begin() + static_cast<int>(label);
  }
};

/*! \brief match error */
struct EvalMultiLogLoss : public EvalMClassBase<EvalMultiLogLoss> {
  const char* Name() const override {
    return "mlogloss";
  }
  XGBOOST_DEVICE static bst_float EvalRow(common::Span<float const> s_label,
                                          common::Span<float const> s_predt,
                                          size_t idx,
                                          size_t nclass) {
    auto pred = s_predt.subspan(idx * nclass, nclass);
    auto label = static_cast<int32_t>(s_label[idx]);
    const bst_float eps = 1e-16f;
    if (pred[label] > eps) {
      return -std::log(pred[label]);
    } else {
      return -std::log(eps);
    }
  }
};

struct EvalMultiLogLossOneHot : public EvalMClassBase<EvalMultiLogLossOneHot> {
  const char* Name() const override {
    return "mtlogloss";
  }
  XGBOOST_DEVICE static bst_float EvalRow(common::Span<float const> s_label,
                                          common::Span<float const> s_predt,
                                          size_t idx,
                                          size_t nclass) {
    auto predt = s_predt.subspan(idx * nclass, nclass);
    auto label = s_label.subspan(idx * nclass, nclass);
    size_t k = 0;
    for (; k < nclass; ++k) {
      if (label[k] == 1) {
        break;
      }
    }
    float ret { 0 };
    if (predt[k] > kRtEps) {
      ret = -std::log(predt[k]);
    } else {
      ret = -std::log(kRtEps);
    }
    return ret;
  }
};

XGBOOST_REGISTER_METRIC(MatchError, "merror")
.describe("Multiclass classification error.")
.set_body([](const char* param) { return new EvalMatchError(); });

XGBOOST_REGISTER_METRIC(MultiLogLoss, "mlogloss")
.describe("Multiclass negative loglikelihood.")
.set_body([](const char* param) { return new EvalMultiLogLoss(); });

XGBOOST_REGISTER_METRIC(MultiLogLossOneHot, "mtlogloss")
.describe("Multiclass negative loglikelihood.")
.set_body([](const char* param) { return new EvalMultiLogLossOneHot(); });
}  // namespace metric
}  // namespace xgboost
