/*!
 * Copyright 2015-2019 by Contributors
 * \file elementwise_metric.cc
 * \brief evaluation metrics for elementwise binary or regression.
 * \author Kailong Chen, Tianqi Chen
 */
#include <rabit/rabit.h>
#include <xgboost/metric.h>
#include <dmlc/registry.h>
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
DMLC_REGISTRY_FILE_TAG(elementwise_metric);

template <typename EvalRow>
class ElementWiseMetricsReduction {
 public:
  explicit ElementWiseMetricsReduction(EvalRow policy) :
#if defined(XGBOOST_USE_CUDA)
    device_(-1),
#endif  // XGBOOST_USE_CUDA
    policy_(std::move(policy)) {}

  PackedReduceResult CpuReduceMetrics(
      const HostDeviceVector<bst_float>& weights,
      const HostDeviceVector<bst_float>& labels,
      const HostDeviceVector<bst_float>& preds) const {
    size_t ndata = labels.Size();

    const auto& h_labels = labels.HostVector();
    const auto& h_weights = weights.HostVector();
    const auto& h_preds = preds.HostVector();

    bst_float residue_sum = 0;
    bst_float weights_sum = 0;

#pragma omp parallel for reduction(+: residue_sum, weights_sum) schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) {
      const bst_float wt = h_weights.size() > 0 ? h_weights[i] : 1.0f;
      residue_sum += policy_.EvalRow(h_labels[i], h_preds[i]) * wt;
      weights_sum += wt;
    }
    PackedReduceResult res { residue_sum, weights_sum };
    return res;
  }

#if defined(XGBOOST_USE_CUDA)

  ~ElementWiseMetricsReduction() {
    if (device_ >= 0) {
      dh::safe_cuda(cudaSetDevice(device_));
      allocator_.Free();
    }
  }

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
        thrust::cuda::par(allocator_),
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
      const GenericParameter &tparam,
      int device,
      const HostDeviceVector<bst_float>& weights,
      const HostDeviceVector<bst_float>& labels,
      const HostDeviceVector<bst_float>& preds) {
    PackedReduceResult result;

    if (device < 0) {
      result = CpuReduceMetrics(weights, labels, preds);
    }
#if defined(XGBOOST_USE_CUDA)
    else {  // NOLINT
      device_ = device;
      if (device_ >= 0) {
        preds.Shard(device_);
        labels.Shard(device_);
        weights.Shard(device_);

        dh::safe_cuda(cudaSetDevice(device_));
        result = DeviceReduceMetrics(device_, 0, weights, labels, preds);
      }
    }
#endif  // defined(XGBOOST_USE_CUDA)
    return result;
  }

 private:
  EvalRow policy_;
#if defined(XGBOOST_USE_CUDA)
  int device_;
  dh::CubMemory allocator_;
#endif  // defined(XGBOOST_USE_CUDA)
};

struct EvalRowRMSE {
  char const *Name() const {
    return "rmse";
  }

  XGBOOST_DEVICE bst_float EvalRow(bst_float label, bst_float pred) const {
    bst_float diff = label - pred;
    return diff * diff;
  }
  static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return std::sqrt(esum / wsum);
  }
};

struct EvalRowRMSLE {
  char const* Name() const {
    return "rmsle";
  }
  XGBOOST_DEVICE bst_float EvalRow(bst_float label, bst_float pred) const {
    bst_float diff = std::log1p(label) - std::log1p(pred);
    return diff * diff;
  }
  static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return std::sqrt(esum / wsum);
  }
};

struct EvalRowMAE {
  const char *Name() const {
    return "mae";
  }

  XGBOOST_DEVICE bst_float EvalRow(bst_float label, bst_float pred) const {
    return std::abs(label - pred);
  }
  static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return esum / wsum;
  }
};

struct EvalRowLogLoss {
  const char *Name() const {
    return "logloss";
  }

  XGBOOST_DEVICE bst_float EvalRow(bst_float y, bst_float py) const {
    const bst_float eps = 1e-16f;
    const bst_float pneg = 1.0f - py;
    if (py < eps) {
      return -y * std::log(eps) - (1.0f - y)  * std::log(1.0f - eps);
    } else if (pneg < eps) {
      return -y * std::log(1.0f - eps) - (1.0f - y)  * std::log(eps);
    } else {
      return -y * std::log(py) - (1.0f - y) * std::log(pneg);
    }
  }

  static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return esum / wsum;
  }
};

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
  const char *Name() const {
    static std::string name;
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

  XGBOOST_DEVICE bst_float EvalRow(
      bst_float label, bst_float pred) const {
    // assume label is in [0,1]
    return pred > threshold_ ? 1.0f - label : label;
  }

  static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return esum / wsum;
  }

 private:
  bst_float threshold_;
  bool has_param_;
};

struct EvalPoissonNegLogLik {
  const char *Name() const {
    return "poisson-nloglik";
  }

  XGBOOST_DEVICE bst_float EvalRow(bst_float y, bst_float py) const {
    const bst_float eps = 1e-16f;
    if (py < eps) py = eps;
    return common::LogGamma(y + 1.0f) + py - std::log(py) * y;
  }

  static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return esum / wsum;
  }
};

struct EvalGammaDeviance {
  const char *Name() const {
    return "gamma-deviance";
  }

  XGBOOST_DEVICE bst_float EvalRow(bst_float label, bst_float pred) const {
    bst_float epsilon = 1.0e-9;
    bst_float tmp = label / (pred + epsilon);
    return tmp - std::log(tmp) - 1;
  }
  static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return 2 * esum;
  }
};

struct EvalGammaNLogLik {
  static const char *Name() {
    return "gamma-nloglik";
  }

  XGBOOST_DEVICE bst_float EvalRow(bst_float y, bst_float py) const {
    bst_float psi = 1.0;
    bst_float theta = -1. / py;
    bst_float a = psi;
    bst_float b = -std::log(-theta);
    bst_float c = 1. / psi * std::log(y/psi) - std::log(y) - common::LogGamma(1. / psi);
    return -((y * theta - b) / a + c);
  }
  static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return esum / wsum;
  }
};

struct EvalTweedieNLogLik {
  explicit EvalTweedieNLogLik(const char* param) {
    CHECK(param != nullptr)
        << "tweedie-nloglik must be in format tweedie-nloglik@rho";
    rho_ = atof(param);
    CHECK(rho_ < 2 && rho_ >= 1)
        << "tweedie variance power must be in interval [1, 2)";
  }
  const char *Name() const {
    static std::string name;
    std::ostringstream os;
    os << "tweedie-nloglik@" << rho_;
    name = os.str();
    return name.c_str();
  }

  XGBOOST_DEVICE bst_float EvalRow(bst_float y, bst_float p) const {
    bst_float a = y * std::exp((1 - rho_) * std::log(p)) / (1 - rho_);
    bst_float b = std::exp((2 - rho_) * std::log(p)) / (2 - rho_);
    return -a + b;
  }
  static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return esum / wsum;
  }

 protected:
  bst_float rho_;
};
/*!
 * \brief base class of element-wise evaluation
 * \tparam Derived the name of subclass
 */
template<typename Policy>
struct EvalEWiseBase : public Metric {
  EvalEWiseBase() : policy_{}, reducer_{policy_} {}
  explicit EvalEWiseBase(char const* policy_param) :
    policy_{policy_param}, reducer_{policy_} {}

  bst_float Eval(const HostDeviceVector<bst_float>& preds,
                 const MetaInfo& info,
                 bool distributed) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size())
        << "label and prediction size not match, "
        << "hint: use merror or mlogloss for multi-class classification";
    const auto ndata = static_cast<omp_ulong>(info.labels_.Size());
    // Dealing with ndata < n_gpus.
    int device = tparam_->gpu_id;

    auto result =
        reducer_.Reduce(*tparam_, device, info.weights_, info.labels_, preds);

    double dat[2] { result.Residue(), result.Weights() };
    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    return Policy::GetFinal(dat[0], dat[1]);
  }

  const char* Name() const override {
    return policy_.Name();
  }

 private:
  Policy policy_;

  ElementWiseMetricsReduction<Policy> reducer_;
};

XGBOOST_REGISTER_METRIC(RMSE, "rmse")
.describe("Rooted mean square error.")
.set_body([](const char* param) { return new EvalEWiseBase<EvalRowRMSE>(); });

XGBOOST_REGISTER_METRIC(RMSLE, "rmsle")
.describe("Rooted mean square log error.")
.set_body([](const char* param) { return new EvalEWiseBase<EvalRowRMSLE>(); });

XGBOOST_REGISTER_METRIC(MAE, "mae")
.describe("Mean absolute error.")
.set_body([](const char* param) { return new EvalEWiseBase<EvalRowMAE>(); });

XGBOOST_REGISTER_METRIC(LogLoss, "logloss")
.describe("Negative loglikelihood for logistic regression.")
.set_body([](const char* param) { return new EvalEWiseBase<EvalRowLogLoss>(); });

XGBOOST_REGISTER_METRIC(PossionNegLoglik, "poisson-nloglik")
.describe("Negative loglikelihood for poisson regression.")
.set_body([](const char* param) { return new EvalEWiseBase<EvalPoissonNegLogLik>(); });

XGBOOST_REGISTER_METRIC(GammaDeviance, "gamma-deviance")
.describe("Residual deviance for gamma regression.")
.set_body([](const char* param) { return new EvalEWiseBase<EvalGammaDeviance>(); });

XGBOOST_REGISTER_METRIC(GammaNLogLik, "gamma-nloglik")
.describe("Negative log-likelihood for gamma regression.")
.set_body([](const char* param) { return new EvalEWiseBase<EvalGammaNLogLik>(); });

XGBOOST_REGISTER_METRIC(Error, "error")
.describe("Binary classification error.")
.set_body([](const char* param) { return new EvalEWiseBase<EvalError>(param); });

XGBOOST_REGISTER_METRIC(TweedieNLogLik, "tweedie-nloglik")
.describe("tweedie-nloglik@rho for tweedie regression.")
.set_body([](const char* param) {
  return new EvalEWiseBase<EvalTweedieNLogLik>(param);
});

}  // namespace metric
}  // namespace xgboost
