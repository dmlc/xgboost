/*!
 * Copyright 2015-2022 by XGBoost Contributors
 * \file elementwise_metric.cc
 * \brief evaluation metrics for elementwise binary or regression.
 * \author Kailong Chen, Tianqi Chen
 *
 *  The expressions like wsum == 0 ? esum : esum / wsum is used to handle empty dataset.
 */
#include <dmlc/registry.h>
#include <xgboost/metric.h>

#include <cmath>

#include "../collective/communicator-inl.h"
#include "../common/common.h"
#include "../common/math.h"
#include "../common/pseudo_huber.h"
#include "../common/threading_utils.h"
#include "metric_common.h"

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

namespace {
/**
 * \brief Reduce function for element wise metrics.
 *
 *   The loss function should handle all the computation for each sample, including
 *   applying the weights.  A tuple of {error_i, weight_i} is expected as return.
 */
template <typename Fn>
PackedReduceResult Reduce(GenericParameter const* ctx, MetaInfo const& info, Fn&& loss) {
  PackedReduceResult result;
  auto labels = info.labels.View(ctx->gpu_id);
  if (ctx->IsCPU()) {
    auto n_threads = ctx->Threads();
    std::vector<double> score_tloc(n_threads, 0.0);
    std::vector<double> weight_tloc(n_threads, 0.0);
    // We sum over losses over all samples and targets instead of performing this for each
    // target since the first one approach more accurate while the second approach is used
    // for approximation in distributed setting.  For rmse:
    // - sqrt(1/w(sum_t0 + sum_t1 + ... + sum_tm))       // multi-target
    // - sqrt(avg_t0) + sqrt(avg_t1) + ... sqrt(avg_tm)  // distributed
    common::ParallelFor(info.labels.Size(), ctx->Threads(), [&](size_t i) {
      auto t_idx = omp_get_thread_num();
      size_t sample_id;
      size_t target_id;
      std::tie(sample_id, target_id) = linalg::UnravelIndex(i, labels.Shape());

      float v, wt;
      std::tie(v, wt) = loss(i, sample_id, target_id);
      score_tloc[t_idx] += v;
      weight_tloc[t_idx] += wt;
    });
    double residue_sum = std::accumulate(score_tloc.cbegin(), score_tloc.cend(), 0.0);
    double weights_sum = std::accumulate(weight_tloc.cbegin(), weight_tloc.cend(), 0.0);
    result = PackedReduceResult{residue_sum, weights_sum};
  } else {
#if defined(XGBOOST_USE_CUDA)
    dh::XGBCachingDeviceAllocator<char> alloc;
    thrust::counting_iterator<size_t> begin(0);
    thrust::counting_iterator<size_t> end = begin + labels.Size();
    result = thrust::transform_reduce(
        thrust::cuda::par(alloc), begin, end,
        [=] XGBOOST_DEVICE(size_t i) {
          auto idx = linalg::UnravelIndex(i, labels.Shape());
          auto sample_id = std::get<0>(idx);
          auto target_id = std::get<1>(idx);
          auto res = loss(i, sample_id, target_id);
          float v{std::get<0>(res)}, wt{std::get<1>(res)};
          return PackedReduceResult{v, wt};
        },
        PackedReduceResult{}, thrust::plus<PackedReduceResult>());
#else
    common::AssertGPUSupport();
#endif  //  defined(XGBOOST_USE_CUDA)
  }
  return result;
}
}  // anonymous namespace

struct EvalRowRMSE {
  char const *Name() const {
    return "rmse";
  }

  XGBOOST_DEVICE bst_float EvalRow(bst_float label, bst_float pred) const {
    bst_float diff = label - pred;
    return diff * diff;
  }
  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? std::sqrt(esum) : std::sqrt(esum / wsum);
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
  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? std::sqrt(esum) : std::sqrt(esum / wsum);
  }
};

struct EvalRowMAE {
  const char *Name() const {
    return "mae";
  }

  XGBOOST_DEVICE bst_float EvalRow(bst_float label, bst_float pred) const {
    return std::abs(label - pred);
  }
  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? esum : esum / wsum;
  }
};

struct EvalRowMAPE {
  const char *Name() const {
    return "mape";
  }
  XGBOOST_DEVICE bst_float EvalRow(bst_float label, bst_float pred) const {
    return std::abs((label - pred) / label);
  }
  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? esum : esum / wsum;
  }
};

namespace {
XGBOOST_DEVICE inline float LogLoss(float y, float py) {
  auto xlogy = [](float x, float y) {
    float eps = 1e-16;
    return (x - 0.0f == 0.0f) ? 0.0f : (x * std::log(std::max(y, eps)));
  };
  const bst_float pneg = 1.0f - py;
  return xlogy(-y, py) + xlogy(-(1.0f - y), pneg);
}
}  // anonymous namespace

struct EvalRowLogLoss {
  const char *Name() const {
    return "logloss";
  }

  XGBOOST_DEVICE bst_float EvalRow(bst_float y, bst_float py) const { return LogLoss(y, py); }
  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? esum : esum / wsum;
  }
};

class PseudoErrorLoss : public Metric {
  PesudoHuberParam param_;

 public:
  const char* Name() const override { return "mphe"; }
  void Configure(Args const& args) override { param_.UpdateAllowUnknown(args); }
  void LoadConfig(Json const& in) override { FromJson(in["pseudo_huber_param"], &param_); }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(this->Name());
    out["pseudo_huber_param"] = ToJson(param_);
  }

  double Eval(const HostDeviceVector<bst_float>& preds, const MetaInfo& info) override {
    CHECK_EQ(info.labels.Shape(0), info.num_row_);
    auto labels = info.labels.View(tparam_->gpu_id);
    preds.SetDevice(tparam_->gpu_id);
    auto predts = tparam_->IsCPU() ? preds.ConstHostSpan() : preds.ConstDeviceSpan();
    info.weights_.SetDevice(tparam_->gpu_id);
    common::OptionalWeights weights(tparam_->IsCPU() ? info.weights_.ConstHostSpan()
                                                     : info.weights_.ConstDeviceSpan());
    float slope = this->param_.huber_slope;
    CHECK_NE(slope, 0.0) << "slope for pseudo huber cannot be 0.";
    PackedReduceResult result =
        Reduce(tparam_, info, [=] XGBOOST_DEVICE(size_t i, size_t sample_id, size_t target_id) {
          float wt = weights[sample_id];
          auto a = labels(sample_id, target_id) - predts[i];
          auto v = common::Sqr(slope) * (std::sqrt((1 + common::Sqr(a / slope))) - 1) * wt;
          return std::make_tuple(v, wt);
        });
    double dat[2]{result.Residue(), result.Weights()};
    if (collective::IsDistributed()) {
      collective::Allreduce<collective::Operation::kSum>(dat, 2);
    }
    return EvalRowMAPE::GetFinal(dat[0], dat[1]);
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

  XGBOOST_DEVICE bst_float EvalRow(bst_float label, bst_float pred) const {
    // assume label is in [0,1]
    return pred > threshold_ ? 1.0f - label : label;
  }

  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? esum : esum / wsum;
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

  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? esum : esum / wsum;
  }
};

/**
 * Gamma deviance
 *
 *   Expected input:
 *   label >= 0
 *   predt >= 0
 */
struct EvalGammaDeviance {
  const char *Name() const { return "gamma-deviance"; }

  XGBOOST_DEVICE bst_float EvalRow(bst_float label, bst_float predt) const {
    predt += kRtEps;
    label += kRtEps;
    return std::log(predt / label) + label / predt - 1;
  }

  static double GetFinal(double esum, double wsum) {
    if (wsum <= 0) {
      wsum = kRtEps;
    }
    return 2 * esum / wsum;
  }
};

struct EvalGammaNLogLik {
  static const char *Name() {
    return "gamma-nloglik";
  }

  XGBOOST_DEVICE bst_float EvalRow(bst_float y, bst_float py) const {
    py = std::max(py, 1e-6f);
    // hardcoded dispersion.
    float constexpr kPsi = 1.0;
    bst_float theta = -1. / py;
    bst_float a = kPsi;
    float b = -std::log(-theta);
    // c = 1. / kPsi^2 * std::log(y/kPsi) - std::log(y) - common::LogGamma(1. / kPsi);
    //   = 1.0f        * std::log(y)      - std::log(y) - 0 = 0
    float c = 0;
    // general form for exponential family.
    return -((y * theta - b) / a + c);
  }
  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? esum : esum / wsum;
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
  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? esum : esum / wsum;
  }

 protected:
  bst_float rho_;
};
/*!
 * \brief base class of element-wise evaluation
 * \tparam Derived the name of subclass
 */
template <typename Policy>
struct EvalEWiseBase : public Metric {
  EvalEWiseBase() = default;
  explicit EvalEWiseBase(char const* policy_param) : policy_{policy_param} {}

  double Eval(HostDeviceVector<bst_float> const& preds, const MetaInfo& info) override {
    CHECK_EQ(preds.Size(), info.labels.Size())
        << "label and prediction size not match, "
        << "hint: use merror or mlogloss for multi-class classification";
    if (info.labels.Size() != 0) {
      CHECK_NE(info.labels.Shape(1), 0);
    }
    auto labels = info.labels.View(tparam_->gpu_id);
    info.weights_.SetDevice(tparam_->gpu_id);
    common::OptionalWeights weights(tparam_->IsCPU() ? info.weights_.ConstHostSpan()
                                                     : info.weights_.ConstDeviceSpan());
    preds.SetDevice(tparam_->gpu_id);
    auto predts = tparam_->IsCPU() ? preds.ConstHostSpan() : preds.ConstDeviceSpan();

    auto d_policy = policy_;
    auto result =
        Reduce(tparam_, info, [=] XGBOOST_DEVICE(size_t i, size_t sample_id, size_t target_id) {
          float wt = weights[sample_id];
          float residue = d_policy.EvalRow(labels(sample_id, target_id), predts[i]);
          residue *= wt;
          return std::make_tuple(residue, wt);
        });

    double dat[2]{result.Residue(), result.Weights()};
    collective::Allreduce<collective::Operation::kSum>(dat, 2);
    return Policy::GetFinal(dat[0], dat[1]);
  }

  const char* Name() const override { return policy_.Name(); }

 private:
  Policy policy_;
};

XGBOOST_REGISTER_METRIC(RMSE, "rmse")
    .describe("Rooted mean square error.")
    .set_body([](const char*) { return new EvalEWiseBase<EvalRowRMSE>(); });

XGBOOST_REGISTER_METRIC(RMSLE, "rmsle")
    .describe("Rooted mean square log error.")
    .set_body([](const char*) { return new EvalEWiseBase<EvalRowRMSLE>(); });

XGBOOST_REGISTER_METRIC(MAE, "mae").describe("Mean absolute error.").set_body([](const char*) {
  return new EvalEWiseBase<EvalRowMAE>();
});

XGBOOST_REGISTER_METRIC(MAPE, "mape")
    .describe("Mean absolute percentage error.")
    .set_body([](const char*) { return new EvalEWiseBase<EvalRowMAPE>(); });

XGBOOST_REGISTER_METRIC(LogLoss, "logloss")
    .describe("Negative loglikelihood for logistic regression.")
    .set_body([](const char*) { return new EvalEWiseBase<EvalRowLogLoss>(); });

XGBOOST_REGISTER_METRIC(PseudoErrorLoss, "mphe")
    .describe("Mean Pseudo-huber error.")
    .set_body([](const char*) { return new PseudoErrorLoss{}; });

XGBOOST_REGISTER_METRIC(PossionNegLoglik, "poisson-nloglik")
    .describe("Negative loglikelihood for poisson regression.")
    .set_body([](const char*) { return new EvalEWiseBase<EvalPoissonNegLogLik>(); });

XGBOOST_REGISTER_METRIC(GammaDeviance, "gamma-deviance")
    .describe("Residual deviance for gamma regression.")
    .set_body([](const char*) { return new EvalEWiseBase<EvalGammaDeviance>(); });

XGBOOST_REGISTER_METRIC(GammaNLogLik, "gamma-nloglik")
    .describe("Negative log-likelihood for gamma regression.")
    .set_body([](const char*) { return new EvalEWiseBase<EvalGammaNLogLik>(); });

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
