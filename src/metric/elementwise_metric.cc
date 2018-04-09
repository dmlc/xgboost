/*!
 * Copyright 2015 by Contributors
 * \file elementwise_metric.cc
 * \brief evaluation metrics for elementwise binary or regression.
 * \author Kailong Chen, Tianqi Chen
 */
#include <xgboost/metric.h>
#include <dmlc/registry.h>
#include <cmath>
#include "../common/math.h"
#include "../common/sync.h"

namespace xgboost {
namespace metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(elementwise_metric);

/*!
 * \brief base class of element-wise evaluation
 * \tparam Derived the name of subclass
 */
template<typename Derived>
struct EvalEWiseBase : public Metric {
  bst_float Eval(const std::vector<bst_float>& preds,
                 const MetaInfo& info,
                 bool distributed) const override {
    CHECK_NE(info.labels_.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.size(), info.labels_.size())
        << "label and prediction size not match, "
        << "hint: use merror or mlogloss for multi-class classification";
    const auto ndata = static_cast<omp_ulong>(info.labels_.size());
    double sum = 0.0, wsum = 0.0;
    #pragma omp parallel for reduction(+: sum, wsum) schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) {
      const bst_float wt = info.GetWeight(i);
      sum += static_cast<const Derived*>(this)->EvalRow(info.labels_[i], preds[i]) * wt;
      wsum += wt;
    }
    double dat[2]; dat[0] = sum, dat[1] = wsum;
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
   */
  inline bst_float EvalRow(bst_float label, bst_float pred) const;
  /*!
   * \brief to be overridden by subclass, final transformation
   * \param esum the sum statistics returned by EvalRow
   * \param wsum sum of weight
   */
  inline static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return esum / wsum;
  }
};

struct EvalRMSE : public EvalEWiseBase<EvalRMSE> {
  const char *Name() const override {
    return "rmse";
  }
  inline bst_float EvalRow(bst_float label, bst_float pred) const {
    bst_float diff = label - pred;
    return diff * diff;
  }
  inline static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return std::sqrt(esum / wsum);
  }
};

struct EvalMAE : public EvalEWiseBase<EvalMAE> {
  const char *Name() const override {
    return "mae";
  }
  inline bst_float EvalRow(bst_float label, bst_float pred) const {
    return std::abs(label - pred);
  }
};

struct EvalLogLoss : public EvalEWiseBase<EvalLogLoss> {
  const char *Name() const override {
    return "logloss";
  }
  inline bst_float EvalRow(bst_float y, bst_float py) const {
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
};

struct EvalError : public EvalEWiseBase<EvalError> {
  explicit EvalError(const char* param) {
    if (param != nullptr) {
      std::ostringstream os;
      os << "error";
      CHECK_EQ(sscanf(param, "%f", &threshold_), 1)
        << "unable to parse the threshold value for the error metric";
      if (threshold_ != 0.5f) os << '@' << threshold_;
      name_ = os.str();
    } else {
      threshold_ = 0.5f;
      name_ = "error";
    }
  }
  const char *Name() const override {
    return name_.c_str();
  }
  inline bst_float EvalRow(bst_float label, bst_float pred) const {
    // assume label is in [0,1]
    return pred > threshold_ ? 1.0f - label : label;
  }
 protected:
  bst_float threshold_;
  std::string name_;
};

struct EvalPoissonNegLogLik : public EvalEWiseBase<EvalPoissonNegLogLik> {
  const char *Name() const override {
    return "poisson-nloglik";
  }
  inline bst_float EvalRow(bst_float y, bst_float py) const {
    const bst_float eps = 1e-16f;
    if (py < eps) py = eps;
    return common::LogGamma(y + 1.0f) + py - std::log(py) * y;
  }
};

struct EvalGammaDeviance : public EvalEWiseBase<EvalGammaDeviance> {
  const char *Name() const override {
    return "gamma-deviance";
  }
  inline bst_float EvalRow(bst_float label, bst_float pred) const {
    bst_float epsilon = 1.0e-9;
    bst_float tmp = label / (pred + epsilon);
    return tmp - std::log(tmp) - 1;
  }
  inline static bst_float GetFinal(bst_float esum, bst_float wsum) {
    return 2 * esum;
  }
};

struct EvalGammaNLogLik: public EvalEWiseBase<EvalGammaNLogLik> {
  const char *Name() const override {
    return "gamma-nloglik";
  }
  inline bst_float EvalRow(bst_float y, bst_float py) const {
    bst_float psi = 1.0;
    bst_float theta = -1. / py;
    bst_float a = psi;
    bst_float b = -std::log(-theta);
    bst_float c = 1. / psi * std::log(y/psi) - std::log(y) - common::LogGamma(1. / psi);
    return -((y * theta - b) / a + c);
  }
};

struct EvalTweedieNLogLik: public EvalEWiseBase<EvalTweedieNLogLik> {
  explicit EvalTweedieNLogLik(const char* param) {
    CHECK(param != nullptr)
        << "tweedie-nloglik must be in format tweedie-nloglik@rho";
    rho_ = atof(param);
    CHECK(rho_ < 2 && rho_ >= 1)
        << "tweedie variance power must be in interval [1, 2)";
    std::ostringstream os;
    os << "tweedie-nloglik@" << rho_;
    name_ = os.str();
  }
  const char *Name() const override {
    return name_.c_str();
  }
  inline bst_float EvalRow(bst_float y, bst_float p) const {
    bst_float a = y * std::exp((1 - rho_) * std::log(p)) / (1 - rho_);
    bst_float b = std::exp((2 - rho_) * std::log(p)) / (2 - rho_);
    return -a + b;
  }
 protected:
  std::string name_;
  bst_float rho_;
};

XGBOOST_REGISTER_METRIC(RMSE, "rmse")
.describe("Rooted mean square error.")
.set_body([](const char* param) { return new EvalRMSE(); });

XGBOOST_REGISTER_METRIC(MAE, "mae")
.describe("Mean absolute error.")
.set_body([](const char* param) { return new EvalMAE(); });

XGBOOST_REGISTER_METRIC(LogLoss, "logloss")
.describe("Negative loglikelihood for logistic regression.")
.set_body([](const char* param) { return new EvalLogLoss(); });

XGBOOST_REGISTER_METRIC(Error, "error")
.describe("Binary classification error.")
.set_body([](const char* param) { return new EvalError(param); });

XGBOOST_REGISTER_METRIC(PossionNegLoglik, "poisson-nloglik")
.describe("Negative loglikelihood for poisson regression.")
.set_body([](const char* param) { return new EvalPoissonNegLogLik(); });

XGBOOST_REGISTER_METRIC(GammaDeviance, "gamma-deviance")
.describe("Residual deviance for gamma regression.")
.set_body([](const char* param) { return new EvalGammaDeviance(); });

XGBOOST_REGISTER_METRIC(GammaNLogLik, "gamma-nloglik")
.describe("Negative log-likelihood for gamma regression.")
.set_body([](const char* param) { return new EvalGammaNLogLik(); });

XGBOOST_REGISTER_METRIC(TweedieNLogLik, "tweedie-nloglik")
.describe("tweedie-nloglik@rho for tweedie regression.")
.set_body([](const char* param) {
  return new EvalTweedieNLogLik(param);
});

}  // namespace metric
}  // namespace xgboost
