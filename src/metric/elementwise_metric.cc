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
  float Eval(const std::vector<float>& preds,
             const MetaInfo& info,
             bool distributed) const override {
    CHECK_NE(info.labels.size(), 0) << "label set cannot be empty";
    CHECK_EQ(preds.size(), info.labels.size())
        << "label and prediction size not match, "
        << "hint: use merror or mlogloss for multi-class classification";
    const omp_ulong ndata = static_cast<omp_ulong>(info.labels.size());
    double sum = 0.0, wsum = 0.0;
    #pragma omp parallel for reduction(+: sum, wsum) schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) {
      const float wt = info.GetWeight(i);
      sum += static_cast<const Derived*>(this)->EvalRow(info.labels[i], preds[i]) * wt;
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
  inline float EvalRow(float label, float pred) const;
  /*!
   * \brief to be overridden by subclass, final transformation
   * \param esum the sum statistics returned by EvalRow
   * \param wsum sum of weight
   */
  inline static float GetFinal(float esum, float wsum) {
    return esum / wsum;
  }
};

struct EvalRMSE : public EvalEWiseBase<EvalRMSE> {
  const char *Name() const override {
    return "rmse";
  }
  inline float EvalRow(float label, float pred) const {
    float diff = label - pred;
    return diff * diff;
  }
  inline static float GetFinal(float esum, float wsum) {
    return std::sqrt(esum / wsum);
  }
};

struct EvalMAE : public EvalEWiseBase<EvalMAE> {
  const char *Name() const override {
    return "mae";
  }
  inline float EvalRow(float label, float pred) const {
    return std::abs(label - pred);
  }
};

struct EvalLogLoss : public EvalEWiseBase<EvalLogLoss> {
  const char *Name() const override {
    return "logloss";
  }
  inline float EvalRow(float y, float py) const {
    const float eps = 1e-16f;
    const float pneg = 1.0f - py;
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
  inline float EvalRow(float label, float pred) const {
    // assume label is in [0,1]
    return pred > threshold_ ? 1.0f - label : label;
  }
 protected:
  float threshold_;
  std::string name_;
};

struct EvalPoissionNegLogLik : public EvalEWiseBase<EvalPoissionNegLogLik> {
  const char *Name() const override {
    return "poisson-nloglik";
  }
  inline float EvalRow(float y, float py) const {
    const float eps = 1e-16f;
    if (py < eps) py = eps;
    return common::LogGamma(y + 1.0f) + py - std::log(py) * y;
  }
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
.set_body([](const char* param) { return new EvalPoissionNegLogLik(); });

/*!
 * \brief base class of element-wise evaluation
 *      with additonal dispersion parameter
 * \tparam Derived the name of subclass
 */
template<typename Derived>
struct EvalEWiseBase2 : public Metric {
  float Eval(const std::vector<float>& preds,
             const MetaInfo& info,
             bool distributed) const override {
    CHECK_NE(info.labels.size(), 0) << "label set cannot be empty";
    CHECK_EQ(preds.size(), info.labels.size())
        << "label and prediction size not match, "
        << "hint: use merror or mlogloss for multi-class classification";
    const omp_ulong ndata = static_cast<omp_ulong>(info.labels.size());

    // Computer dispersion
    double sum = 0.0, wsum = 0.0;
    #pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) {
      const float wt = info.GetWeight(i);
      sum += static_cast<const Derived*>(this)->EvalDispersion(info.labels[i], preds[i]) * wt;
      wsum += wt;
    }
    double dat[2]; dat[0] = sum, dat[1] = wsum;
    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    double dispersion = dat[0] / (dat[1] - info.num_col);

    // Computer metric
    sum = 0.0, wsum = 0.0;
    #pragma omp parallel for reduction(+: sum, wsum) schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) {
      const float wt = info.GetWeight(i);
      sum += static_cast<const Derived*>(this)->EvalRow(info.labels[i], preds[i], dispersion) * wt;
      wsum += wt;
    }
    dat[0] = sum, dat[1] = wsum;
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
  inline float EvalRow(float label, float pred, float dispersion) const;
  /*!
   * \brief to be overridden by subclass, final transformation
   * \param esum the sum statistics returned by EvalRow
   * \param wsum sum of weight
   */
  inline static float GetFinal(float esum, float wsum) {
    return esum / wsum;
  }
  inline float EvalDispersion(float label, float pred) const;
};

struct EvalGammaNegLogLik : public EvalEWiseBase2<EvalGammaNegLogLik> {
  const char *Name() const override {
    return "gamma-nloglik";
  }
  inline float EvalRow(float y, float py, float psi) const {
    double theta = -1. / py;
    double a = psi;
    double b = -std::log(-theta);
    double c = 1. / psi * std::log(y/psi) - std::log(y) - common::LogGamma(1. / psi);
    return -((y * theta - b) / a + c);
  }
  inline float EvalDispersion(float y, float py) const {
    return ((y - py) * (y - py)) / (py * py);
  }
};

XGBOOST_REGISTER_METRIC(GammaNegLoglik, "gamma-nloglik")
.describe("Negative loglikelihood for gamma regression.")
.set_body([](const char* param) { return new EvalGammaNegLogLik(); });

}  // namespace metric
}  // namespace xgboost
