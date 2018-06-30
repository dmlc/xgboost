/*!
 * Copyright 2015 by Contributors
 * \file regression_obj.cc
 * \brief Definition of single-value regression and classification objectives.
 * \author Tianqi Chen, Kailong Chen
 */
#include <dmlc/omp.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <algorithm>
#include <utility>
#include "../common/math.h"
#include "../common/avx_helpers.h"
#include "./regression_loss.h"

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(regression_obj);

struct RegLossParam : public dmlc::Parameter<RegLossParam> {
  float scale_pos_weight;
  // declare parameters
  DMLC_DECLARE_PARAMETER(RegLossParam) {
    DMLC_DECLARE_FIELD(scale_pos_weight).set_default(1.0f).set_lower_bound(0.0f)
        .describe("Scale the weight of positive examples by this factor");
  }
};

// regression loss function
template <typename Loss>
class RegLossObj : public ObjFunction {
 public:
  RegLossObj()  = default;

  void Configure(
      const std::vector<std::pair<std::string, std::string> > &args) override {
    param_.InitAllowUnknown(args);
  }
  void GetGradient(HostDeviceVector<bst_float> *preds, const MetaInfo &info,
                   int iter, HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds->Size(), info.labels_.size())
        << "labels are not correctly provided"
        << "preds.size=" << preds->Size()
        << ", label.size=" << info.labels_.size();
    auto& preds_h = preds->HostVector();

    this->LazyCheckLabels(info.labels_);
    out_gpair->Resize(preds_h.size());
    auto& gpair = out_gpair->HostVector();
    const auto n = static_cast<omp_ulong>(preds_h.size());
    auto gpair_ptr = out_gpair->HostPointer();
    avx::Float8 scale(param_.scale_pos_weight);

    const omp_ulong remainder = n % 8;
#pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < n - remainder; i += 8) {
      avx::Float8 y(&info.labels_[i]);
      avx::Float8 p = Loss::PredTransform(avx::Float8(&preds_h[i]));
      avx::Float8 w = info.weights_.empty() ? avx::Float8(1.0f)
                                           : avx::Float8(&info.weights_[i]);
      // Adjust weight
      w += y * (scale * w - w);
      avx::Float8 grad = Loss::FirstOrderGradient(p, y);
      avx::Float8 hess = Loss::SecondOrderGradient(p, y);
      avx::StoreGpair(gpair_ptr + i, grad * w, hess * w);
    }
    for (omp_ulong i = n - remainder; i < n; ++i) {
      auto y = info.labels_[i];
      bst_float p = Loss::PredTransform(preds_h[i]);
      bst_float w = info.GetWeight(i);
      w += y * ((param_.scale_pos_weight * w) - w);
      gpair[i] = GradientPair(Loss::FirstOrderGradient(p, y) * w,
                           Loss::SecondOrderGradient(p, y) * w);
    }
  }
  const char *DefaultEvalMetric() const override {
    return Loss::DefaultEvalMetric();
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    std::vector<bst_float> &preds = io_preds->HostVector();
    const auto ndata = static_cast<bst_omp_uint>(preds.size());
#pragma omp parallel for schedule(static)
    for (bst_omp_uint j = 0; j < ndata; ++j) {
      preds[j] = Loss::PredTransform(preds[j]);
    }
  }
  bst_float ProbToMargin(bst_float base_score) const override {
    return Loss::ProbToMargin(base_score);
  }

 protected:
  void LazyCheckLabels(const std::vector<float> &labels) {
    if (labels_checked_) return;
    for (auto &y : labels) {
      CHECK(Loss::CheckLabel(y)) << Loss::LabelErrorMsg();
    }
    labels_checked_ = true;
  }
  RegLossParam param_;
  bool labels_checked_{false};
};

// register the objective functions
DMLC_REGISTER_PARAMETER(RegLossParam);

XGBOOST_REGISTER_OBJECTIVE(LinearRegression, "reg:linear")
.describe("Linear regression.")
.set_body([]() { return new RegLossObj<LinearSquareLoss>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticRegression, "reg:logistic")
.describe("Logistic regression for probability regression task.")
.set_body([]() { return new RegLossObj<LogisticRegression>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticClassification, "binary:logistic")
.describe("Logistic regression for binary classification task.")
.set_body([]() { return new RegLossObj<LogisticClassification>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticRaw, "binary:logitraw")
.describe("Logistic regression for classification, output score before logistic transformation")
.set_body([]() { return new RegLossObj<LogisticRaw>(); });

// declare parameter
struct PoissonRegressionParam : public dmlc::Parameter<PoissonRegressionParam> {
  float max_delta_step;
  DMLC_DECLARE_PARAMETER(PoissonRegressionParam) {
    DMLC_DECLARE_FIELD(max_delta_step).set_lower_bound(0.0f).set_default(0.7f)
        .describe("Maximum delta step we allow each weight estimation to be." \
                  " This parameter is required for possion regression.");
  }
};

// poisson regression for count
class PoissonRegression : public ObjFunction {
 public:
  // declare functions
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }

  void GetGradient(HostDeviceVector<bst_float> *preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds->Size(), info.labels_.size()) << "labels are not correctly provided";
    auto& preds_h = preds->HostVector();
    out_gpair->Resize(preds->Size());
    auto& gpair = out_gpair->HostVector();
    // check if label in range
    bool label_correct = true;
    // start calculating gradient
    const omp_ulong ndata = static_cast<omp_ulong>(preds_h.size()); // NOLINT(*)
#pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) { // NOLINT(*)
      bst_float p = preds_h[i];
      bst_float w = info.GetWeight(i);
      bst_float y = info.labels_[i];
      if (y >= 0.0f) {
        gpair[i] = GradientPair((std::exp(p) - y) * w,
                             std::exp(p + param_.max_delta_step) * w);
      } else {
        label_correct = false;
      }
    }
    CHECK(label_correct) << "PoissonRegression: label must be nonnegative";
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    std::vector<bst_float> &preds = io_preds->HostVector();
    const long ndata = static_cast<long>(preds.size()); // NOLINT(*)
#pragma omp parallel for schedule(static)
    for (long j = 0; j < ndata; ++j) {  // NOLINT(*)
      preds[j] = std::exp(preds[j]);
    }
  }
  void EvalTransform(HostDeviceVector<bst_float> *io_preds) override {
    PredTransform(io_preds);
  }
  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }
  const char* DefaultEvalMetric() const override {
    return "poisson-nloglik";
  }

 private:
  PoissonRegressionParam param_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(PoissonRegressionParam);

XGBOOST_REGISTER_OBJECTIVE(PoissonRegression, "count:poisson")
.describe("Possion regression for count data.")
.set_body([]() { return new PoissonRegression(); });

// cox regression for survival data (negative values mean they are censored)
class CoxRegression : public ObjFunction {
 public:
  // declare functions
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {}
  void GetGradient(HostDeviceVector<bst_float> *preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds->Size(), info.labels_.size()) << "labels are not correctly provided";
    auto& preds_h = preds->HostVector();
    out_gpair->Resize(preds_h.size());
    auto& gpair = out_gpair->HostVector();
    const std::vector<size_t> &label_order = info.LabelAbsSort();

    const omp_ulong ndata = static_cast<omp_ulong>(preds_h.size()); // NOLINT(*)

    // pre-compute a sum
    double exp_p_sum = 0;  // we use double because we might need the precision with large datasets
    for (omp_ulong i = 0; i < ndata; ++i) {
      exp_p_sum += std::exp(preds_h[label_order[i]]);
    }

    // start calculating grad and hess
    double r_k = 0;
    double s_k = 0;
    double last_exp_p = 0.0;
    double last_abs_y = 0.0;
    double accumulated_sum = 0;
    for (omp_ulong i = 0; i < ndata; ++i) { // NOLINT(*)
      const size_t ind = label_order[i];
      const double p = preds_h[ind];
      const double exp_p = std::exp(p);
      const double w = info.GetWeight(ind);
      const double y = info.labels_[ind];
      const double abs_y = std::abs(y);

      // only update the denominator after we move forward in time (labels are sorted)
      // this is Breslow's method for ties
      accumulated_sum += last_exp_p;
      if (last_abs_y < abs_y) {
        exp_p_sum -= accumulated_sum;
        accumulated_sum = 0;
      } else {
        CHECK(last_abs_y <= abs_y) << "CoxRegression: labels must be in sorted order, " <<
                                      "MetaInfo::LabelArgsort failed!";
      }

      if (y > 0) {
        r_k += 1.0/exp_p_sum;
        s_k += 1.0/(exp_p_sum*exp_p_sum);
      }

      const double grad = exp_p*r_k - static_cast<bst_float>(y > 0);
      const double hess = exp_p*r_k - exp_p*exp_p * s_k;
      gpair.at(ind) = GradientPair(grad * w, hess * w);

      last_abs_y = abs_y;
      last_exp_p = exp_p;
    }
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    std::vector<bst_float> &preds = io_preds->HostVector();
    const long ndata = static_cast<long>(preds.size()); // NOLINT(*)
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < ndata; ++j) {  // NOLINT(*)
      preds[j] = std::exp(preds[j]);
    }
  }
  void EvalTransform(HostDeviceVector<bst_float> *io_preds) override {
    PredTransform(io_preds);
  }
  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }
  const char* DefaultEvalMetric() const override {
    return "cox-nloglik";
  }
};

// register the objective function
XGBOOST_REGISTER_OBJECTIVE(CoxRegression, "survival:cox")
.describe("Cox regression for censored survival data (negative labels are considered censored).")
.set_body([]() { return new CoxRegression(); });

// gamma regression
class GammaRegression : public ObjFunction {
 public:
  // declare functions
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
  }

  void GetGradient(HostDeviceVector<bst_float> *preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds->Size(), info.labels_.size()) << "labels are not correctly provided";
    auto& preds_h = preds->HostVector();
    out_gpair->Resize(preds_h.size());
    auto& gpair = out_gpair->HostVector();
    // check if label in range
    bool label_correct = true;
    // start calculating gradient
    const omp_ulong ndata = static_cast<omp_ulong>(preds_h.size()); // NOLINT(*)
    #pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) { // NOLINT(*)
      bst_float p = preds_h[i];
      bst_float w = info.GetWeight(i);
      bst_float y = info.labels_[i];
      if (y >= 0.0f) {
        gpair[i] = GradientPair((1 - y / std::exp(p)) * w, y / std::exp(p) * w);
      } else {
        label_correct = false;
      }
    }
    CHECK(label_correct) << "GammaRegression: label must be positive";
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    std::vector<bst_float> &preds = io_preds->HostVector();
    const long ndata = static_cast<long>(preds.size()); // NOLINT(*)
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < ndata; ++j) {  // NOLINT(*)
      preds[j] = std::exp(preds[j]);
    }
  }
  void EvalTransform(HostDeviceVector<bst_float> *io_preds) override {
    PredTransform(io_preds);
  }
  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }
  const char* DefaultEvalMetric() const override {
    return "gamma-nloglik";
  }
};

// register the objective functions
XGBOOST_REGISTER_OBJECTIVE(GammaRegression, "reg:gamma")
.describe("Gamma regression for severity data.")
.set_body([]() { return new GammaRegression(); });

// declare parameter
struct TweedieRegressionParam : public dmlc::Parameter<TweedieRegressionParam> {
  float tweedie_variance_power;
  DMLC_DECLARE_PARAMETER(TweedieRegressionParam) {
    DMLC_DECLARE_FIELD(tweedie_variance_power).set_range(1.0f, 2.0f).set_default(1.5f)
      .describe("Tweedie variance power.  Must be between in range [1, 2).");
  }
};

// tweedie regression
class TweedieRegression : public ObjFunction {
 public:
  // declare functions
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }

  void GetGradient(HostDeviceVector<bst_float> *preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds->Size(), info.labels_.size()) << "labels are not correctly provided";
    auto& preds_h = preds->HostVector();
    out_gpair->Resize(preds->Size());
    auto& gpair = out_gpair->HostVector();
    // check if label in range
    bool label_correct = true;
    // start calculating gradient
    const omp_ulong ndata = static_cast<omp_ulong>(preds->Size()); // NOLINT(*)
    #pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) { // NOLINT(*)
      bst_float p = preds_h[i];
      bst_float w = info.GetWeight(i);
      bst_float y = info.labels_[i];
      float rho = param_.tweedie_variance_power;
      if (y >= 0.0f) {
        bst_float grad = -y * std::exp((1 - rho) * p) + std::exp((2 - rho) * p);
        bst_float hess = -y * (1 - rho) * \
          std::exp((1 - rho) * p) + (2 - rho) * std::exp((2 - rho) * p);
        gpair[i] = GradientPair(grad * w, hess * w);
      } else {
        label_correct = false;
      }
    }
    CHECK(label_correct) << "TweedieRegression: label must be nonnegative";
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    std::vector<bst_float> &preds = io_preds->HostVector();
    const long ndata = static_cast<long>(preds.size()); // NOLINT(*)
#pragma omp parallel for schedule(static)
    for (long j = 0; j < ndata; ++j) {  // NOLINT(*)
      preds[j] = std::exp(preds[j]);
    }
  }

  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }

  const char* DefaultEvalMetric() const override {
    std::ostringstream os;
    os << "tweedie-nloglik@" << param_.tweedie_variance_power;
    std::string metric = os.str();
    return metric.c_str();
  }

 private:
  TweedieRegressionParam param_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(TweedieRegressionParam);

XGBOOST_REGISTER_OBJECTIVE(TweedieRegression, "reg:tweedie")
.describe("Tweedie regression for insurance data.")
.set_body([]() { return new TweedieRegression(); });
}  // namespace obj
}  // namespace xgboost
