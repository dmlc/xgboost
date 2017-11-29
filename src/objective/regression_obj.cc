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

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(regression_obj);

// common regressions
// linear regression
struct LinearSquareLoss {
  template <typename T>
  static T PredTransform(T x) { return x; }
  static bool CheckLabel(bst_float x) { return true; }
  template <typename T>
  static T FirstOrderGradient(T predt, T label) { return predt - label; }
  template <typename T>
  static T SecondOrderGradient(T predt, T label) { return T(1.0f); }
  static bst_float ProbToMargin(bst_float base_score) { return base_score; }
  static const char* LabelErrorMsg() { return ""; }
  static const char* DefaultEvalMetric() { return "rmse"; }
};
// logistic loss for probability regression task
struct LogisticRegression {
  template <typename T>
  static T PredTransform(T x) { return common::Sigmoid(x); }
  static bool CheckLabel(bst_float x) { return x >= 0.0f && x <= 1.0f; }
  template <typename T>
  static T FirstOrderGradient(T predt, T label) { return predt - label; }
  template <typename T>
  static T SecondOrderGradient(T predt, T label) {
    const T eps = T(1e-16f);
    return std::max(predt * (T(1.0f) - predt), eps);
  }
  static bst_float ProbToMargin(bst_float base_score) {
    CHECK(base_score > 0.0f && base_score < 1.0f)
        << "base_score must be in (0,1) for logistic loss";
    return -std::log(1.0f / base_score - 1.0f);
  }
  static const char* LabelErrorMsg() {
    return "label must be in [0,1] for logistic regression";
  }
  static const char* DefaultEvalMetric() { return "rmse"; }
};
// logistic loss for binary classification task.
struct LogisticClassification : public LogisticRegression {
  static const char* DefaultEvalMetric() { return "error"; }
};
// logistic loss, but predict un-transformed margin
struct LogisticRaw : public LogisticRegression {
  template <typename T>
  static T PredTransform(T x) { return x; }
  template <typename T>
  static T FirstOrderGradient(T predt, T label) {
    predt = common::Sigmoid(predt);
    return predt - label;
  }
  template <typename T>
  static T SecondOrderGradient(T predt, T label) {
    const T eps = T(1e-16f);
    predt = common::Sigmoid(predt);
    return std::max(predt * (T(1.0f) - predt), eps);
  }
  static const char* DefaultEvalMetric() { return "auc"; }
};

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
  RegLossObj() : labels_checked(false) {}

  void Configure(
      const std::vector<std::pair<std::string, std::string> > &args) override {
    param_.InitAllowUnknown(args);
  }
  void GetGradient(const std::vector<bst_float> &preds, const MetaInfo &info,
                   int iter, std::vector<bst_gpair> *out_gpair) override {
    CHECK_NE(info.labels.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.size(), info.labels.size())
        << "labels are not correctly provided"
        << "preds.size=" << preds.size()
        << ", label.size=" << info.labels.size();

    this->LazyCheckLabels(info.labels);
    out_gpair->resize(preds.size());
    const omp_ulong n = static_cast<omp_ulong>(preds.size());
    auto gpair_ptr = out_gpair->data();
    avx::Float8 scale(param_.scale_pos_weight);

    const omp_ulong remainder = n % 8;
    int nthread = omp_get_max_threads();
    // Use a maximum of 8 threads
#pragma omp parallel for schedule(static) num_threads(std::min(8, nthread))
    for (int i = 0; i < n - remainder; i += 8) {
      avx::Float8 y(&info.labels[i]);
      avx::Float8 p = Loss::PredTransform(avx::Float8(&preds[i]));
      avx::Float8 w = info.weights.empty() ? avx::Float8(1.0f)
                                           : avx::Float8(&info.weights[i]);
      // Adjust weight
      w += y * (scale * w - w);
      avx::Float8 grad = Loss::FirstOrderGradient(p, y);
      avx::Float8 hess = Loss::SecondOrderGradient(p, y);
      avx::StoreGpair(gpair_ptr + i, grad * w, hess * w);
    }
    for (omp_ulong i = n - remainder; i < n; ++i) {
      auto y = info.labels[i];
      bst_float p = Loss::PredTransform(preds[i]);
      bst_float w = info.GetWeight(i);
      w += y * ((param_.scale_pos_weight * w) - w);
      (*out_gpair)[i] = bst_gpair(Loss::FirstOrderGradient(p, y) * w,
                                  Loss::SecondOrderGradient(p, y) * w);
    }

    // Reset omp max threads
    omp_set_num_threads(nthread);
  }
  const char *DefaultEvalMetric() const override {
    return Loss::DefaultEvalMetric();
  }
  void PredTransform(std::vector<bst_float> *io_preds) override {
    std::vector<bst_float> &preds = *io_preds;
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(preds.size());
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
    if (labels_checked) return;
    for (auto &y : labels) {
      CHECK(Loss::CheckLabel(y)) << Loss::LabelErrorMsg();
    }
    labels_checked = true;
  }
  RegLossParam param_;
  bool labels_checked;
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

  void GetGradient(const std::vector<bst_float> &preds,
                   const MetaInfo &info,
                   int iter,
                   std::vector<bst_gpair> *out_gpair) override {
    CHECK_NE(info.labels.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.size(), info.labels.size()) << "labels are not correctly provided";
    out_gpair->resize(preds.size());
    // check if label in range
    bool label_correct = true;
    // start calculating gradient
    const omp_ulong ndata = static_cast<omp_ulong>(preds.size()); // NOLINT(*)
#pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) { // NOLINT(*)
      bst_float p = preds[i];
      bst_float w = info.GetWeight(i);
      bst_float y = info.labels[i];
      if (y >= 0.0f) {
        (*out_gpair)[i] = bst_gpair((std::exp(p) - y) * w,
                                     std::exp(p + param_.max_delta_step) * w);
      } else {
        label_correct = false;
      }
    }
    CHECK(label_correct) << "PoissonRegression: label must be nonnegative";
  }
  void PredTransform(std::vector<bst_float> *io_preds) override {
    std::vector<bst_float> &preds = *io_preds;
    const long ndata = static_cast<long>(preds.size()); // NOLINT(*)
#pragma omp parallel for schedule(static)
    for (long j = 0; j < ndata; ++j) {  // NOLINT(*)
      preds[j] = std::exp(preds[j]);
    }
  }
  void EvalTransform(std::vector<bst_float> *io_preds) override {
    PredTransform(io_preds);
  }
  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }
  const char* DefaultEvalMetric(void) const override {
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

// gamma regression
class GammaRegression : public ObjFunction {
 public:
  // declare functions
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
  }

  void GetGradient(const std::vector<bst_float> &preds,
                   const MetaInfo &info,
                   int iter,
                   std::vector<bst_gpair> *out_gpair) override {
    CHECK_NE(info.labels.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.size(), info.labels.size()) << "labels are not correctly provided";
    out_gpair->resize(preds.size());
    // check if label in range
    bool label_correct = true;
    // start calculating gradient
    const omp_ulong ndata = static_cast<omp_ulong>(preds.size()); // NOLINT(*)
    #pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) { // NOLINT(*)
      bst_float p = preds[i];
      bst_float w = info.GetWeight(i);
      bst_float y = info.labels[i];
      if (y >= 0.0f) {
        (*out_gpair)[i] = bst_gpair((1 - y / std::exp(p)) * w, y / std::exp(p) * w);
      } else {
        label_correct = false;
      }
    }
    CHECK(label_correct) << "GammaRegression: label must be positive";
  }
  void PredTransform(std::vector<bst_float> *io_preds) override {
    std::vector<bst_float> &preds = *io_preds;
    const long ndata = static_cast<long>(preds.size()); // NOLINT(*)
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < ndata; ++j) {  // NOLINT(*)
      preds[j] = std::exp(preds[j]);
    }
  }
  void EvalTransform(std::vector<bst_float> *io_preds) override {
    PredTransform(io_preds);
  }
  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }
  const char* DefaultEvalMetric(void) const override {
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

  void GetGradient(const std::vector<bst_float> &preds,
                   const MetaInfo &info,
                   int iter,
                   std::vector<bst_gpair> *out_gpair) override {
    CHECK_NE(info.labels.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.size(), info.labels.size()) << "labels are not correctly provided";
    out_gpair->resize(preds.size());
    // check if label in range
    bool label_correct = true;
    // start calculating gradient
    const omp_ulong ndata = static_cast<omp_ulong>(preds.size()); // NOLINT(*)
    #pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) { // NOLINT(*)
      bst_float p = preds[i];
      bst_float w = info.GetWeight(i);
      bst_float y = info.labels[i];
      float rho = param_.tweedie_variance_power;
      if (y >= 0.0f) {
        bst_float grad = -y * std::exp((1 - rho) * p) + std::exp((2 - rho) * p);
        bst_float hess = -y * (1 - rho) * \
          std::exp((1 - rho) * p) + (2 - rho) * std::exp((2 - rho) * p);
        (*out_gpair)[i] = bst_gpair(grad * w, hess * w);
      } else {
        label_correct = false;
      }
    }
    CHECK(label_correct) << "TweedieRegression: label must be nonnegative";
  }
  void PredTransform(std::vector<bst_float> *io_preds) override {
    std::vector<bst_float> &preds = *io_preds;
    const long ndata = static_cast<long>(preds.size()); // NOLINT(*)
#pragma omp parallel for schedule(static)
    for (long j = 0; j < ndata; ++j) {  // NOLINT(*)
      preds[j] = std::exp(preds[j]);
    }
  }
  const char* DefaultEvalMetric(void) const override {
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
