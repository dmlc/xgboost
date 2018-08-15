/*!
 * Copyright 2015-2018 by Contributors
 * \file regression_obj.cu
 * \brief Definition of single-value regression and classification objectives.
 * \author Tianqi Chen, Kailong Chen
 */

#include <dmlc/omp.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <cmath>
#include <memory>
#include <vector>

#include "../common/span.h"
#include "../common/transform.h"
#include "../common/gpu_set.h"
#include "../common/host_device_vector.h"
#include "./regression_loss.h"


namespace xgboost {
namespace obj {

#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(regression_obj_gpu);
#endif

struct RegLossParam : public dmlc::Parameter<RegLossParam> {
  float scale_pos_weight;
  int n_gpus;
  int gpu_id;
  // declare parameters
  DMLC_DECLARE_PARAMETER(RegLossParam) {
    DMLC_DECLARE_FIELD(scale_pos_weight).set_default(1.0f).set_lower_bound(0.0f)
      .describe("Scale the weight of positive examples by this factor");
    DMLC_DECLARE_FIELD(n_gpus).set_default(-1).set_lower_bound(-1)
      .describe("Number of GPUs to use for multi-gpu algorithms (NOT IMPLEMENTED)");
    DMLC_DECLARE_FIELD(gpu_id)
      .set_lower_bound(0)
      .set_default(0)
      .describe("gpu to use for objective function evaluation");
  }
};

// regression loss function for evaluation on GPU (eventually)
template<typename Loss>
class RegLossObj : public ObjFunction {
 protected:
  bool copied_;
  HostDeviceVector<bst_float> labels_, weights_;
  HostDeviceVector<unsigned int> label_correct_;

  // allocate device data for n elements, do nothing if memory is allocated already
  void LazyResize(size_t n, size_t n_weights) {
    if (labels_.Size() == n && weights_.Size() == n_weights) {
      return;
    }
    copied_ = false;

    labels_.Reshard(devices_);
    weights_.Reshard(devices_);
    label_correct_.Reshard(devices_);

    if (labels_.Size() != n) {
      labels_.Resize(n);
      if (devices_ == GPUSet::Empty()) {
        label_correct_.Resize(1);
      } else {
        label_correct_.Resize(devices_.Size());
      }
    }
    if (weights_.Size() != n_weights) {
      weights_.Resize(n_weights);
    }
  }

 public:
  RegLossObj() : copied_(false) {}

  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
    // CHECK(param_.n_gpus != 0) << "Must have at least one device";
    devices_ = GPUSet::All(param_.n_gpus).Normalised(param_.gpu_id);
  }

  void GetGradient(HostDeviceVector<float>* preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    CHECK_NE(info.labels_.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds->Size(), info.labels_.size())
      << "labels are not correctly provided"
      << "preds.size=" << preds->Size() << ", label.size=" << info.labels_.size();
    size_t ndata = preds->Size();
    preds->Reshard(devices_);
    out_gpair->Reshard(devices_);
    out_gpair->Resize(ndata);
    LazyResize(ndata, info.weights_.size());

    label_correct_.Fill(1);
    // only copy the labels and weights once, similar to how the data is copied
    if (!copied_) {
      labels_.Copy(info.labels_);
      if (info.weights_.size() > 0) {
        weights_.Copy(info.weights_);
      }
      copied_ = true;
    }

    bool is_null_weight = info.weights_.size() == 0;
    auto scale_pos_weight = param_.scale_pos_weight;
    common::SegTransform(
        [=] XGBOOST_DEVICE (unsigned int* _label_correct,  // NOLINT
                            common::Span<GradientPair> _out_gpair,
                            common::Span<const float> _preds,
                            common::Span<const float> _labels,
                            common::Span<const bst_float> _weights) {
          float p = Loss::PredTransform(_preds[0]);
          float w = is_null_weight ? 1.0f : _weights[0];
          float label = _labels[0];
          if (label == 1.0f) {
            w *= scale_pos_weight;
          }
          if (!Loss::CheckLabel(label)) {
            // FIXME: I'm not sure if atomicAnd is actually needed.
            // atomicAnd(_label_correct.data(), 0);
            *_label_correct = 0;
          }
          _out_gpair[0] = GradientPair(Loss::FirstOrderGradient(p, label) * w,
                                       Loss::SecondOrderGradient(p, label) * w);
        },
        common::Range{0, static_cast<int64_t>(ndata), 1},
        &label_correct_, devices_,
        out_gpair, preds, &labels_, &weights_);

    // copy "label correct" flags back to host
    std::vector<unsigned int>& label_correct_h = label_correct_.HostVector();
    for (auto const flag : label_correct_h) {
      if (flag == 0) {
        LOG(FATAL) << Loss::LabelErrorMsg();
      }
    }
  }

 public:
  const char* DefaultEvalMetric() const override {
    return Loss::DefaultEvalMetric();
  }

  void PredTransform(HostDeviceVector<float> *io_preds) override {
    io_preds->Reshard(devices_);
    size_t ndata = io_preds->Size();
    common::SegTransform(
        [] XGBOOST_DEVICE (common::Span<float> _preds) {  // NOLINT
          _preds[0] = Loss::PredTransform(_preds[0]);
        }, common::Range{0, static_cast<int64_t>(io_preds->Size()), 1},
        devices_, io_preds);
  }

  float ProbToMargin(float base_score) const override {
    return Loss::ProbToMargin(base_score);
  }

 protected:
  RegLossParam param_;
  GPUSet devices_;
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
.describe("Logistic regression for classification, output score "
          "before logistic transformation.")
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
