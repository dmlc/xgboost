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
#include "../common/common.h"
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
    DMLC_DECLARE_FIELD(n_gpus).set_default(1).set_lower_bound(-1)
      .describe("Number of GPUs to use for multi-gpu algorithms.");
    DMLC_DECLARE_FIELD(gpu_id)
      .set_lower_bound(0)
      .set_default(0)
      .describe("gpu to use for objective function evaluation");
  }
};

template<typename Loss>
class RegLossObj : public ObjFunction {
 protected:
  HostDeviceVector<int> label_correct_;

 public:
  RegLossObj() = default;

  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
    CHECK(param_.n_gpus != 0) << "Must have at least one device";  // Default is -1
    devices_ = GPUSet::All(param_.n_gpus).Normalised(param_.gpu_id);
    label_correct_.Resize(devices_.IsEmpty() ? 1 : devices_.Size());
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size())
        << "labels are not correctly provided"
        << "preds.size=" << preds.Size() << ", label.size=" << info.labels_.Size();
    size_t ndata = preds.Size();
    out_gpair->Resize(ndata);
    label_correct_.Fill(1);

    bool is_null_weight = info.weights_.Size() == 0;
    auto scale_pos_weight = param_.scale_pos_weight;
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t _idx,
                           common::Span<int> _label_correct,
                           common::Span<GradientPair> _out_gpair,
                           common::Span<const bst_float> _preds,
                           common::Span<const bst_float> _labels,
                           common::Span<const bst_float> _weights) {
          bst_float p = Loss::PredTransform(_preds[_idx]);
          bst_float w = is_null_weight ? 1.0f : _weights[_idx];
          bst_float label = _labels[_idx];
          if (label == 1.0f) {
            w *= scale_pos_weight;
          }
          if (!Loss::CheckLabel(label)) {
            // If there is an incorrect label, the host code will know.
            _label_correct[0] = 0;
          }
          _out_gpair[_idx] = GradientPair(Loss::FirstOrderGradient(p, label) * w,
                                          Loss::SecondOrderGradient(p, label) * w);
        },
        common::Range{0, static_cast<int64_t>(ndata)}, devices_).Eval(
            &label_correct_, out_gpair, &preds, &info.labels_, &info.weights_);

    // copy "label correct" flags back to host
    std::vector<int>& label_correct_h = label_correct_.HostVector();
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
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<float> _preds) {
          _preds[_idx] = Loss::PredTransform(_preds[_idx]);
        }, common::Range{0, static_cast<int64_t>(io_preds->Size())},
        devices_).Eval(io_preds);
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

// Deprecated GPU functions
XGBOOST_REGISTER_OBJECTIVE(GPULinearRegression, "gpu:reg:linear")
.describe("Deprecated. Linear regression (computed on GPU).")
.set_body([]() {
    LOG(WARNING) << "gpu:reg:linear is now deprecated, use reg:linear instead.";
    return new RegLossObj<LinearSquareLoss>(); });

XGBOOST_REGISTER_OBJECTIVE(GPULogisticRegression, "gpu:reg:logistic")
.describe("Deprecated. Logistic regression for probability regression task (computed on GPU).")
.set_body([]() {
    LOG(WARNING) << "gpu:reg:logistic is now deprecated, use reg:logistic instead.";
    return new RegLossObj<LogisticRegression>(); });

XGBOOST_REGISTER_OBJECTIVE(GPULogisticClassification, "gpu:binary:logistic")
.describe("Deprecated. Logistic regression for binary classification task (computed on GPU).")
.set_body([]() {
    LOG(WARNING) << "gpu:binary:logistic is now deprecated, use binary:logistic instead.";
    return new RegLossObj<LogisticClassification>(); });

XGBOOST_REGISTER_OBJECTIVE(GPULogisticRaw, "gpu:binary:logitraw")
.describe("Deprecated. Logistic regression for classification, output score "
          "before logistic transformation (computed on GPU)")
.set_body([]() {
    LOG(WARNING) << "gpu:binary:logitraw is now deprecated, use binary:logitraw instead.";
    return new RegLossObj<LogisticRaw>(); });
// End deprecated

// declare parameter
struct PoissonRegressionParam : public dmlc::Parameter<PoissonRegressionParam> {
  float max_delta_step;
  int n_gpus;
  int gpu_id;
  DMLC_DECLARE_PARAMETER(PoissonRegressionParam) {
    DMLC_DECLARE_FIELD(max_delta_step).set_lower_bound(0.0f).set_default(0.7f)
        .describe("Maximum delta step we allow each weight estimation to be." \
                  " This parameter is required for possion regression.");
    DMLC_DECLARE_FIELD(n_gpus).set_default(-1).set_lower_bound(-1)
        .describe("Number of GPUs to use for multi-gpu algorithms.");
    DMLC_DECLARE_FIELD(gpu_id)
        .set_lower_bound(0)
        .set_default(0)
        .describe("gpu to use for objective function evaluation");
  }
};

// poisson regression for count
class PoissonRegression : public ObjFunction {
 public:
  // declare functions
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
    CHECK(param_.n_gpus != 0) << "Must have at least one device";  // Default is -1
    devices_ = GPUSet::All(param_.n_gpus).Normalised(param_.gpu_id);
    label_correct_.Resize(devices_.IsEmpty() ? 1 : devices_.Size());
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size()) << "labels are not correctly provided";
    size_t ndata = preds.Size();
    out_gpair->Resize(ndata);
    label_correct_.Fill(1);

    bool is_null_weight = info.weights_.Size() == 0;
    bst_float max_delta_step = param_.max_delta_step;
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t _idx,
                           common::Span<int> _label_correct,
                           common::Span<GradientPair> _out_gpair,
                           common::Span<const bst_float> _preds,
                           common::Span<const bst_float> _labels,
                           common::Span<const bst_float> _weights) {
          bst_float p = _preds[_idx];
          bst_float w = is_null_weight ? 1.0f : _weights[_idx];
          bst_float y = _labels[_idx];
          if (y < 0.0f) {
            _label_correct[0] = 0;
          }
          _out_gpair[_idx] = GradientPair{(expf(p) - y) * w,
                                          expf(p + max_delta_step) * w};
        },
        common::Range{0, static_cast<int64_t>(ndata)}, devices_).Eval(
            &label_correct_, out_gpair, &preds, &info.labels_, &info.weights_);
    // copy "label correct" flags back to host
    std::vector<int>& label_correct_h = label_correct_.HostVector();
    for (auto const flag : label_correct_h) {
      if (flag == 0) {
        LOG(FATAL) << "PoissonRegression: label must be nonnegative";
      }
    }
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<bst_float> _preds) {
          _preds[_idx] = expf(_preds[_idx]);
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size())}, devices_)
        .Eval(io_preds);
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
  GPUSet devices_;
  PoissonRegressionParam param_;
  HostDeviceVector<int> label_correct_;
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
  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size()) << "labels are not correctly provided";
    const auto& preds_h = preds.HostVector();
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
    const auto& labels = info.labels_.HostVector();
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
      const double y = labels[ind];
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


struct GammaRegressionParam : public dmlc::Parameter<GammaRegressionParam> {
  int n_gpus;
  int gpu_id;
  DMLC_DECLARE_PARAMETER(GammaRegressionParam) {
    DMLC_DECLARE_FIELD(n_gpus).set_default(-1).set_lower_bound(-1)
        .describe("Number of GPUs to use for multi-gpu algorithms.");
    DMLC_DECLARE_FIELD(gpu_id)
        .set_lower_bound(0)
        .set_default(0)
        .describe("gpu to use for objective function evaluation");
  }
};

// gamma regression
class GammaRegression : public ObjFunction {
 public:
  // declare functions
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
    CHECK(param_.n_gpus != 0) << "Must have at least one device";  // Default is -1
    devices_ = GPUSet::All(param_.n_gpus).Normalised(param_.gpu_id);
    label_correct_.Resize(devices_.IsEmpty() ? 1 : devices_.Size());
  }

  void GetGradient(const HostDeviceVector<bst_float> &preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size()) << "labels are not correctly provided";
    const size_t ndata = preds.Size();
    out_gpair->Resize(ndata);
    label_correct_.Fill(1);

    const bool is_null_weight = info.weights_.Size() == 0;
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t _idx,
                           common::Span<int> _label_correct,
                           common::Span<GradientPair> _out_gpair,
                           common::Span<const bst_float> _preds,
                           common::Span<const bst_float> _labels,
                           common::Span<const bst_float> _weights) {
          bst_float p = _preds[_idx];
          bst_float w = is_null_weight ? 1.0f : _weights[_idx];
          bst_float y = _labels[_idx];
          if (y < 0.0f) {
            _label_correct[0] = 0;
          }
          _out_gpair[_idx] = GradientPair((1 - y / expf(p)) * w, y / expf(p) * w);
        },
        common::Range{0, static_cast<int64_t>(ndata)}, devices_).Eval(
            &label_correct_, out_gpair, &preds, &info.labels_, &info.weights_);

    // copy "label correct" flags back to host
    std::vector<int>& label_correct_h = label_correct_.HostVector();
    for (auto const flag : label_correct_h) {
      if (flag == 0) {
        LOG(FATAL) << "GammaRegression: label must be nonnegative";
      }
    }
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<bst_float> _preds) {
          _preds[_idx] = expf(_preds[_idx]);
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size())}, devices_)
        .Eval(io_preds);
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

 private:
  GPUSet devices_;
  GammaRegressionParam param_;
  HostDeviceVector<int> label_correct_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(GammaRegressionParam);
// register the objective functions
XGBOOST_REGISTER_OBJECTIVE(GammaRegression, "reg:gamma")
.describe("Gamma regression for severity data.")
.set_body([]() { return new GammaRegression(); });


// declare parameter
struct TweedieRegressionParam : public dmlc::Parameter<TweedieRegressionParam> {
  float tweedie_variance_power;
  int n_gpus;
  int gpu_id;
  DMLC_DECLARE_PARAMETER(TweedieRegressionParam) {
    DMLC_DECLARE_FIELD(tweedie_variance_power).set_range(1.0f, 2.0f).set_default(1.5f)
      .describe("Tweedie variance power.  Must be between in range [1, 2).");
    DMLC_DECLARE_FIELD(n_gpus).set_default(-1).set_lower_bound(-1)
        .describe("Number of GPUs to use for multi-gpu algorithms.");
    DMLC_DECLARE_FIELD(gpu_id)
        .set_lower_bound(0)
        .set_default(0)
        .describe("gpu to use for objective function evaluation");
  }
};

// tweedie regression
class TweedieRegression : public ObjFunction {
 public:
  // declare functions
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
    CHECK(param_.n_gpus != 0) << "Must have at least one device";  // Default is -1
    devices_ = GPUSet::All(param_.n_gpus).Normalised(param_.gpu_id);
    label_correct_.Resize(devices_.IsEmpty() ? 1 : devices_.Size());
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size()) << "labels are not correctly provided";
    const size_t ndata = preds.Size();
    out_gpair->Resize(ndata);
    label_correct_.Fill(1);

    const bool is_null_weight = info.weights_.Size() == 0;
    const float rho = param_.tweedie_variance_power;
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t _idx,
                           common::Span<int> _label_correct,
                           common::Span<GradientPair> _out_gpair,
                           common::Span<const bst_float> _preds,
                           common::Span<const bst_float> _labels,
                           common::Span<const bst_float> _weights) {
          bst_float p = _preds[_idx];
          bst_float w = is_null_weight ? 1.0f : _weights[_idx];
          bst_float y = _labels[_idx];
          if (y < 0.0f) {
            _label_correct[0] = 0;
          }
          bst_float grad = -y * expf((1 - rho) * p) + expf((2 - rho) * p);
          bst_float hess =
              -y * (1 - rho) * \
              std::exp((1 - rho) * p) + (2 - rho) * expf((2 - rho) * p);
          _out_gpair[_idx] = GradientPair(grad * w, hess * w);
        },
        common::Range{0, static_cast<int64_t>(ndata), 1}, devices_)
        .Eval(&label_correct_, out_gpair, &preds, &info.labels_, &info.weights_);

    // copy "label correct" flags back to host
    std::vector<int>& label_correct_h = label_correct_.HostVector();
    for (auto const flag : label_correct_h) {
      if (flag == 0) {
        LOG(FATAL) << "TweedieRegression: label must be nonnegative";
      }
    }
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<bst_float> _preds) {
          _preds[_idx] = expf(_preds[_idx]);
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size())}, devices_)
        .Eval(io_preds);
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
  GPUSet devices_;
  TweedieRegressionParam param_;
  HostDeviceVector<int> label_correct_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(TweedieRegressionParam);

XGBOOST_REGISTER_OBJECTIVE(TweedieRegression, "reg:tweedie")
.describe("Tweedie regression for insurance data.")
.set_body([]() { return new TweedieRegression(); });

}  // namespace obj
}  // namespace xgboost
