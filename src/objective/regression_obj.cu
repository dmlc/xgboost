/*!
 * Copyright 2015-2022 by XGBoost Contributors
 * \file regression_obj.cu
 * \brief Definition of single-value regression and classification objectives.
 * \author Tianqi Chen, Kailong Chen
 */

#include <dmlc/omp.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <xgboost/tree_model.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "../common/common.h"
#include "../common/linalg_op.h"
#include "../common/pseudo_huber.h"
#include "../common/stats.h"
#include "../common/threading_utils.h"
#include "../common/transform.h"
#include "./regression_loss.h"
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/generic_parameters.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/linalg.h"
#include "xgboost/parameter.h"
#include "xgboost/span.h"

#if defined(XGBOOST_USE_CUDA)
#include "../common/linalg_op.cuh"
#endif  // defined(XGBOOST_USE_CUDA)

namespace xgboost {
namespace obj {
namespace {
void CheckRegInputs(MetaInfo const& info, HostDeviceVector<bst_float> const& preds) {
  CHECK_EQ(info.labels.Shape(0), info.num_row_) << "Invalid shape of labels.";
  CHECK_EQ(info.labels.Size(), preds.Size()) << "Invalid shape of labels.";
  if (!info.weights_.Empty()) {
    CHECK_EQ(info.weights_.Size(), info.num_row_)
        << "Number of weights should be equal to number of data points.";
  }
}
}  // anonymous namespace

#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(regression_obj_gpu);
#endif  // defined(XGBOOST_USE_CUDA)

struct RegLossParam : public XGBoostParameter<RegLossParam> {
  float scale_pos_weight;
  // declare parameters
  DMLC_DECLARE_PARAMETER(RegLossParam) {
    DMLC_DECLARE_FIELD(scale_pos_weight).set_default(1.0f).set_lower_bound(0.0f)
      .describe("Scale the weight of positive examples by this factor");
  }
};

template<typename Loss>
class RegLossObj : public ObjFunction {
 protected:
  HostDeviceVector<float> additional_input_;

 public:
  // 0 - label_correct flag, 1 - scale_pos_weight, 2 - is_null_weight
  RegLossObj(): additional_input_(3) {}

  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.UpdateAllowUnknown(args);
  }

  ObjInfo Task() const override { return Loss::Info(); }

  uint32_t Targets(MetaInfo const& info) const override {
    // Multi-target regression.
    return std::max(static_cast<size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info, int,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    CheckRegInputs(info, preds);
    size_t const ndata = preds.Size();
    out_gpair->Resize(ndata);
    auto device = ctx_->gpu_id;
    additional_input_.HostVector().begin()[0] = 1;  // Fill the label_correct flag

    bool is_null_weight = info.weights_.Size() == 0;
    auto scale_pos_weight = param_.scale_pos_weight;
    additional_input_.HostVector().begin()[1] = scale_pos_weight;
    additional_input_.HostVector().begin()[2] = is_null_weight;

    const size_t nthreads = ctx_->Threads();
    bool on_device = device >= 0;
    // On CPU we run the transformation each thread processing a contigious block of data
    // for better performance.
    const size_t n_data_blocks = std::max(static_cast<size_t>(1), (on_device ? ndata : nthreads));
    const size_t block_size = ndata / n_data_blocks + !!(ndata % n_data_blocks);
    auto const n_targets = std::max(info.labels.Shape(1), static_cast<size_t>(1));

    common::Transform<>::Init(
        [block_size, ndata, n_targets] XGBOOST_DEVICE(
            size_t data_block_idx, common::Span<float> _additional_input,
            common::Span<GradientPair> _out_gpair,
            common::Span<const bst_float> _preds,
            common::Span<const bst_float> _labels,
            common::Span<const bst_float> _weights) {
          const bst_float* preds_ptr = _preds.data();
          const bst_float* labels_ptr = _labels.data();
          const bst_float* weights_ptr = _weights.data();
          GradientPair* out_gpair_ptr = _out_gpair.data();
          const size_t begin = data_block_idx*block_size;
          const size_t end = std::min(ndata, begin + block_size);
          const float _scale_pos_weight = _additional_input[1];
          const bool _is_null_weight = _additional_input[2];

          for (size_t idx = begin; idx < end; ++idx) {
            bst_float p = Loss::PredTransform(preds_ptr[idx]);
            bst_float w = _is_null_weight ? 1.0f : weights_ptr[idx / n_targets];
            bst_float label = labels_ptr[idx];
            if (label == 1.0f) {
              w *= _scale_pos_weight;
            }
            if (!Loss::CheckLabel(label)) {
              // If there is an incorrect label, the host code will know.
              _additional_input[0] = 0;
            }
            out_gpair_ptr[idx] = GradientPair(Loss::FirstOrderGradient(p, label) * w,
                                              Loss::SecondOrderGradient(p, label) * w);
          }
        },
        common::Range{0, static_cast<int64_t>(n_data_blocks)}, nthreads, device)
        .Eval(&additional_input_, out_gpair, &preds, info.labels.Data(),
              &info.weights_);

    auto const flag = additional_input_.HostVector().begin()[0];
    if (flag == 0) {
      LOG(FATAL) << Loss::LabelErrorMsg();
    }
  }

 public:
  const char* DefaultEvalMetric() const override {
    return Loss::DefaultEvalMetric();
  }

  void PredTransform(HostDeviceVector<float> *io_preds) const override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<float> _preds) {
          _preds[_idx] = Loss::PredTransform(_preds[_idx]);
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size())}, this->ctx_->Threads(),
        io_preds->DeviceIdx())
        .Eval(io_preds);
  }

  float ProbToMargin(float base_score) const override {
    return Loss::ProbToMargin(base_score);
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(Loss::Name());
    out["reg_loss_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    FromJson(in["reg_loss_param"], &param_);
  }

 protected:
  RegLossParam param_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(RegLossParam);

XGBOOST_REGISTER_OBJECTIVE(SquaredLossRegression, LinearSquareLoss::Name())
.describe("Regression with squared error.")
.set_body([]() { return new RegLossObj<LinearSquareLoss>(); });

XGBOOST_REGISTER_OBJECTIVE(SquareLogError, SquaredLogError::Name())
.describe("Regression with root mean squared logarithmic error.")
.set_body([]() { return new RegLossObj<SquaredLogError>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticRegression, LogisticRegression::Name())
.describe("Logistic regression for probability regression task.")
.set_body([]() { return new RegLossObj<LogisticRegression>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticClassification, LogisticClassification::Name())
.describe("Logistic regression for binary classification task.")
.set_body([]() { return new RegLossObj<LogisticClassification>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticRaw, LogisticRaw::Name())
.describe("Logistic regression for classification, output score "
          "before logistic transformation.")
.set_body([]() { return new RegLossObj<LogisticRaw>(); });

// Deprecated functions
XGBOOST_REGISTER_OBJECTIVE(LinearRegression, "reg:linear")
.describe("Regression with squared error.")
.set_body([]() {
    LOG(WARNING) << "reg:linear is now deprecated in favor of reg:squarederror.";
    return new RegLossObj<LinearSquareLoss>(); });
// End deprecated

class PseudoHuberRegression : public ObjFunction {
  PesudoHuberParam param_;

 public:
  void Configure(Args const& args) override { param_.UpdateAllowUnknown(args); }
  ObjInfo Task() const override { return ObjInfo::kRegression; }
  uint32_t Targets(MetaInfo const& info) const override {
    return std::max(static_cast<size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(HostDeviceVector<bst_float> const& preds, const MetaInfo& info, int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    CheckRegInputs(info, preds);
    auto slope = param_.huber_slope;
    CHECK_NE(slope, 0.0) << "slope for pseudo huber cannot be 0.";
    auto labels = info.labels.View(ctx_->gpu_id);

    out_gpair->SetDevice(ctx_->gpu_id);
    out_gpair->Resize(info.labels.Size());
    auto gpair = linalg::MakeVec(out_gpair);

    preds.SetDevice(ctx_->gpu_id);
    auto predt = linalg::MakeVec(&preds);

    info.weights_.SetDevice(ctx_->gpu_id);
    common::OptionalWeights weight{ctx_->IsCPU() ? info.weights_.ConstHostSpan()
                                                 : info.weights_.ConstDeviceSpan()};

    linalg::ElementWiseKernel(ctx_, labels, [=] XGBOOST_DEVICE(size_t i, float const y) mutable {
      auto sample_id = std::get<0>(linalg::UnravelIndex(i, labels.Shape()));
      const float z = predt(i) - y;
      const float scale_sqrt = std::sqrt(1 + common::Sqr(z) / common::Sqr(slope));
      float grad = z / scale_sqrt;

      auto scale = common::Sqr(slope) + common::Sqr(z);
      float hess = common::Sqr(slope) / (scale * scale_sqrt);

      auto w = weight[sample_id];
      gpair(i) = {grad * w, hess * w};
    });
  }

  const char* DefaultEvalMetric() const override { return "mphe"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:pseudohubererror");
    out["pseduo_huber_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    if (config.find("pseduo_huber_param") == config.cend()) {
      // The parameter is added in 1.6.
      return;
    }
    FromJson(in["pseduo_huber_param"], &param_);
  }
};

XGBOOST_REGISTER_OBJECTIVE(PseudoHuberRegression, "reg:pseudohubererror")
    .describe("Regression Pseudo Huber error.")
    .set_body([]() { return new PseudoHuberRegression(); });

// declare parameter
struct PoissonRegressionParam : public XGBoostParameter<PoissonRegressionParam> {
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
    param_.UpdateAllowUnknown(args);
  }

  ObjInfo Task() const override { return ObjInfo::kRegression; }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info, int,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels.Size()) << "labels are not correctly provided";
    size_t const ndata = preds.Size();
    out_gpair->Resize(ndata);
    auto device = ctx_->gpu_id;
    label_correct_.Resize(1);
    label_correct_.Fill(1);

    bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }
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
        common::Range{0, static_cast<int64_t>(ndata)}, this->ctx_->Threads(), device).Eval(
            &label_correct_, out_gpair, &preds, info.labels.Data(), &info.weights_);
    // copy "label correct" flags back to host
    std::vector<int>& label_correct_h = label_correct_.HostVector();
    for (auto const flag : label_correct_h) {
      if (flag == 0) {
        LOG(FATAL) << "PoissonRegression: label must be nonnegative";
      }
    }
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) const override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<bst_float> _preds) {
          _preds[_idx] = expf(_preds[_idx]);
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size())}, this->ctx_->Threads(),
        io_preds->DeviceIdx())
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

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("count:poisson");
    out["poisson_regression_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    FromJson(in["poisson_regression_param"], &param_);
  }

 private:
  PoissonRegressionParam param_;
  HostDeviceVector<int> label_correct_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(PoissonRegressionParam);

XGBOOST_REGISTER_OBJECTIVE(PoissonRegression, "count:poisson")
.describe("Poisson regression for count data.")
.set_body([]() { return new PoissonRegression(); });


// cox regression for survival data (negative values mean they are censored)
class CoxRegression : public ObjFunction {
 public:
  void Configure(Args const&) override {}
  ObjInfo Task() const override { return ObjInfo::kRegression; }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info, int,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels.Size()) << "labels are not correctly provided";
    const auto& preds_h = preds.HostVector();
    out_gpair->Resize(preds_h.size());
    auto& gpair = out_gpair->HostVector();
    const std::vector<size_t> &label_order = info.LabelAbsSort();

    const omp_ulong ndata = static_cast<omp_ulong>(preds_h.size()); // NOLINT(*)
    const bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }

    // pre-compute a sum
    double exp_p_sum = 0;  // we use double because we might need the precision with large datasets
    for (omp_ulong i = 0; i < ndata; ++i) {
      exp_p_sum += std::exp(preds_h[label_order[i]]);
    }

    // start calculating grad and hess
    const auto& labels = info.labels.HostView();
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
      const double y = labels(ind);
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
  void PredTransform(HostDeviceVector<bst_float> *io_preds) const override {
    std::vector<bst_float> &preds = io_preds->HostVector();
    const long ndata = static_cast<long>(preds.size()); // NOLINT(*)
    common::ParallelFor(ndata, ctx_->Threads(), [&](long j) { // NOLINT(*)
      preds[j] = std::exp(preds[j]);
    });
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

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("survival:cox");
  }
  void LoadConfig(Json const&) override {}
};

// register the objective function
XGBOOST_REGISTER_OBJECTIVE(CoxRegression, "survival:cox")
.describe("Cox regression for censored survival data (negative labels are considered censored).")
.set_body([]() { return new CoxRegression(); });

// gamma regression
class GammaRegression : public ObjFunction {
 public:
  void Configure(Args const&) override {}
  ObjInfo Task() const override { return ObjInfo::kRegression; }

  void GetGradient(const HostDeviceVector<bst_float> &preds,
                   const MetaInfo &info, int,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels.Size()) << "labels are not correctly provided";
    const size_t ndata = preds.Size();
    auto device = ctx_->gpu_id;
    out_gpair->Resize(ndata);
    label_correct_.Resize(1);
    label_correct_.Fill(1);

    const bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }
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
          if (y <= 0.0f) {
            _label_correct[0] = 0;
          }
          _out_gpair[_idx] = GradientPair((1 - y / expf(p)) * w, y / expf(p) * w);
        },
        common::Range{0, static_cast<int64_t>(ndata)}, this->ctx_->Threads(), device).Eval(
            &label_correct_, out_gpair, &preds, info.labels.Data(), &info.weights_);

    // copy "label correct" flags back to host
    std::vector<int>& label_correct_h = label_correct_.HostVector();
    for (auto const flag : label_correct_h) {
      if (flag == 0) {
        LOG(FATAL) << "GammaRegression: label must be positive.";
      }
    }
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) const override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<bst_float> _preds) {
          _preds[_idx] = expf(_preds[_idx]);
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size())}, this->ctx_->Threads(),
        io_preds->DeviceIdx())
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
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:gamma");
  }
  void LoadConfig(Json const&) override {}

 private:
  HostDeviceVector<int> label_correct_;
};

// register the objective functions
XGBOOST_REGISTER_OBJECTIVE(GammaRegression, "reg:gamma")
.describe("Gamma regression for severity data.")
.set_body([]() { return new GammaRegression(); });


// declare parameter
struct TweedieRegressionParam : public XGBoostParameter<TweedieRegressionParam> {
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
    param_.UpdateAllowUnknown(args);
    std::ostringstream os;
    os << "tweedie-nloglik@" << param_.tweedie_variance_power;
    metric_ = os.str();
  }

  ObjInfo Task() const override { return ObjInfo::kRegression; }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info, int,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels.Size()) << "labels are not correctly provided";
    const size_t ndata = preds.Size();
    out_gpair->Resize(ndata);

    auto device = ctx_->gpu_id;
    label_correct_.Resize(1);
    label_correct_.Fill(1);

    const bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }

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
        common::Range{0, static_cast<int64_t>(ndata), 1}, this->ctx_->Threads(), device)
        .Eval(&label_correct_, out_gpair, &preds, info.labels.Data(), &info.weights_);

    // copy "label correct" flags back to host
    std::vector<int>& label_correct_h = label_correct_.HostVector();
    for (auto const flag : label_correct_h) {
      if (flag == 0) {
        LOG(FATAL) << "TweedieRegression: label must be nonnegative";
      }
    }
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) const override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<bst_float> _preds) {
          _preds[_idx] = expf(_preds[_idx]);
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size())}, this->ctx_->Threads(),
        io_preds->DeviceIdx())
        .Eval(io_preds);
  }

  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }

  const char* DefaultEvalMetric() const override {
    return metric_.c_str();
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:tweedie");
    out["tweedie_regression_param"] = ToJson(param_);
  }
  void LoadConfig(Json const& in) override {
    FromJson(in["tweedie_regression_param"], &param_);
  }

 private:
  std::string metric_;
  TweedieRegressionParam param_;
  HostDeviceVector<int> label_correct_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(TweedieRegressionParam);

XGBOOST_REGISTER_OBJECTIVE(TweedieRegression, "reg:tweedie")
.describe("Tweedie regression for insurance data.")
.set_body([]() { return new TweedieRegression(); });

void SegmentedPercentile(Context const* ctx, double alpha, RowIndexCache const& row_index,
                         MetaInfo const& info, HostDeviceVector<float> const& predt,
                         HostDeviceVector<float>* quantiles) {
  CHECK(alpha >= 0 && alpha <= 1);

  auto d_predt = predt.ConstDeviceSpan();
  auto d_labels = info.labels.View(ctx->gpu_id);
  linalg::Tensor<float, 2> residue{d_labels.Shape(), ctx->gpu_id};
  auto d_residue = residue.View(ctx->gpu_id);
  CHECK_EQ(d_predt.size(), d_labels.Size());
  linalg::ElementWiseKernel(ctx, d_labels, [=] XGBOOST_DEVICE(size_t i, float y) mutable {
    size_t sample_id, target_id;
    std::tie(sample_id, target_id) = linalg::UnravelIndex(i, d_labels.Shape());
    d_residue(sample_id, target_id) = y - d_predt[i];
  });

  dh::device_vector<size_t> segment_idx(row_index.indptr.size() + 1, 0);
  auto d_segment_idx = dh::ToSpan(segment_idx);
  dh::device_vector<RowIndexCache::Segment> indptr(row_index.indptr);
  auto d_indptr = dh::ToSpan(indptr);
  dh::LaunchN(d_segment_idx.size(), [=] XGBOOST_DEVICE(size_t i) {
    if (i == d_segment_idx.size() - 1) {
      d_segment_idx[i] = d_indptr[i].begin + d_indptr[i].n;
      return;
    }
    d_segment_idx[i] = d_indptr[i].begin;
  });

  dh::XGBDeviceAllocator<char> alloc;
  dh::device_vector<size_t> sorted_idx(d_labels.Shape(0));
  dh::Iota(dh::ToSpan(sorted_idx));
  using Tup = thrust::tuple<size_t, float>;
  auto key_it = dh::MakeTransformIterator<Tup>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) -> Tup {
        size_t sample_id, target_id;
        std::tie(sample_id, target_id) = linalg::UnravelIndex(i, d_labels.Shape());
        auto leaf_idx = dh::SegmentId(d_segment_idx, sample_id);
        auto residue = d_residue(sample_id, target_id);
        return thrust::make_tuple(leaf_idx, residue);
      });
  dh::device_vector<Tup> keys(residue.Size());
  dh::XGBCachingDeviceAllocator<char> caching;
  thrust::copy(thrust::cuda::par(caching), key_it, key_it + keys.size(), keys.begin());

  thrust::stable_sort_by_key(thrust::cuda::par(alloc), keys.begin(), keys.end(), sorted_idx.begin(),
                             [=] XGBOOST_DEVICE(Tup const& l, Tup const& r) {
                               if (thrust::get<0>(l) != thrust::get<0>(r)) {
                                 return thrust::get<0>(l) < thrust::get<0>(r);  // segment index
                               }
                               return thrust::get<1>(l) < thrust::get<1>(r);  // residue
                             });

  dh::caching_device_vector<RowIndexCache::Segment> segment(row_index.indptr.size());
  thrust::copy(row_index.indptr.cbegin(), row_index.indptr.cend(), segment.begin());
  auto d_segments = dh::ToSpan(segment);

  quantiles->Resize(row_index.indptr.size());
  auto d_results = quantiles->DeviceSpan();

  auto d_row_index = row_index.row_index.ConstDeviceSpan();
  auto d_sorted_idx = dh::ToSpan(sorted_idx);
  auto d_keys = dh::ToSpan(keys);

  dh::LaunchN(residue.Size(), [=] XGBOOST_DEVICE(size_t i) {
    size_t sample_id, target_id;
    std::tie(sample_id, target_id) = linalg::UnravelIndex(i, d_labels.Shape());
    // each segment is the index of a leaf.
    size_t seg_idx = thrust::get<0>(d_keys[i]);
    auto seg = d_segments[seg_idx];

    double x = alpha * static_cast<double>(seg.n + 1);
    double k = std::floor(x) - 1;
    double d = (x - 1) - k;

    if (i == seg.begin) {
      auto v0 = d_residue(d_row_index[d_sorted_idx[static_cast<size_t>(k)]], target_id);
      auto v1 = d_residue(d_row_index[d_sorted_idx[static_cast<size_t>(k) + 1]], target_id);
      d_results[seg_idx] = v0 + d * (v1 - v0);
    }
  });
}

void UpdateTreeLeafDevice(Context const* ctx, common::Span<RowIndexCache const> row_index,
                          MetaInfo const& info, HostDeviceVector<float> const& prediction,
                          uint32_t target, float alpha, RegTree* p_tree) {
  dh::safe_cuda(cudaSetDevice(ctx->gpu_id));
  CHECK_EQ(row_index.size(), 1)
      << "External memory with GPU hist should have only 1 row partition.";
  auto const& part = row_index.front();

  HostDeviceVector<float> results;
  SegmentedPercentile(ctx, alpha, part, info, prediction, &results);

  auto const& h_results = results.HostVector();
  auto& tree = *p_tree;
  for (size_t i = 0; i < row_index.front().indptr.size(); ++i) {
    auto seg = row_index.front().indptr[i];
    auto q = h_results[i];
    CHECK(tree[seg.nidx].IsLeaf());
    tree[seg.nidx].SetLeaf(q);  // fixme: exact tree method
  }
}

void UpdateTreeLeafHost(Context const* ctx, common::Span<RowIndexCache const> row_index,
                        MetaInfo const& info, HostDeviceVector<float> const& prediction,
                        uint32_t target, float alpha, RegTree* p_tree) {
  auto& tree = *p_tree;
  std::vector<float> quantiles;
  for (auto const& part : row_index) {
    std::vector<float> results(part.indptr.size());
    common::ParallelFor(part.indptr.size(), ctx->Threads(), [&](size_t k) {
      auto const& seg = part.indptr[k];
      CHECK(tree[seg.nidx].IsLeaf());
      auto h_row_set = part.row_index.HostSpan().subspan(seg.begin, seg.n);
      float q{0};
      auto h_labels = info.labels.HostView().Slice(linalg::All(), target);
      auto const& h_prediction = prediction.ConstHostVector();
      auto iter = common::MakeIndexTransformIter([&](size_t i) -> float {
        auto row_idx = h_row_set[i];
        return h_labels(row_idx) - h_prediction[row_idx];
      });

      if (info.weights_.Empty()) {
        q = common::Percentile(alpha, iter, iter + h_row_set.size());
      } else {
        q = common::WeightedPercentile(alpha, h_row_set,
                                       info.labels.HostView().Slice(linalg::All(), target),
                                       linalg::MakeVec(&info.weights_));
      }
      results.at(k) = q;
    });

    // fixme: verify this is correct for external memory
    if (quantiles.empty()) {
      quantiles.resize(results.size(), 0);
    }
    for (size_t i = 0; i < results.size(); ++i) {
      quantiles[i] += results[i];
    }
  }

  // use the mean value
  rabit::Allreduce<rabit::op::Sum>(quantiles.data(), quantiles.size());
  auto world = rabit::GetWorldSize();
  std::transform(quantiles.begin(), quantiles.end(), quantiles.begin(),
                 [&](float q) { return q / world; });

  // fixme: verify this is correct for external memory
  for (size_t i = 0; i < row_index.front().indptr.size(); ++i) {
    auto seg = row_index.front().indptr[i];
    auto q = quantiles[i];
    CHECK(tree[seg.nidx].IsLeaf());
    tree[seg.nidx].SetLeaf(q);  // fixme: exact tree method
  }
}

class MeanAbsoluteError : public ObjFunction {
 public:
  void Configure(Args const&) override {}

  uint32_t Targets(MetaInfo const& info) const override {
    return std::max(static_cast<size_t>(1), info.labels.Shape(1));
  }

  struct ObjInfo Task() const override {
    return {ObjInfo::kRegression, true, true};
  }

  void GetGradient(HostDeviceVector<bst_float> const& preds, const MetaInfo& info, int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    auto labels = info.labels.View(ctx_->gpu_id);

    out_gpair->SetDevice(ctx_->gpu_id);
    out_gpair->Resize(info.labels.Size());
    auto gpair = linalg::MakeVec(out_gpair);

    preds.SetDevice(ctx_->gpu_id);
    auto predt = linalg::MakeVec(&preds);
    info.weights_.SetDevice(ctx_->gpu_id);
    common::OptionalWeights weight{ctx_->IsCPU() ? info.weights_.ConstHostSpan()
                                                 : info.weights_.ConstDeviceSpan()};

    linalg::ElementWiseKernel(ctx_, labels, [=] XGBOOST_DEVICE(size_t i, float const y) mutable {
      auto sign = [](auto x) {
        return (x > static_cast<decltype(x)>(0)) - (x < static_cast<decltype(x)>(0));
      };
      auto sample_id = std::get<0>(linalg::UnravelIndex(i, labels.Shape()));
      auto grad = sign(predt(i) - y) * weight[i];
      auto hess = weight[sample_id];
      gpair(i) = GradientPair{grad, hess};
    });
  }

  void UpdateTreeLeaf(common::Span<RowIndexCache const> row_index, MetaInfo const& info,
                      HostDeviceVector<float> const& prediction, uint32_t target,
                      RegTree* p_tree) const override {
    if (ctx_->IsCPU()) {
      UpdateTreeLeafHost(ctx_, row_index, info, prediction, target, 0.5, p_tree);
    } else {
      UpdateTreeLeafDevice(ctx_, row_index, info, prediction, target, 0.5, p_tree);
    }
  }

  const char* DefaultEvalMetric() const override { return "mae"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:absoluteerror");
  }

  void LoadConfig(Json const& in) override {}
};

XGBOOST_REGISTER_OBJECTIVE(MeanAbsoluteError, "reg:absoluteerror")
    .describe("Mean absoluate error.")
    .set_body([]() { return new MeanAbsoluteError(); });
}  // namespace obj
}  // namespace xgboost
