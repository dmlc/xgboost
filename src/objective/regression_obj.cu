/**
 * Copyright 2015-2025, XGBoost Contributors
 * \file regression_obj.cu
 * \brief Definition of single-value regression and classification objectives.
 * \author Tianqi Chen, Kailong Chen
 */
#include <dmlc/omp.h>

#include <algorithm>  // for all_of
#include <cmath>
#include <cstdint>  // for  int32_t
#include <vector>   // for vector

#include "../common/common.h"
#include "../common/linalg_op.h"
#include "../common/numeric.h"          // Reduce
#include "../common/optional_weight.h"  // OptionalWeights
#include "../common/pseudo_huber.h"
#include "../common/stats.h"
#include "../common/threading_utils.h"
#include "../common/transform.h"
#include "../common/utils.h"  // for NoOp
#include "./regression_loss.h"
#include "adaptive.h"
#include "init_estimation.h"  // FitIntercept
#include "regression_param.h"
#include "xgboost/base.h"
#include "xgboost/context.h"  // Context
#include "xgboost/data.h"     // MetaInfo
#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/linalg.h"
#include "xgboost/logging.h"
#include "xgboost/objective.h"  // ObjFunction
#include "xgboost/parameter.h"
#include "xgboost/span.h"
#include "xgboost/tree_model.h"  // RegTree

#if defined(XGBOOST_USE_CUDA)
#include "../common/algorithm.cuh"       // for AllOf
#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/device_helpers.cuh"  // for MakeIndexTransformIter
#include "../common/linalg_op.cuh"
#endif  // defined(XGBOOST_USE_CUDA)

#if defined(XGBOOST_USE_SYCL)
#include "../../plugin/sycl/common/linalg_op.h"
#endif

namespace xgboost::obj {
namespace {
void CheckRegInputs(MetaInfo const& info, HostDeviceVector<float> const& preds) {
  CheckInitInputs(info);
  CHECK_EQ(info.labels.Size(), preds.Size()) << "Invalid shape of labels.";
}

template <typename Loss>
void ValidateLabel(Context const* ctx, MetaInfo const& info) {
  auto label = info.labels.View(ctx->Device());
  auto valid = ctx->DispatchDevice(
      [&] {
        return std::all_of(linalg::cbegin(label), linalg::cend(label),
                           [](float y) -> bool { return Loss::CheckLabel(y); });
      },
      [&] {
#if defined(XGBOOST_USE_CUDA)
        auto it = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) -> float {
          auto [m, n] = linalg::UnravelIndex(i, label.Shape());
          return label(m, n);
        });
        return common::AllOf(ctx->CUDACtx()->CTP(), it, it + label.Size(),
                             [] XGBOOST_DEVICE(float y) { return Loss::CheckLabel(y); });
#else
        common::AssertGPUSupport();
        return false;
#endif  // defined(XGBOOST_USE_CUDA)
      },
      [&] {
#if defined(XGBOOST_USE_SYCL)
        return sycl::linalg::Validate(ctx->Device(), label,
                                      [](float y) -> bool { return Loss::CheckLabel(y); });
#else
        common::AssertSYCLSupport();
        return false;
#endif  // defined(XGBOOST_USE_SYCL)
      });
  if (!valid) {
    LOG(FATAL) << Loss::LabelErrorMsg();
  }
  if (!info.weights_.Empty()) {
    CHECK_EQ(info.weights_.Size(), info.num_row_)
        << "Number of weights should be equal to the number of data points.";
  }
}

template <typename Fn, typename Chk = common::NoOp<bool>, typename Err = common::NoOp<StringView>>
void ProbToMarginImpl(Context const* ctx, linalg::Vector<float>* base_score, Fn&& fn,
                      Chk check = common::NoOp{true}, Err error = common::NoOp<StringView>{{}}) {
  auto intercept = base_score->View(ctx->Device());
  bool is_valid = ctx->DispatchDevice(
      [&] { return std::all_of(linalg::cbegin(intercept), linalg::cend(intercept), check); },
      [&] {
#if defined(XGBOOST_USE_CUDA)
        return common::AllOf(ctx->CUDACtx()->CTP(), linalg::tcbegin(intercept),
                             linalg::tcend(intercept), check);
#else
        common::AssertGPUSupport();
        return false;
#endif  // defined(XGBOOST_USE_CUDA)
      });
  CHECK(is_valid) << error();
  linalg::ElementWiseKernel(ctx, intercept, [=] XGBOOST_DEVICE(std::size_t i) mutable {
    intercept(i) = fn(intercept(i));
  });
}
}  // anonymous namespace

#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(regression_obj_gpu);
#endif  // defined(XGBOOST_USE_CUDA)

template<typename Loss>
class RegLossObj : public FitInterceptGlmLike {
 protected:
  HostDeviceVector<float> additional_input_;

 public:
  // 0 - scale_pos_weight, 1 - is_null_weight
  RegLossObj() : additional_input_(2) {}

  void Configure(Args const& args) override { param_.UpdateAllowUnknown(args); }

  [[nodiscard]] ObjInfo Task() const override { return Loss::Info(); }

  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    // Multi-target regression.
    return std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(const HostDeviceVector<float>& preds, const MetaInfo& info, std::int32_t iter,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    CheckRegInputs(info, preds);
    if (iter == 0) {
      ValidateLabel<Loss>(this->ctx_, info);
    }

    size_t const ndata = preds.Size();
    out_gpair->SetDevice(ctx_->Device());
    auto device = ctx_->Device();

    bool is_null_weight = info.weights_.Size() == 0;
    auto scale_pos_weight = param_.scale_pos_weight;
    additional_input_.HostVector().begin()[0] = scale_pos_weight;
    additional_input_.HostVector().begin()[1] = is_null_weight;

    const size_t nthreads = ctx_->Threads();
    bool on_device = !device.IsCPU();
    // On CPU we run the transformation each thread processing a contigious block of data
    // for better performance.
    const size_t n_data_blocks = std::max(static_cast<size_t>(1), (on_device ? ndata : nthreads));
    const size_t block_size = ndata / n_data_blocks + !!(ndata % n_data_blocks);
    auto const n_targets = this->Targets(info);
    out_gpair->Reshape(info.num_row_, n_targets);

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
          const float _scale_pos_weight = _additional_input[0];
          const bool _is_null_weight = _additional_input[1];

          for (size_t idx = begin; idx < end; ++idx) {
            bst_float p = Loss::PredTransform(preds_ptr[idx]);
            bst_float w = _is_null_weight ? 1.0f : weights_ptr[idx / n_targets];
            bst_float label = labels_ptr[idx];
            if (label == 1.0f) {
              w *= _scale_pos_weight;
            }
            out_gpair_ptr[idx] = GradientPair(Loss::FirstOrderGradient(p, label) * w,
                                              Loss::SecondOrderGradient(p, label) * w);
          }
        },
        common::Range{0, static_cast<int64_t>(n_data_blocks)}, nthreads, device)
        .Eval(&additional_input_, out_gpair->Data(), &preds, info.labels.Data(),
              &info.weights_);
  }

 public:
  [[nodiscard]] const char* DefaultEvalMetric() const override {
    return Loss::DefaultEvalMetric();
  }

  void PredTransform(HostDeviceVector<float> *io_preds) const override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<float> _preds) {
          _preds[_idx] = Loss::PredTransform(_preds[_idx]);
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size())}, this->ctx_->Threads(),
        io_preds->Device())
        .Eval(io_preds);
  }

  void InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) const override {
    if (std::abs(this->param_.scale_pos_weight - 1.0f) > kRtEps) {
      // Use newton method if `scale_pos_weight` is present. The alternative is to use
      // weighted mean, but we also need to take sample weight into account.
      FitIntercept::InitEstimation(info, base_score);
    } else {
      FitInterceptGlmLike::InitEstimation(info, base_score);
    }
  }

  void ProbToMargin(linalg::Vector<float>* base_score) const override {
    ProbToMarginImpl(
        this->ctx_, base_score, [] XGBOOST_DEVICE(float v) { return Loss::ProbToMargin(v); },
        [] XGBOOST_DEVICE(float v) { return Loss::CheckIntercept(v); }, Loss::InterceptErrorMsg);
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(Loss::Name());
    out["reg_loss_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    auto obj = get<Object const>(in);
    auto it = obj.find("reg_loss_param");
    if (it != obj.cend()) {
      FromJson(it->second, &param_);
    }
  }

 protected:
  RegLossParam param_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(RegLossParam);

XGBOOST_REGISTER_OBJECTIVE(SquaredLossRegression, LinearSquareLoss::Name())
.describe("Regression with squared error.")
.set_body([]() { return new RegLossObj<LinearSquareLoss>(); });

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

XGBOOST_REGISTER_OBJECTIVE(GammaRegression, GammaDeviance::Name())
    .describe("Gamma regression using the gamma deviance loss with log link.")
    .set_body([]() { return new RegLossObj<GammaDeviance>(); });

// Deprecated functions
XGBOOST_REGISTER_OBJECTIVE(LinearRegression, "reg:linear")
.describe("Regression with squared error.")
.set_body([]() {
    LOG(WARNING) << "reg:linear is now deprecated in favor of reg:squarederror.";
    return new RegLossObj<LinearSquareLoss>(); });
// End deprecated

class SquaredLogErrorRegression : public FitIntercept {
 public:
  static auto Name() { return SquaredLogError::Name(); }

  void Configure(Args const&) override {}
  [[nodiscard]] ObjInfo Task() const override { return ObjInfo::kRegression; }
  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    return std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
  }
  void GetGradient(HostDeviceVector<bst_float> const& preds, const MetaInfo& info,
                   std::int32_t iter, linalg::Matrix<GradientPair>* out_gpair) override {
    if (iter == 0) {
      ValidateLabel<SquaredLogError>(this->ctx_, info);
    }
    auto labels = info.labels.View(ctx_->Device());

    out_gpair->SetDevice(ctx_->Device());
    out_gpair->Reshape(info.num_row_, this->Targets(info));
    auto gpair = out_gpair->View(ctx_->Device());

    preds.SetDevice(ctx_->Device());
    auto predt = linalg::MakeTensorView(ctx_, &preds, info.num_row_, this->Targets(info));

    info.weights_.SetDevice(ctx_->Device());
    common::OptionalWeights weight{ctx_->IsCPU() ? info.weights_.ConstHostSpan()
                                                 : info.weights_.ConstDeviceSpan()};
    linalg::ElementWiseKernel(this->ctx_, labels,
                              [=] XGBOOST_DEVICE(std::size_t i, std::size_t j) mutable {
                                auto p = predt(i, j);
                                auto y = labels(i, j);
                                auto w = weight[i];
                                auto grad = SquaredLogError::FirstOrderGradient(p, y);
                                auto hess = SquaredLogError::SecondOrderGradient(p, y);
                                gpair(i) = {grad * w, hess * w};
                              });
  }
  [[nodiscard]] const char* DefaultEvalMetric() const override { return "rmsle"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(Name());
  }
  void LoadConfig(Json const&) override {}
};

XGBOOST_REGISTER_OBJECTIVE(SquaredLogErrorRegression, SquaredLogErrorRegression::Name())
    .describe("Root mean squared log error.")
    .set_body([]() { return new SquaredLogErrorRegression(); });

class PseudoHuberRegression : public FitIntercept {
  PseudoHuberParam param_;

 public:
  void Configure(Args const& args) override { param_.UpdateAllowUnknown(args); }
  [[nodiscard]] ObjInfo Task() const override { return ObjInfo::kRegression; }
  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    return std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(HostDeviceVector<bst_float> const& preds, const MetaInfo& info, int /*iter*/,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    CheckRegInputs(info, preds);
    auto slope = param_.huber_slope;
    CHECK_NE(slope, 0.0) << "slope for pseudo huber cannot be 0.";
    auto labels = info.labels.View(ctx_->Device());

    out_gpair->SetDevice(ctx_->Device());
    out_gpair->Reshape(info.num_row_, this->Targets(info));
    auto gpair = out_gpair->View(ctx_->Device());

    preds.SetDevice(ctx_->Device());
    auto predt = linalg::MakeTensorView(ctx_, &preds, info.num_row_, this->Targets(info));

    info.weights_.SetDevice(ctx_->Device());
    common::OptionalWeights weight{ctx_->IsCPU() ? info.weights_.ConstHostSpan()
                                                 : info.weights_.ConstDeviceSpan()};

    linalg::ElementWiseKernel(
        ctx_, labels, [=] XGBOOST_DEVICE(std::size_t i, std::size_t j) mutable {
          float z = predt(i, j) - labels(i, j);
          float scale_sqrt = std::sqrt(1 + common::Sqr(z) / common::Sqr(slope));
          float grad = z / scale_sqrt;

          auto scale = common::Sqr(slope) + common::Sqr(z);
          float hess = common::Sqr(slope) / (scale * scale_sqrt);

          auto w = weight[i];
          gpair(i) = {grad * w, hess * w};
        });
  }

  [[nodiscard]] const char* DefaultEvalMetric() const override { return "mphe"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:pseudohubererror");
    out["pseudo_huber_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    if (config.find("pseudo_huber_param") == config.cend()) {
      // The parameter is added in 1.6.
      return;
    }
    FromJson(in["pseudo_huber_param"], &param_);
  }
  [[nodiscard]] Json DefaultMetricConfig() const override {
    CHECK(param_.GetInitialised());
    Json config{Object{}};
    config["name"] = String{this->DefaultEvalMetric()};
    config["pseudo_huber_param"] = ToJson(param_);
    return config;
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
class PoissonRegression : public FitInterceptGlmLike {
 public:
  // declare functions
  void Configure(Args const& args) override { param_.UpdateAllowUnknown(args); }

  [[nodiscard]] ObjInfo Task() const override { return ObjInfo::kRegression; }

  void GetGradient(const HostDeviceVector<bst_float>& preds, const MetaInfo& info, int,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    CHECK_NE(info.labels.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels.Size()) << "labels are not correctly provided";
    size_t const ndata = preds.Size();
    out_gpair->SetDevice(ctx_->Device());
    out_gpair->Reshape(info.num_row_, this->Targets(info));
    auto device = ctx_->Device();
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
            &label_correct_, out_gpair->Data(), &preds, info.labels.Data(), &info.weights_);
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
        io_preds->Device())
        .Eval(io_preds);
  }
  void EvalTransform(HostDeviceVector<bst_float> *io_preds) override {
    PredTransform(io_preds);
  }
  void ProbToMargin(linalg::Vector<float>* base_score) const override {
    ProbToMarginImpl(this->ctx_, base_score, [] XGBOOST_DEVICE(float v) { return std::log(v); });
  }
  [[nodiscard]] const char* DefaultEvalMetric() const override {
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
class CoxRegression : public FitIntercept {
 public:
  void Configure(Args const&) override {}
  [[nodiscard]] ObjInfo Task() const override { return ObjInfo::kRegression; }

  void GetGradient(const HostDeviceVector<bst_float>& preds, const MetaInfo& info, int,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    CHECK_NE(info.labels.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels.Size()) << "labels are not correctly provided";
    const auto& preds_h = preds.HostVector();
    out_gpair->Reshape(info.num_row_, this->Targets(info));
    auto gpair = out_gpair->HostView();
    const std::vector<size_t> &label_order = info.LabelAbsSort(ctx_);

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
      const double hess = exp_p * r_k - exp_p * exp_p * s_k;
      gpair(ind) = GradientPair(grad * w, hess * w);

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
  void ProbToMargin(linalg::Vector<float>* base_score) const override {
    ProbToMarginImpl(this->ctx_, base_score, [] XGBOOST_DEVICE(float v) { return std::log(v); });
  }
  [[nodiscard]] const char* DefaultEvalMetric() const override {
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


// declare parameter
struct TweedieRegressionParam : public XGBoostParameter<TweedieRegressionParam> {
  float tweedie_variance_power;
  DMLC_DECLARE_PARAMETER(TweedieRegressionParam) {
    DMLC_DECLARE_FIELD(tweedie_variance_power).set_range(1.0f, 2.0f).set_default(1.5f)
      .describe("Tweedie variance power.  Must be between in range [1, 2).");
  }
};

// tweedie regression
class TweedieRegression : public FitInterceptGlmLike {
 public:
  // declare functions
  void Configure(Args const& args) override {
    param_.UpdateAllowUnknown(args);
    std::ostringstream os;
    os << "tweedie-nloglik@" << param_.tweedie_variance_power;
    metric_ = os.str();
  }

  [[nodiscard]] ObjInfo Task() const override { return ObjInfo::kRegression; }

  void GetGradient(const HostDeviceVector<bst_float>& preds, const MetaInfo& info, std::int32_t,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    CHECK_NE(info.labels.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels.Size()) << "labels are not correctly provided";
    const size_t ndata = preds.Size();
    out_gpair->SetDevice(ctx_->Device());
    out_gpair->Reshape(info.num_row_, this->Targets(info));

    auto device = ctx_->Device();
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
        .Eval(&label_correct_, out_gpair->Data(), &preds, info.labels.Data(), &info.weights_);

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
        io_preds->Device())
        .Eval(io_preds);
  }
  void ProbToMargin(linalg::Vector<float>* base_score) const override {
    ProbToMarginImpl(this->ctx_, base_score, [] XGBOOST_DEVICE(float v) { return std::log(v); });
  }

  [[nodiscard]] const char* DefaultEvalMetric() const override {
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

class MeanAbsoluteError : public ObjFunction {
 public:
  void Configure(Args const&) override {}
  [[nodiscard]] ObjInfo Task() const override { return {ObjInfo::kRegression, true, true}; }
  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    return std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(HostDeviceVector<float> const& preds, const MetaInfo& info,
                   std::int32_t /*iter*/, linalg::Matrix<GradientPair>* out_gpair) override {
    CheckRegInputs(info, preds);
    auto labels = info.labels.View(ctx_->Device());

    out_gpair->SetDevice(ctx_->Device());
    out_gpair->Reshape(info.num_row_, this->Targets(info));
    auto gpair = out_gpair->View(ctx_->Device());

    preds.SetDevice(ctx_->Device());
    auto predt = linalg::MakeTensorView(ctx_, &preds, info.num_row_, this->Targets(info));
    info.weights_.SetDevice(ctx_->Device());
    common::OptionalWeights weight{ctx_->IsCPU() ? info.weights_.ConstHostSpan()
                                                 : info.weights_.ConstDeviceSpan()};

    linalg::ElementWiseKernel(
        ctx_, labels, [=] XGBOOST_DEVICE(std::size_t i, std::size_t j) mutable {
          auto sign = [](auto x) {
            return (x > static_cast<decltype(x)>(0)) - (x < static_cast<decltype(x)>(0));
          };
          auto y = labels(i, j);
          auto hess = weight[i];
          auto grad = sign(predt(i, j) - y) * hess;
          gpair(i, j) = GradientPair{grad, hess};
        });
  }

  void InitEstimation(MetaInfo const& info, linalg::Tensor<float, 1>* base_score) const override {
    CheckInitInputs(info);
    base_score->Reshape(this->Targets(info));

    double sum_weight{0.0};
    if (info.weights_.Empty()) {
      sum_weight = static_cast<double>(info.num_row_);
    } else {
      sum_weight = common::Reduce(ctx_, info.weights_);
    }

    if (info.num_row_ == 0) {
      auto out = base_score->HostView();
      std::fill(linalg::begin(out), linalg::end(out), 0.0f);
    } else {
      common::Median(ctx_, info.labels, info.weights_, base_score);
    }

    auto intercept = base_score->View(this->ctx_->Device());
    // weighted avg
    linalg::VecScaMul(this->ctx_, intercept, sum_weight);
    auto rc = collective::GlobalSum(ctx_, info, intercept, &sum_weight);
    collective::SafeColl(rc);

    if (common::CloseTo(sum_weight, 0.0)) {
      // Mostly for handling empty dataset test.
      LOG(WARNING) << "Sum of weights is close to 0.0, skipping base score estimation.";
      *base_score = linalg::Zeros<float>(ctx_, base_score->Shape(0));
      return;
    }

    linalg::VecScaDiv(this->ctx_, intercept, sum_weight);
  }

  void UpdateTreeLeaf(HostDeviceVector<bst_node_t> const& position, MetaInfo const& info,
                      float learning_rate, HostDeviceVector<float> const& prediction,
                      std::int32_t group_idx, RegTree* p_tree) const override {
    ::xgboost::obj::UpdateTreeLeaf(ctx_, position, group_idx, info, learning_rate, prediction, 0.5,
                                   p_tree);
  }

  [[nodiscard]] const char* DefaultEvalMetric() const override { return "mae"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:absoluteerror");
  }

  void LoadConfig(Json const& in) override {
    CHECK_EQ(StringView{get<String const>(in["name"])}, StringView{"reg:absoluteerror"});
  }
};

XGBOOST_REGISTER_OBJECTIVE(MeanAbsoluteError, "reg:absoluteerror")
    .describe("Mean absoluate error.")
    .set_body([]() { return new MeanAbsoluteError(); });
}  // namespace xgboost::obj
