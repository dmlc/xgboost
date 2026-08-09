/**
 * Copyright 2015-2026, XGBoost Contributors
 * \file regression_obj.cu
 * \brief Definition of single-value regression and classification objectives.
 * \author Tianqi Chen, Kailong Chen
 */
#include <dmlc/omp.h>

#include <algorithm>  // for all_of
#include <cmath>
#include <cstdint>  // for int32_t
#include <memory>   // for unique_ptr
#include <vector>   // for vector

#include "../collective/aggregator.h"
#include "../common/common.h"
#include "../common/expectile_loss_utils.h"  // for ExpectileLossParam
#include "../common/linalg_op.h"             // for ElementWiseKernel
#include "../common/numeric.h"               // for Reduce
#include "../common/optional_weight.h"       // for MakeOptionalWeights
#include "../common/stats.h"
#include "../common/threading_utils.h"
#include "../common/transform.h"
#include "../common/utils.h"  // for NoOp
#include "../tree/fit_stump.h"
#include "./regression_loss.h"
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

#if defined(XGBOOST_USE_CUDA)
#include "../common/algorithm.cuh"       // for AllOf
#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/device_helpers.cuh"  // for MakeIndexTransformIter
#endif                                   // defined(XGBOOST_USE_CUDA)

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
      },
      [&] {
#if defined(XGBOOST_USE_SYCL)
        return sycl::linalg::Validate(ctx->Device(), intercept, check);
#else
        common::AssertSYCLSupport();
        return false;
#endif  // defined(XGBOOST_USE_SYCL)
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

template <typename Loss>
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
            common::Span<GradientPair> _out_gpair, common::Span<const bst_float> _preds,
            common::Span<const bst_float> _labels, common::Span<const bst_float> _weights) {
          const bst_float* preds_ptr = _preds.data();
          const bst_float* labels_ptr = _labels.data();
          const bst_float* weights_ptr = _weights.data();
          GradientPair* out_gpair_ptr = _out_gpair.data();
          const size_t begin = data_block_idx * block_size;
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
        .Eval(&additional_input_, out_gpair->Data(), &preds, info.labels.Data(), &info.weights_);
  }

 public:
  [[nodiscard]] const char* DefaultEvalMetric() const override { return Loss::DefaultEvalMetric(); }

  void PredTransform(HostDeviceVector<float>* io_preds) const override {
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
    .describe(
        "Logistic regression for classification, output score "
        "before logistic transformation.")
    .set_body([]() { return new RegLossObj<LogisticRaw>(); });

// Deprecated functions
XGBOOST_REGISTER_OBJECTIVE(LinearRegression, "reg:linear")
    .describe("Regression with squared error.")
    .set_body([]() {
      LOG(WARNING) << "reg:linear is now deprecated in favor of reg:squarederror.";
      return new RegLossObj<LinearSquareLoss>();
    });
// End deprecated

class ExpectileRegression : public FitIntercept {
  common::ExpectileLossParam param_;
  HostDeviceVector<float> alpha_;

  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    auto const& alpha = param_.expectile_alpha.Get();
    CHECK_EQ(alpha.size(), alpha_.Size()) << "The objective is not yet configured.";
    CHECK_EQ(info.labels.Shape(1), 1) << "Multi-target is not yet supported by the expectile loss.";
    CHECK(!alpha.empty());
    auto n_y = std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
    return alpha_.Size() * n_y;
  }

 public:
  void Configure(Args const& args) override {
    param_.UpdateAllowUnknown(args);
    param_.Validate();
    alpha_.HostVector() = param_.expectile_alpha.Get();
  }

  [[nodiscard]] ObjInfo Task() const override { return ObjInfo::kRegression; }

  void GetGradient(HostDeviceVector<float> const& preds, const MetaInfo& info, std::int32_t iter,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    if (iter == 0) {
      CheckInitInputs(info);
    }
    CHECK_EQ(param_.expectile_alpha.Get().size(), alpha_.Size());

    using SizeT = decltype(info.num_row_);
    SizeT n_targets = this->Targets(info);
    SizeT n_alphas = alpha_.Size();
    CHECK_NE(n_alphas, 0);
    CHECK_GE(n_targets, n_alphas);
    CHECK_EQ(preds.Size(), info.num_row_ * n_targets);

    auto labels = info.labels.View(ctx_->Device());

    out_gpair->SetDevice(ctx_->Device());
    CHECK_EQ(info.labels.Shape(1), 1)
        << "Multi-target for expectile regression is not yet supported.";
    out_gpair->Reshape(info.num_row_, n_targets);
    auto gpair = out_gpair->View(ctx_->Device());

    info.weights_.SetDevice(ctx_->Device());
    auto weights = common::MakeOptionalWeights(ctx_->Device(), info.weights_);

    preds.SetDevice(ctx_->Device());
    auto predt = linalg::MakeTensorView(ctx_, &preds, info.num_row_, n_targets);

    alpha_.SetDevice(ctx_->Device());
    auto alpha = ctx_->IsCPU() ? alpha_.ConstHostSpan() : alpha_.ConstDeviceSpan();

    linalg::ElementWiseKernel(
        ctx_, gpair, [=] XGBOOST_DEVICE(std::size_t i, std::size_t j) mutable {
          auto label = labels(i, 0);
          auto sample_weight = weights[i];
          float pred = predt(i, 0);
          float grad_sum{0.0f};
          float hess_sum{0.0f};
          for (std::size_t k = 0; k < n_alphas; ++k) {
            if (k > 0) {
              pred += kRtEps + common::SoftPlus(predt(i, k));
            }
            if (k >= j) {
              auto diff = pred - label;
              auto expectile = alpha[k];
              auto weight_scale = diff >= 0.0f ? (1.0f - expectile) : expectile;
              grad_sum += weight_scale * diff * sample_weight;
              hess_sum += weight_scale * sample_weight;
            }
          }

          auto scale = j == 0 ? 1.0f : common::Sigmoid(predt(i, j));
          auto grad = scale * grad_sum;
          // Diagonal Gauss-Newton approximation for the transformed margin.
          auto hess = scale * scale * hess_sum;
          gpair(i, j) = GradientPair{grad, hess};
        });
  }

  void InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) const override {
    CHECK(!alpha_.Empty());
    auto n_targets = this->Targets(info);
    base_score->SetDevice(ctx_->Device());
    base_score->Reshape(n_targets);

    linalg::Vector<float> label_mean;
    if (info.weights_.Empty()) {
      common::SampleMean(ctx_, info.labels, &label_mean);
    } else {
      common::WeightedSampleMean(ctx_, info.labels, info.weights_, &label_mean);
    }
    CHECK_EQ(label_mean.Size(), 1);

    auto mean = label_mean.HostView()(0);

    linalg::Matrix<GradientPair> gpair;
    gpair.SetDevice(ctx_->Device());
    gpair.Reshape(info.num_row_, n_targets);
    auto gpair_view = gpair.View(ctx_->Device());

    auto labels = info.labels.View(ctx_->Device());
    info.weights_.SetDevice(ctx_->Device());
    auto weights = common::MakeOptionalWeights(ctx_->Device(), info.weights_);
    alpha_.SetDevice(ctx_->Device());
    auto alpha = ctx_->IsCPU() ? alpha_.ConstHostSpan() : alpha_.ConstDeviceSpan();

    linalg::ElementWiseKernel(ctx_, gpair_view,
                              [=] XGBOOST_DEVICE(std::size_t i, std::size_t j) mutable {
                                auto diff = mean - labels(i, 0);
                                auto expectile = alpha[j];
                                auto weight_scale = diff >= 0.0f ? (1.0f - expectile) : expectile;
                                auto sample_weight = weights[i];
                                auto grad = weight_scale * diff * sample_weight;
                                auto hess = weight_scale * sample_weight;
                                gpair_view(i, j) = GradientPair{grad, hess};
                              });

    tree::FitStump(ctx_, gpair, n_targets, base_score);

    auto out = base_score->HostView();
    for (std::size_t j = 0; j < n_targets; ++j) {
      out(j) += mean;
    }
    for (std::size_t j = 1; j < n_targets; ++j) {
      out(j) = std::max(out(j), out(j - 1));
    }
  }

  void PredTransform(HostDeviceVector<float>* io_preds) const override {
    auto n_alphas = alpha_.Size();
    CHECK_NE(n_alphas, 0);
    CHECK_EQ(io_preds->Size() % n_alphas, 0);
    auto n_samples = io_preds->Size() / n_alphas;
    auto device = io_preds->Device();
    auto predt = linalg::MakeTensorView(
        device, device.IsCPU() ? io_preds->HostSpan() : io_preds->DeviceSpan(), n_samples,
        n_alphas);
    auto rows = predt.Slice(linalg::All(), 0);
    linalg::ElementWiseKernel(ctx_, device, rows, [=] XGBOOST_DEVICE(std::size_t i) mutable {
      auto point = predt.Slice(i, linalg::All());
      float pred = point(0);
      for (std::size_t j = 1; j < n_alphas; ++j) {
        pred += kRtEps + common::SoftPlus(point(j));
        point(j) = pred;
      }
    });
  }

  void ProbToMargin(linalg::Vector<float>* base_score) const override {
    CHECK_EQ(base_score->Size(), alpha_.Size());
    auto margin = base_score->HostView();
    for (std::size_t j = margin.Size() - 1; j > 0; --j) {
      auto gap = margin(j) - margin(j - 1);
      margin(j) = common::SoftPlusInv(gap - kRtEps);
    }
  }

  [[nodiscard]] const char* DefaultEvalMetric() const override { return "expectile"; }
  [[nodiscard]] Json DefaultMetricConfig() const override {
    CHECK(param_.GetInitialised());
    Json config{Object{}};
    config["name"] = String{this->DefaultEvalMetric()};
    config["expectile_loss_param"] = ToJson(param_);
    return config;
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:expectileerror");
    out["expectile_loss_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    CHECK_EQ(get<String const>(in["name"]), "reg:expectileerror");
    auto const& obj = get<Object const>(in);
    auto it = obj.find("expectile_loss_param");
    if (it != obj.cend()) {
      FromJson(it->second, &param_);
      alpha_.HostVector() = param_.expectile_alpha.Get();
    }
  }
};

XGBOOST_REGISTER_OBJECTIVE(ExpectileRegression, "reg:expectileerror")
    .describe("Regression with expectile loss.")
    .set_body([]() { return new ExpectileRegression(); });

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
    const std::vector<size_t>& label_order = info.LabelAbsSort(ctx_);

    const omp_ulong ndata = static_cast<omp_ulong>(preds_h.size());  // NOLINT(*)
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
    for (omp_ulong i = 0; i < ndata; ++i) {  // NOLINT(*)
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
        CHECK(last_abs_y <= abs_y) << "CoxRegression: labels must be in sorted order, "
                                   << "MetaInfo::LabelArgsort failed!";
      }

      if (y > 0) {
        r_k += 1.0 / exp_p_sum;
        s_k += 1.0 / (exp_p_sum * exp_p_sum);
      }

      const double grad = exp_p * r_k - static_cast<bst_float>(y > 0);
      const double hess = exp_p * r_k - exp_p * exp_p * s_k;
      gpair(ind) = GradientPair(grad * w, hess * w);

      last_abs_y = abs_y;
      last_exp_p = exp_p;
    }
  }
  void PredTransform(HostDeviceVector<bst_float>* io_preds) const override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t i, common::Span<bst_float> preds) { preds[i] = expf(preds[i]); },
        common::Range{0, static_cast<int64_t>(io_preds->Size())}, this->ctx_->Threads(),
        io_preds->Device())
        .Eval(io_preds);
  }
  void EvalTransform(HostDeviceVector<bst_float>* io_preds) override { PredTransform(io_preds); }
  void ProbToMargin(linalg::Vector<float>* base_score) const override {
    ProbToMarginImpl(this->ctx_, base_score, [] XGBOOST_DEVICE(float v) { return std::log(v); });
  }
  [[nodiscard]] const char* DefaultEvalMetric() const override { return "cox-nloglik"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("survival:cox");
  }
  void LoadConfig(Json const&) override {}
};

// register the objective function
XGBOOST_REGISTER_OBJECTIVE(CoxRegression, "survival:cox")
    .describe(
        "Cox regression for censored survival data (negative labels are considered censored).")
    .set_body([]() { return new CoxRegression(); });

}  // namespace xgboost::obj
