/**
 * Copyright 2015-2025, XGBoost Contributors
 * \file multi_class.cc
 * \brief Definition of multi-class classification objectives.
 * \author Tianqi Chen
 */
#include <dmlc/omp.h>

#include <cassert>  // for assert
#include <limits>

#include "../collective/aggregator.h"  // for GlobalSum
#include "../common/common.h"          // for AssertGPUSupport
#include "../common/linalg_op.h"
#include "../common/math.h"
#include "../common/optional_weight.h"  // for MakeOptionalWeights
#include "../common/transform.h"
#include "xgboost/data.h"
#include "xgboost/json.h"
#include "xgboost/logging.h"
#include "xgboost/objective.h"

#if defined(XGBOOST_USE_CUDA)

#include "../common/algorithm.cuh"     // for AllOf
#include "../common/cuda_context.cuh"  // for CUDAContext
#include "../common/linalg_op.cuh"     // for tcbegin

#endif  // defined(XGBOOST_USE_CUDA)

#if defined(XGBOOST_USE_SYCL)
#include "../../plugin/sycl/common/linalg_op.h"
#endif

#include "multiclass_param.h"

namespace xgboost::obj {
#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(multiclass_obj_gpu);
#endif  // defined(XGBOOST_USE_CUDA)

namespace {
void ValidateLabel(Context const* ctx, MetaInfo const& info, std::int64_t n_classes) {
  auto label = info.labels.View(ctx->Device());
  CHECK_LE(label.Shape(1), 1) << "multi-class-multi-label is not yet supported.";
  auto check = [=] XGBOOST_DEVICE(float y) -> bool {
    return y >= 0 && y < n_classes && std::floor(y) == y;
  };
  auto valid = ctx->DispatchDevice(
      [&] { return std::all_of(linalg::cbegin(label), linalg::cend(label), check); },
      [&] {
#if defined(XGBOOST_USE_CUDA)
        return common::AllOf(ctx->CUDACtx()->CTP(), linalg::tcbegin(label), linalg::tcend(label),
                             check);
#else
        common::AssertGPUSupport();
        return false;
#endif  // defined(XGBOOST_USE_CUDA)
      },
      [&] {
#if defined(XGBOOST_USE_SYCL)
        return sycl::linalg::Validate(ctx->Device(), label, check);
#else
        common::AssertSYCLSupport();
        return false;
#endif  // defined(XGBOOST_USE_SYCL)
      });
  CHECK(valid)
      << "SoftmaxMultiClassObj: label must be discrete values in the range of [0, num_class).";
}
}  // namespace

class SoftmaxMultiClassObj : public ObjFunction {
 public:
  explicit SoftmaxMultiClassObj(bool output_prob) : output_prob_(output_prob) {}

  void Configure(Args const& args) override { param_.UpdateAllowUnknown(args); }

  ObjInfo Task() const override { return ObjInfo::kClassification; }

  void GetGradient(HostDeviceVector<float> const& preds, const MetaInfo& info, std::int32_t iter,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    if (info.labels.Size() == 0) {
      return;
    }
    std::int64_t n_classes = param_.num_class;
    CHECK(preds.Size() == (static_cast<std::size_t>(n_classes) * info.labels.Size()))
        << "SoftmaxMultiClassObj: label size and pred size does not match.\n"
        << "label.Size() * num_class: " << info.labels.Size() * n_classes << "\n"
        << "num_class: " << param_.num_class << "\n"
        << "preds.Size(): " << preds.Size();

    if (iter == 0) {
      ValidateLabel(this->ctx_, info, n_classes);
    }

    const auto n_samples = preds.Size() / n_classes;
    CHECK_EQ(n_samples, info.num_row_);

    // fallback to cpu if current device doesn't supports fp64
    auto device = ctx_->DeviceFP64();
    auto labels = info.labels.View(device);

    out_gpair->SetDevice(device);
    out_gpair->Reshape(info.num_row_, n_classes);
    auto gpair = out_gpair->View(device);

    if (!info.weights_.Empty()) {
      CHECK_EQ(info.weights_.Size(), n_samples)
          << "Number of weights should be equal to number of data points.";
    }
    info.weights_.SetDevice(device);
    auto weights = common::MakeOptionalWeights(this->ctx_->Device(), info.weights_);

    preds.SetDevice(device);
    auto predt = linalg::MakeTensorView(this->ctx_, &preds, n_samples, n_classes);
    CHECK_EQ(labels.Shape(1), 1);
    auto y1d = labels.Slice(linalg::All(), 0);
    CHECK_EQ(y1d.Shape(0), info.num_row_);
    linalg::ElementWiseKernel(this->ctx_, y1d, [=] XGBOOST_DEVICE(std::size_t idx) mutable {
      auto point = predt.Slice(idx, linalg::All());
      assert(point.Size() == static_cast<std::size_t>(n_classes));

      // Part of the common::Softmax function
      float wmax = std::numeric_limits<float>::min();
      for (std::size_t k = 0, m = point.Size(); k < m; ++k) {
        wmax = fmaxf(point(k), wmax);
      }
      double wsum = 0.0f;
      for (std::size_t k = 0, m = point.Size(); k < m; ++k) {
        wsum += expf(point(k) - wmax);
      }
      auto label = y1d(idx);

      float wt = weights[idx];
      for (decltype(n_classes) k = 0; k < n_classes; ++k) {
        // Computation duplicated to avoid creating a cache.
        float p = expf(point(k) - wmax) / static_cast<float>(wsum);
        constexpr float kEps = 1e-16f;
        float h = fmax(2.0f * p * (1.0f - p) * wt, kEps);
        p = label == k ? p - 1.0f : p;
        gpair(idx, k) = GradientPair{p * wt, h};
      }
    });
  }

  void PredTransform(HostDeviceVector<float>* io_preds) const override {
    this->Transform(io_preds, output_prob_);
  }
  void EvalTransform(HostDeviceVector<float>* io_preds) override {
    this->Transform(io_preds, true);
  }
  const char* DefaultEvalMetric() const override { return "mlogloss"; }

  void Transform(HostDeviceVector<float>* io_preds, bool prob) const {
    const int n_classes = param_.num_class;
    const auto n_samples = static_cast<int64_t>(io_preds->Size() / n_classes);

    auto device = io_preds->Device();
    if (prob) {
      common::Transform<>::Init(
          [=] XGBOOST_DEVICE(size_t _idx, common::Span<float> _preds) {
            common::Span<float> point = _preds.subspan(_idx * n_classes, n_classes);
            common::Softmax(point.begin(), point.end());
          },
          common::Range{0, n_samples}, this->ctx_->Threads(), device)
          .Eval(io_preds);
    } else {
      io_preds->SetDevice(device);
      HostDeviceVector<float> max_preds;
      max_preds.SetDevice(device);
      max_preds.Resize(n_samples);
      common::Transform<>::Init(
          [=] XGBOOST_DEVICE(size_t _idx, common::Span<const float> _preds,
                             common::Span<float> _max_preds) {
            common::Span<const float> point = _preds.subspan(_idx * n_classes, n_classes);
            _max_preds[_idx] = common::FindMaxIndex(point.cbegin(), point.cend()) - point.cbegin();
          },
          common::Range{0, n_samples}, this->ctx_->Threads(), device)
          .Eval(io_preds, &max_preds);
      io_preds->Resize(max_preds.Size());
      io_preds->Copy(max_preds);
    }
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    if (this->output_prob_) {
      out["name"] = String("multi:softprob");
    } else {
      out["name"] = String("multi:softmax");
    }
    out["softmax_multiclass_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override { FromJson(in["softmax_multiclass_param"], &param_); }

  void InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) const override {
    std::int64_t n_classes = this->param_.num_class;
    ValidateLabel(this->ctx_, info, n_classes);

    *base_score = linalg::Zeros<float>(this->ctx_, n_classes);

    std::size_t n = info.labels.Size();

    auto labels = info.labels.View(ctx_->Device());
    auto weights = common::MakeOptionalWeights(this->ctx_->Device(), info.weights_);
    auto intercept = base_score->View(ctx_->Device());
    CHECK_EQ(intercept.Size(), n_classes);
    CHECK_EQ(n, info.num_row_);
    linalg::SmallHistogram(ctx_, labels, weights, intercept);
    auto sum_weight = common::SumOptionalWeights(this->ctx_, weights, n);
    auto status = collective::GlobalSum(this->ctx_, info, intercept, &sum_weight);
    collective::SafeColl(status);
    CHECK_GE(sum_weight, kRtEps);
    linalg::VecScaDiv(this->ctx_, intercept, sum_weight);
  }

 private:
  // output probability
  bool const output_prob_;
  // parameter
  SoftmaxMultiClassParam param_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(SoftmaxMultiClassParam);

XGBOOST_REGISTER_OBJECTIVE(SoftmaxMultiClass, "multi:softmax")
    .describe("Softmax for multi-class classification, output class index.")
    .set_body([]() { return new SoftmaxMultiClassObj(false); });

XGBOOST_REGISTER_OBJECTIVE(SoftprobMultiClass, "multi:softprob")
    .describe("Softmax for multi-class classification, output probability distribution.")
    .set_body([]() { return new SoftmaxMultiClassObj(true); });
}  // namespace xgboost::obj
