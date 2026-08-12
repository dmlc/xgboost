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

#include "../common/common.h"
#include "../common/linalg_op.h"  // for ElementWiseKernel
#include "../common/math.h"       // for CloseTo
#include "../common/threading_utils.h"
#include "../common/transform.h"
#include "../common/utils.h"  // for NoOp
#include "init_estimation.h"  // FitIntercept
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
#include "../common/algorithm.cuh"     // for AllOf
#include "../common/cuda_context.cuh"  // for CUDAContext
#endif                                 // defined(XGBOOST_USE_CUDA)

namespace xgboost::obj {
namespace {
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

// cox regression for survival data (negative values mean they are censored)
class CoxRegression : public FitIntercept {
 public:
  std::set<std::string> Configure(Args const&) override { return {}; }
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
