/**
 * Copyright 2019-2025, XGBoost Contributors
 * \file aft_obj.cu
 * \brief Definition of AFT loss for survival analysis.
 * \author Avinash Barnwal, Hyunsu Cho and Toby Hocking
 */

#include <cmath>    // for log
#include <cstddef>  // for size_t


#include "../common/survival_util.h"
#include "../common/transform.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/logging.h"
#include "xgboost/objective.h"
#include "xgboost/span.h"

#if defined(XGBOOST_USE_CUDA)
#include "../common/linalg_op.cuh"  // for ElementWiseKernel
#elif defined(XGBOOST_USE_SYCL)
#include "../../plugin/sycl/common/linalg_op.h"
#else
#include "../common/linalg_op.h"  // for ElementWiseKernel
#endif

using AFTParam = xgboost::common::AFTParam;
using ProbabilityDistributionType = xgboost::common::ProbabilityDistributionType;
template <typename Distribution>
using AFTLoss = xgboost::common::AFTLoss<Distribution>;

namespace xgboost {
namespace obj {

#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(aft_obj_gpu);
#endif  // defined(XGBOOST_USE_CUDA)

class AFTObj : public ObjFunction {
 public:
  void Configure(Args const& args) override {
    param_.UpdateAllowUnknown(args);
  }

  ObjInfo Task() const override { return ObjInfo::kSurvival; }

  template <typename Distribution>
  void GetGradientImpl(const HostDeviceVector<bst_float>& preds, const MetaInfo& info,
                       linalg::Matrix<GradientPair>* out_gpair, size_t ndata, DeviceOrd device,
                       bool is_null_weight, float aft_loss_distribution_scale) {
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t _idx,
        common::Span<GradientPair> _out_gpair,
        common::Span<const bst_float> _preds,
        common::Span<const bst_float> _labels_lower_bound,
        common::Span<const bst_float> _labels_upper_bound,
        common::Span<const bst_float> _weights) {
      const double pred = static_cast<double>(_preds[_idx]);
      const double label_lower_bound = static_cast<double>(_labels_lower_bound[_idx]);
      const double label_upper_bound = static_cast<double>(_labels_upper_bound[_idx]);
      const float grad = static_cast<float>(
          AFTLoss<Distribution>::Gradient(label_lower_bound, label_upper_bound,
                                          pred, aft_loss_distribution_scale));
      const float hess = static_cast<float>(
          AFTLoss<Distribution>::Hessian(label_lower_bound, label_upper_bound,
                                         pred, aft_loss_distribution_scale));
      const bst_float w = is_null_weight ? 1.0f : _weights[_idx];
      _out_gpair[_idx] = GradientPair(grad * w, hess * w);
    },
    common::Range{0, static_cast<int64_t>(ndata)}, this->ctx_->Threads(), device).Eval(
        out_gpair->Data(), &preds, &info.labels_lower_bound_, &info.labels_upper_bound_,
        &info.weights_);
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds, const MetaInfo& info, int /*iter*/,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    const size_t ndata = preds.Size();
    CHECK_EQ(info.labels_lower_bound_.Size(), ndata);
    CHECK_EQ(info.labels_upper_bound_.Size(), ndata);
    out_gpair->SetDevice(ctx_->Device());
    out_gpair->Reshape(ndata, 1);
    const auto device = ctx_->Device();
    const float aft_loss_distribution_scale = param_.aft_loss_distribution_scale;
    const bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
        << "Number of weights should be equal to number of data points.";
    }

    switch (param_.aft_loss_distribution) {
    case common::ProbabilityDistributionType::kNormal:
      GetGradientImpl<common::NormalDistribution>(preds, info, out_gpair, ndata, device,
                                                  is_null_weight, aft_loss_distribution_scale);
      break;
    case common::ProbabilityDistributionType::kLogistic:
      GetGradientImpl<common::LogisticDistribution>(preds, info, out_gpair, ndata, device,
                                                    is_null_weight, aft_loss_distribution_scale);
      break;
    case common::ProbabilityDistributionType::kExtreme:
      GetGradientImpl<common::ExtremeDistribution>(preds, info, out_gpair, ndata, device,
                                                   is_null_weight, aft_loss_distribution_scale);
      break;
    default:
      LOG(FATAL) << "Unrecognized distribution";
    }
  }

  void PredTransform(HostDeviceVector<bst_float> *io_preds) const override {
    // Trees give us a prediction in log scale, so exponentiate
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<bst_float> _preds) {
          _preds[_idx] = exp(_preds[_idx]);
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size())}, this->ctx_->Threads(),
        io_preds->Device())
        .Eval(io_preds);
  }

  void EvalTransform(HostDeviceVector<bst_float>* /*io_preds*/) override {
    // do nothing here, since the AFT metric expects untransformed prediction score
  }

  void ProbToMargin(linalg::Vector<float>* base_score) const override {
    auto intercept = base_score->View(this->ctx_->Device());
    linalg::ElementWiseKernel(ctx_, intercept, [=] XGBOOST_DEVICE(std::size_t i) mutable {
      intercept(i) = std::log(intercept(i));
    });
  }

  const char* DefaultEvalMetric() const override {
    return "aft-nloglik";
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("survival:aft");
    out["aft_loss_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    FromJson(in["aft_loss_param"], &param_);
  }
  Json DefaultMetricConfig() const override {
    Json config{Object{}};
    config["name"] = String{this->DefaultEvalMetric()};
    config["aft_loss_param"] = ToJson(param_);
    return config;
  }

 private:
  AFTParam param_;
};

// register the objective functions
XGBOOST_REGISTER_OBJECTIVE(AFTObj, "survival:aft")
    .describe("AFT loss function")
    .set_body([]() { return new AFTObj(); });

}  // namespace obj
}  // namespace xgboost
