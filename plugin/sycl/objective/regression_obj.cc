/*!
 * Copyright 2015-2023 by Contributors
 * \file regression_obj.cc
 * \brief Definition of regression objectives.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#pragma GCC diagnostic pop

#include <cmath>
#include <memory>
#include <vector>

#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/parameter.h"
#include "xgboost/span.h"

#include "../../src/common/transform.h"
#include "../../src/common/common.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include "../../../src/objective/regression_loss.h"
#pragma GCC diagnostic pop
#include "../../../src/objective/regression_param.h"
#include "../../../src/objective/init_estimation.h"
#include "../../../src/objective/adaptive.h"
#include "../../../src/common/optional_weight.h"  // OptionalWeights

#include "../common/linalg_op.h"

#include "../device_manager.h"
#include "../data.h"

#include <CL/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace obj {

DMLC_REGISTRY_FILE_TAG(regression_obj_sycl);

template<typename Loss>
class RegLossObj : public ObjFunction {
 public:
  RegLossObj() = default;

  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.UpdateAllowUnknown(args);
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info,
                   int iter,
                   xgboost::linalg::Matrix<GradientPair>* out_gpair) override {
    if (qu_ == nullptr) {
      qu_ = device_manager.GetQueue(ctx_->Device());
    }
    if (info.labels.Size() == 0) return;
    CHECK_EQ(preds.Size(), info.labels.Size())
        << " " << "labels are not correctly provided"
        << "preds.size=" << preds.Size() << ", label.size=" << info.labels.Size() << ", "
        << "Loss: " << Loss::Name();

    size_t const ndata = preds.Size();
    auto const n_targets = this->Targets(info);
    out_gpair->Reshape(info.num_row_, n_targets);

    bool is_null_weight = info.weights_.Size() == 0;

    auto scale_pos_weight = param_.scale_pos_weight;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), info.labels.Shape(0))
        << "Number of weights should be equal to number of data points.";
    }

    out_gpair->Data()->SetDevice(ctx_->Device());
    preds.SetDevice(ctx_->Device());
    info.labels.Data()->SetDevice(ctx_->Device());
    info.weights_.SetDevice(ctx_->Device());

    GradientPair* out_gpair_ptr = out_gpair->Data()->DevicePointer();
    const bst_float* preds_ptr = preds.ConstDevicePointer();
    const bst_float* label_ptr = info.labels.Data()->ConstDevicePointer();
    const bst_float* weights_ptr = info.weights_.ConstDevicePointer();

    int flag = 1;
    const size_t wg_size = 32;
    const size_t nwgs = ndata / wg_size + (ndata % wg_size > 0);
    linalg::GroupWiseKernel(qu_, &flag, {}, {nwgs, wg_size},
      [=] (size_t idx, auto flag) {
        if (idx < ndata) {
          const bst_float pred = Loss::PredTransform(preds_ptr[idx]);
          bst_float weight = is_null_weight ? 1.0f : weights_ptr[idx/n_targets];
          const bst_float label = label_ptr[idx];
          if (label == 1.0f) {
            weight *= scale_pos_weight;
          }
          if (!Loss::CheckLabel(label)) {
            AtomicRef<int> flag_ref(flag[0]);
            flag_ref = 0;
          }
          out_gpair_ptr[idx] = GradientPair(Loss::FirstOrderGradient(pred, label) * weight,
                                            Loss::SecondOrderGradient(pred, label) * weight);
        }
    });
    qu_->wait_and_throw();

    if (flag == 0) {
      LOG(FATAL) << Loss::LabelErrorMsg();
    }
  }

 public:
  const char* DefaultEvalMetric() const override {
    return Loss::DefaultEvalMetric();
  }

  void PredTransform(HostDeviceVector<bst_float> *io_preds) const override {
    if (qu_ == nullptr) {
      LOG(WARNING) << ctx_->Device();
      qu_ = device_manager.GetQueue(ctx_->Device());
    }
    size_t const ndata = io_preds->Size();
    if (ndata == 0) return;

    io_preds->SetDevice(ctx_->Device());
    bst_float* io_preds_ptr = io_preds->DevicePointer();
    qu_->submit([&](::sycl::handler& cgh) {
      cgh.parallel_for<>(::sycl::range<1>(ndata), [=](::sycl::id<1> pid) {
        int idx = pid[0];
        io_preds_ptr[idx] = Loss::PredTransform(io_preds_ptr[idx]);
      });
    });

    qu_->wait_and_throw();
  }

  float ProbToMargin(float base_score) const override {
    return Loss::ProbToMargin(base_score);
  }

  struct ObjInfo Task() const override {
    return Loss::Info();
  };

  uint32_t Targets(MetaInfo const& info) const override {
    // Multi-target regression.
    return std::max(static_cast<size_t>(1), info.labels.Shape(1));
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
  xgboost::obj::RegLossParam param_;
  sycl::DeviceManager device_manager;

  mutable ::sycl::queue* qu_ = nullptr;
};

XGBOOST_REGISTER_OBJECTIVE(SquaredLossRegression,
                           std::string(xgboost::obj::LinearSquareLoss::Name()) + "_sycl")
.describe("Regression with squared error with SYCL backend.")
.set_body([]() { return new RegLossObj<xgboost::obj::LinearSquareLoss>(); });

XGBOOST_REGISTER_OBJECTIVE(SquareLogError,
                           std::string(xgboost::obj::SquaredLogError::Name()) + "_sycl")
.describe("Regression with root mean squared logarithmic error with SYCL backend.")
.set_body([]() { return new RegLossObj<xgboost::obj::SquaredLogError>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticRegression,
                           std::string(xgboost::obj::LogisticRegression::Name()) + "_sycl")
.describe("Logistic regression for probability regression task with SYCL backend.")
.set_body([]() { return new RegLossObj<xgboost::obj::LogisticRegression>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticClassification,
                           std::string(xgboost::obj::LogisticClassification::Name()) + "_sycl")
.describe("Logistic regression for binary classification task with SYCL backend.")
.set_body([]() { return new RegLossObj<xgboost::obj::LogisticClassification>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticRaw,
                           std::string(xgboost::obj::LogisticRaw::Name()) + "_sycl")
.describe("Logistic regression for classification, output score "
          "before logistic transformation with SYCL backend.")
.set_body([]() { return new RegLossObj<xgboost::obj::LogisticRaw>(); });

class MeanAbsoluteError : public ObjFunction {
 public:
  void Configure(Args const&) override {}

  ObjInfo Task() const override {
    return {ObjInfo::kRegression, true, true};
  }

  bst_target_t Targets(MetaInfo const& info) const override {
    return std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(HostDeviceVector<float> const& preds, const MetaInfo& info,
                   std::int32_t, xgboost::linalg::Matrix<GradientPair>* out_gpair) override {
    if (qu_ == nullptr) {
      qu_ = device_manager.GetQueue(ctx_->Device());
    }

    size_t const ndata = preds.Size();
    auto const n_targets = this->Targets(info);

    xgboost::obj::CheckInitInputs(info);
    CHECK_EQ(info.labels.Size(), preds.Size()) << "Invalid shape of labels.";
    const bst_float* label_ptr = info.labels.Data()->ConstDevicePointer();

    out_gpair->SetDevice(ctx_->Device());
    out_gpair->Reshape(info.num_row_, this->Targets(info));
    GradientPair* out_gpair_ptr  = out_gpair->Data()->DevicePointer();

    preds.SetDevice(ctx_->Device());
    const bst_float* preds_ptr = preds.ConstDevicePointer();
    auto predt = xgboost::linalg::MakeTensorView(ctx_, &preds, info.num_row_, this->Targets(info));
    info.weights_.SetDevice(ctx_->Device());
    common::OptionalWeights weight{ctx_->IsCPU() ? info.weights_.ConstHostSpan()
                                                 : info.weights_.ConstDeviceSpan()};

    qu_->submit([&](::sycl::handler& cgh) {
      cgh.parallel_for<>(::sycl::range<1>(ndata), [=](::sycl::id<1> pid) {
        int idx = pid[0];
        auto sign = [](auto x) {
          return (x > static_cast<decltype(x)>(0)) - (x < static_cast<decltype(x)>(0));
        };
        const bst_float pred = preds_ptr[idx];
        const bst_float label = label_ptr[idx];

        bst_float hess = weight[idx/n_targets];
        bst_float grad = sign(pred - label) * hess;
        out_gpair_ptr[idx] = GradientPair{grad, hess};
      });
    });
    qu_->wait_and_throw();
  }

  void UpdateTreeLeaf(HostDeviceVector<bst_node_t> const& position, MetaInfo const& info,
                      float learning_rate, HostDeviceVector<float> const& prediction,
                      std::int32_t group_idx, RegTree* p_tree) const override {
    ::xgboost::obj::UpdateTreeLeaf(ctx_, position, group_idx, info, learning_rate, prediction, 0.5,
                                   p_tree);
  }

  const char* DefaultEvalMetric() const override { return "mae"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:absoluteerror");
  }

  void LoadConfig(Json const& in) override {
    CHECK_EQ(StringView{get<String const>(in["name"])}, StringView{"reg:absoluteerror"});
  }

 protected:
  sycl::DeviceManager device_manager;
  mutable ::sycl::queue* qu_ = nullptr;
};

XGBOOST_REGISTER_OBJECTIVE(MeanAbsoluteError, "reg:absoluteerror_sycl")
    .describe("Mean absoluate error.")
    .set_body([]() { return new MeanAbsoluteError(); });

}  // namespace obj
}  // namespace sycl
}  // namespace xgboost
