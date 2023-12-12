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
#include <rabit/rabit.h>

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

#include "../device_manager.h"

#include <CL/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace obj {

DMLC_REGISTRY_FILE_TAG(regression_obj_sycl);

template<typename Loss>
class RegLossObj : public ObjFunction {
 protected:
  HostDeviceVector<int> label_correct_;

 public:
  RegLossObj() = default;

  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.UpdateAllowUnknown(args);
    qu_ = device_manager.GetQueue(ctx_->Device());
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info,
                   int iter,
                   linalg::Matrix<GradientPair>* out_gpair) override {
  if (info.labels.Size() == 0) return;
  CHECK_EQ(preds.Size(), info.labels.Size())
      << " " << "labels are not correctly provided"
      << "preds.size=" << preds.Size() << ", label.size=" << info.labels.Size() << ", "
      << "Loss: " << Loss::Name();

  size_t const ndata = preds.Size();
  auto const n_targets = this->Targets(info);
  out_gpair->Reshape(info.num_row_, n_targets);

  // TODO(razdoburdin): add label_correct check
  label_correct_.Resize(1);
  label_correct_.Fill(1);

  bool is_null_weight = info.weights_.Size() == 0;

  ::sycl::buffer<bst_float, 1> preds_buf(preds.HostPointer(), preds.Size());
  ::sycl::buffer<bst_float, 1> labels_buf(info.labels.Data()->HostPointer(), info.labels.Size());
  ::sycl::buffer<GradientPair, 1> out_gpair_buf(out_gpair->Data()->HostPointer(),
                                                out_gpair->Size());
  ::sycl::buffer<bst_float, 1> weights_buf(is_null_weight ? NULL : info.weights_.HostPointer(),
                                           is_null_weight ? 1    : info.weights_.Size());

  auto scale_pos_weight = param_.scale_pos_weight;
  if (!is_null_weight) {
    CHECK_EQ(info.weights_.Size(), info.labels.Shape(0))
      << "Number of weights should be equal to number of data points.";
  }

  int flag = 1;
  {
    ::sycl::buffer<int, 1> flag_buf(&flag, 1);
    qu_.submit([&](::sycl::handler& cgh) {
        auto preds_acc     = preds_buf.get_access<::sycl::access::mode::read>(cgh);
        auto labels_acc    = labels_buf.get_access<::sycl::access::mode::read>(cgh);
        auto weights_acc   = weights_buf.get_access<::sycl::access::mode::read>(cgh);
        auto out_gpair_acc = out_gpair_buf.get_access<::sycl::access::mode::write>(cgh);
        auto flag_buf_acc  = flag_buf.get_access<::sycl::access::mode::write>(cgh);
        cgh.parallel_for<>(::sycl::range<1>(ndata), [=](::sycl::id<1> pid) {
          int idx = pid[0];
          bst_float p = Loss::PredTransform(preds_acc[idx]);
          bst_float w = is_null_weight ? 1.0f : weights_acc[idx/n_targets];
          bst_float label = labels_acc[idx];
          if (label == 1.0f) {
            w *= scale_pos_weight;
          }
          if (!Loss::CheckLabel(label)) {
            // If there is an incorrect label, the host code will know.
            flag_buf_acc[0] = 0;
          }
          out_gpair_acc[idx] = GradientPair(Loss::FirstOrderGradient(p, label) * w,
                                            Loss::SecondOrderGradient(p, label) * w);
        });
      }).wait();
  }
  // flag_buf is destroyed, content is copyed to the "flag"

    if (flag == 0) {
      LOG(FATAL) << Loss::LabelErrorMsg();
    }
  }

 public:
  const char* DefaultEvalMetric() const override {
    return Loss::DefaultEvalMetric();
  }

  void PredTransform(HostDeviceVector<float> *io_preds) const override {
    size_t const ndata = io_preds->Size();
    if (ndata == 0) return;
    ::sycl::buffer<bst_float, 1> io_preds_buf(io_preds->HostPointer(), io_preds->Size());

    qu_.submit([&](::sycl::handler& cgh) {
      auto io_preds_acc = io_preds_buf.get_access<::sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<>(::sycl::range<1>(ndata), [=](::sycl::id<1> pid) {
        int idx = pid[0];
        io_preds_acc[idx] = Loss::PredTransform(io_preds_acc[idx]);
      });
    }).wait();
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

  mutable ::sycl::queue qu_;
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

}  // namespace obj
}  // namespace sycl
}  // namespace xgboost
