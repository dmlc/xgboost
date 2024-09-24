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
 protected:
  HostDeviceVector<int> label_correct_;
  mutable bool are_buffs_init = false;

  void InitBuffers() const {
    if (!are_buffs_init) {
      batch_processor_.InitBuffers(qu_, {1, 1, 1, 1});
      are_buffs_init = true;
    }
  }

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
      LOG(WARNING) << ctx_->Device();
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

    // TODO(razdoburdin): add label_correct check
    label_correct_.Resize(1);
    label_correct_.Fill(1);

    bool is_null_weight = info.weights_.Size() == 0;

    auto scale_pos_weight = param_.scale_pos_weight;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), info.labels.Shape(0))
        << "Number of weights should be equal to number of data points.";
    }

    int flag = 1;
    auto objective_fn = [=, &flag]
                        (const std::vector<::sycl::event>& events,
                         size_t ndata,
                         GradientPair* out_gpair,
                         const bst_float* preds,
                         const bst_float* labels,
                         const bst_float* weights) {
      const size_t wg_size = 32;
      const size_t nwgs = ndata / wg_size + (ndata % wg_size > 0);
      return linalg::GroupWiseKernel(qu_, &flag, events, {nwgs, wg_size},
        [=] (size_t idx, auto flag) {
          const bst_float pred = Loss::PredTransform(preds[idx]);
          bst_float weight = is_null_weight ? 1.0f : weights[idx/n_targets];
          const bst_float label = labels[idx];
          if (label == 1.0f) {
            weight *= scale_pos_weight;
          }
          if (!Loss::CheckLabel(label)) {
            AtomicRef<int> flag_ref(flag[0]);
            flag_ref = 0;
          }
          out_gpair[idx] = GradientPair(Loss::FirstOrderGradient(pred, label) * weight,
                                        Loss::SecondOrderGradient(pred, label) * weight);
      });
    };

    InitBuffers();
    if (is_null_weight) {
      // Output is passed by pointer
      // Inputs are passed by const reference
      batch_processor_.Calculate(std::move(objective_fn),
                                 out_gpair->Data(),
                                 preds,
                                 *(info.labels.Data()));
    } else {
      batch_processor_.Calculate(std::move(objective_fn),
                                 out_gpair->Data(),
                                 preds,
                                 *(info.labels.Data()),
                                 info.weights_);
    }
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
    InitBuffers();

    batch_processor_.Calculate([=] (const std::vector<::sycl::event>& events,
                                    size_t ndata,
                                    bst_float* io_preds) {
       return qu_->submit([&](::sycl::handler& cgh) {
        cgh.depends_on(events);
        cgh.parallel_for<>(::sycl::range<1>(ndata), [=](::sycl::id<1> pid) {
          int idx = pid[0];
          io_preds[idx] = Loss::PredTransform(io_preds[idx]);
        });
      });
    }, io_preds);
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
  static constexpr size_t kBatchSize = 1u << 22;
  mutable linalg::BatchProcessingHelper<GradientPair, bst_float, kBatchSize, 3> batch_processor_;
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
