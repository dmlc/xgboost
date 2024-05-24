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
  static constexpr size_t kBatchSize = 1u << 22;
  mutable bool are_buffs_init = false;

  void InitBuffers() const {
    if (!are_buffs_init) {
      events_.resize(5);
      preds_.Resize(&qu_, kBatchSize);
      labels_.Resize(&qu_, kBatchSize);
      weights_.Resize(&qu_, kBatchSize);
      out_gpair_.Resize(&qu_, kBatchSize);
      are_buffs_init = true;
    }
  }

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

    InitBuffers();
    size_t const ndata = preds.Size();
    auto const n_targets = this->Targets(info);
    out_gpair->Reshape(info.num_row_, n_targets);

    // TODO(razdoburdin): add label_correct check
    label_correct_.Resize(1);
    label_correct_.Fill(1);

    bool is_null_weight = info.weights_.Size() == 0;

    bst_float* preds_ptr = preds_.Data();
    bst_float* labels_ptr = labels_.Data();
    bst_float* weights_ptr = weights_.Data();
    GradientPair* out_gpair_ptr = out_gpair_.Data();

    auto scale_pos_weight = param_.scale_pos_weight;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), info.labels.Shape(0))
        << "Number of weights should be equal to number of data points.";
    }

    int flag = 1;
    const int wg_size = 32;
    const size_t nBatch = ndata / kBatchSize + (ndata % kBatchSize > 0);
    {
      ::sycl::buffer<int, 1> flag_buf(&flag, 1);
      for (size_t batch = 0; batch < nBatch; ++batch) {
        const size_t begin = batch * kBatchSize;
        const size_t end = (batch == nBatch - 1) ? ndata : begin + kBatchSize;
        const size_t batch_size = end - begin;
        int nwgs = (batch_size / wg_size + (batch_size % wg_size > 0));

        events_[0] = qu_.memcpy(preds_ptr, preds.HostPointer() + begin,
                                batch_size * sizeof(bst_float), events_[3]);
        events_[1] = qu_.memcpy(labels_ptr, info.labels.Data()->HostPointer() + begin,
                                batch_size * sizeof(bst_float), events_[3]);
        if (!is_null_weight) {
          events_[2] = qu_.memcpy(weights_ptr, info.weights_.HostPointer() + begin,
                                  info.weights_.Size() * sizeof(bst_float), events_[3]);
        }

        events_[3] = qu_.submit([&](::sycl::handler& cgh) {
          cgh.depends_on(events_);
          auto flag_buf_acc  = flag_buf.get_access<::sycl::access::mode::write>(cgh);
          cgh.parallel_for_work_group<>(::sycl::range<1>(nwgs), ::sycl::range<1>(wg_size),
                                        [=](::sycl::group<1> group) {
            group.parallel_for_work_item([&](::sycl::h_item<1> item) {
              const size_t idx = item.get_global_id()[0];

              const bst_float pred = Loss::PredTransform(preds_ptr[idx]);
              bst_float weight = is_null_weight ? 1.0f : weights_ptr[idx/n_targets];
              const bst_float label = labels_ptr[idx];
              if (label == 1.0f) {
                weight *= scale_pos_weight;
              }
              if (!Loss::CheckLabel(label)) {
                AtomicRef<int> flag_ref(flag_buf_acc[0]);
                flag_ref = 0;
              }
              out_gpair_ptr[idx] = GradientPair(Loss::FirstOrderGradient(pred, label) * weight,
                                                Loss::SecondOrderGradient(pred, label) * weight);
            });
          });
        });
        events_[4] = qu_.memcpy(out_gpair->Data()->HostPointer() + begin, out_gpair_ptr,
                                batch_size * sizeof(GradientPair), events_[3]);
      }
      qu_.wait_and_throw();
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
    InitBuffers();

    ::sycl::event event;
    bst_float* preds_ptr = preds_.Data();
    const size_t nBatch = ndata / kBatchSize + (ndata % kBatchSize > 0);
    for (size_t batch = 0; batch < nBatch; ++batch) {
      const size_t begin = batch * kBatchSize;
      const size_t end = (batch == nBatch - 1) ? ndata : begin + kBatchSize;
      const size_t batch_size = end - begin;

      event = qu_.memcpy(preds_ptr, io_preds->HostPointer() + begin,
                         batch_size * sizeof(bst_float), event);

      event = qu_.submit([&](::sycl::handler& cgh) {
        cgh.depends_on(event);
        cgh.parallel_for<>(::sycl::range<1>(ndata), [=](::sycl::id<1> pid) {
          int idx = pid[0];
          preds_ptr[idx] = Loss::PredTransform(preds_ptr[idx]);
        });
      });
      event = qu_.memcpy(io_preds->HostPointer(), preds_ptr, batch_size*sizeof(bst_float), event);
    }
    qu_.wait_and_throw();
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
  mutable std::vector<::sycl::event> events_;
  // Buffers
  mutable USMVector<bst_float, MemoryType::on_device> preds_;
  mutable USMVector<bst_float, MemoryType::on_device> labels_;
  mutable USMVector<bst_float, MemoryType::on_device> weights_;
  mutable USMVector<GradientPair, MemoryType::on_device> out_gpair_;
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
