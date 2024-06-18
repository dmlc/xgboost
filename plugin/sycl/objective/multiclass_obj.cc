/*!
 * Copyright 2015-2023 by Contributors
 * \file multiclass_obj.cc
 * \brief Definition of multi-class classification objectives.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <rabit/rabit.h>
#pragma GCC diagnostic pop

#include <vector>
#include <algorithm>
#include <limits>
#include <utility>

#include "xgboost/parameter.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include "xgboost/data.h"
#include "../../src/common/math.h"
#pragma GCC diagnostic pop
#include "xgboost/logging.h"
#include "xgboost/objective.h"
#include "xgboost/json.h"
#include "xgboost/span.h"

#include "../../../src/objective/multiclass_param.h"

#include "../device_manager.h"
#include <CL/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace obj {

DMLC_REGISTRY_FILE_TAG(multiclass_obj_sycl);

class SoftmaxMultiClassObj : public ObjFunction {
 public:
  explicit SoftmaxMultiClassObj(bool output_prob)
  : output_prob_(output_prob) {}

  void Configure(Args const& args) override {
    param_.UpdateAllowUnknown(args);
    qu_ = device_manager.GetQueue(ctx_->Device());
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo& info,
                   int iter,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    if (preds.Size() == 0) return;
    if (info.labels.Size() == 0) return;

    CHECK(preds.Size() == (static_cast<size_t>(param_.num_class) * info.labels.Size()))
        << "SoftmaxMultiClassObj: label size and pred size does not match.\n"
        << "label.Size() * num_class: "
        << info.labels.Size() * static_cast<size_t>(param_.num_class) << "\n"
        << "num_class: " << param_.num_class << "\n"
        << "preds.Size(): " << preds.Size();

    const int nclass = param_.num_class;
    const auto ndata = static_cast<int64_t>(preds.Size() / nclass);

    out_gpair->Reshape(info.num_row_, static_cast<std::uint64_t>(nclass));

    const bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }

    ::sycl::buffer<bst_float, 1> preds_buf(preds.HostPointer(), preds.Size());
    ::sycl::buffer<bst_float, 1> labels_buf(info.labels.Data()->HostPointer(), info.labels.Size());
    ::sycl::buffer<GradientPair, 1> out_gpair_buf(out_gpair->Data()->HostPointer(),
                                                  out_gpair->Size());
    ::sycl::buffer<bst_float, 1> weights_buf(is_null_weight ? NULL : info.weights_.HostPointer(),
                                             is_null_weight ? 1 : info.weights_.Size());

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

          bst_float const * point = &preds_acc[idx * nclass];

          // Part of Softmax function
          bst_float wmax = std::numeric_limits<bst_float>::min();
          for (int k = 0; k < nclass; k++) { wmax = ::sycl::max(point[k], wmax); }
          float wsum = 0.0f;
          for (int k = 0; k < nclass; k++) { wsum += ::sycl::exp(point[k] - wmax); }
          auto label = labels_acc[idx];
          if (label < 0 || label >= nclass) {
            flag_buf_acc[0] = 0;
            label = 0;
          }
          bst_float wt = is_null_weight ? 1.0f : weights_acc[idx];
          for (int k = 0; k < nclass; ++k) {
            bst_float p = expf(point[k] - wmax) / static_cast<float>(wsum);
            const float eps = 1e-16f;
            const bst_float h = ::sycl::max(2.0f * p * (1.0f - p) * wt, eps);
            p = label == k ? p - 1.0f : p;
            out_gpair_acc[idx * nclass + k] = GradientPair(p * wt, h);
          }
        });
      }).wait();
    }
    // flag_buf is destroyed, content is copyed to the "flag"

    if (flag == 0) {
      LOG(FATAL) << "SYCL::SoftmaxMultiClassObj: label must be in [0, num_class).";
    }
  }
  void PredTransform(HostDeviceVector<bst_float>* io_preds) const override {
    this->Transform(io_preds, output_prob_);
  }
  void EvalTransform(HostDeviceVector<bst_float>* io_preds) override {
    this->Transform(io_preds, true);
  }
  const char* DefaultEvalMetric() const override {
    return "mlogloss";
  }

  inline void Transform(HostDeviceVector<bst_float> *io_preds, bool prob) const {
    if (io_preds->Size() == 0) return;
    const int nclass = param_.num_class;
    const auto ndata = static_cast<int64_t>(io_preds->Size() / nclass);
    max_preds_.Resize(ndata);

    {
      ::sycl::buffer<bst_float, 1> io_preds_buf(io_preds->HostPointer(), io_preds->Size());

      if (prob) {
        qu_.submit([&](::sycl::handler& cgh) {
          auto io_preds_acc = io_preds_buf.get_access<::sycl::access::mode::read_write>(cgh);
          cgh.parallel_for<>(::sycl::range<1>(ndata), [=](::sycl::id<1> pid) {
            int idx = pid[0];
            auto it = io_preds_acc.begin() + idx * nclass;
            common::Softmax(it, it + nclass);
          });
        }).wait();
      } else {
        ::sycl::buffer<bst_float, 1> max_preds_buf(max_preds_.HostPointer(), max_preds_.Size());

        qu_.submit([&](::sycl::handler& cgh) {
          auto io_preds_acc = io_preds_buf.get_access<::sycl::access::mode::read>(cgh);
          auto max_preds_acc = max_preds_buf.get_access<::sycl::access::mode::read_write>(cgh);
          cgh.parallel_for<>(::sycl::range<1>(ndata), [=](::sycl::id<1> pid) {
            int idx = pid[0];
            auto it = io_preds_acc.begin() + idx * nclass;
            max_preds_acc[idx] = common::FindMaxIndex(it, it + nclass) - it;
          });
        }).wait();
      }
    }

    if (!prob) {
      io_preds->Resize(max_preds_.Size());
      io_preds->Copy(max_preds_);
    }
  }

  struct ObjInfo Task() const override {return {ObjInfo::kClassification}; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    if (this->output_prob_) {
      out["name"] = String("multi:softprob");
    } else {
      out["name"] = String("multi:softmax");
    }
    out["softmax_multiclass_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    FromJson(in["softmax_multiclass_param"], &param_);
  }

 private:
  // output probability
  bool output_prob_;
  // parameter
  xgboost::obj::SoftmaxMultiClassParam param_;
  // Cache for max_preds
  mutable HostDeviceVector<bst_float> max_preds_;

  sycl::DeviceManager device_manager;

  mutable ::sycl::queue qu_;
};

XGBOOST_REGISTER_OBJECTIVE(SoftmaxMultiClass, "multi:softmax_sycl")
.describe("Softmax for multi-class classification, output class index.")
.set_body([]() { return new SoftmaxMultiClassObj(false); });

XGBOOST_REGISTER_OBJECTIVE(SoftprobMultiClass, "multi:softprob_sycl")
.describe("Softmax for multi-class classification, output probability distribution.")
.set_body([]() { return new SoftmaxMultiClassObj(true); });

}  // namespace obj
}  // namespace sycl
}  // namespace xgboost
