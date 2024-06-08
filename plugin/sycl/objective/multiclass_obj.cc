/*!
 * Copyright 2015-2023 by Contributors
 * \file multiclass_obj.cc
 * \brief Definition of multi-class classification objectives.
 */
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include "xgboost/parameter.h"
#include "xgboost/data.h"
#include "../../src/common/math.h"
#pragma GCC diagnostic pop
#include "xgboost/logging.h"
#include "xgboost/objective.h"
#include "xgboost/json.h"
#include "xgboost/span.h"

#include "../../../src/objective/multiclass_param.h"

#include "../common/linalg_op.h"

#include "../device_manager.h"
#include "../data.h"
#include <CL/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace obj {

DMLC_REGISTRY_FILE_TAG(multiclass_obj_sycl);

class SoftmaxMultiClassObj : public ObjFunction {
  mutable bool are_buffs_init = false;

  void InitBuffers(const std::vector<int>& sample_rate) const {
    if (!are_buffs_init) {
      batch_processor_.InitBuffers(&qu_, sample_rate);
      are_buffs_init = true;
    }
  }

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
                   xgboost::linalg::Matrix<GradientPair>* out_gpair) override {
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
      return linalg::GroupWiseKernel(&qu_, &flag, events, {nwgs, wg_size},
        [=] (size_t idx, auto flag) {
          const bst_float* pred = preds + idx * nclass;

          // Part of Softmax function
          bst_float wmax = std::numeric_limits<bst_float>::min();
          for (int k = 0; k < nclass; k++) { wmax = ::sycl::max(pred[k], wmax); }
          bst_float wsum = 0.0f;
          for (int k = 0; k < nclass; k++) { wsum += ::sycl::exp(pred[k] - wmax); }
          bst_float label = labels[idx];

          if (label < 0 || label >= nclass) {
            AtomicRef<int> flag_ref(flag[0]);
            flag_ref = 0;
            label = 0;
          }

          bst_float wt = is_null_weight ? 1.0f : weights[idx];
          for (int k = 0; k < nclass; ++k) {
            bst_float p = expf(pred[k] - wmax) / static_cast<float>(wsum);
            const float eps = 1e-16f;
            const bst_float h = ::sycl::max(2.0f * p * (1.0f - p) * wt, eps);
            p = label == k ? p - 1.0f : p;
            out_gpair[idx * nclass + k] = GradientPair(p * wt, h);
          }
      });
    };

    // out_gpair and preds have nclass points per sample
    // labels and weights have 1 points per sample
    InitBuffers({nclass, nclass, 1, 1});
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
    qu_.wait_and_throw();

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
  static constexpr size_t kBatchSize = 1u << 22;
  mutable linalg::BatchProcessingHelper<GradientPair, bst_float, kBatchSize, 3> batch_processor_;
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
