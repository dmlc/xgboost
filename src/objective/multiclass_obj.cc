/*!
 * Copyright 2015 by Contributors
 * \file multi_class.cc
 * \brief Definition of multi-class classification objectives.
 * \author Tianqi Chen
 */
#include <dmlc/omp.h>
#include <dmlc/parameter.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <algorithm>
#include <utility>
#include "../common/math.h"

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(multiclass_obj);

struct SoftmaxMultiClassParam : public dmlc::Parameter<SoftmaxMultiClassParam> {
  int num_class;
  // declare parameters
  DMLC_DECLARE_PARAMETER(SoftmaxMultiClassParam) {
    DMLC_DECLARE_FIELD(num_class).set_lower_bound(1)
        .describe("Number of output class in the multi-class classification.");
  }
};

class SoftmaxMultiClassObj : public ObjFunction {
 public:
  explicit SoftmaxMultiClassObj(bool output_prob)
      : output_prob_(output_prob) {
  }
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }
  void GetGradient(HostDeviceVector<bst_float>* preds,
                   const MetaInfo& info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    CHECK_NE(info.labels_.size(), 0U) << "label set cannot be empty";
    CHECK(preds->Size() == (static_cast<size_t>(param_.num_class) * info.labels_.size()))
        << "SoftmaxMultiClassObj: label size and pred size does not match";
    std::vector<bst_float>& preds_h = preds->HostVector();
    out_gpair->Resize(preds_h.size());
    std::vector<GradientPair>& gpair = out_gpair->HostVector();
    const int nclass = param_.num_class;
    const auto ndata = static_cast<omp_ulong>(preds_h.size() / nclass);

    int label_error = 0;
    #pragma omp parallel
    {
      std::vector<bst_float> rec(nclass);
      #pragma omp for schedule(static)
      for (omp_ulong i = 0; i < ndata; ++i) {
        for (int k = 0; k < nclass; ++k) {
          rec[k] = preds_h[i * nclass + k];
        }
        common::Softmax(&rec);
        auto label = static_cast<int>(info.labels_[i]);
        if (label < 0 || label >= nclass)  {
          label_error = label; label = 0;
        }
        const bst_float wt = info.GetWeight(i);
        for (int k = 0; k < nclass; ++k) {
          bst_float p = rec[k];
          const float eps = 1e-16f;
          const bst_float h = fmax(2.0f * p * (1.0f - p) * wt, eps);
          if (label == k) {
            gpair[i * nclass + k] = GradientPair((p - 1.0f) * wt, h);
          } else {
            gpair[i * nclass + k] = GradientPair(p* wt, h);
          }
        }
      }
    }
    CHECK(label_error >= 0 && label_error < nclass)
        << "SoftmaxMultiClassObj: label must be in [0, num_class),"
        << " num_class=" << nclass
        << " but found " << label_error << " in label.";
  }
  void PredTransform(HostDeviceVector<bst_float>* io_preds) override {
    this->Transform(io_preds, output_prob_);
  }
  void EvalTransform(HostDeviceVector<bst_float>* io_preds) override {
    this->Transform(io_preds, true);
  }
  const char* DefaultEvalMetric() const override {
    return "merror";
  }

 private:
  inline void Transform(HostDeviceVector<bst_float> *io_preds, bool prob) {
    std::vector<bst_float> &preds = io_preds->HostVector();
    std::vector<bst_float> tmp;
    const int nclass = param_.num_class;
    const auto ndata = static_cast<omp_ulong>(preds.size() / nclass);
    if (!prob) tmp.resize(ndata);

    #pragma omp parallel
    {
      std::vector<bst_float> rec(nclass);
      #pragma omp for schedule(static)
      for (omp_ulong j = 0; j < ndata; ++j) {
        for (int k = 0; k < nclass; ++k) {
          rec[k] = preds[j * nclass + k];
        }
        if (!prob) {
          tmp[j] = static_cast<bst_float>(
              common::FindMaxIndex(rec.begin(), rec.end()) - rec.begin());
        } else {
          common::Softmax(&rec);
          for (int k = 0; k < nclass; ++k) {
            preds[j * nclass + k] = rec[k];
          }
        }
      }
    }
    if (!prob) preds = tmp;
  }
  // output probability
  bool output_prob_;
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

}  // namespace obj
}  // namespace xgboost
