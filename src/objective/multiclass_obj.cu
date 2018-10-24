/*!
 * Copyright 2015-2018 by Contributors
 * \file multi_class.cc
 * \brief Definition of multi-class classification objectives.
 * \author Tianqi Chen
 */
#include <dmlc/omp.h>
#include <dmlc/parameter.h>
#include <xgboost/data.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>
#include "../common/math.h"
#include "../common/transform.h"

namespace xgboost {
namespace obj {

#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(multiclass_obj_gpu);
#endif

struct SoftmaxMultiClassParam : public dmlc::Parameter<SoftmaxMultiClassParam> {
  int num_class;
  int n_gpus;
  int gpu_id;
  // declare parameters
  DMLC_DECLARE_PARAMETER(SoftmaxMultiClassParam) {
    DMLC_DECLARE_FIELD(num_class).set_lower_bound(1)
        .describe("Number of output class in the multi-class classification.");
    DMLC_DECLARE_FIELD(n_gpus).set_default(1).set_lower_bound(-1)
        .describe("Number of GPUs to use for multi-gpu algorithms.");
    DMLC_DECLARE_FIELD(gpu_id)
        .set_lower_bound(0)
        .set_default(0)
        .describe("gpu to use for objective function evaluation");
  }
};
// TODO(trivialfis): Currently the resharding in softmax is less than ideal
// due to repeated copying data between CPU and GPUs.  Maybe we just use single
// GPU?
class SoftmaxMultiClassObj : public ObjFunction {
 public:
  explicit SoftmaxMultiClassObj(bool output_prob)
  : output_prob_(output_prob) {
  }
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
    CHECK(param_.n_gpus != 0) << "Must have at least one device";  // Default is -1
    devices_ = GPUSet::All(param_.n_gpus).Normalised(param_.gpu_id);
    label_correct_.Resize(devices_.IsEmpty() ? 1 : devices_.Size());
  }
  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo& info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK(preds.Size() == (static_cast<size_t>(param_.num_class) * info.labels_.Size()))
        << "SoftmaxMultiClassObj: label size and pred size does not match";

    const int nclass = param_.num_class;
    const auto ndata = static_cast<int64_t>(preds.Size() / nclass);

    out_gpair->Reshard(GPUDistribution::Granular(devices_, nclass));
    info.labels_.Reshard(GPUDistribution::Block(devices_));
    info.weights_.Reshard(GPUDistribution::Block(devices_));
    preds.Reshard(GPUDistribution::Granular(devices_, nclass));
    label_correct_.Reshard(GPUDistribution::Block(devices_));

    out_gpair->Resize(preds.Size());
    label_correct_.Fill(1);

    const bool is_null_weight = info.weights_.Size() == 0;
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t idx,
                           common::Span<GradientPair> gpair,
                           common::Span<bst_float const> labels,
                           common::Span<bst_float const> preds,
                           common::Span<bst_float const> weights,
                           common::Span<int> _label_correct) {
          common::Span<bst_float const> point = preds.subspan(idx * nclass, nclass);

          // Part of Softmax function
          bst_float wmax = std::numeric_limits<bst_float>::min();
          for (auto const i : point) { wmax = fmaxf(i, wmax); }
          double wsum = 0.0f;
          for (auto const i : point) { wsum += expf(i - wmax); }
          auto label = labels[idx];
          if (label < 0 || label >= nclass) {
            _label_correct[0] = 0;
            label = 0;
          }
          bst_float wt = is_null_weight ? 1.0f : weights[idx];
          for (int k = 0; k < nclass; ++k) {
            // Computation duplicated to avoid creating a cache.
            bst_float p = expf(point[k] - wmax) / static_cast<float>(wsum);
            const float eps = 1e-16f;
            const bst_float h = fmax(2.0f * p * (1.0f - p) * wt, eps);
            p = label == k ? p - 1.0f : p;
            gpair[idx * nclass + k] = GradientPair(p * wt, h);
          }
        }, common::Range{0, ndata}, devices_, false)
        .Eval(out_gpair, &info.labels_, &preds, &info.weights_, &label_correct_);

    std::vector<int>& label_correct_h = label_correct_.HostVector();
    for (auto const flag : label_correct_h) {
      if (flag != 1) {
        LOG(FATAL) << "SoftmaxMultiClassObj: label must be in [0, num_class).";
      }
    }
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

  inline void Transform(HostDeviceVector<bst_float> *io_preds, bool prob) {
    const int nclass = param_.num_class;
    const auto ndata = static_cast<int64_t>(io_preds->Size() / nclass);
    max_preds_.Resize(ndata);

    if (prob) {
      common::Transform<>::Init(
          [=] XGBOOST_DEVICE(size_t _idx, common::Span<bst_float> _preds) {
            common::Span<bst_float> point =
                _preds.subspan(_idx * nclass, nclass);
            common::Softmax(point.begin(), point.end());
          },
          common::Range{0, ndata}, GPUDistribution::Granular(devices_, nclass))
        .Eval(io_preds);
    } else {
      io_preds->Reshard(GPUDistribution::Granular(devices_, nclass));
      max_preds_.Reshard(GPUDistribution::Block(devices_));
      common::Transform<>::Init(
          [=] XGBOOST_DEVICE(size_t _idx,
                             common::Span<const bst_float> _preds,
                             common::Span<bst_float> _max_preds) {
            common::Span<const bst_float> point =
                _preds.subspan(_idx * nclass, nclass);
            _max_preds[_idx] =
                common::FindMaxIndex(point.cbegin(),
                                     point.cend()) - point.cbegin();
          },
          common::Range{0, ndata}, devices_, false)
        .Eval(io_preds, &max_preds_);
    }
    if (!prob) {
      io_preds->Resize(max_preds_.Size());
      io_preds->Copy(max_preds_);
    }
  }

 private:
  // output probability
  bool output_prob_;
  // parameter
  SoftmaxMultiClassParam param_;
  GPUSet devices_;
  // Cache for max_preds
  HostDeviceVector<bst_float> max_preds_;
  HostDeviceVector<int> label_correct_;
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
