/*!
 * Copyright 2018 by Contributors
 * \file hinge.cc
 * \brief Provides an implementation of the hinge loss function
 * \author Henry Gouk
 */
#include <xgboost/objective.h>
#include "../common/math.h"
#include "../common/transform.h"
#include "../common/common.h"
#include "../common/span.h"
#include "../common/host_device_vector.h"

namespace xgboost {
namespace obj {

#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(hinge_obj_gpu);
#endif

struct HingeObjParam : public dmlc::Parameter<HingeObjParam> {
  int n_gpus;
  int gpu_id;
  DMLC_DECLARE_PARAMETER(HingeObjParam) {
    DMLC_DECLARE_FIELD(n_gpus).set_default(1).set_lower_bound(-1)
        .describe("Number of GPUs to use for multi-gpu algorithms.");
    DMLC_DECLARE_FIELD(gpu_id)
        .set_lower_bound(0)
        .set_default(0)
        .describe("gpu to use for objective function evaluation");
  }
};

class HingeObj : public ObjFunction {
 public:
  HingeObj() = default;

  void Configure(
      const std::vector<std::pair<std::string, std::string> > &args) override {
    param_.InitAllowUnknown(args);
    devices_ = GPUSet::All(param_.n_gpus).Normalised(param_.gpu_id);
    label_correct_.Resize(devices_.IsEmpty() ? 1 : devices_.Size());
  }

  void GetGradient(const HostDeviceVector<bst_float> &preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size())
        << "labels are not correctly provided"
        << "preds.size=" << preds.Size()
        << ", label.size=" << info.labels_.Size();

    const bool is_null_weight = info.weights_.Size() == 0;
    const size_t ndata = preds.Size();
    out_gpair->Resize(ndata);
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t _idx,
                           common::Span<int> _label_correct,
                           common::Span<GradientPair> _out_gpair,
                           common::Span<const bst_float> _preds,
                           common::Span<const bst_float> _labels,
                           common::Span<const bst_float> _weights) {
          bst_float p = _preds[_idx];
          bst_float w = is_null_weight ? 1.0f : _weights[_idx];
          bst_float y = _labels[_idx] * 2.0 - 1.0;
          bst_float g, h;
          if (p * y < 1.0) {
            g = -y * w;
            h = w;
          } else {
            g = 0.0;
            h = std::numeric_limits<bst_float>::min();
          }
          _out_gpair[_idx] = GradientPair(g, h);
        },
        common::Range{0, static_cast<int64_t>(ndata)}, devices_).Eval(
            &label_correct_, out_gpair, &preds, &info.labels_, &info.weights_);
  }

  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<bst_float> _preds) {
          _preds[_idx] = _preds[_idx] > 0.0 ? 1.0 : 0.0;
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size()), 1}, devices_)
        .Eval(io_preds);
  }

  const char* DefaultEvalMetric() const override {
    return "error";
  }

 private:
  GPUSet devices_;
  HostDeviceVector<int> label_correct_;
  HingeObjParam param_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(HingeObjParam);
// register the objective functions
XGBOOST_REGISTER_OBJECTIVE(HingeObj, "binary:hinge")
.describe("Hinge loss. Expects labels to be in [0,1f]")
.set_body([]() { return new HingeObj(); });

}  // namespace obj
}  // namespace xgboost
