/*!
 * Copyright 2018-2022 by XGBoost Contributors
 * \file hinge.cc
 * \brief Provides an implementation of the hinge loss function
 * \author Henry Gouk
 */
#include "xgboost/objective.h"
#include "xgboost/json.h"
#include "xgboost/span.h"
#include "xgboost/host_device_vector.h"

#include "../common/math.h"
#include "../common/transform.h"
#include "../common/common.h"

namespace xgboost {
namespace obj {

#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(hinge_obj_gpu);
#endif  // defined(XGBOOST_USE_CUDA)

class HingeObj : public ObjFunction {
 public:
  HingeObj() = default;

  void Configure(Args const&) override {}
  ObjInfo Task() const override { return ObjInfo::kRegression; }

  void GetGradient(const HostDeviceVector<bst_float> &preds, const MetaInfo &info, int /*iter*/,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels.Size())
        << "labels are not correctly provided"
        << "preds.size=" << preds.Size()
        << ", label.size=" << info.labels.Size();

    const size_t ndata = preds.Size();
    const bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }
    out_gpair->Resize(ndata);
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t _idx,
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
        common::Range{0, static_cast<int64_t>(ndata)}, this->ctx_->Threads(),
        ctx_->gpu_id).Eval(
            out_gpair, &preds, info.labels.Data(), &info.weights_);
  }

  void PredTransform(HostDeviceVector<bst_float> *io_preds) const override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<bst_float> _preds) {
          _preds[_idx] = _preds[_idx] > 0.0 ? 1.0 : 0.0;
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size()), 1}, this->ctx_->Threads(),
        io_preds->DeviceIdx())
        .Eval(io_preds);
  }

  const char* DefaultEvalMetric() const override {
    return "error";
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("binary:hinge");
  }
  void LoadConfig(Json const &) override {}
};

// register the objective functions
XGBOOST_REGISTER_OBJECTIVE(HingeObj, "binary:hinge")
.describe("Hinge loss. Expects labels to be in [0,1f]")
.set_body([]() { return new HingeObj(); });

}  // namespace obj
}  // namespace xgboost
