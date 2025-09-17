/**
 * Copyright 2018-2023, XGBoost Contributors
 * \file hinge.cc
 * \brief Provides an implementation of the hinge loss function
 * \author Henry Gouk
 */
#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t

#include "../common/common.h"  // for Range
#if defined(XGBOOST_USE_CUDA)
#include "../common/linalg_op.cuh"
#endif
#if defined(XGBOOST_USE_SYCL)
#include "../../plugin/sycl/common/linalg_op.h"
#endif
#include "../common/linalg_op.h"
#include "../common/optional_weight.h"   // for OptionalWeights
#include "../common/transform.h"         // for Transform
#include "init_estimation.h"             // for FitIntercept
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // HostDeviceVector
#include "xgboost/json.h"                // for Json
#include "xgboost/linalg.h"              // for UnravelIndex
#include "xgboost/span.h"                // for Span

namespace xgboost::obj {
#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(hinge_obj_gpu);
#endif  // defined(XGBOOST_USE_CUDA)

class HingeObj : public FitIntercept {
 public:
  HingeObj() = default;

  void Configure(Args const &) override {}
  ObjInfo Task() const override { return ObjInfo::kRegression; }

  [[nodiscard]] bst_target_t Targets(MetaInfo const &info) const override {
    // Multi-target regression.
    return std::max(static_cast<std::size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(HostDeviceVector<float> const &preds, MetaInfo const &info,
                   std::int32_t /*iter*/, linalg::Matrix<GradientPair> *out_gpair) override {
    CheckInitInputs(info);
    CHECK_EQ(info.labels.Size(), preds.Size()) << "Invalid shape of labels.";
    if (!info.weights_.Empty()) {
      CHECK_EQ(info.weights_.Size(), info.num_row_)
          << "Number of weights should be equal to number of data points.";
    }

    bst_target_t n_targets = this->Targets(info);
    out_gpair->Reshape(info.num_row_, n_targets);
    auto gpair = out_gpair->View(ctx_->Device());

    preds.SetDevice(ctx_->Device());
    auto predt = linalg::MakeTensorView(ctx_, &preds, info.num_row_, n_targets);

    auto labels = info.labels.View(ctx_->Device());

    info.weights_.SetDevice(ctx_->Device());
    common::OptionalWeights weight{ctx_->IsCPU() ? info.weights_.ConstHostSpan()
                                                 : info.weights_.ConstDeviceSpan()};

    linalg::ElementWiseKernel(this->ctx_, labels,
                              [=] XGBOOST_DEVICE(std::size_t i, std::size_t j) mutable {
                                auto w = weight[i];

                                auto p = predt(i, j);
                                auto y = labels(i, j) * 2.0 - 1.0;

                                float g, h;
                                if (p * y < 1.0) {
                                  g = -y * w;
                                  h = w;
                                } else {
                                  g = 0.0;
                                  h = std::numeric_limits<float>::min();
                                }
                                gpair(i, j) = GradientPair{g, h};
                              });
  }

  void PredTransform(HostDeviceVector<float> *io_preds) const override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(std::size_t _idx, common::Span<float> _preds) {
          _preds[_idx] = _preds[_idx] > 0.0 ? 1.0 : 0.0;
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size()), 1}, this->ctx_->Threads(),
        io_preds->Device())
        .Eval(io_preds);
  }

  [[nodiscard]] const char *DefaultEvalMetric() const override { return "error"; }

  void SaveConfig(Json *p_out) const override {
    auto &out = *p_out;
    out["name"] = String("binary:hinge");
  }
  void LoadConfig(Json const &) override {}
};

// register the objective functions
XGBOOST_REGISTER_OBJECTIVE(HingeObj, "binary:hinge")
    .describe("Hinge loss. Expects labels to be in [0,1f]")
    .set_body([]() { return new HingeObj(); });

}  // namespace xgboost::obj
