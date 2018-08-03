/*!
 * Copyright 2018 by Contributors
 * \file hinge.cc
 * \brief Provides an implementation of the hinge loss function
 * \author Henry Gouk
 */
#include <xgboost/objective.h>
#include "../common/math.h"

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(hinge);

class HingeObj : public ObjFunction {
 public:
  HingeObj() = default;

  void Configure(
      const std::vector<std::pair<std::string, std::string> > &args) override {
    // This objective does not take any parameters
  }

  void GetGradient(HostDeviceVector<bst_float> *preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds->Size(), info.labels_.size())
        << "labels are not correctly provided"
        << "preds.size=" << preds->Size()
        << ", label.size=" << info.labels_.size();
    auto& preds_h = preds->HostVector();

    out_gpair->Resize(preds_h.size());
    auto& gpair = out_gpair->HostVector();

    for (size_t i = 0; i < preds_h.size(); ++i) {
      auto y = info.labels_[i] * 2.0 - 1.0;
      bst_float p = preds_h[i];
      bst_float w = info.GetWeight(i);
      bst_float g, h;
      if (p * y < 1.0) {
        g = -y * w;
        h = w;
      } else {
        g = 0.0;
        h = std::numeric_limits<bst_float>::min();
      }
      gpair[i] = GradientPair(g, h);
    }
  }

  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    std::vector<bst_float> &preds = io_preds->HostVector();
    for (auto& p : preds) {
      p = p > 0.0 ? 1.0 : 0.0;
    }
  }

  const char* DefaultEvalMetric() const override {
    return "error";
  }
};

XGBOOST_REGISTER_OBJECTIVE(HingeObj, "binary:hinge")
.describe("Hinge loss. Expects labels to be in [0,1f]")
.set_body([]() { return new HingeObj(); });

}  // namespace obj
}  // namespace xgboost
