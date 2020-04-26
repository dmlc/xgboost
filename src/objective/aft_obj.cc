/*!
 * Copyright 2015 by Contributors
 * \file rank.cc
 * \brief Definition of aft loss.
 */

#include <dmlc/omp.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <memory>
#include <utility>
#include <cmath>

#include "xgboost/json.h"

#include "../common/math.h"
#include "../common/random.h"
#include "../common/survival_util.h"

using AFTParam = xgboost::common::AFTParam;
using AFTLoss = xgboost::common::AFTLoss;

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(aft_obj);

class AFTObj : public ObjFunction {
 public:
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.UpdateAllowUnknown(args);
    loss_.reset(new AFTLoss(param_.aft_loss_distribution));
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo& info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    /* Boilerplate */
    CHECK_EQ(preds.Size(), info.labels_lower_bound_.Size());
    CHECK_EQ(preds.Size(), info.labels_upper_bound_.Size());

    const auto& yhat = preds.HostVector();
    const auto& y_lower = info.labels_lower_bound_.HostVector();
    const auto& y_upper = info.labels_upper_bound_.HostVector();
    const auto& weights = info.weights_.HostVector();
    const bool is_null_weight = weights.empty();

    out_gpair->Resize(yhat.size());
    std::vector<GradientPair>& gpair = out_gpair->HostVector();
    CHECK_LE(yhat.size(), static_cast<size_t>(std::numeric_limits<omp_ulong>::max()))
      << "yhat is too big";
    const omp_ulong nsize = static_cast<omp_ulong>(yhat.size());
    const float aft_loss_distribution_scale = param_.aft_loss_distribution_scale;

    #pragma omp parallel for \
      shared(weights, y_lower, y_upper, yhat, gpair)
    for (omp_ulong i = 0; i < nsize; ++i) {
      // If weights are empty, data is unweighted so we use 1.0 everywhere
      const double w = is_null_weight ? 1.0 : weights[i];
      const double grad = loss_->Gradient(y_lower[i], y_upper[i],
                                          yhat[i], aft_loss_distribution_scale);
      const double hess = loss_->Hessian(y_lower[i], y_upper[i],
                                         yhat[i], aft_loss_distribution_scale);
      gpair[i] = GradientPair(grad * w, hess * w);
    }
  }

  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    // Trees give us a prediction in log scale, so exponentiate
    std::vector<bst_float> &preds = io_preds->HostVector();
    const long ndata = static_cast<long>(preds.size()); // NOLINT(*)
    #pragma omp parallel for shared(preds)
    for (long j = 0; j < ndata; ++j) {  // NOLINT(*)
      preds[j] = std::exp(preds[j]);
    }
  }

  void EvalTransform(HostDeviceVector<bst_float> *io_preds) override {
    // do nothing here, since the AFT metric expects untransformed prediction score
  }

  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }

  const char* DefaultEvalMetric() const override {
    return "aft-nloglik";
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("survival:aft");
    out["aft_loss_param"] = ToJson(param_);
  }

  void LoadConfig(Json const& in) override {
    FromJson(in["aft_loss_param"], &param_);
    loss_.reset(new AFTLoss(param_.aft_loss_distribution));
  }

 private:
  AFTParam param_;
  std::unique_ptr<AFTLoss> loss_;
};

// register the objective functions
XGBOOST_REGISTER_OBJECTIVE(AFTObj, "survival:aft")
.describe("AFT loss function")
.set_body([]() { return new AFTObj(); });

}  // namespace obj
}  // namespace xgboost
