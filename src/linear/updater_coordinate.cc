/**
 * Copyright 2018-2025, XGBoost Contributors
 * \author Rory Mitchell
 */

#include <xgboost/linear_updater.h>

#include "../common/timer.h"
#include "./param.h"
#include "coordinate_common.h"
#include "xgboost/json.h"

namespace xgboost::linear {

DMLC_REGISTER_PARAMETER(CoordinateParam);
DMLC_REGISTRY_FILE_TAG(updater_coordinate);

// training parameter
/**
 * \class CoordinateUpdater
 *
 * \brief Coordinate descent algorithm that updates one feature per iteration
 */

class CoordinateUpdater : public LinearUpdater {
 public:
  // set training parameter
  void Configure(Args const &args) override {
    const std::vector<std::pair<std::string, std::string> > rest{tparam_.UpdateAllowUnknown(args)};
    cparam_.UpdateAllowUnknown(rest);
    selector_.reset(FeatureSelector::Create(tparam_.feature_selector));
    monitor_.Init("CoordinateUpdater");
  }

  void LoadConfig(Json const &in) override {
    auto const &config = get<Object const>(in);
    FromJson(config.at("linear_train_param"), &tparam_);
    FromJson(config.at("coordinate_param"), &cparam_);
  }
  void SaveConfig(Json *p_out) const override {
    LOG(DEBUG) << "Save config for CPU updater.";
    auto &out = *p_out;
    out["linear_train_param"] = ToJson(tparam_);
    out["coordinate_param"] = ToJson(cparam_);
  }

  void Update(linalg::Matrix<GradientPair> *in_gpair, DMatrix *p_fmat, gbm::GBLinearModel *model,
              double sum_instance_weight) override {
    auto gpair = in_gpair->Data();
    tparam_.DenormalizePenalties(sum_instance_weight);
    auto n_targets = model->learner_model_param->NumTargets();
    // update bias
    for (bst_target_t target_idx = 0; target_idx < n_targets; ++target_idx) {
      auto grad = GetBiasGradientParallel(target_idx, n_targets, gpair->ConstHostVector(), p_fmat,
                                          ctx_->Threads());
      auto dbias =
          static_cast<float>(tparam_.learning_rate * CoordinateDeltaBias(grad.first, grad.second));
      model->Bias()[target_idx] += dbias;
      UpdateBiasResidualParallel(ctx_, target_idx, n_targets, dbias, &gpair->HostVector(), p_fmat);
    }
    // prepare for updating the weights
    selector_->Setup(ctx_, *model, gpair->ConstHostVector(), p_fmat, tparam_.reg_alpha_denorm,
                     tparam_.reg_lambda_denorm, cparam_.top_k);
    // update weights
    for (bst_target_t target_idx = 0; target_idx < n_targets; ++target_idx) {
      for (unsigned i = 0U; i < model->learner_model_param->num_feature; i++) {
        int fidx =
            selector_->NextFeature(ctx_, i, *model, target_idx, gpair->ConstHostVector(), p_fmat,
                                   tparam_.reg_alpha_denorm, tparam_.reg_lambda_denorm);
        if (fidx < 0) break;
        this->UpdateFeature(fidx, target_idx, &gpair->HostVector(), p_fmat, model);
      }
    }
    monitor_.Stop("UpdateFeature");
  }

  void UpdateFeature(int fidx, bst_target_t target_idx, std::vector<GradientPair> *in_gpair,
                     DMatrix *p_fmat, gbm::GBLinearModel *model) {
    auto n_targets = model->learner_model_param->NumTargets();
    bst_float &w = (*model)[fidx][target_idx];
    auto gradient = GetGradientParallel(ctx_, target_idx, n_targets, fidx, *in_gpair, p_fmat);
    auto dw =
        static_cast<float>(tparam_.learning_rate * CoordinateDelta(gradient.first, gradient.second,
                                                                   w, tparam_.reg_alpha_denorm,
                                                                   tparam_.reg_lambda_denorm));
    w += dw;
    UpdateResidualParallel(ctx_, fidx, target_idx, n_targets, dw, in_gpair, p_fmat);
  }

 private:
  CoordinateParam cparam_;
  // training parameter
  LinearTrainParam tparam_;
  std::unique_ptr<FeatureSelector> selector_;
  common::Monitor monitor_;
};

XGBOOST_REGISTER_LINEAR_UPDATER(CoordinateUpdater, "coord_descent")
    .describe("Update linear model according to coordinate descent algorithm.")
    .set_body([]() { return new CoordinateUpdater(); });
}  // namespace xgboost::linear
