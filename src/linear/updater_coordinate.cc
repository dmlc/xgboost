/*!
 * Copyright 2018 by Contributors
 * \author Rory Mitchell
 */

#include <xgboost/linear_updater.h>
#include "./param.h"
#include "../common/timer.h"
#include "coordinate_common.h"

namespace xgboost {
namespace linear {

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
  void Init(
      const std::vector<std::pair<std::string, std::string> > &args) override {
    param.InitAllowUnknown(args);
    selector.reset(FeatureSelector::Create(param.feature_selector));
    monitor.Init("CoordinateUpdater");
  }
  void Update(HostDeviceVector<GradientPair> *in_gpair, DMatrix *p_fmat,
              gbm::GBLinearModel *model, double sum_instance_weight) override {
    param.DenormalizePenalties(sum_instance_weight);
    const int ngroup = model->param.num_output_group;
    // update bias
    for (int group_idx = 0; group_idx < ngroup; ++group_idx) {
      auto grad = GetBiasGradientParallel(group_idx, ngroup,
                                          in_gpair->ConstHostVector(), p_fmat);
      auto dbias = static_cast<float>(param.learning_rate *
                                      CoordinateDeltaBias(grad.first, grad.second));
      model->bias()[group_idx] += dbias;
      UpdateBiasResidualParallel(group_idx, ngroup,
                                 dbias, &in_gpair->HostVector(), p_fmat);
    }
    // prepare for updating the weights
    selector->Setup(*model, in_gpair->ConstHostVector(), p_fmat, param.reg_alpha_denorm,
                    param.reg_lambda_denorm, param.top_k);
    // update weights
    for (int group_idx = 0; group_idx < ngroup; ++group_idx) {
      for (unsigned i = 0U; i < model->param.num_feature; i++) {
        int fidx = selector->NextFeature
          (i, *model, group_idx, in_gpair->ConstHostVector(), p_fmat,
           param.reg_alpha_denorm, param.reg_lambda_denorm);
        if (fidx < 0) break;
        this->UpdateFeature(fidx, group_idx, &in_gpair->HostVector(), p_fmat, model);
      }
    }
    monitor.Stop("UpdateFeature");
  }

  inline void UpdateFeature(int fidx, int group_idx, std::vector<GradientPair> *in_gpair,
                            DMatrix *p_fmat, gbm::GBLinearModel *model) {
    const int ngroup = model->param.num_output_group;
    bst_float &w = (*model)[fidx][group_idx];
    auto gradient =
        GetGradientParallel(group_idx, ngroup, fidx, *in_gpair, p_fmat);
    auto dw = static_cast<float>(
        param.learning_rate *
        CoordinateDelta(gradient.first, gradient.second, w, param.reg_alpha_denorm,
                        param.reg_lambda_denorm));
    w += dw;
    UpdateResidualParallel(fidx, group_idx, ngroup, dw, in_gpair, p_fmat);
  }

  // training parameter
  LinearTrainParam param;
  std::unique_ptr<FeatureSelector> selector;
  common::Monitor monitor;
};

XGBOOST_REGISTER_LINEAR_UPDATER(CoordinateUpdater, "coord_descent")
    .describe("Update linear model according to coordinate descent algorithm.")
    .set_body([]() { return new CoordinateUpdater(); });
}  // namespace linear
}  // namespace xgboost
