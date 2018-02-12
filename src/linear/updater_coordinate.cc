/*!
 * Copyright 2018 by Contributors
 * \author Rory Mitchell
 */

#include <xgboost/linear_updater.h>
#include "../common/timer.h"
#include "coordinate_common.h"

namespace xgboost {
namespace linear {

DMLC_REGISTRY_FILE_TAG(updater_coordinate);

// training parameter
struct CoordinateTrainParam : public dmlc::Parameter<CoordinateTrainParam> {
  /*! \brief learning_rate */
  float learning_rate;
  /*! \brief regularization weight for L2 norm */
  float reg_lambda;
  /*! \brief regularization weight for L1 norm */
  float reg_alpha;
  std::string coordinate_selection;
  float maximum_weight;
  int debug_verbose;
  // declare parameters
  DMLC_DECLARE_PARAMETER(CoordinateTrainParam) {
    DMLC_DECLARE_FIELD(learning_rate)
        .set_lower_bound(0.0f)
        .set_default(1.0f)
        .describe("Learning rate of each update.");
    DMLC_DECLARE_FIELD(reg_lambda)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe("L2 regularization on weights.");
    DMLC_DECLARE_FIELD(reg_alpha)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe("L1 regularization on weights.");
    DMLC_DECLARE_FIELD(coordinate_selection)
        .set_default("cyclic")
        .describe(
            "Coordinate selection algorithm, one of cyclic/random/greedy");
    DMLC_DECLARE_FIELD(debug_verbose)
        .set_lower_bound(0)
        .set_default(0)
        .describe("flag to print out detailed breakdown of runtime");
    // alias of parameters
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_alpha, alpha);
  }
};

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
    selector.reset(CoordinateSelector::Create(param.coordinate_selection));
    monitor.Init("CoordinateUpdater", param.debug_verbose);
  }
  void Update(std::vector<bst_gpair> *in_gpair, DMatrix *p_fmat,
              gbm::GBLinearModel *model, double sum_instance_weight) override {
    // Calculate bias
    for (int group_idx = 0; group_idx < model->param.num_output_group;
         ++group_idx) {
      auto grad = GetBiasGradientParallel(
          group_idx, model->param.num_output_group, *in_gpair, p_fmat);
      auto dbias = static_cast<float>(
          param.learning_rate * CoordinateDeltaBias(grad.first, grad.second));
      model->bias()[group_idx] += dbias;
      UpdateBiasResidualParallel(group_idx, model->param.num_output_group,
                                 dbias, in_gpair, p_fmat);
    }
    for (int group_idx = 0; group_idx < model->param.num_output_group;
         ++group_idx) {
      for (auto i = 0U; i < model->param.num_feature; i++) {
        int fidx = selector->SelectNextCoordinate(
            i, *model, group_idx, *in_gpair, p_fmat, param.reg_alpha,
            param.reg_lambda, sum_instance_weight);
        this->UpdateFeature(fidx, group_idx, in_gpair, p_fmat, model,
                            sum_instance_weight);
      }
    }
  }

  void UpdateFeature(int fidx, int group_idx, std::vector<bst_gpair> *in_gpair,
                     DMatrix *p_fmat, gbm::GBLinearModel *model,
                     double sum_instance_weight) {
    bst_float &w = (*model)[fidx][group_idx];
    monitor.Start("GetGradientParallel");
    auto gradient = GetGradientParallel(
        group_idx, model->param.num_output_group, fidx, *in_gpair, p_fmat);
    monitor.Stop("GetGradientParallel");
    auto dw = static_cast<float>(
        param.learning_rate *
        CoordinateDelta(gradient.first, gradient.second, w, param.reg_lambda,
                        param.reg_alpha, sum_instance_weight));
    w += dw;
    monitor.Start("UpdateResidualParallel");
    UpdateResidualParallel(fidx, group_idx, model->param.num_output_group, dw,
                           in_gpair, p_fmat);
    monitor.Stop("UpdateResidualParallel");
  }

  // training parameter
  CoordinateTrainParam param;
  std::unique_ptr<CoordinateSelector> selector;
  common::Monitor monitor;
};

DMLC_REGISTER_PARAMETER(CoordinateTrainParam);
XGBOOST_REGISTER_LINEAR_UPDATER(CoordinateUpdater, "updater_coordinate")
    .describe("Update linear model according to coordinate descent algorithm.")
    .set_body([]() { return new CoordinateUpdater(); });
}  // namespace linear
}  // namespace xgboost
