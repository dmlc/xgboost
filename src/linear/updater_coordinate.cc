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
  int feature_selector;
  int top_k;
  int debug_verbose;
  // declare parameters
  DMLC_DECLARE_PARAMETER(CoordinateTrainParam) {
    DMLC_DECLARE_FIELD(learning_rate)
        .set_lower_bound(0.0f)
        .set_default(0.5f)
        .describe("Learning rate of each update.");
    DMLC_DECLARE_FIELD(reg_lambda)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe("L2 regularization on weights.");
    DMLC_DECLARE_FIELD(reg_alpha)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe("L1 regularization on weights.");
    DMLC_DECLARE_FIELD(feature_selector)
        .set_default(kCyclic)
        .add_enum("cyclic", kCyclic)
        .add_enum("shuffle", kShuffle)
        .add_enum("thrifty", kThrifty)
        .add_enum("greedy", kGreedy)
        .add_enum("random", kRandom)
        .describe("Feature selection or ordering method.");
    DMLC_DECLARE_FIELD(top_k)
        .set_lower_bound(0)
        .set_default(0)
        .describe("The number of top features to select in 'thrifty' feature_selector. "
                  "The value of zero means using all the features.");
    DMLC_DECLARE_FIELD(debug_verbose)
        .set_lower_bound(0)
        .set_default(0)
        .describe("flag to print out detailed breakdown of runtime");
    // alias of parameters
    DMLC_DECLARE_ALIAS(learning_rate, eta);
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_alpha, alpha);
  }
  /*! \brief Denormalizes the regularization penalties - to be called at each update */
  void DenormalizePenalties(double sum_instance_weight) {
    reg_lambda_denorm = reg_lambda * sum_instance_weight;
    reg_alpha_denorm = reg_alpha * sum_instance_weight;
  }
  // denormalizated regularization penalties
  float reg_lambda_denorm;
  float reg_alpha_denorm;
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
    selector.reset(FeatureSelector::Create(param.feature_selector));
    monitor.Init("CoordinateUpdater", param.debug_verbose);
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
  CoordinateTrainParam param;
  std::unique_ptr<FeatureSelector> selector;
  common::Monitor monitor;
};

DMLC_REGISTER_PARAMETER(CoordinateTrainParam);
XGBOOST_REGISTER_LINEAR_UPDATER(CoordinateUpdater, "coord_descent")
    .describe("Update linear model according to coordinate descent algorithm.")
    .set_body([]() { return new CoordinateUpdater(); });
}  // namespace linear
}  // namespace xgboost
