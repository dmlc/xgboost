/*!
 * Copyright 2018 by Contributors
 * \author Tianqi Chen, Rory Mitchell
 */

#include <xgboost/linear_updater.h>
#include "coordinate_common.h"

namespace xgboost {
namespace linear {

DMLC_REGISTRY_FILE_TAG(updater_shotgun);

// training parameter
struct ShotgunTrainParam : public dmlc::Parameter<ShotgunTrainParam> {
  /*! \brief learning_rate */
  float learning_rate;
  /*! \brief regularization weight for L2 norm */
  float reg_lambda;
  /*! \brief regularization weight for L1 norm */
  float reg_alpha;
  int feature_selector;
  // declare parameters
  DMLC_DECLARE_PARAMETER(ShotgunTrainParam) {
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
        .describe("Feature selection or ordering method.");
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

class ShotgunUpdater : public LinearUpdater {
 public:
  // set training parameter
  void Init(const std::vector<std::pair<std::string, std::string> > &args) override {
    param_.InitAllowUnknown(args);
    selector_.reset(FeatureSelector::Create(param_.feature_selector));
  }
  void Update(HostDeviceVector<GradientPair> *in_gpair, DMatrix *p_fmat,
              gbm::GBLinearModel *model, double sum_instance_weight) override {
    auto &gpair = in_gpair->HostVector();
    param_.DenormalizePenalties(sum_instance_weight);
    const int ngroup = model->param.num_output_group;

    // update bias
    for (int gid = 0; gid < ngroup; ++gid) {
      auto grad = GetBiasGradientParallel(gid, ngroup,
                                          in_gpair->ConstHostVector(), p_fmat);
      auto dbias = static_cast<bst_float>(param_.learning_rate *
                               CoordinateDeltaBias(grad.first, grad.second));
      model->bias()[gid] += dbias;
      UpdateBiasResidualParallel(gid, ngroup, dbias, &in_gpair->HostVector(), p_fmat);
    }

    // lock-free parallel updates of weights
    selector_->Setup(*model, in_gpair->ConstHostVector(), p_fmat,
                     param_.reg_alpha_denorm, param_.reg_lambda_denorm, 0);
    for (const auto &batch : p_fmat->GetColumnBatches()) {
      const auto nfeat = static_cast<bst_omp_uint>(batch.Size());
#pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nfeat; ++i) {
        int ii = selector_->NextFeature
          (i, *model, 0, in_gpair->ConstHostVector(), p_fmat, param_.reg_alpha_denorm,
           param_.reg_lambda_denorm);
        if (ii < 0) continue;
        const bst_uint fid = ii;
        auto col = batch[ii];
        for (int gid = 0; gid < ngroup; ++gid) {
          double sum_grad = 0.0, sum_hess = 0.0;
          for (auto& c : col) {
            const GradientPair &p = gpair[c.index * ngroup + gid];
            if (p.GetHess() < 0.0f) continue;
            const bst_float v = c.fvalue;
            sum_grad += p.GetGrad() * v;
            sum_hess += p.GetHess() * v * v;
          }
          bst_float &w = (*model)[fid][gid];
          auto dw = static_cast<bst_float>(
              param_.learning_rate *
              CoordinateDelta(sum_grad, sum_hess, w, param_.reg_alpha_denorm,
                              param_.reg_lambda_denorm));
          if (dw == 0.f) continue;
          w += dw;
          // update grad values
          for (auto& c : col) {
            GradientPair &p = gpair[c.index * ngroup + gid];
            if (p.GetHess() < 0.0f) continue;
            p += GradientPair(p.GetHess() * c.fvalue * dw, 0);
          }
        }
      }
    }
  }

 protected:
  // training parameters
  ShotgunTrainParam param_;

  std::unique_ptr<FeatureSelector> selector_;
};

DMLC_REGISTER_PARAMETER(ShotgunTrainParam);

XGBOOST_REGISTER_LINEAR_UPDATER(ShotgunUpdater, "shotgun")
    .describe(
        "Update linear model according to shotgun coordinate descent "
        "algorithm.")
    .set_body([]() { return new ShotgunUpdater(); });
}  // namespace linear
}  // namespace xgboost
