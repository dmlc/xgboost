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
    param.InitAllowUnknown(args);
    selector.reset(FeatureSelector::Create(param.feature_selector));
  }

  void Update(std::vector<bst_gpair> *in_gpair, DMatrix *p_fmat,
              gbm::GBLinearModel *model, double sum_instance_weight) override {
    param.DenormalizePenalties(sum_instance_weight);
    std::vector<bst_gpair> &gpair = *in_gpair;
    const int ngroup = model->param.num_output_group;

    // update bias
    for (int gid = 0; gid < ngroup; ++gid) {
      auto grad = GetBiasGradientParallel(gid, ngroup, *in_gpair, p_fmat);
      auto dbias = static_cast<bst_float>(param.learning_rate *
                               CoordinateDeltaBias(grad.first, grad.second));
      model->bias()[gid] += dbias;
      UpdateBiasResidualParallel(gid, ngroup, dbias, in_gpair, p_fmat);
    }

    // lock-free parallel updates of weights
    selector->Setup(*model, *in_gpair, p_fmat, param.reg_alpha_denorm, param.reg_lambda_denorm, 0);
    dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator();
    while (iter->Next()) {
      const ColBatch &batch = iter->Value();
      const bst_omp_uint nfeat = static_cast<bst_omp_uint>(batch.size);
#pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nfeat; ++i) {
        int ii = selector->NextFeature(i, *model, 0, *in_gpair, p_fmat,
                                       param.reg_alpha_denorm, param.reg_lambda_denorm);
        if (ii < 0) continue;
        const bst_uint fid = batch.col_index[ii];
        ColBatch::Inst col = batch[ii];
        for (int gid = 0; gid < ngroup; ++gid) {
          double sum_grad = 0.0, sum_hess = 0.0;
          for (bst_uint j = 0; j < col.length; ++j) {
            bst_gpair &p = gpair[col[j].index * ngroup + gid];
            if (p.GetHess() < 0.0f) continue;
            const bst_float v = col[j].fvalue;
            sum_grad += p.GetGrad() * v;
            sum_hess += p.GetHess() * v * v;
          }
          bst_float &w = (*model)[fid][gid];
          bst_float dw = static_cast<bst_float>(
              param.learning_rate *
              CoordinateDelta(sum_grad, sum_hess, w, param.reg_alpha_denorm,
                              param.reg_lambda_denorm));
          if (dw == 0.f) continue;
          w += dw;
          // update grad values
          for (bst_uint j = 0; j < col.length; ++j) {
            bst_gpair &p = gpair[col[j].index * ngroup + gid];
            if (p.GetHess() < 0.0f) continue;
            p += bst_gpair(p.GetHess() * col[j].fvalue * dw, 0);
          }
        }
      }
    }
  }

 protected:
  // training parameters
  ShotgunTrainParam param;

  std::unique_ptr<FeatureSelector> selector;
};

DMLC_REGISTER_PARAMETER(ShotgunTrainParam);

XGBOOST_REGISTER_LINEAR_UPDATER(ShotgunUpdater, "shotgun")
    .describe(
        "Update linear model according to shotgun coordinate descent "
        "algorithm.")
    .set_body([]() { return new ShotgunUpdater(); });
}  // namespace linear
}  // namespace xgboost
