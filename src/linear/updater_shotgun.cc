/*!
 * Copyright 2018 by Contributors
 * \author Tianqi Chen, Rory Mitchell
 */

#include <xgboost/linear_updater.h>
#include "coordinate_common.h"

namespace xgboost {
namespace linear {

DMLC_REGISTRY_FILE_TAG(updater_shotgun);

class ShotgunUpdater : public LinearUpdater {
 public:
  // set training parameter
  void Configure(Args const& args) override {
    param_.UpdateAllowUnknown(args);
    if (param_.feature_selector != kCyclic &&
        param_.feature_selector != kShuffle) {
      LOG(FATAL) << "Unsupported feature selector for shotgun updater.\n"
                 << "Supported options are: {cyclic, shuffle}";
    }
    selector_.reset(FeatureSelector::Create(param_.feature_selector));
  }
  void Update(HostDeviceVector<GradientPair> *in_gpair, DMatrix *p_fmat,
              gbm::GBLinearModel *model, double sum_instance_weight) override {
    auto &gpair = in_gpair->HostVector();
    param_.DenormalizePenalties(sum_instance_weight);
    const int ngroup = model->learner_model_param_->num_output_group;

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
    for (const auto &batch : p_fmat->GetBatches<CSCPage>()) {
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
  LinearTrainParam param_;

  std::unique_ptr<FeatureSelector> selector_;
};

XGBOOST_REGISTER_LINEAR_UPDATER(ShotgunUpdater, "shotgun")
    .describe(
        "Update linear model according to shotgun coordinate descent "
        "algorithm.")
    .set_body([]() { return new ShotgunUpdater(); });
}  // namespace linear
}  // namespace xgboost
