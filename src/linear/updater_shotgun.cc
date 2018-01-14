/*!
 * Copyright 2018 by Contributors
 * \author Tianqi Chen
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
  // declare parameters
  DMLC_DECLARE_PARAMETER(ShotgunTrainParam) {
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
    // alias of parameters
    DMLC_DECLARE_ALIAS(learning_rate, eta);
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_alpha, alpha);
  }
};

class ShotgunUpdater : public LinearUpdater {
 public:
  // set training parameter
  void Init(
      const std::vector<std::pair<std::string, std::string> > &args) override {
    param.InitAllowUnknown(args);
  }
  void Update(std::vector<bst_gpair> *in_gpair, DMatrix *p_fmat,
              gbm::GBLinearModel *model, double sum_instance_weight) override {
    std::vector<bst_gpair> &gpair = *in_gpair;
    const int ngroup = model->param.num_output_group;
    const RowSet &rowset = p_fmat->buffered_rowset();
    // for all the output group
    for (int gid = 0; gid < ngroup; ++gid) {
      double sum_grad = 0.0, sum_hess = 0.0;
      const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
#pragma omp parallel for schedule(static) reduction(+ : sum_grad, sum_hess)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        bst_gpair &p = gpair[rowset[i] * ngroup + gid];
        if (p.GetHess() >= 0.0f) {
          sum_grad += p.GetGrad();
          sum_hess += p.GetHess();
        }
      }
      // remove bias effect
      bst_float dw = static_cast<bst_float>(
          param.learning_rate * CoordinateDeltaBias(sum_grad, sum_hess));
      model->bias()[gid] += dw;
// update grad value
#pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        bst_gpair &p = gpair[rowset[i] * ngroup + gid];
        if (p.GetHess() >= 0.0f) {
          p += bst_gpair(p.GetHess() * dw, 0);
        }
      }
    }
    dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator();
    while (iter->Next()) {
      // number of features
      const ColBatch &batch = iter->Value();
      const bst_omp_uint nfeat = static_cast<bst_omp_uint>(batch.size);
#pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nfeat; ++i) {
        const bst_uint fid = batch.col_index[i];
        ColBatch::Inst col = batch[i];
        for (int gid = 0; gid < ngroup; ++gid) {
          double sum_grad = 0.0, sum_hess = 0.0;
          for (bst_uint j = 0; j < col.length; ++j) {
            const bst_float v = col[j].fvalue;
            bst_gpair &p = gpair[col[j].index * ngroup + gid];
            if (p.GetHess() < 0.0f) continue;
            sum_grad += p.GetGrad() * v;
            sum_hess += p.GetHess() * v * v;
          }
          bst_float &w = (*model)[fid][gid];
          bst_float dw = static_cast<bst_float>(
              param.learning_rate *
              CoordinateDelta(sum_grad, sum_hess, w, param.reg_lambda,
                              param.reg_alpha, sum_instance_weight));
          w += dw;
          // update grad value
          for (bst_uint j = 0; j < col.length; ++j) {
            bst_gpair &p = gpair[col[j].index * ngroup + gid];
            if (p.GetHess() < 0.0f) continue;
            p += bst_gpair(p.GetHess() * col[j].fvalue * dw, 0);
          }
        }
      }
    }
  }

  // training parameter
  ShotgunTrainParam param;
};

DMLC_REGISTER_PARAMETER(ShotgunTrainParam);

XGBOOST_REGISTER_LINEAR_UPDATER(ShotgunUpdater, "updater_shotgun")
    .describe(
        "Update linear model according to shotgun coordinate descent "
        "algorithm.")
    .set_body([]() { return new ShotgunUpdater(); });
}  // namespace linear
}  // namespace xgboost
