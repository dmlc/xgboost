/*!
 * Copyright 2014 by Contributors
 * \file gblinear.cc
 * \brief Implementation of Linear booster, with L1/L2 regularization: Elastic Net
 *        the update rule is parallel coordinate descent (shotgun)
 * \author Tianqi Chen
 */
#include <dmlc/omp.h>
#include <dmlc/parameter.h>
#include <xgboost/gbm.h>
#include <xgboost/logging.h>
#include <xgboost/linear_updater.h>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include "../common/timer.h"

namespace xgboost {
namespace gbm {

DMLC_REGISTRY_FILE_TAG(gblinear);

// training parameters
struct GBLinearTrainParam : public dmlc::Parameter<GBLinearTrainParam> {
  std::string updater;
  float tolerance;
  size_t max_row_perbatch;
  int debug_verbose;
  DMLC_DECLARE_PARAMETER(GBLinearTrainParam) {
    DMLC_DECLARE_FIELD(updater)
        .set_default("shotgun")
        .describe("Update algorithm for linear model. One of shotgun/coord_descent");
    DMLC_DECLARE_FIELD(tolerance)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe("Stop if largest weight update is smaller than this number.");
    DMLC_DECLARE_FIELD(max_row_perbatch)
        .set_default(std::numeric_limits<size_t>::max())
        .describe("Maximum rows per batch.");
    DMLC_DECLARE_FIELD(debug_verbose)
        .set_lower_bound(0)
        .set_default(0)
        .describe("flag to print out detailed breakdown of runtime");
  }
};
/*!
 * \brief gradient boosted linear model
 */
class GBLinear : public GradientBooster {
 public:
  explicit GBLinear(const std::vector<std::shared_ptr<DMatrix> > &cache,
                    bst_float base_margin)
      : base_margin_(base_margin),
        sum_instance_weight_(0),
        sum_weight_complete_(false),
        is_converged_(false) {
    // Add matrices to the prediction cache
    for (auto &d : cache) {
      PredictionCacheEntry e;
      e.data = d;
      cache_[d.get()] = std::move(e);
    }
  }
  void Configure(const std::vector<std::pair<std::string, std::string> >& cfg) override {
    if (model_.weight.size() == 0) {
      model_.param.InitAllowUnknown(cfg);
    }
    param_.InitAllowUnknown(cfg);
    updater_.reset(LinearUpdater::Create(param_.updater));
    updater_->Init(cfg);
    monitor_.Init("GBLinear ", param_.debug_verbose);
  }
  void Load(dmlc::Stream* fi) override {
    model_.Load(fi);
  }
  void Save(dmlc::Stream* fo) const override {
    model_.Save(fo);
  }

  void DoBoost(DMatrix *p_fmat,
               HostDeviceVector<GradientPair> *in_gpair,
               ObjFunction* obj) override {
    monitor_.Start("DoBoost");

    if (!p_fmat->HaveColAccess(false)) {
      monitor_.Start("InitColAccess");
      std::vector<bool> enabled(p_fmat->Info().num_col_, true);
      p_fmat->InitColAccess(param_.max_row_perbatch, false);
      monitor_.Stop("InitColAccess");
    }

    model_.LazyInitModel();
    this->LazySumWeights(p_fmat);

    if (!this->CheckConvergence()) {
      updater_->Update(in_gpair, p_fmat, &model_, sum_instance_weight_);
    }
    this->UpdatePredictionCache();

    monitor_.Stop("DoBoost");
  }

  void PredictBatch(DMatrix *p_fmat,
                    HostDeviceVector<bst_float> *out_preds,
                    unsigned ntree_limit) override {
    monitor_.Start("PredictBatch");
    CHECK_EQ(ntree_limit, 0U)
        << "GBLinear::Predict ntrees is only valid for gbtree predictor";

    // Try to predict from cache
    auto it = cache_.find(p_fmat);
    if (it != cache_.end() && it->second.predictions.size() != 0) {
      std::vector<bst_float> &y = it->second.predictions;
      out_preds->Resize(y.size());
      std::copy(y.begin(), y.end(), out_preds->HostVector().begin());
    } else {
      this->PredictBatchInternal(p_fmat, &out_preds->HostVector());
    }
    monitor_.Stop("PredictBatch");
  }
  // add base margin
  void PredictInstance(const SparsePage::Inst &inst,
               std::vector<bst_float> *out_preds,
               unsigned ntree_limit,
               unsigned root_index) override {
    const int ngroup = model_.param.num_output_group;
    for (int gid = 0; gid < ngroup; ++gid) {
      this->Pred(inst, dmlc::BeginPtr(*out_preds), gid, base_margin_);
    }
  }

  void PredictLeaf(DMatrix *p_fmat,
                   std::vector<bst_float> *out_preds,
                   unsigned ntree_limit) override {
    LOG(FATAL) << "gblinear does not support prediction of leaf index";
  }

  void PredictContribution(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           unsigned ntree_limit, bool approximate, int condition = 0,
                           unsigned condition_feature = 0) override {
    model_.LazyInitModel();
    CHECK_EQ(ntree_limit, 0U)
        << "GBLinear::PredictContribution: ntrees is only valid for gbtree predictor";
    const std::vector<bst_float>& base_margin = p_fmat->Info().base_margin_;
    const int ngroup = model_.param.num_output_group;
    const size_t ncolumns = model_.param.num_feature + 1;
    // allocate space for (#features + bias) times #groups times #rows
    std::vector<bst_float>& contribs = *out_contribs;
    contribs.resize(p_fmat->Info().num_row_ * ncolumns * ngroup);
    // make sure contributions is zeroed, we could be reusing a previously allocated one
    std::fill(contribs.begin(), contribs.end(), 0);
    // start collecting the contributions
     auto iter = p_fmat->RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
       auto &batch = iter->Value();
      // parallel over local batch
      const auto nsize = static_cast<bst_omp_uint>(batch.Size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
         auto inst = batch[i];
        auto row_idx = static_cast<size_t>(batch.base_rowid + i);
        // loop over output groups
        for (int gid = 0; gid < ngroup; ++gid) {
          bst_float *p_contribs = &contribs[(row_idx * ngroup + gid) * ncolumns];
          // calculate linear terms' contributions
          for (bst_uint c = 0; c < inst.length; ++c) {
            if (inst[c].index >= model_.param.num_feature) continue;
            p_contribs[inst[c].index] = inst[c].fvalue * model_[inst[c].index][gid];
          }
          // add base margin to BIAS
          p_contribs[ncolumns - 1] = model_.bias()[gid] +
            ((base_margin.size() != 0) ? base_margin[row_idx * ngroup + gid] : base_margin_);
        }
      }
    }
  }

  void PredictInteractionContributions(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           unsigned ntree_limit, bool approximate) override {
                             std::vector<bst_float>& contribs = *out_contribs;

     // linear models have no interaction effects
     const size_t nelements = model_.param.num_feature*model_.param.num_feature;
     contribs.resize(p_fmat->Info().num_row_ * nelements * model_.param.num_output_group);
     std::fill(contribs.begin(), contribs.end(), 0);
  }

  std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                     bool with_stats,
                                     std::string format) const override {
    return model_.DumpModel(fmap, with_stats, format);
  }

 protected:
  void PredictBatchInternal(DMatrix *p_fmat,
               std::vector<bst_float> *out_preds) {
    monitor_.Start("PredictBatchInternal");
      model_.LazyInitModel();
    std::vector<bst_float> &preds = *out_preds;
    const std::vector<bst_float>& base_margin = p_fmat->Info().base_margin_;
    // start collecting the prediction
     auto iter = p_fmat->RowIterator();
    const int ngroup = model_.param.num_output_group;
    preds.resize(p_fmat->Info().num_row_ * ngroup);
    while (iter->Next()) {
       auto &batch = iter->Value();
      // output convention: nrow * k, where nrow is number of rows
      // k is number of group
      // parallel over local batch
      const auto nsize = static_cast<omp_ulong>(batch.Size());
      #pragma omp parallel for schedule(static)
      for (omp_ulong i = 0; i < nsize; ++i) {
        const size_t ridx = batch.base_rowid + i;
        // loop over output groups
        for (int gid = 0; gid < ngroup; ++gid) {
          bst_float margin =  (base_margin.size() != 0) ?
              base_margin[ridx * ngroup + gid] : base_margin_;
          this->Pred(batch[i], &preds[ridx * ngroup], gid, margin);
        }
      }
    }
    monitor_.Stop("PredictBatchInternal");
  }
  void UpdatePredictionCache() {
    // update cache entry
    for (auto &kv : cache_) {
      PredictionCacheEntry &e = kv.second;
      if (e.predictions.size() == 0) {
        size_t n = model_.param.num_output_group * e.data->Info().num_row_;
        e.predictions.resize(n);
      }
      this->PredictBatchInternal(e.data.get(), &e.predictions);
    }
  }

  bool CheckConvergence() {
    if (param_.tolerance == 0.0f) return false;
    if (is_converged_) return true;
    if (previous_model_.weight.size() != model_.weight.size()) {
      previous_model_ = model_;
      return false;
    }
    float largest_dw = 0.0;
    for (size_t i = 0; i < model_.weight.size(); i++) {
      largest_dw = std::max(
          largest_dw, std::abs(model_.weight[i] - previous_model_.weight[i]));
    }
    previous_model_ = model_;

    is_converged_ = largest_dw <= param_.tolerance;
    return is_converged_;
  }

  void LazySumWeights(DMatrix *p_fmat) {
    if (!sum_weight_complete_) {
      auto &info = p_fmat->Info();
      for (size_t i = 0; i < info.num_row_; i++) {
        sum_instance_weight_ += info.GetWeight(i);
      }
      sum_weight_complete_ = true;
    }
  }

  inline void Pred(const SparsePage::Inst &inst, bst_float *preds, int gid,
                   bst_float base) {
    bst_float psum = model_.bias()[gid] + base;
    for (bst_uint i = 0; i < inst.length; ++i) {
      if (inst[i].index >= model_.param.num_feature) continue;
      psum += inst[i].fvalue * model_[inst[i].index][gid];
    }
    preds[gid] = psum;
  }
  // biase margin score
  bst_float base_margin_;
  // model field
  GBLinearModel model_;
  GBLinearModel previous_model_;
  GBLinearTrainParam param_;
  std::unique_ptr<LinearUpdater> updater_;
  double sum_instance_weight_;
  bool sum_weight_complete_;
  common::Monitor monitor_;
  bool is_converged_;

  /**
   * \struct  PredictionCacheEntry
   *
   * \brief Contains pointer to input matrix and associated cached predictions.
   */
  struct PredictionCacheEntry {
    std::shared_ptr<DMatrix> data;
    std::vector<bst_float> predictions;
  };

  /**
   * \brief Map of matrices and associated cached predictions to facilitate
   * storing and looking up predictions.
   */
  std::unordered_map<DMatrix*, PredictionCacheEntry> cache_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(GBLinearModelParam);
DMLC_REGISTER_PARAMETER(GBLinearTrainParam);

XGBOOST_REGISTER_GBM(GBLinear, "gblinear")
    .describe("Linear booster, implement generalized linear model.")
    .set_body([](const std::vector<std::shared_ptr<DMatrix> > &cache,
                 bst_float base_margin) {
      return new GBLinear(cache, base_margin);
    });
}  // namespace gbm
}  // namespace xgboost
