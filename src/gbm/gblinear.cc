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

// training parameter
struct GBLinearTrainParam : public dmlc::Parameter<GBLinearTrainParam> {
  /*! \brief learning_rate */
  std::string updater;
  // flag to print out detailed breakdown of runtime
  int debug_verbose;
  float tolerance;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GBLinearTrainParam) {
    DMLC_DECLARE_FIELD(updater)
        .set_default("shotgun")
        .describe("Update algorithm for linear model. One of shotgun/coord_descent");
    DMLC_DECLARE_FIELD(tolerance)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe("Stop if largest weight update is smaller than this number.");
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
        sum_instance_weight(0),
        sum_weight_complete(false),
        is_converged(false) {
    // Add matrices to the prediction cache
    for (auto &d : cache) {
      PredictionCacheEntry e;
      e.data = d;
      cache_[d.get()] = std::move(e);
    }
  }
  void Configure(const std::vector<std::pair<std::string, std::string> >& cfg) override {
    if (model.weight.size() == 0) {
      model.param.InitAllowUnknown(cfg);
    }
    param.InitAllowUnknown(cfg);
    updater.reset(LinearUpdater::Create(param.updater));
    updater->Init(cfg);
    monitor.Init("GBLinear ", param.debug_verbose);
  }
  void Load(dmlc::Stream* fi) override {
    model.Load(fi);
  }
  void Save(dmlc::Stream* fo) const override {
    model.Save(fo);
  }
  void DoBoost(DMatrix *p_fmat, std::vector<bst_gpair> *in_gpair,
               ObjFunction *obj) override {
    monitor.Start("DoBoost");

    if (!p_fmat->HaveColAccess(false)) {
      std::vector<bool> enabled(p_fmat->info().num_col, true);
      p_fmat->InitColAccess(enabled, 1.0f, 32ul << 10ul, false);
    }

    model.LazyInitModel();

    this->LazySumWeights(p_fmat);

    if (!this->CheckConvergence()) {
      updater->Update(in_gpair, p_fmat, &model, sum_instance_weight);
    }
    this->UpdatePredictionCache();

    monitor.Stop("DoBoost");
  }

  void PredictBatch(DMatrix *p_fmat, std::vector<bst_float> *out_preds,
                    unsigned ntree_limit) override {
    monitor.Start("PredictBatch");
    CHECK_EQ(ntree_limit, 0U)
        << "GBLinear::Predict ntrees is only valid for gbtree predictor";

    // Try to predict from cache
    auto it = cache_.find(p_fmat);
    if (it != cache_.end() && it->second.predictions.size() != 0) {
      std::vector<bst_float> &y = it->second.predictions;
      out_preds->resize(y.size());
      std::copy(y.begin(), y.end(), out_preds->begin());
    } else {
      this->PredictBatchInternal(p_fmat, out_preds);
    }
    monitor.Stop("PredictBatch");
  }
  // add base margin
  void PredictInstance(const SparseBatch::Inst &inst,
               std::vector<bst_float> *out_preds,
               unsigned ntree_limit,
               unsigned root_index) override {
    const int ngroup = model.param.num_output_group;
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
    model.LazyInitModel();
    CHECK_EQ(ntree_limit, 0U)
        << "GBLinear::PredictContribution: ntrees is only valid for gbtree predictor";
    const std::vector<bst_float>& base_margin = p_fmat->info().base_margin;
    const int ngroup = model.param.num_output_group;
    const size_t ncolumns = model.param.num_feature + 1;
    // allocate space for (#features + bias) times #groups times #rows
    std::vector<bst_float>& contribs = *out_contribs;
    contribs.resize(p_fmat->info().num_row * ncolumns * ngroup);
    // make sure contributions is zeroed, we could be reusing a previously allocated one
    std::fill(contribs.begin(), contribs.end(), 0);
    // start collecting the contributions
    dmlc::DataIter<RowBatch>* iter = p_fmat->RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch& batch = iter->Value();
      // parallel over local batch
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        const RowBatch::Inst &inst = batch[i];
        size_t row_idx = static_cast<size_t>(batch.base_rowid + i);
        // loop over output groups
        for (int gid = 0; gid < ngroup; ++gid) {
          bst_float *p_contribs = &contribs[(row_idx * ngroup + gid) * ncolumns];
          // calculate linear terms' contributions
          for (bst_uint c = 0; c < inst.length; ++c) {
            if (inst[c].index >= model.param.num_feature) continue;
            p_contribs[inst[c].index] = inst[c].fvalue * model[inst[c].index][gid];
          }
          // add base margin to BIAS
          p_contribs[ncolumns - 1] = model.bias()[gid] +
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
     const size_t nelements = model.param.num_feature*model.param.num_feature;
     contribs.resize(p_fmat->info().num_row * nelements * model.param.num_output_group);
     std::fill(contribs.begin(), contribs.end(), 0);
  }

  std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                     bool with_stats,
                                     std::string format) const override {
    const int ngroup = model.param.num_output_group;
    const unsigned nfeature = model.param.num_feature;

    std::stringstream fo("");
    if (format == "json") {
      fo << "  { \"bias\": [" << std::endl;
      for (int gid = 0; gid < ngroup; ++gid) {
        if (gid != 0) fo << "," << std::endl;
        fo << "      " << model.bias()[gid];
      }
      fo << std::endl << "    ]," << std::endl
         << "    \"weight\": [" << std::endl;
      for (unsigned i = 0; i < nfeature; ++i) {
        for (int gid = 0; gid < ngroup; ++gid) {
          if (i != 0 || gid != 0) fo << "," << std::endl;
          fo << "      " << model[i][gid];
        }
      }
      fo << std::endl << "    ]" << std::endl << "  }";
    } else {
      fo << "bias:\n";
      for (int gid = 0; gid < ngroup; ++gid) {
        fo << model.bias()[gid] << std::endl;
      }
      fo << "weight:\n";
      for (unsigned i = 0; i < nfeature; ++i) {
        for (int gid = 0; gid < ngroup; ++gid) {
          fo << model[i][gid] << std::endl;
        }
      }
    }
    std::vector<std::string> v;
    v.push_back(fo.str());
    return v;
  }

 protected:
  void PredictBatchInternal(DMatrix *p_fmat,
               std::vector<bst_float> *out_preds) {
    monitor.Start("PredictBatchInternal");
      model.LazyInitModel();
    std::vector<bst_float> &preds = *out_preds;
    const std::vector<bst_float>& base_margin = p_fmat->info().base_margin;
    // start collecting the prediction
    dmlc::DataIter<RowBatch> *iter = p_fmat->RowIterator();
    const int ngroup = model.param.num_output_group;
    preds.resize(p_fmat->info().num_row * ngroup);
    while (iter->Next()) {
      const RowBatch &batch = iter->Value();
      // output convention: nrow * k, where nrow is number of rows
      // k is number of group
      // parallel over local batch
      const omp_ulong nsize = static_cast<omp_ulong>(batch.size);
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
    monitor.Stop("PredictBatchInternal");
  }
  void UpdatePredictionCache() {
    // update cache entry
    for (auto &kv : cache_) {
      PredictionCacheEntry &e = kv.second;
      if (e.predictions.size() == 0) {
        size_t n = model.param.num_output_group * e.data->info().num_row;
        e.predictions.resize(n);
      }
      this->PredictBatchInternal(e.data.get(), &e.predictions);
    }
  }

  bool CheckConvergence() {
    if (param.tolerance == 0.0f) return false;
    if (is_converged) return true;
    if (previous_model.weight.size() != model.weight.size()) return false;
    float largest_dw = 0.0;
    for (auto i = 0; i < model.weight.size(); i++) {
      largest_dw = std::max(
          largest_dw, std::abs(model.weight[i] - previous_model.weight[i]));
    }
    previous_model = model;

    is_converged = largest_dw <= param.tolerance;
    return is_converged;
  }

  void LazySumWeights(DMatrix *p_fmat) {
    if (!sum_weight_complete) {
      auto &info = p_fmat->info();
      for (int i = 0; i < info.num_row; i++) {
        sum_instance_weight += info.GetWeight(i);
      }
      sum_weight_complete = true;
    }
  }

  inline void Pred(const RowBatch::Inst &inst, bst_float *preds, int gid,
                   bst_float base) {
    bst_float psum = model.bias()[gid] + base;
    for (bst_uint i = 0; i < inst.length; ++i) {
      if (inst[i].index >= model.param.num_feature) continue;
      psum += inst[i].fvalue * model[inst[i].index][gid];
    }
    preds[gid] = psum;
  }
  // biase margin score
  bst_float base_margin_;
  // model field
  GBLinearModel model;
  GBLinearModel previous_model;
  GBLinearTrainParam param;
  std::unique_ptr<LinearUpdater> updater;
  double sum_instance_weight;
  bool sum_weight_complete;
  common::Monitor monitor;
  bool is_converged;

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
