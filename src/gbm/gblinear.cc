/*!
 * Copyright 2014-2022 by XGBoost Contributors
 * \file gblinear.cc
 * \brief Implementation of Linear booster, with L1/L2 regularization: Elastic Net
 *        the update rule is parallel coordinate descent (shotgun)
 * \author Tianqi Chen
 */
#include <dmlc/omp.h>
#include <dmlc/parameter.h>

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>

#include "xgboost/gbm.h"
#include "xgboost/json.h"
#include "xgboost/predictor.h"
#include "xgboost/linear_updater.h"
#include "xgboost/logging.h"
#include "xgboost/learner.h"
#include "xgboost/linalg.h"

#include "gblinear_model.h"
#include "../common/timer.h"
#include "../common/common.h"
#include "../common/threading_utils.h"

namespace xgboost {
namespace gbm {

DMLC_REGISTRY_FILE_TAG(gblinear);

// training parameters
struct GBLinearTrainParam : public XGBoostParameter<GBLinearTrainParam> {
  std::string updater;
  float tolerance;
  size_t max_row_perbatch;

  void CheckGPUSupport() {
    auto n_gpus = common::AllVisibleGPUs();
    if (n_gpus == 0 && this->updater == "gpu_coord_descent") {
      common::AssertGPUSupport();
      this->UpdateAllowUnknown(Args{{"updater", "coord_descent"}});
      LOG(WARNING) << "Loading configuration on a CPU only machine.   Changing "
                      "updater to `coord_descent`.";
    }
  }

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
  }
};

void LinearCheckLayer(unsigned layer_begin) {
  CHECK_EQ(layer_begin, 0) << "Linear booster does not support prediction range.";
}

/*!
 * \brief gradient boosted linear model
 */
class GBLinear : public GradientBooster {
 public:
  explicit GBLinear(LearnerModelParam const* learner_model_param, Context const* ctx)
      : GradientBooster{ctx},
        learner_model_param_{learner_model_param},
        model_{learner_model_param},
        previous_model_{learner_model_param} {}

  void Configure(const Args& cfg) override {
    if (model_.weight.size() == 0) {
      model_.Configure(cfg);
    }
    param_.UpdateAllowUnknown(cfg);
    param_.CheckGPUSupport();
    updater_.reset(LinearUpdater::Create(param_.updater, ctx_));
    updater_->Configure(cfg);
    monitor_.Init("GBLinear");
  }

  int32_t BoostedRounds() const override {
    return model_.num_boosted_rounds;
  }

  bool ModelFitted() const override { return BoostedRounds() != 0; }

  void Load(dmlc::Stream* fi) override {
    model_.Load(fi);
  }
  void Save(dmlc::Stream* fo) const override {
    model_.Save(fo);
  }

  void SaveModel(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String{"gblinear"};

    out["model"] = Object();
    auto& model = out["model"];
    model_.SaveModel(&model);
  }
  void LoadModel(Json const& in) override {
    CHECK_EQ(get<String>(in["name"]), "gblinear");
    auto const& model = in["model"];
    model_.LoadModel(model);
  }

  void LoadConfig(Json const& in) override {
    CHECK_EQ(get<String>(in["name"]), "gblinear");
    FromJson(in["gblinear_train_param"], &param_);
    param_.CheckGPUSupport();
    updater_.reset(LinearUpdater::Create(param_.updater, ctx_));
    this->updater_->LoadConfig(in["updater"]);
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String{"gblinear"};
    out["gblinear_train_param"] = ToJson(param_);

    out["updater"] = Object();
    auto& j_updater = out["updater"];
    CHECK(this->updater_);
    this->updater_->SaveConfig(&j_updater);
  }

  void DoBoost(DMatrix* p_fmat, HostDeviceVector<GradientPair>* in_gpair, PredictionCacheEntry*,
               ObjFunction const*) override {
    monitor_.Start("DoBoost");

    model_.LazyInitModel();
    this->LazySumWeights(p_fmat);

    if (!this->CheckConvergence()) {
      updater_->Update(in_gpair, p_fmat, &model_, sum_instance_weight_);
    }
    model_.num_boosted_rounds++;
    monitor_.Stop("DoBoost");
  }

  void PredictBatch(DMatrix* p_fmat, PredictionCacheEntry* predts, bool /*training*/,
                    bst_layer_t layer_begin, bst_layer_t) override {
    monitor_.Start("PredictBatch");
    LinearCheckLayer(layer_begin);
    auto* out_preds = &predts->predictions;
    this->PredictBatchInternal(p_fmat, &out_preds->HostVector());
    monitor_.Stop("PredictBatch");
  }
  // add base margin
  void PredictInstance(const SparsePage::Inst& inst, std::vector<bst_float>* out_preds,
                       uint32_t layer_begin, uint32_t) override {
    LinearCheckLayer(layer_begin);
    const int ngroup = model_.learner_model_param->num_output_group;

    auto base_score = learner_model_param_->BaseScore(ctx_);
    for (int gid = 0; gid < ngroup; ++gid) {
      this->Pred(inst, dmlc::BeginPtr(*out_preds), gid, base_score(0));
    }
  }

  void PredictLeaf(DMatrix *, HostDeviceVector<bst_float> *, unsigned, unsigned) override {
    LOG(FATAL) << "gblinear does not support prediction of leaf index";
  }

  void PredictContribution(DMatrix* p_fmat, HostDeviceVector<bst_float>* out_contribs,
                           uint32_t layer_begin, uint32_t /*layer_end*/, bool, int,
                           unsigned) override {
    model_.LazyInitModel();
    LinearCheckLayer(layer_begin);
    auto base_margin = p_fmat->Info().base_margin_.View(Context::kCpuId);
    const int ngroup = model_.learner_model_param->num_output_group;
    const size_t ncolumns = model_.learner_model_param->num_feature + 1;
    // allocate space for (#features + bias) times #groups times #rows
    std::vector<bst_float>& contribs = out_contribs->HostVector();
    contribs.resize(p_fmat->Info().num_row_ * ncolumns * ngroup);
    // make sure contributions is zeroed, we could be reusing a previously allocated one
    std::fill(contribs.begin(), contribs.end(), 0);
    auto base_score = learner_model_param_->BaseScore(ctx_);
    // start collecting the contributions
    for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
      // parallel over local batch
      const auto nsize = static_cast<bst_omp_uint>(batch.Size());
      auto page = batch.GetView();
      common::ParallelFor(nsize, ctx_->Threads(), [&](bst_omp_uint i) {
        auto inst = page[i];
        auto row_idx = static_cast<size_t>(batch.base_rowid + i);
        // loop over output groups
        for (int gid = 0; gid < ngroup; ++gid) {
          bst_float *p_contribs = &contribs[(row_idx * ngroup + gid) * ncolumns];
          // calculate linear terms' contributions
          for (auto& ins : inst) {
            if (ins.index >= model_.learner_model_param->num_feature) continue;
            p_contribs[ins.index] = ins.fvalue * model_[ins.index][gid];
          }
          // add base margin to BIAS
          p_contribs[ncolumns - 1] =
              model_.Bias()[gid] +
              ((base_margin.Size() != 0) ? base_margin(row_idx, gid) : base_score(0));
        }
      });
    }
  }

  void PredictInteractionContributions(DMatrix* p_fmat, HostDeviceVector<bst_float>* out_contribs,
                                       unsigned layer_begin, unsigned /*layer_end*/,
                                       bool) override {
    LinearCheckLayer(layer_begin);
    std::vector<bst_float>& contribs = out_contribs->HostVector();

    // linear models have no interaction effects
    const size_t nelements = model_.learner_model_param->num_feature *
                             model_.learner_model_param->num_feature;
    contribs.resize(p_fmat->Info().num_row_ * nelements *
                    model_.learner_model_param->num_output_group);
    std::fill(contribs.begin(), contribs.end(), 0);
  }

  std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                     bool with_stats,
                                     std::string format) const override {
    return model_.DumpModel(fmap, with_stats, format);
  }

  void FeatureScore(std::string const &importance_type,
                    common::Span<int32_t const> trees,
                    std::vector<bst_feature_t> *out_features,
                    std::vector<float> *out_scores) const override {
    CHECK(!model_.weight.empty()) << "Model is not initialized";
    CHECK(trees.empty()) << "gblinear doesn't support number of trees for feature importance.";
    CHECK_EQ(importance_type, "weight")
        << "gblinear only has `weight` defined for feature importance.";
    out_features->resize(this->learner_model_param_->num_feature, 0);
    std::iota(out_features->begin(), out_features->end(), 0);
    // Don't include the bias term in the feature importance scores
    // The bias is the last weight
    out_scores->resize(model_.weight.size() - learner_model_param_->num_output_group, 0);
    auto n_groups = learner_model_param_->num_output_group;
    linalg::TensorView<float, 2> scores{
        *out_scores,
        {learner_model_param_->num_feature, n_groups},
        Context::kCpuId};
    for (size_t i = 0; i < learner_model_param_->num_feature; ++i) {
      for (bst_group_t g = 0; g < n_groups; ++g) {
        scores(i, g) = model_[i][g];
      }
    }
  }

  bool UseGPU() const override {
    if (param_.updater == "gpu_coord_descent") {
      return true;
    } else {
      return false;
    }
  }

 protected:
  void PredictBatchInternal(DMatrix *p_fmat,
                            std::vector<bst_float> *out_preds) {
    monitor_.Start("PredictBatchInternal");
    model_.LazyInitModel();
    std::vector<bst_float> &preds = *out_preds;
    auto base_margin = p_fmat->Info().base_margin_.View(Context::kCpuId);
    // start collecting the prediction
    const int ngroup = model_.learner_model_param->num_output_group;
    preds.resize(p_fmat->Info().num_row_ * ngroup);

    auto base_score = learner_model_param_->BaseScore(Context::kCpuId);
    for (const auto &page : p_fmat->GetBatches<SparsePage>()) {
      auto const& batch = page.GetView();
      // output convention: nrow * k, where nrow is number of rows
      // k is number of group
      // parallel over local batch
      const auto nsize = static_cast<omp_ulong>(batch.Size());
      if (base_margin.Size() != 0) {
        CHECK_EQ(base_margin.Size(), nsize * ngroup);
      }
      common::ParallelFor(nsize, ctx_->Threads(), [&](omp_ulong i) {
        const size_t ridx = page.base_rowid + i;
        // loop over output groups
        for (int gid = 0; gid < ngroup; ++gid) {
          float margin = (base_margin.Size() != 0) ? base_margin(ridx, gid) : base_score(0);
          this->Pred(batch[i], &preds[ridx * ngroup], gid, margin);
        }
      });
    }
    monitor_.Stop("PredictBatchInternal");
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

  void Pred(const SparsePage::Inst &inst, bst_float *preds, int gid,
            bst_float base) {
    bst_float psum = model_.Bias()[gid] + base;
    for (const auto& ins : inst) {
      if (ins.index >= model_.learner_model_param->num_feature) continue;
      psum += ins.fvalue * model_[ins.index][gid];
    }
    preds[gid] = psum;
  }

  // biase margin score
  LearnerModelParam const* learner_model_param_;
  // model field
  GBLinearModel model_;
  GBLinearModel previous_model_;
  GBLinearTrainParam param_;
  std::unique_ptr<LinearUpdater> updater_;
  double sum_instance_weight_{};
  bool sum_weight_complete_{false};
  common::Monitor monitor_;
  bool is_converged_{false};
};

// register the objective functions
DMLC_REGISTER_PARAMETER(GBLinearTrainParam);

XGBOOST_REGISTER_GBM(GBLinear, "gblinear")
    .describe("Linear booster, implement generalized linear model.")
    .set_body([](LearnerModelParam const* booster_config, Context const* ctx) {
      return new GBLinear(booster_config, ctx);
    });
}  // namespace gbm
}  // namespace xgboost
