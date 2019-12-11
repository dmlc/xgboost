/*!
 * Copyright 2014-2019 by Contributors
 * \file gbtree.cc
 * \brief gradient boosted tree implementation.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_GBM_GBTREE_H_
#define XGBOOST_GBM_GBTREE_H_

#include <dmlc/omp.h>

#include <vector>
#include <map>
#include <memory>
#include <utility>
#include <string>
#include <unordered_map>

#include "xgboost/logging.h"
#include "xgboost/gbm.h"
#include "xgboost/predictor.h"
#include "xgboost/tree_updater.h"
#include "xgboost/parameter.h"
#include "xgboost/json.h"
#include "xgboost/host_device_vector.h"

#include "gbtree_model.h"
#include "../common/common.h"
#include "../common/timer.h"

namespace xgboost {
enum class TreeMethod : int {
  kAuto = 0, kApprox = 1, kExact = 2, kHist = 3,
  kGPUHist = 5
};

// boosting process types
enum class TreeProcessType : int {
  kDefault = 0,
  kUpdate = 1
};

enum class PredictorType : int {
  kAuto = 0,
  kCPUPredictor,
  kGPUPredictor
};
}  // namespace xgboost

DECLARE_FIELD_ENUM_CLASS(xgboost::TreeMethod);
DECLARE_FIELD_ENUM_CLASS(xgboost::TreeProcessType);
DECLARE_FIELD_ENUM_CLASS(xgboost::PredictorType);

namespace xgboost {
namespace gbm {

/*! \brief training parameters */
struct GBTreeTrainParam : public XGBoostParameter<GBTreeTrainParam> {
  /*!
   * \brief number of parallel trees constructed each iteration
   *  use this option to support boosted random forest
   */
  int num_parallel_tree;
  /*! \brief tree updater sequence */
  std::string updater_seq;
  /*! \brief type of boosting process to run */
  TreeProcessType process_type;
  // predictor type
  PredictorType predictor;
  // tree construction method
  TreeMethod tree_method;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GBTreeTrainParam) {
    DMLC_DECLARE_FIELD(num_parallel_tree)
        .set_default(1)
        .set_lower_bound(1)
        .describe("Number of parallel trees constructed during each iteration."\
                  " This option is used to support boosted random forest.");
    DMLC_DECLARE_FIELD(updater_seq)
        .set_default("grow_colmaker,prune")
        .describe("Tree updater sequence.");
    DMLC_DECLARE_FIELD(process_type)
        .set_default(TreeProcessType::kDefault)
        .add_enum("default", TreeProcessType::kDefault)
        .add_enum("update", TreeProcessType::kUpdate)
        .describe("Whether to run the normal boosting process that creates new trees,"\
                  " or to update the trees in an existing model.");
    DMLC_DECLARE_ALIAS(updater_seq, updater);
    DMLC_DECLARE_FIELD(predictor)
        .set_default(PredictorType::kAuto)
        .add_enum("auto", PredictorType::kAuto)
        .add_enum("cpu_predictor", PredictorType::kCPUPredictor)
        .add_enum("gpu_predictor", PredictorType::kGPUPredictor)
        .describe("Predictor algorithm type");
    DMLC_DECLARE_FIELD(tree_method)
        .set_default(TreeMethod::kAuto)
        .add_enum("auto",      TreeMethod::kAuto)
        .add_enum("approx",    TreeMethod::kApprox)
        .add_enum("exact",     TreeMethod::kExact)
        .add_enum("hist",      TreeMethod::kHist)
        .add_enum("gpu_hist",  TreeMethod::kGPUHist)
        .describe("Choice of tree construction method.");
  }
};

/*! \brief training parameters */
struct DartTrainParam : public XGBoostParameter<DartTrainParam> {
  /*! \brief type of sampling algorithm */
  int sample_type;
  /*! \brief type of normalization algorithm */
  int normalize_type;
  /*! \brief fraction of trees to drop during the dropout */
  float rate_drop;
  /*! \brief whether at least one tree should always be dropped during the dropout */
  bool one_drop;
  /*! \brief probability of skipping the dropout during an iteration */
  float skip_drop;
  /*! \brief learning step size for a time */
  float learning_rate;
  // declare parameters
  DMLC_DECLARE_PARAMETER(DartTrainParam) {
    DMLC_DECLARE_FIELD(sample_type)
        .set_default(0)
        .add_enum("uniform", 0)
        .add_enum("weighted", 1)
        .describe("Different types of sampling algorithm.");
    DMLC_DECLARE_FIELD(normalize_type)
        .set_default(0)
        .add_enum("tree", 0)
        .add_enum("forest", 1)
        .describe("Different types of normalization algorithm.");
    DMLC_DECLARE_FIELD(rate_drop)
        .set_range(0.0f, 1.0f)
        .set_default(0.0f)
        .describe("Fraction of trees to drop during the dropout.");
    DMLC_DECLARE_FIELD(one_drop)
        .set_default(false)
        .describe("Whether at least one tree should always be dropped during the dropout.");
    DMLC_DECLARE_FIELD(skip_drop)
        .set_range(0.0f, 1.0f)
        .set_default(0.0f)
        .describe("Probability of skipping the dropout during a boosting iteration.");
    DMLC_DECLARE_FIELD(learning_rate)
        .set_lower_bound(0.0f)
        .set_default(0.3f)
        .describe("Learning rate(step size) of update.");
    DMLC_DECLARE_ALIAS(learning_rate, eta);
  }
};

// gradient boosted trees
class GBTree : public GradientBooster {
 public:
  explicit GBTree(LearnerModelParam const* booster_config) : model_(booster_config) {}

  void InitCache(const std::vector<std::shared_ptr<DMatrix> > &cache) {
    cache_ = std::make_shared<std::unordered_map<DMatrix*, PredictionCacheEntry>>();
    for (std::shared_ptr<DMatrix> const& d : cache) {
      (*cache_)[d.get()].data = d;
    }
  }

  void Configure(const Args& cfg) override;
  // Revise `tree_method` and `updater` parameters after seeing the training
  // data matrix, only useful when tree_method is auto.
  void PerformTreeMethodHeuristic(DMatrix* fmat);
  /*! \brief Map `tree_method` parameter to `updater` parameter */
  void ConfigureUpdaters();
  void ConfigureWithKnownData(Args const& cfg, DMatrix* fmat);

  /*! \brief Carry out one iteration of boosting */
  void DoBoost(DMatrix* p_fmat,
               HostDeviceVector<GradientPair>* in_gpair,
               ObjFunction* obj) override;

  bool UseGPU() const override {
    return
        tparam_.predictor == PredictorType::kGPUPredictor ||
        tparam_.tree_method == TreeMethod::kGPUHist;
  }

  GBTreeTrainParam const& GetTrainParam() const {
    return tparam_;
  }

  void Load(dmlc::Stream* fi) override {
    model_.Load(fi);
    this->cfg_.clear();
  }

  void Save(dmlc::Stream* fo) const override {
    model_.Save(fo);
  }

  void LoadConfig(Json const& in) override;
  void SaveConfig(Json* p_out) const override;

  void SaveModel(Json* p_out) const override;
  void LoadModel(Json const& in) override;

  bool AllowLazyCheckPoint() const override {
    return model_.learner_model_param_->num_output_group == 1 ||
        tparam_.updater_seq.find("distcol") != std::string::npos;
  }

  void PredictBatch(DMatrix* p_fmat,
                    HostDeviceVector<bst_float>* out_preds,
                    unsigned ntree_limit) override {
    CHECK(configured_);
    GetPredictor(out_preds, p_fmat)->PredictBatch(p_fmat, out_preds, model_, 0, ntree_limit);
  }

  void PredictInstance(const SparsePage::Inst& inst,
                       std::vector<bst_float>* out_preds,
                       unsigned ntree_limit) override {
    CHECK(configured_);
    cpu_predictor_->PredictInstance(inst, out_preds, model_,
                                    ntree_limit);
  }

  void PredictLeaf(DMatrix* p_fmat,
                   std::vector<bst_float>* out_preds,
                   unsigned ntree_limit) override {
    CHECK(configured_);
    cpu_predictor_->PredictLeaf(p_fmat, out_preds, model_, ntree_limit);
  }

  void PredictContribution(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           unsigned ntree_limit, bool approximate, int condition,
                           unsigned condition_feature) override {
    CHECK(configured_);
    cpu_predictor_->PredictContribution(p_fmat, out_contribs, model_,
                                        ntree_limit, nullptr, approximate);
  }

  void PredictInteractionContributions(DMatrix* p_fmat,
                                       std::vector<bst_float>* out_contribs,
                                       unsigned ntree_limit, bool approximate) override {
    CHECK(configured_);
    cpu_predictor_->PredictInteractionContributions(p_fmat, out_contribs, model_,
                                                    ntree_limit, nullptr, approximate);
  }

  std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                     bool with_stats,
                                     std::string format) const override {
    return model_.DumpModel(fmap, with_stats, format);
  }

 protected:
  // initialize updater before using them
  void InitUpdater(Args const& cfg);

  // do group specific group
  void BoostNewTrees(HostDeviceVector<GradientPair>* gpair,
                     DMatrix *p_fmat,
                     int bst_group,
                     std::vector<std::unique_ptr<RegTree> >* ret);

  std::unique_ptr<Predictor> const& GetPredictor(HostDeviceVector<float> const* out_pred = nullptr,
                                                 DMatrix* f_dmat = nullptr) const {
    CHECK(configured_);
    if (tparam_.predictor != PredictorType::kAuto) {
      if (tparam_.predictor == PredictorType::kGPUPredictor) {
#if defined(XGBOOST_USE_CUDA)
        CHECK(gpu_predictor_);
        return gpu_predictor_;
#else
        this->AssertGPUSupport();
#endif  // defined(XGBOOST_USE_CUDA)
      }
      CHECK(cpu_predictor_);
      return cpu_predictor_;
    }

    auto on_device = f_dmat && (*(f_dmat->GetBatches<SparsePage>().begin())).data.DeviceCanRead();

    // Use GPU Predictor if data is already on device.
    if (on_device) {
#if defined(XGBOOST_USE_CUDA)
      CHECK(gpu_predictor_);
      return gpu_predictor_;
#else
      LOG(FATAL) << "Data is on CUDA device, but XGBoost is not compiled with CUDA support.";
      return cpu_predictor_;
#endif  // defined(XGBOOST_USE_CUDA)
    }

    // GPU_Hist by default has prediction cache calculated from quantile values, so GPU
    // Predictor is not used for training dataset.  But when XGBoost performs continue
    // training with an existing model, the prediction cache is not availbale and number
    // of trees doesn't equal zero, the whole training dataset got copied into GPU for
    // precise prediction.  This condition tries to avoid such copy by calling CPU
    // Predictor instead.
    if ((out_pred && out_pred->Size() == 0) &&
        (model_.param.num_trees != 0) &&
        // FIXME(trivialfis): Implement a better method for testing whether data is on
        // device after DMatrix refactoring is done.
        !on_device) {
      CHECK(cpu_predictor_);
      return cpu_predictor_;
    }

    if (tparam_.tree_method == TreeMethod::kGPUHist) {
#if defined(XGBOOST_USE_CUDA)
      CHECK(gpu_predictor_);
      return gpu_predictor_;
#else
      this->AssertGPUSupport();
      return cpu_predictor_;
#endif  // defined(XGBOOST_USE_CUDA)
    }

    CHECK(cpu_predictor_);
    return cpu_predictor_;
  }

  // commit new trees all at once
  virtual void CommitModel(std::vector<std::vector<std::unique_ptr<RegTree>>>&& new_trees);

  // --- data structure ---
  GBTreeModel model_;
  // training parameter
  GBTreeTrainParam tparam_;
  // ----training fields----
  bool showed_updater_warning_ {false};
  bool specified_updater_   {false};
  bool configured_ {false};
  // configurations for tree
  Args cfg_;
  // the updaters that can be applied to each of tree
  std::vector<std::unique_ptr<TreeUpdater>> updaters_;
  /**
   * \brief Map of matrices and associated cached predictions to facilitate
   * storing and looking up predictions.
   */
  std::shared_ptr<std::unordered_map<DMatrix*, PredictionCacheEntry>> cache_;
  // Predictors
  std::unique_ptr<Predictor> cpu_predictor_;
#if defined(XGBOOST_USE_CUDA)
  std::unique_ptr<Predictor> gpu_predictor_;
#endif  // defined(XGBOOST_USE_CUDA)
  common::Monitor monitor_;
};

}  // namespace gbm
}  // namespace xgboost

#endif  // XGBOOST_GBM_GBTREE_H_
