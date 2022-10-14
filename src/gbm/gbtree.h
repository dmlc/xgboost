/*!
 * Copyright 2014-2022 by Contributors
 * \file gbtree.cc
 * \brief gradient boosted tree implementation.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_GBM_GBTREE_H_
#define XGBOOST_GBM_GBTREE_H_

#include <dmlc/omp.h>

#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <utility>
#include <string>
#include <unordered_map>

#include "xgboost/base.h"
#include "xgboost/data.h"
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
  kGPUPredictor,
  kOneAPIPredictor
};
}  // namespace xgboost

DECLARE_FIELD_ENUM_CLASS(xgboost::TreeMethod);
DECLARE_FIELD_ENUM_CLASS(xgboost::TreeProcessType);
DECLARE_FIELD_ENUM_CLASS(xgboost::PredictorType);

namespace xgboost {
namespace gbm {

/*! \brief training parameters */
struct GBTreeTrainParam : public XGBoostParameter<GBTreeTrainParam> {
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
        .add_enum("oneapi_predictor", PredictorType::kOneAPIPredictor)
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

namespace detail {
// From here on, layer becomes concrete trees.
inline std::pair<uint32_t, uint32_t> LayerToTree(gbm::GBTreeModel const &model,
                                                 size_t layer_begin,
                                                 size_t layer_end) {
  bst_group_t groups = model.learner_model_param->num_output_group;
  uint32_t tree_begin = layer_begin * groups * model.param.num_parallel_tree;
  uint32_t tree_end = layer_end * groups * model.param.num_parallel_tree;
  if (tree_end == 0) {
    tree_end = static_cast<uint32_t>(model.trees.size());
  }
  if (model.trees.size() != 0) {
    CHECK_LE(tree_begin, tree_end);
  }
  return {tree_begin, tree_end};
}

// Call fn for each pair of input output tree.  Return true if index is out of bound.
template <typename Func>
bool SliceTrees(int32_t layer_begin, int32_t layer_end, int32_t step, GBTreeModel const& model,
                uint32_t layer_trees, Func fn) {
  uint32_t tree_begin, tree_end;
  std::tie(tree_begin, tree_end) = detail::LayerToTree(model, layer_begin, layer_end);
  if (tree_end > model.trees.size()) {
    return true;
  }

  layer_end = layer_end == 0 ? model.trees.size() / layer_trees : layer_end;
  uint32_t n_layers = (layer_end - layer_begin) / step;
  int32_t in_it = tree_begin;
  int32_t out_it = 0;
  for (uint32_t l = 0; l < n_layers; ++l) {
    for (uint32_t i = 0; i < layer_trees; ++i) {
      CHECK_LT(in_it, tree_end);
      fn(in_it, out_it);
      out_it++;
      in_it++;
    }
    in_it += (step - 1) * layer_trees;
  }
  return false;
}
}  // namespace detail

// gradient boosted trees
class GBTree : public GradientBooster {
 public:
  explicit GBTree(LearnerModelParam const* booster_config, GenericParameter const* ctx)
      : GradientBooster{ctx}, model_(booster_config, ctx_) {}

  void Configure(const Args& cfg) override;
  // Revise `tree_method` and `updater` parameters after seeing the training
  // data matrix, only useful when tree_method is auto.
  void PerformTreeMethodHeuristic(DMatrix* fmat);
  /*! \brief Map `tree_method` parameter to `updater` parameter */
  void ConfigureUpdaters();
  void ConfigureWithKnownData(Args const& cfg, DMatrix* fmat);

  /**
   * \brief Optionally update the leaf value.
   */
  void UpdateTreeLeaf(DMatrix const* p_fmat, HostDeviceVector<float> const& predictions,
                      ObjFunction const* obj, std::vector<std::unique_ptr<RegTree>>* p_trees);

  /*! \brief Carry out one iteration of boosting */
  void DoBoost(DMatrix* p_fmat, HostDeviceVector<GradientPair>* in_gpair,
               PredictionCacheEntry* predt, ObjFunction const* obj) override;

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

  // Number of trees per layer.
  auto LayerTrees() const {
    auto n_trees = model_.learner_model_param->num_output_group * model_.param.num_parallel_tree;
    return n_trees;
  }

  // slice the trees, out must be already allocated
  void Slice(int32_t layer_begin, int32_t layer_end, int32_t step,
             GradientBooster *out, bool* out_of_bound) const override;

  int32_t BoostedRounds() const override {
    CHECK_NE(model_.param.num_parallel_tree, 0);
    CHECK_NE(model_.learner_model_param->num_output_group, 0);
    return model_.trees.size() / this->LayerTrees();
  }

  bool ModelFitted() const override {
    return !model_.trees.empty() || !model_.trees_to_update.empty();
  }

  void PredictBatch(DMatrix *p_fmat, PredictionCacheEntry *out_preds,
                    bool training, unsigned layer_begin, unsigned layer_end) override;

  void InplacePredict(std::shared_ptr<DMatrix> p_m, float missing, PredictionCacheEntry* out_preds,
                      uint32_t layer_begin, unsigned layer_end) const override {
    CHECK(configured_);
    uint32_t tree_begin, tree_end;
    std::tie(tree_begin, tree_end) = detail::LayerToTree(model_, layer_begin, layer_end);
    CHECK_LE(tree_end, model_.trees.size()) << "Invalid number of trees.";
    std::vector<Predictor const *> predictors{
      cpu_predictor_.get(),
#if defined(XGBOOST_USE_CUDA)
      gpu_predictor_.get()
#endif  // defined(XGBOOST_USE_CUDA)
    };
    StringView msg{"Unsupported data type for inplace predict."};
    if (tparam_.predictor == PredictorType::kAuto) {
      // Try both predictor implementations
      for (auto const &p : predictors) {
        if (p && p->InplacePredict(p_m, model_, missing, out_preds, tree_begin, tree_end)) {
          return;
        }
      }
      LOG(FATAL) << msg;
    } else {
      bool success = this->GetPredictor()->InplacePredict(p_m, model_, missing, out_preds,
                                                          tree_begin, tree_end);
      CHECK(success) << msg << std::endl
                     << "Current Predictor: "
                     << (tparam_.predictor == PredictorType::kCPUPredictor
                             ? "cpu_predictor"
                             : "gpu_predictor");
    }
  }

  void FeatureScore(std::string const& importance_type, common::Span<int32_t const> trees,
                    std::vector<bst_feature_t>* features,
                    std::vector<float>* scores) const override {
    // Because feature with no importance doesn't appear in the return value so
    // we need to set up another pair of vectors to store the values during
    // computation.
    std::vector<size_t> split_counts(this->model_.learner_model_param->num_feature, 0);
    std::vector<float> gain_map(this->model_.learner_model_param->num_feature, 0);
    std::vector<int32_t> tree_idx;
    if (trees.empty()) {
      tree_idx.resize(this->model_.trees.size());
      std::iota(tree_idx.begin(), tree_idx.end(), 0);
      trees = common::Span<int32_t const>(tree_idx);
    }

    auto total_n_trees = model_.trees.size();
    auto add_score = [&](auto fn) {
      for (auto idx : trees) {
        CHECK_LE(idx, total_n_trees) << "Invalid tree index.";
        auto const& p_tree = model_.trees[idx];
        p_tree->WalkTree([&](bst_node_t nidx) {
          auto const& node = (*p_tree)[nidx];
          if (!node.IsLeaf()) {
            split_counts[node.SplitIndex()]++;
            fn(p_tree, nidx, node.SplitIndex());
          }
          return true;
        });
      }
    };

    if (importance_type == "weight") {
      add_score([&](auto const&, bst_node_t, bst_feature_t split) {
        gain_map[split] = split_counts[split];
      });
    } else if (importance_type == "gain" || importance_type == "total_gain") {
      add_score([&](auto const &p_tree, bst_node_t nidx, bst_feature_t split) {
        gain_map[split] += p_tree->Stat(nidx).loss_chg;
      });
    } else if (importance_type == "cover" || importance_type == "total_cover") {
      add_score([&](auto const &p_tree, bst_node_t nidx, bst_feature_t split) {
        gain_map[split] += p_tree->Stat(nidx).sum_hess;
      });
    } else {
      LOG(FATAL)
          << "Unknown feature importance type, expected one of: "
          << R"({"weight", "total_gain", "total_cover", "gain", "cover"}, got: )"
          << importance_type;
    }
    if (importance_type == "gain" || importance_type == "cover") {
      for (size_t i = 0; i < gain_map.size(); ++i) {
        gain_map[i] /= std::max(1.0f, static_cast<float>(split_counts[i]));
      }
    }

    features->clear();
    scores->clear();
    for (size_t i = 0; i < split_counts.size(); ++i) {
      if (split_counts[i] != 0) {
        features->push_back(i);
        scores->push_back(gain_map[i]);
      }
    }
  }

  void PredictInstance(const SparsePage::Inst& inst,
                       std::vector<bst_float>* out_preds,
                       uint32_t layer_begin, uint32_t layer_end) override {
    CHECK(configured_);
    uint32_t tree_begin, tree_end;
    std::tie(tree_begin, tree_end) = detail::LayerToTree(model_, layer_begin, layer_end);
    cpu_predictor_->PredictInstance(inst, out_preds, model_, tree_end);
  }

  void PredictLeaf(DMatrix* p_fmat,
                   HostDeviceVector<bst_float>* out_preds,
                   uint32_t layer_begin, uint32_t layer_end) override {
    uint32_t tree_begin, tree_end;
    std::tie(tree_begin, tree_end) = detail::LayerToTree(model_, layer_begin, layer_end);
    CHECK_EQ(tree_begin, 0) << "Predict leaf supports only iteration end: (0, "
                               "n_iteration), use model slicing instead.";
    this->GetPredictor()->PredictLeaf(p_fmat, out_preds, model_, tree_end);
  }

  void PredictContribution(DMatrix* p_fmat,
                           HostDeviceVector<bst_float>* out_contribs,
                           uint32_t layer_begin, uint32_t layer_end, bool approximate,
                           int, unsigned) override {
    CHECK(configured_);
    uint32_t tree_begin, tree_end;
    std::tie(tree_begin, tree_end) = detail::LayerToTree(model_, layer_begin, layer_end);
    CHECK_EQ(tree_begin, 0)
        << "Predict contribution supports only iteration end: (0, "
           "n_iteration), using model slicing instead.";
    this->GetPredictor()->PredictContribution(
        p_fmat, out_contribs, model_, tree_end, nullptr, approximate);
  }

  void PredictInteractionContributions(
      DMatrix *p_fmat, HostDeviceVector<bst_float> *out_contribs,
      uint32_t layer_begin, uint32_t layer_end, bool approximate) override {
    CHECK(configured_);
    uint32_t tree_begin, tree_end;
    std::tie(tree_begin, tree_end) = detail::LayerToTree(model_, layer_begin, layer_end);
    CHECK_EQ(tree_begin, 0)
        << "Predict interaction contribution supports only iteration end: (0, "
           "n_iteration), using model slicing instead.";
    this->GetPredictor()->PredictInteractionContributions(
        p_fmat, out_contribs, model_, tree_end, nullptr, approximate);
  }

  std::vector<std::string> DumpModel(const FeatureMap& fmap, bool with_stats,
                                     std::string format) const override {
    return model_.DumpModel(fmap, with_stats, this->ctx_->Threads(), format);
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
                                                 DMatrix* f_dmat = nullptr) const;

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
  // The node position for each row, 1 HDV for each tree in the forest.  Note that the
  // position is negated if the row is sampled out.
  std::vector<HostDeviceVector<bst_node_t>> node_position_;
  // Predictors
  std::unique_ptr<Predictor> cpu_predictor_;
#if defined(XGBOOST_USE_CUDA)
  std::unique_ptr<Predictor> gpu_predictor_;
#endif  // defined(XGBOOST_USE_CUDA)
#if defined(XGBOOST_USE_ONEAPI)
  std::unique_ptr<Predictor> oneapi_predictor_;
#endif  // defined(XGBOOST_USE_ONEAPI)
  common::Monitor monitor_;
};

}  // namespace gbm
}  // namespace xgboost

#endif  // XGBOOST_GBM_GBTREE_H_
