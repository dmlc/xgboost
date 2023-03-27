/**
 * Copyright 2014-2023 by Contributors
 * \file gbtree.cc
 * \brief gradient boosted tree implementation.
 * \author Tianqi Chen
 */
#include "gbtree.h"

#include <dmlc/omp.h>
#include <dmlc/parameter.h>

#include <algorithm>
#include <cinttypes>  // for uint32_t
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../common/common.h"
#include "../common/random.h"
#include "../common/threading_utils.h"
#include "../common/timer.h"
#include "gbtree_model.h"
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/gbm.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/logging.h"
#include "xgboost/model.h"
#include "xgboost/objective.h"
#include "xgboost/predictor.h"
#include "xgboost/string_view.h"  // for StringView
#include "xgboost/tree_model.h"   // for RegTree
#include "xgboost/tree_updater.h"

namespace xgboost::gbm {
DMLC_REGISTRY_FILE_TAG(gbtree);

void GBTree::Configure(Args const& cfg) {
  this->cfg_ = cfg;
  std::string updater_seq = tparam_.updater_seq;
  tparam_.UpdateAllowUnknown(cfg);
  tree_param_.UpdateAllowUnknown(cfg);

  model_.Configure(cfg);

  // for the 'update' process_type, move trees into trees_to_update
  if (tparam_.process_type == TreeProcessType::kUpdate) {
    model_.InitTreesToUpdate();
  }

  // configure predictors
  if (!cpu_predictor_) {
    cpu_predictor_ = std::unique_ptr<Predictor>(
        Predictor::Create("cpu_predictor", this->ctx_));
  }
  cpu_predictor_->Configure(cfg);
#if defined(XGBOOST_USE_CUDA)
  auto n_gpus = common::AllVisibleGPUs();
  if (!gpu_predictor_ && n_gpus != 0) {
    gpu_predictor_ = std::unique_ptr<Predictor>(
        Predictor::Create("gpu_predictor", this->ctx_));
  }
  if (n_gpus != 0) {
    gpu_predictor_->Configure(cfg);
  }
#endif  // defined(XGBOOST_USE_CUDA)

#if defined(XGBOOST_USE_ONEAPI)
  if (!oneapi_predictor_) {
    oneapi_predictor_ = std::unique_ptr<Predictor>(
        Predictor::Create("oneapi_predictor", this->ctx_));
  }
  oneapi_predictor_->Configure(cfg);
#endif  // defined(XGBOOST_USE_ONEAPI)

  monitor_.Init("GBTree");

  specified_updater_ = std::any_of(cfg.cbegin(), cfg.cend(),
                   [](std::pair<std::string, std::string> const& arg) {
                     return arg.first == "updater";
                   });

  if (specified_updater_ && !showed_updater_warning_) {
    LOG(WARNING) << "DANGER AHEAD: You have manually specified `updater` "
        "parameter. The `tree_method` parameter will be ignored. "
        "Incorrect sequence of updaters will produce undefined "
        "behavior. For common uses, we recommend using "
        "`tree_method` parameter instead.";
    // Don't drive users to silent XGBOost.
    showed_updater_warning_ = true;
  }

  this->ConfigureUpdaters();
  if (updater_seq != tparam_.updater_seq) {
    updaters_.clear();
    this->InitUpdater(cfg);
  } else {
    for (auto &up : updaters_) {
      up->Configure(cfg);
    }
  }

  configured_ = true;
}

// FIXME(trivialfis): This handles updaters.  Because the choice of updaters depends on
// whether external memory is used and how large is dataset.  We can remove the dependency
// on DMatrix once `hist` tree method can handle external memory so that we can make it
// default.
void GBTree::ConfigureWithKnownData(Args const& cfg, DMatrix* fmat) {
  CHECK(this->configured_);
  std::string updater_seq = tparam_.updater_seq;
  CHECK(tparam_.GetInitialised());

  tparam_.UpdateAllowUnknown(cfg);

  this->PerformTreeMethodHeuristic(fmat);
  this->ConfigureUpdaters();

  // initialize the updaters only when needed.
  if (updater_seq != tparam_.updater_seq) {
    LOG(DEBUG) << "Using updaters: " << tparam_.updater_seq;
    this->updaters_.clear();
    this->InitUpdater(cfg);
  }
}

void GBTree::PerformTreeMethodHeuristic(DMatrix* fmat) {
  if (specified_updater_) {
    // This method is disabled when `updater` parameter is explicitly
    // set, since only experts are expected to do so.
    return;
  }
  if (model_.learner_model_param->IsVectorLeaf()) {
    CHECK(tparam_.tree_method == TreeMethod::kHist)
        << "Only the hist tree method is supported for building multi-target trees with vector "
           "leaf.";
  }

  // tparam_ is set before calling this function.
  if (tparam_.tree_method != TreeMethod::kAuto) {
    return;
  }

  if (collective::IsDistributed()) {
    LOG(INFO) << "Tree method is automatically selected to be 'approx' "
                 "for distributed training.";
    tparam_.tree_method = TreeMethod::kApprox;
  } else if (!fmat->SingleColBlock()) {
    LOG(INFO) << "Tree method is automatically set to 'approx' "
                 "since external-memory data matrix is used.";
    tparam_.tree_method = TreeMethod::kApprox;
  } else if (fmat->Info().num_row_ >= (4UL << 20UL)) {
    /* Choose tree_method='approx' automatically for large data matrix */
    LOG(INFO) << "Tree method is automatically selected to be "
                 "'approx' for faster speed. To use old behavior "
                 "(exact greedy algorithm on single machine), "
                 "set tree_method to 'exact'.";
    tparam_.tree_method = TreeMethod::kApprox;
  } else {
    tparam_.tree_method = TreeMethod::kExact;
  }
  LOG(DEBUG) << "Using tree method: " << static_cast<int>(tparam_.tree_method);
}

void GBTree::ConfigureUpdaters() {
  if (specified_updater_) {
    return;
  }
  // `updater` parameter was manually specified
  /* Choose updaters according to tree_method parameters */
  switch (tparam_.tree_method) {
    case TreeMethod::kAuto:
      // Use heuristic to choose between 'exact' and 'approx' This
      // choice is carried out in PerformTreeMethodHeuristic() before
      // calling this function.
      break;
    case TreeMethod::kApprox:
      tparam_.updater_seq = "grow_histmaker";
      break;
    case TreeMethod::kExact:
      tparam_.updater_seq = "grow_colmaker,prune";
      break;
    case TreeMethod::kHist: {
      LOG(INFO) << "Tree method is selected to be 'hist', which uses a single updater "
                   "grow_quantile_histmaker.";
      tparam_.updater_seq = "grow_quantile_histmaker";
      break;
    }
    case TreeMethod::kGPUHist: {
      common::AssertGPUSupport();
      tparam_.updater_seq = "grow_gpu_hist";
      break;
    }
    default:
      LOG(FATAL) << "Unknown tree_method ("
                 << static_cast<int>(tparam_.tree_method) << ") detected";
  }
}

void GPUCopyGradient(HostDeviceVector<GradientPair> const*, bst_group_t, bst_group_t,
                     HostDeviceVector<GradientPair>*)
#if defined(XGBOOST_USE_CUDA)
    ;  // NOLINT
#else
{
  common::AssertGPUSupport();
}
#endif

void CopyGradient(HostDeviceVector<GradientPair> const* in_gpair, int32_t n_threads,
                  bst_group_t n_groups, bst_group_t group_id,
                  HostDeviceVector<GradientPair>* out_gpair) {
  if (in_gpair->DeviceIdx() != Context::kCpuId) {
    GPUCopyGradient(in_gpair, n_groups, group_id, out_gpair);
  } else {
    std::vector<GradientPair> &tmp_h = out_gpair->HostVector();
    const auto& gpair_h = in_gpair->ConstHostVector();
    common::ParallelFor(out_gpair->Size(), n_threads,
                        [&](auto i) { tmp_h[i] = gpair_h[i * n_groups + group_id]; });
  }
}

void GBTree::UpdateTreeLeaf(DMatrix const* p_fmat, HostDeviceVector<float> const& predictions,
                            ObjFunction const* obj, std::int32_t group_idx,
                            std::vector<HostDeviceVector<bst_node_t>> const& node_position,
                            TreesOneGroup* p_trees) {
  CHECK(!updaters_.empty());
  if (!updaters_.back()->HasNodePosition()) {
    return;
  }
  if (!obj || !obj->Task().UpdateTreeLeaf()) {
    return;
  }

  auto& trees = *p_trees;
  CHECK_EQ(model_.param.num_parallel_tree, trees.size());
  CHECK_EQ(model_.param.num_parallel_tree, 1)
      << "Boosting random forest is not supported for current objective.";
  CHECK(!trees.front()->IsMultiTarget()) << "Update tree leaf" << MTNotImplemented();
  CHECK_EQ(trees.size(), model_.param.num_parallel_tree);
  for (std::size_t tree_idx = 0; tree_idx < trees.size(); ++tree_idx) {
    auto const& position = node_position.at(tree_idx);
    obj->UpdateTreeLeaf(position, p_fmat->Info(), tree_param_.learning_rate / trees.size(),
                        predictions, group_idx, trees[tree_idx].get());
  }
}

void GBTree::DoBoost(DMatrix* p_fmat, HostDeviceVector<GradientPair>* in_gpair,
                     PredictionCacheEntry* predt, ObjFunction const* obj) {
  TreesOneIter new_trees;
  bst_target_t const n_groups = model_.learner_model_param->OutputLength();
  ConfigureWithKnownData(this->cfg_, p_fmat);
  monitor_.Start("BoostNewTrees");

  // Weird case that tree method is cpu-based but gpu_id is set.  Ideally we should let
  // `gpu_id` be the single source of determining what algorithms to run, but that will
  // break a lots of existing code.
  auto device = tparam_.tree_method != TreeMethod::kGPUHist ? Context::kCpuId : ctx_->gpu_id;
  auto out = linalg::MakeTensorView(
      device,
      device == Context::kCpuId ? predt->predictions.HostSpan() : predt->predictions.DeviceSpan(),
      p_fmat->Info().num_row_, model_.learner_model_param->OutputLength());
  CHECK_NE(n_groups, 0);

  if (!p_fmat->SingleColBlock() && obj->Task().UpdateTreeLeaf()) {
    LOG(FATAL) << "Current objective doesn't support external memory.";
  }

  // The node position for each row, 1 HDV for each tree in the forest.  Note that the
  // position is negated if the row is sampled out.
  std::vector<HostDeviceVector<bst_node_t>> node_position;

  if (model_.learner_model_param->IsVectorLeaf()) {
    TreesOneGroup ret;
    BoostNewTrees(in_gpair, p_fmat, 0, &node_position, &ret);
    UpdateTreeLeaf(p_fmat, predt->predictions, obj, 0, node_position, &ret);
    std::size_t num_new_trees = ret.size();
    new_trees.push_back(std::move(ret));
    if (updaters_.size() > 0 && num_new_trees == 1 && predt->predictions.Size() > 0 &&
        updaters_.back()->UpdatePredictionCache(p_fmat, out)) {
      predt->Update(1);
    }
  } else if (model_.learner_model_param->OutputLength() == 1u) {
    TreesOneGroup ret;
    BoostNewTrees(in_gpair, p_fmat, 0, &node_position, &ret);
    UpdateTreeLeaf(p_fmat, predt->predictions, obj, 0, node_position, &ret);
    const size_t num_new_trees = ret.size();
    new_trees.push_back(std::move(ret));
    if (updaters_.size() > 0 && num_new_trees == 1 && predt->predictions.Size() > 0 &&
        updaters_.back()->UpdatePredictionCache(p_fmat, out)) {
      predt->Update(1);
    }
  } else {
    CHECK_EQ(in_gpair->Size() % n_groups, 0U) << "must have exactly ngroup * nrow gpairs";
    HostDeviceVector<GradientPair> tmp(in_gpair->Size() / n_groups, GradientPair(),
                                       in_gpair->DeviceIdx());
    bool update_predict = true;
    for (bst_target_t gid = 0; gid < n_groups; ++gid) {
      node_position.clear();
      CopyGradient(in_gpair, ctx_->Threads(), n_groups, gid, &tmp);
      TreesOneGroup ret;
      BoostNewTrees(&tmp, p_fmat, gid, &node_position, &ret);
      UpdateTreeLeaf(p_fmat, predt->predictions, obj, gid, node_position, &ret);
      const size_t num_new_trees = ret.size();
      new_trees.push_back(std::move(ret));
      auto v_predt = out.Slice(linalg::All(), linalg::Range(gid, gid + 1));
      if (!(updaters_.size() > 0 && predt->predictions.Size() > 0 && num_new_trees == 1 &&
            updaters_.back()->UpdatePredictionCache(p_fmat, v_predt))) {
        update_predict = false;
      }
    }
    if (update_predict) {
      predt->Update(1);
    }
  }

  monitor_.Stop("BoostNewTrees");
  this->CommitModel(std::move(new_trees));
}

void GBTree::InitUpdater(Args const& cfg) {
  std::string tval = tparam_.updater_seq;
  std::vector<std::string> ups = common::Split(tval, ',');

  if (updaters_.size() != 0) {
    // Assert we have a valid set of updaters.
    CHECK_EQ(ups.size(), updaters_.size());
    for (auto const& up : updaters_) {
      bool contains = std::any_of(ups.cbegin(), ups.cend(),
                        [&up](std::string const& name) {
                          return name == up->Name();
                        });
      if (!contains) {
        std::stringstream ss;
        ss << "Internal Error: " << " mismatched updater sequence.\n";
        ss << "Specified updaters: ";
        std::for_each(ups.cbegin(), ups.cend(),
                      [&ss](std::string const& name){
                        ss << name << " ";
                      });
        ss << "\n" << "Actual updaters: ";
        std::for_each(updaters_.cbegin(), updaters_.cend(),
                      [&ss](std::unique_ptr<TreeUpdater> const& updater){
                        ss << updater->Name() << " ";
                      });
        LOG(FATAL) << ss.str();
      }
    }
    // Do not push new updater in.
    return;
  }

  // create new updaters
  for (const std::string& pstr : ups) {
    std::unique_ptr<TreeUpdater> up(
        TreeUpdater::Create(pstr.c_str(), ctx_, &model_.learner_model_param->task));
    up->Configure(cfg);
    updaters_.push_back(std::move(up));
  }
}

void GBTree::BoostNewTrees(HostDeviceVector<GradientPair>* gpair, DMatrix* p_fmat, int bst_group,
                           std::vector<HostDeviceVector<bst_node_t>>* out_position,
                           TreesOneGroup* ret) {
  std::vector<RegTree*> new_trees;
  ret->clear();
  // create the trees
  for (int i = 0; i < model_.param.num_parallel_tree; ++i) {
    if (tparam_.process_type == TreeProcessType::kDefault) {
      CHECK(!updaters_.front()->CanModifyTree())
          << "Updater: `" << updaters_.front()->Name() << "` "
          << "can not be used to create new trees. "
          << "Set `process_type` to `update` if you want to update existing "
             "trees.";
      // create new tree
      std::unique_ptr<RegTree> ptr(new RegTree{this->model_.learner_model_param->LeafLength(),
                                               this->model_.learner_model_param->num_feature});
      new_trees.push_back(ptr.get());
      ret->push_back(std::move(ptr));
    } else if (tparam_.process_type == TreeProcessType::kUpdate) {
      for (auto const& up : updaters_) {
        CHECK(up->CanModifyTree())
            << "Updater: `" << up->Name() << "` "
            << "can not be used to modify existing trees. "
            << "Set `process_type` to `default` if you want to build new trees.";
      }
      CHECK_LT(model_.trees.size(), model_.trees_to_update.size())
          << "No more tree left for updating.  For updating existing trees, "
          << "boosting rounds can not exceed previous training rounds";
      // move an existing tree from trees_to_update
      auto t = std::move(model_.trees_to_update[model_.trees.size() +
                                                bst_group * model_.param.num_parallel_tree + i]);
      new_trees.push_back(t.get());
      ret->push_back(std::move(t));
    }
  }

  // update the trees
  auto n_out = model_.learner_model_param->OutputLength() * p_fmat->Info().num_row_;
  StringView msg{
      "Mismatching size between number of rows from input data and size of gradient vector."};
  if (!model_.learner_model_param->IsVectorLeaf() && p_fmat->Info().num_row_ != 0) {
    CHECK_EQ(n_out % gpair->Size(), 0) << msg;
  } else {
    CHECK_EQ(gpair->Size(), n_out) << msg;
  }

  out_position->resize(new_trees.size());

  // Rescale learning rate according to the size of trees
  auto lr = tree_param_.learning_rate;
  tree_param_.learning_rate /= static_cast<float>(new_trees.size());
  for (auto& up : updaters_) {
    up->Update(&tree_param_, gpair, p_fmat,
               common::Span<HostDeviceVector<bst_node_t>>{*out_position}, new_trees);
  }
  tree_param_.learning_rate = lr;
}

void GBTree::CommitModel(TreesOneIter&& new_trees) {
  monitor_.Start("CommitModel");
  model_.CommitModel(std::forward<TreesOneIter>(new_trees));
  monitor_.Stop("CommitModel");
}

void GBTree::LoadConfig(Json const& in) {
  CHECK_EQ(get<String>(in["name"]), "gbtree");
  FromJson(in["gbtree_train_param"], &tparam_);
  FromJson(in["tree_train_param"], &tree_param_);

  // Process type cannot be kUpdate from loaded model
  // This would cause all trees to be pushed to trees_to_update
  // e.g. updating a model, then saving and loading it would result in an empty model
  tparam_.process_type = TreeProcessType::kDefault;
  int32_t const n_gpus = xgboost::common::AllVisibleGPUs();
  if (n_gpus == 0 && tparam_.predictor == PredictorType::kGPUPredictor) {
    LOG(WARNING) << "Loading from a raw memory buffer on CPU only machine.  "
                    "Changing predictor to auto.";
    tparam_.UpdateAllowUnknown(Args{{"predictor", "auto"}});
  }

  auto msg = StringView{
      R"(
  Loading from a raw memory buffer (like pickle in Python, RDS in R) on a CPU-only
  machine. Consider using `save_model/load_model` instead. See:

    https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html

  for more details about differences between saving model and serializing.)"};

  if (n_gpus == 0 && tparam_.tree_method == TreeMethod::kGPUHist) {
    tparam_.UpdateAllowUnknown(Args{{"tree_method", "hist"}});
    LOG(WARNING) << msg << "  Changing `tree_method` to `hist`.";
  }

  auto const& j_updaters = get<Object const>(in["updater"]);
  updaters_.clear();

  for (auto const& kv : j_updaters) {
    auto name = kv.first;
    if (n_gpus == 0 && name == "grow_gpu_hist") {
      name = "grow_quantile_histmaker";
      LOG(WARNING) << "Changing updater from `grow_gpu_hist` to `grow_quantile_histmaker`.";
    }
    std::unique_ptr<TreeUpdater> up{
        TreeUpdater::Create(name, ctx_, &model_.learner_model_param->task)};
    up->LoadConfig(kv.second);
    updaters_.push_back(std::move(up));
  }

  specified_updater_ = get<Boolean>(in["specified_updater"]);
}

void GBTree::SaveConfig(Json* p_out) const {
  auto& out = *p_out;
  out["name"] = String("gbtree");
  out["gbtree_train_param"] = ToJson(tparam_);
  out["tree_train_param"] = ToJson(tree_param_);

  // Process type cannot be kUpdate from loaded model
  // This would cause all trees to be pushed to trees_to_update
  // e.g. updating a model, then saving and loading it would result in an empty
  // model
  out["gbtree_train_param"]["process_type"] = String("default");
  // Duplicated from SaveModel so that user can get `num_parallel_tree` without parsing
  // the model. We might remove this once we can deprecate `best_ntree_limit` so that the
  // language binding doesn't need to know about the forest size.
  out["gbtree_model_param"] = ToJson(model_.param);

  out["updater"] = Object();

  auto& j_updaters = out["updater"];
  for (auto const& up : updaters_) {
    j_updaters[up->Name()] = Object();
    auto& j_up = j_updaters[up->Name()];
    up->SaveConfig(&j_up);
  }
  out["specified_updater"] = Boolean{specified_updater_};
}

void GBTree::LoadModel(Json const& in) {
  CHECK_EQ(get<String>(in["name"]), "gbtree");
  model_.LoadModel(in["model"]);
}

void GBTree::SaveModel(Json* p_out) const {
  auto& out = *p_out;
  out["name"] = String("gbtree");
  out["model"] = Object();
  auto& model = out["model"];
  model_.SaveModel(&model);
}

void GBTree::Slice(bst_layer_t begin, bst_layer_t end, bst_layer_t step, GradientBooster* out,
                   bool* out_of_bound) const {
  CHECK(configured_);
  CHECK(out);

  auto p_gbtree = dynamic_cast<GBTree*>(out);
  CHECK(p_gbtree);
  GBTreeModel& out_model = p_gbtree->model_;
  CHECK(this->model_.learner_model_param->Initialized());

  end = end == 0 ? model_.BoostedRounds() : end;
  CHECK_GE(step, 1);
  CHECK_NE(end, begin) << "Empty slice is not allowed.";

  if (step > (end - begin)) {
    *out_of_bound = true;
    return;
  }

  auto& out_indptr = out_model.iteration_indptr;
  TreesOneGroup& out_trees = out_model.trees;
  std::vector<int32_t>& out_trees_info = out_model.tree_info;

  bst_layer_t n_layers = (end - begin) / step;
  out_indptr.resize(n_layers + 1, 0);

  if (!this->model_.trees_to_update.empty()) {
    CHECK_EQ(this->model_.trees_to_update.size(), this->model_.trees.size())
        << "Not all trees are updated, "
        << this->model_.trees_to_update.size() - this->model_.trees.size()
        << " trees remain.  Slice the model before making update if you only "
           "want to update a portion of trees.";
  }

  *out_of_bound =
      detail::SliceTrees(begin, end, step, this->model_, [&](auto in_tree_idx, auto out_l) {
        auto new_tree = std::make_unique<RegTree>(*this->model_.trees.at(in_tree_idx));
        out_trees.emplace_back(std::move(new_tree));

        bst_group_t group = this->model_.tree_info[in_tree_idx];
        out_trees_info.push_back(group);

        out_model.iteration_indptr[out_l + 1]++;
      });

  std::partial_sum(out_indptr.cbegin(), out_indptr.cend(), out_indptr.begin());
  CHECK_EQ(out_model.iteration_indptr.front(), 0);

  out_model.param.num_trees = out_model.trees.size();
  out_model.param.num_parallel_tree = model_.param.num_parallel_tree;
}

void GBTree::PredictBatch(DMatrix* p_fmat, PredictionCacheEntry* out_preds, bool,
                          bst_layer_t layer_begin, bst_layer_t layer_end) {
  CHECK(configured_);
  if (layer_end == 0) {
    layer_end = this->BoostedRounds();
  }
  if (layer_begin != 0 || layer_end < static_cast<bst_layer_t>(out_preds->version)) {
    // cache is dropped.
    out_preds->version = 0;
  }
  bool reset = false;
  if (layer_begin == 0) {
    layer_begin = out_preds->version;
  } else {
    // When begin layer is not 0, the cache is not useful.
    reset = true;
  }
  if (out_preds->predictions.Size() == 0 && p_fmat->Info().num_row_ != 0) {
    CHECK_EQ(out_preds->version, 0);
  }

  auto const& predictor = GetPredictor(&out_preds->predictions, p_fmat);
  if (out_preds->version == 0) {
    // out_preds->Size() can be non-zero as it's initialized here before any
    // tree is built at the 0^th iterator.
    predictor->InitOutPredictions(p_fmat->Info(), &out_preds->predictions, model_);
  }

  auto [tree_begin, tree_end] = detail::LayerToTree(model_, layer_begin, layer_end);
  CHECK_LE(tree_end, model_.trees.size()) << "Invalid number of trees.";
  if (tree_end > tree_begin) {
    predictor->PredictBatch(p_fmat, out_preds, model_, tree_begin, tree_end);
  }
  if (reset) {
    out_preds->version = 0;
  } else {
    std::uint32_t delta = layer_end - out_preds->version;
    out_preds->Update(delta);
  }
}

std::unique_ptr<Predictor> const &
GBTree::GetPredictor(HostDeviceVector<float> const *out_pred,
                     DMatrix *f_dmat) const {
  CHECK(configured_);
  if (tparam_.predictor != PredictorType::kAuto) {
    if (tparam_.predictor == PredictorType::kGPUPredictor) {
#if defined(XGBOOST_USE_CUDA)
      CHECK_GE(common::AllVisibleGPUs(), 1) << "No visible GPU is found for XGBoost.";
      CHECK(gpu_predictor_);
      return gpu_predictor_;
#else
      common::AssertGPUSupport();
#endif  // defined(XGBOOST_USE_CUDA)
    }
    if (tparam_.predictor == PredictorType::kOneAPIPredictor) {
#if defined(XGBOOST_USE_ONEAPI)
      CHECK(oneapi_predictor_);
      return oneapi_predictor_;
#else
      common::AssertOneAPISupport();
#endif  // defined(XGBOOST_USE_ONEAPI)
    }
    CHECK(cpu_predictor_);
    return cpu_predictor_;
  }

  // Data comes from Device DMatrix.
  auto is_ellpack = f_dmat && f_dmat->PageExists<EllpackPage>() &&
                    !f_dmat->PageExists<SparsePage>();
  // Data comes from device memory, like CuDF or CuPy.
  auto is_from_device =
      f_dmat && f_dmat->PageExists<SparsePage>() &&
      (*(f_dmat->GetBatches<SparsePage>().begin())).data.DeviceCanRead();
  auto on_device = is_ellpack || is_from_device;

  // Use GPU Predictor if data is already on device and gpu_id is set.
  if (on_device && ctx_->gpu_id >= 0) {
#if defined(XGBOOST_USE_CUDA)
    CHECK_GE(common::AllVisibleGPUs(), 1) << "No visible GPU is found for XGBoost.";
    CHECK(gpu_predictor_);
    return gpu_predictor_;
#else
    LOG(FATAL) << "Data is on CUDA device, but XGBoost is not compiled with "
                  "CUDA support.";
    return cpu_predictor_;
#endif  // defined(XGBOOST_USE_CUDA)
  }

  // GPU_Hist by default has prediction cache calculated from quantile values,
  // so GPU Predictor is not used for training dataset.  But when XGBoost
  // performs continue training with an existing model, the prediction cache is
  // not available and number of trees doesn't equal zero, the whole training
  // dataset got copied into GPU for precise prediction.  This condition tries
  // to avoid such copy by calling CPU Predictor instead.
  if ((out_pred && out_pred->Size() == 0) && (model_.param.num_trees != 0) &&
      // FIXME(trivialfis): Implement a better method for testing whether data
      // is on device after DMatrix refactoring is done.
      !on_device) {
    CHECK(cpu_predictor_);
    return cpu_predictor_;
  }

  if (tparam_.tree_method == TreeMethod::kGPUHist) {
#if defined(XGBOOST_USE_CUDA)
    CHECK_GE(common::AllVisibleGPUs(), 1) << "No visible GPU is found for XGBoost.";
    CHECK(gpu_predictor_);
    return gpu_predictor_;
#else
    common::AssertGPUSupport();
    return cpu_predictor_;
#endif  // defined(XGBOOST_USE_CUDA)
  }

  CHECK(cpu_predictor_);
  return cpu_predictor_;
}

/** Increment the prediction on GPU.
 *
 * \param out_predts Prediction for the whole model.
 * \param predts     Prediction for current tree.
 * \param tree_w     Tree weight.
 */
void GPUDartPredictInc(common::Span<float>, common::Span<float>, float, size_t, bst_group_t,
                       bst_group_t)
#if defined(XGBOOST_USE_CUDA)
    ;  // NOLINT
#else
{
  common::AssertGPUSupport();
}
#endif

void GPUDartInplacePredictInc(common::Span<float> /*out_predts*/, common::Span<float> /*predts*/,
                              float /*tree_w*/, size_t /*n_rows*/,
                              linalg::TensorView<float const, 1> /*base_score*/,
                              bst_group_t /*n_groups*/, bst_group_t /*group*/)
#if defined(XGBOOST_USE_CUDA)
    ;  // NOLINT
#else
{
  common::AssertGPUSupport();
}
#endif


class Dart : public GBTree {
 public:
  explicit Dart(LearnerModelParam const* booster_config, Context const* ctx)
      : GBTree(booster_config, ctx) {}

  void Configure(const Args& cfg) override {
    GBTree::Configure(cfg);
    dparam_.UpdateAllowUnknown(cfg);
  }

  void Slice(int32_t layer_begin, int32_t layer_end, int32_t step,
             GradientBooster *out, bool* out_of_bound) const final {
    GBTree::Slice(layer_begin, layer_end, step, out, out_of_bound);
    if (*out_of_bound) {
      return;
    }
    auto p_dart = dynamic_cast<Dart*>(out);
    CHECK(p_dart);
    CHECK(p_dart->weight_drop_.empty());
    detail::SliceTrees(layer_begin, layer_end, step, model_, [&](auto const& in_it, auto const&) {
      p_dart->weight_drop_.push_back(this->weight_drop_.at(in_it));
    });
  }

  void SaveModel(Json *p_out) const override {
    auto &out = *p_out;
    out["name"] = String("dart");
    out["gbtree"] = Object();
    GBTree::SaveModel(&(out["gbtree"]));

    std::vector<Json> j_weight_drop(weight_drop_.size());
    for (size_t i = 0; i < weight_drop_.size(); ++i) {
      j_weight_drop[i] = Number(weight_drop_[i]);
    }
    out["weight_drop"] = Array(std::move(j_weight_drop));
  }
  void LoadModel(Json const& in) override {
    CHECK_EQ(get<String>(in["name"]), "dart");
    auto const& gbtree = in["gbtree"];
    GBTree::LoadModel(gbtree);

    auto const& j_weight_drop = get<Array>(in["weight_drop"]);
    weight_drop_.resize(j_weight_drop.size());
    for (size_t i = 0; i < weight_drop_.size(); ++i) {
      weight_drop_[i] = get<Number const>(j_weight_drop[i]);
    }
  }

  void Load(dmlc::Stream* fi) override {
    GBTree::Load(fi);
    weight_drop_.resize(model_.param.num_trees);
    if (model_.param.num_trees != 0) {
      fi->Read(&weight_drop_);
    }
  }
  void Save(dmlc::Stream* fo) const override {
    GBTree::Save(fo);
    if (weight_drop_.size() != 0) {
      fo->Write(weight_drop_);
    }
  }

  void LoadConfig(Json const& in) override {
    CHECK_EQ(get<String>(in["name"]), "dart");
    auto const& gbtree = in["gbtree"];
    GBTree::LoadConfig(gbtree);
    FromJson(in["dart_train_param"], &dparam_);
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("dart");
    out["gbtree"] = Object();
    auto& gbtree = out["gbtree"];
    GBTree::SaveConfig(&gbtree);
    out["dart_train_param"] = ToJson(dparam_);
  }

  // An independent const function to make sure it's thread safe.
  void PredictBatchImpl(DMatrix *p_fmat, PredictionCacheEntry *p_out_preds,
                        bool training, unsigned layer_begin,
                        unsigned layer_end) const {
    CHECK(!this->model_.learner_model_param->IsVectorLeaf()) << "dart" << MTNotImplemented();
    auto &predictor = this->GetPredictor(&p_out_preds->predictions, p_fmat);
    CHECK(predictor);
    predictor->InitOutPredictions(p_fmat->Info(), &p_out_preds->predictions,
                                  model_);
    p_out_preds->version = 0;
    auto [tree_begin, tree_end] = detail::LayerToTree(model_, layer_begin, layer_end);
    auto n_groups = model_.learner_model_param->num_output_group;

    PredictionCacheEntry predts;  // temporary storage for prediction
    if (ctx_->gpu_id != Context::kCpuId) {
      predts.predictions.SetDevice(ctx_->gpu_id);
    }
    predts.predictions.Resize(p_fmat->Info().num_row_ * n_groups, 0);
    // multi-target is not yet supported.
    auto layer_trees = [&]() {
      return model_.param.num_parallel_tree * model_.learner_model_param->OutputLength();
    };

    for (bst_tree_t i = tree_begin; i < tree_end; i += 1) {
      if (training && std::binary_search(idx_drop_.cbegin(), idx_drop_.cend(), i)) {
        continue;
      }

      CHECK_GE(i, p_out_preds->version);
      auto version = i / layer_trees();
      p_out_preds->version = version;
      predts.predictions.Fill(0);
      predictor->PredictBatch(p_fmat, &predts, model_, i, i + 1);

      // Multiple the weight to output prediction.
      auto w = this->weight_drop_.at(i);
      auto group = model_.tree_info.at(i);
      CHECK_EQ(p_out_preds->predictions.Size(), predts.predictions.Size());

      size_t n_rows = p_fmat->Info().num_row_;
      if (predts.predictions.DeviceIdx() != Context::kCpuId) {
        p_out_preds->predictions.SetDevice(predts.predictions.DeviceIdx());
        GPUDartPredictInc(p_out_preds->predictions.DeviceSpan(),
                          predts.predictions.DeviceSpan(), w, n_rows, n_groups,
                          group);
      } else {
        auto &h_out_predts = p_out_preds->predictions.HostVector();
        auto &h_predts = predts.predictions.HostVector();
        common::ParallelFor(p_fmat->Info().num_row_, ctx_->Threads(), [&](auto ridx) {
          const size_t offset = ridx * n_groups + group;
          h_out_predts[offset] += (h_predts[offset] * w);
        });
      }
    }
  }

  void PredictBatch(DMatrix* p_fmat, PredictionCacheEntry* p_out_preds, bool training,
                    bst_layer_t layer_begin, bst_layer_t layer_end) override {
    DropTrees(training);
    this->PredictBatchImpl(p_fmat, p_out_preds, training, layer_begin, layer_end);
  }

  void InplacePredict(std::shared_ptr<DMatrix> p_fmat, float missing,
                      PredictionCacheEntry* p_out_preds, bst_layer_t layer_begin,
                      bst_layer_t layer_end) const override {
    CHECK(!this->model_.learner_model_param->IsVectorLeaf()) << "dart" << MTNotImplemented();
    auto [tree_begin, tree_end] = detail::LayerToTree(model_, layer_begin, layer_end);
    auto n_groups = model_.learner_model_param->num_output_group;

    std::vector<Predictor const*> predictors {
      cpu_predictor_.get(),
#if defined(XGBOOST_USE_CUDA)
      gpu_predictor_.get()
#endif  // defined(XGBOOST_USE_CUDA)
    };
    Predictor const* predictor{nullptr};
    StringView msg{"Unsupported data type for inplace predict."};

    PredictionCacheEntry predts;
    if (ctx_->gpu_id != Context::kCpuId) {
      predts.predictions.SetDevice(ctx_->gpu_id);
    }
    predts.predictions.Resize(p_fmat->Info().num_row_ * n_groups, 0);

    auto predict_impl = [&](size_t i) {
      predts.predictions.Fill(0);
      if (tparam_.predictor == PredictorType::kAuto) {
        // Try both predictor implementations
        bool success = false;
        for (auto const& p : predictors) {
          if (p && p->InplacePredict(p_fmat, model_, missing, &predts, i, i + 1)) {
            success = true;
            predictor = p;
            break;
          }
        }
        CHECK(success) << msg;
      } else {
        predictor = this->GetPredictor().get();
        bool success = predictor->InplacePredict(p_fmat, model_, missing, &predts, i, i + 1);
        CHECK(success) << msg << std::endl
                       << "Current Predictor: "
                       << (tparam_.predictor == PredictorType::kCPUPredictor ? "cpu_predictor"
                                                                             : "gpu_predictor");
      }
    };

    // Inplace predict is not used for training, so no need to drop tree.
    for (bst_tree_t i = tree_begin; i < tree_end; ++i) {
      predict_impl(i);
      if (i == tree_begin) {
        predictor->InitOutPredictions(p_fmat->Info(), &p_out_preds->predictions, model_);
      }
      // Multiple the tree weight
      auto w = this->weight_drop_.at(i);
      auto group = model_.tree_info.at(i);
      CHECK_EQ(predts.predictions.Size(), p_out_preds->predictions.Size());

      size_t n_rows = p_fmat->Info().num_row_;
      if (predts.predictions.DeviceIdx() != Context::kCpuId) {
        p_out_preds->predictions.SetDevice(predts.predictions.DeviceIdx());
        auto base_score = model_.learner_model_param->BaseScore(predts.predictions.DeviceIdx());
        GPUDartInplacePredictInc(p_out_preds->predictions.DeviceSpan(),
                                 predts.predictions.DeviceSpan(), w, n_rows, base_score, n_groups,
                                 group);
      } else {
        auto base_score = model_.learner_model_param->BaseScore(Context::kCpuId);
        auto& h_predts = predts.predictions.HostVector();
        auto& h_out_predts = p_out_preds->predictions.HostVector();
        common::ParallelFor(n_rows, ctx_->Threads(), [&](auto ridx) {
          const size_t offset = ridx * n_groups + group;
          h_out_predts[offset] += (h_predts[offset] - base_score(0)) * w;
        });
      }
    }
  }

  void PredictInstance(const SparsePage::Inst &inst,
                       std::vector<bst_float> *out_preds,
                       unsigned layer_begin, unsigned layer_end) override {
    DropTrees(false);
    auto &predictor = this->GetPredictor();
    uint32_t _, tree_end;
    std::tie(_, tree_end) = detail::LayerToTree(model_, layer_begin, layer_end);
    predictor->PredictInstance(inst, out_preds, model_, tree_end);
  }

  void PredictContribution(DMatrix* p_fmat,
                           HostDeviceVector<bst_float>* out_contribs,
                           unsigned layer_begin, unsigned layer_end, bool approximate, int,
                           unsigned) override {
    CHECK(configured_);
    auto [tree_begin, tree_end] = detail::LayerToTree(model_, layer_begin, layer_end);
    cpu_predictor_->PredictContribution(p_fmat, out_contribs, model_, tree_end, &weight_drop_,
                                        approximate);
  }

  void PredictInteractionContributions(
      DMatrix *p_fmat, HostDeviceVector<bst_float> *out_contribs,
      unsigned layer_begin, unsigned layer_end, bool approximate) override {
    CHECK(configured_);
    auto [tree_begin, tree_end] = detail::LayerToTree(model_, layer_begin, layer_end);
    cpu_predictor_->PredictInteractionContributions(p_fmat, out_contribs, model_, tree_end,
                                                    &weight_drop_, approximate);
  }

 protected:
  // commit new trees all at once
  void CommitModel(TreesOneIter&& new_trees) override {
    auto n_new_trees = model_.CommitModel(std::forward<TreesOneIter>(new_trees));
    size_t num_drop = NormalizeTrees(n_new_trees);
    LOG(INFO) << "drop " << num_drop << " trees, "
              << "weight = " << weight_drop_.back();
  }

  // Select which trees to drop.
  inline void DropTrees(bool is_training) {
    if (!is_training) {
      // This function should be thread safe when it's not training.
      return;
    }
    idx_drop_.clear();

    std::uniform_real_distribution<> runif(0.0, 1.0);
    auto& rnd = common::GlobalRandom();
    bool skip = false;
    if (dparam_.skip_drop > 0.0) skip = (runif(rnd) < dparam_.skip_drop);
    // sample some trees to drop
    if (!skip) {
      if (dparam_.sample_type == 1) {
        bst_float sum_weight = 0.0;
        for (auto elem : weight_drop_) {
          sum_weight += elem;
        }
        for (size_t i = 0; i < weight_drop_.size(); ++i) {
          if (runif(rnd) < dparam_.rate_drop * weight_drop_.size() * weight_drop_[i] / sum_weight) {
            idx_drop_.push_back(i);
          }
        }
        if (dparam_.one_drop && idx_drop_.empty() && !weight_drop_.empty()) {
          // the expression below is an ugly but MSVC2013-friendly equivalent of
          // size_t i = std::discrete_distribution<size_t>(weight_drop.begin(),
          //                                               weight_drop.end())(rnd);
          size_t i = std::discrete_distribution<size_t>(
            weight_drop_.size(), 0., static_cast<double>(weight_drop_.size()),
            [this](double x) -> double {
              return weight_drop_[static_cast<size_t>(x)];
            })(rnd);
          idx_drop_.push_back(i);
        }
      } else {
        for (size_t i = 0; i < weight_drop_.size(); ++i) {
          if (runif(rnd) < dparam_.rate_drop) {
            idx_drop_.push_back(i);
          }
        }
        if (dparam_.one_drop && idx_drop_.empty() && !weight_drop_.empty()) {
          size_t i = std::uniform_int_distribution<size_t>(0, weight_drop_.size() - 1)(rnd);
          idx_drop_.push_back(i);
        }
      }
    }
  }

  // set normalization factors
  std::size_t NormalizeTrees(size_t size_new_trees) {
    CHECK(tree_param_.GetInitialised());
    float lr = 1.0 * tree_param_.learning_rate / size_new_trees;
    size_t num_drop = idx_drop_.size();
    if (num_drop == 0) {
      for (size_t i = 0; i < size_new_trees; ++i) {
        weight_drop_.push_back(1.0);
      }
    } else {
      if (dparam_.normalize_type == 1) {
        // normalize_type 1
        float factor = 1.0 / (1.0 + lr);
        for (auto i : idx_drop_) {
          weight_drop_[i] *= factor;
        }
        for (size_t i = 0; i < size_new_trees; ++i) {
          weight_drop_.push_back(factor);
        }
      } else {
        // normalize_type 0
        float factor = 1.0 * num_drop / (num_drop + lr);
        for (auto i : idx_drop_) {
          weight_drop_[i] *= factor;
        }
        for (size_t i = 0; i < size_new_trees; ++i) {
          weight_drop_.push_back(1.0 / (num_drop + lr));
        }
      }
    }
    // reset
    idx_drop_.clear();
    return num_drop;
  }

  // init thread buffers
  inline void InitThreadTemp(int nthread) {
    int prev_thread_temp_size = thread_temp_.size();
    if (prev_thread_temp_size < nthread) {
      thread_temp_.resize(nthread, RegTree::FVec());
      for (int i = prev_thread_temp_size; i < nthread; ++i) {
        thread_temp_[i].Init(model_.learner_model_param->num_feature);
      }
    }
  }

  // --- data structure ---
  // training parameter
  DartTrainParam dparam_;
  /*! \brief prediction buffer */
  std::vector<bst_float> weight_drop_;
  // indexes of dropped trees
  std::vector<size_t> idx_drop_;
  // temporal storage for per thread
  std::vector<RegTree::FVec> thread_temp_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(GBTreeModelParam);
DMLC_REGISTER_PARAMETER(GBTreeTrainParam);
DMLC_REGISTER_PARAMETER(DartTrainParam);

XGBOOST_REGISTER_GBM(GBTree, "gbtree")
    .describe("Tree booster, gradient boosted trees.")
    .set_body([](LearnerModelParam const* booster_config, Context const* ctx) {
      auto* p = new GBTree(booster_config, ctx);
      return p;
    });
XGBOOST_REGISTER_GBM(Dart, "dart")
    .describe("Tree booster, dart.")
    .set_body([](LearnerModelParam const* booster_config, Context const* ctx) {
      GBTree* p = new Dart(booster_config, ctx);
      return p;
    });
}  // namespace xgboost::gbm
