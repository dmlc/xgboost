/**
 * Copyright 2014-2026, XGBoost Contributors
 *
 * \file gbtree.cc
 * \brief gradient boosted tree implementation.
 * \author Tianqi Chen
 */
#include "gbtree.h"

#include <dmlc/omp.h>
#include <dmlc/parameter.h>

#include <algorithm>  // for equal, any_of
#include <cstdint>    // for uint32_t
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../common/common.h"
#include "../common/cuda_rt_utils.h"  // for AllVisibleGPUs
#include "../common/error_msg.h"  // for UnknownDevice, WarnOldSerialization, InplacePredictProxy
#include "../common/threading_utils.h"
#include "../common/timer.h"
#include "../data/proxy_dmatrix.h"  // for DMatrixProxy, HostAdapterDispatch
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

namespace {
/** @brief Map the `tree_method` parameter to the `updater` parameter. */
std::string MapTreeMethodToUpdaters(Context const* ctx, TreeMethod tree_method) {
  // Choose updaters according to tree_method parameters
  if (ctx->IsCUDA()) {
    common::AssertGPUSupport();
  }

  switch (tree_method) {
    case TreeMethod::kAuto:  // Use hist as default in 2.0
    case TreeMethod::kHist: {
      return ctx->DispatchDevice([] { return "grow_quantile_histmaker"; },
                                 [] { return "grow_gpu_hist"; },
                                 [] { return "grow_quantile_histmaker_sycl"; });
    }
    case TreeMethod::kApprox: {
      return ctx->DispatchDevice([] { return "grow_histmaker"; }, [] { return "grow_gpu_approx"; });
    }
    case TreeMethod::kExact:
      CHECK(ctx->IsCPU()) << "The `exact` tree method is not supported on GPU.";
      return "grow_colmaker,prune";
    default:
      auto tm = static_cast<std::underlying_type_t<TreeMethod>>(tree_method);
      LOG(FATAL) << "Unknown tree_method: `" << tm << "`.";
  }

  LOG(FATAL) << "unreachable";
  return "";
}

bool UpdatersMatched(std::vector<std::string> updater_seq,
                     std::vector<std::unique_ptr<TreeUpdater>> const& updaters) {
  if (updater_seq.size() != updaters.size()) {
    return false;
  }

  return std::equal(updater_seq.cbegin(), updater_seq.cend(), updaters.cbegin(),
                    [](std::string const& name, std::unique_ptr<TreeUpdater> const& up) {
                      return name == up->Name();
                    });
}

void ScaleTreeLeaves(RegTree* p_tree, float scale) {
  if (!p_tree->IsMultiTarget()) {
    for (bst_node_t nidx = 0; nidx < p_tree->NumNodes(); ++nidx) {
      auto& node = (*p_tree)[nidx];
      if (!node.IsDeleted() && node.IsLeaf()) {
        node.SetLeaf(node.LeafValue() * scale, node.RightChild());
      }
    }
    return;
  }

  auto const* tree = p_tree->GetMultiTargetTree();
  std::vector<bst_node_t> leaves;
  std::vector<float> values;
  for (bst_node_t nidx = 0; nidx < static_cast<bst_node_t>(tree->Size()); ++nidx) {
    if (tree->IsLeaf(nidx)) {
      leaves.push_back(nidx);
      auto leaf = tree->LeafValue(nidx);
      for (auto value : leaf.Values()) {
        values.push_back(value * scale);
      }
    }
  }
  p_tree->SetLeaves(std::move(leaves), values);
}

void FoldTreeWeights(GBTreeModel* model) {
  CHECK_LE(model->weight_drop.size(), model->trees.size());
  for (std::size_t i = 0; i < model->weight_drop.size(); ++i) {
    ScaleTreeLeaves(model->trees[i].get(), model->weight_drop[i]);
  }
  model->weight_drop.clear();
}

}  // namespace

void GBTree::Configure(Args const& cfg) {
  tparam_.UpdateAllowUnknown(cfg);
  dparam_.UpdateAllowUnknown(cfg);
  auto has_param = [&cfg](std::string const& name) {
    return std::any_of(cfg.cbegin(), cfg.cend(),
                       [&name](auto const& arg) { return arg.first == name; });
  };
  auto has_dropout_rate = has_param("dropout_rate");
  if (has_param("skip_drop")) {
    if (!has_dropout_rate) {
      dparam_.dropout_rate = dparam_.skip_drop;
      LOG(WARNING) << "`skip_drop` has been removed and is interpreted as `dropout_rate`.";
    } else {
      LOG(WARNING) << "`skip_drop` has been removed and is ignored because `dropout_rate` is set.";
    }
  }
  for (auto const* removed : {"rate_drop", "one_drop", "sample_type", "normalize_type"}) {
    if (has_param(removed)) {
      LOG(WARNING) << "`" << removed << "` has been removed and is ignored.";
    }
  }
  CHECK_LT(dparam_.dropout_rate, 1.0f)
      << "`dropout_rate` must be less than 1 so retained predictions can be scaled by "
         "1 / (1 - dropout_rate).";
  tree_param_.UpdateAllowUnknown(cfg);

  model_.Configure(cfg);

  // for the 'update' process_type, move trees into trees_to_update
  if (tparam_.process_type == TreeProcessType::kUpdate) {
    model_.InitTreesToUpdate();
  }

  // configure predictors
  if (!cpu_predictor_) {
    cpu_predictor_ = std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor", this->ctx_));
  }
  cpu_predictor_->Configure(cfg);
#if defined(XGBOOST_USE_CUDA)
  auto n_gpus = curt::AllVisibleGPUs();
  if (!gpu_predictor_) {
    gpu_predictor_ = std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", this->ctx_));
  }
  if (n_gpus != 0) {
    gpu_predictor_->Configure(cfg);
  }
#endif  // defined(XGBOOST_USE_CUDA)

#if defined(XGBOOST_USE_SYCL)
  if (!sycl_predictor_) {
    sycl_predictor_ = std::unique_ptr<Predictor>(Predictor::Create("sycl_predictor", this->ctx_));
  }
  sycl_predictor_->Configure(cfg);
#endif  // defined(XGBOOST_USE_SYCL)

  // `updater` parameter was manually specified
  specified_updater_ =
      std::any_of(cfg.cbegin(), cfg.cend(), [](auto const& arg) { return arg.first == "updater"; });
  if (specified_updater_) {
    error::WarnManualUpdater();
  }
  LOG(DEBUG) << "Using tree method: " << static_cast<int>(tparam_.tree_method);

  if (!specified_updater_) {
    this->tparam_.updater_seq = MapTreeMethodToUpdaters(ctx_, tparam_.tree_method);
  }

  auto up_names = common::Split(tparam_.updater_seq, ',');
  if (!UpdatersMatched(up_names, updaters_)) {
    updaters_.clear();
    for (auto const& name : up_names) {
      std::unique_ptr<TreeUpdater> up(
          TreeUpdater::Create(name.c_str(), ctx_, &model_.learner_model_param->task));
      updaters_.push_back(std::move(up));
    }
  }

  for (auto& up : updaters_) {
    up->Configure(cfg);
  }
}

void GBTreeModel::InitTreesToUpdate() {
  if (trees_to_update.empty()) {
    for (auto& tree : trees) {
      trees_to_update.push_back(std::move(tree));
    }

    trees.clear();
    param.num_trees = 0;
    tree_info.HostVector().clear();

    iteration_indptr.clear();
    iteration_indptr.push_back(0);
  }
}

void GPUCopyGradient(Context const*, linalg::Matrix<GradientPair> const*, bst_group_t,
                     linalg::Matrix<GradientPair>*)
#if defined(XGBOOST_USE_CUDA)
    ;  // NOLINT
#else
{
  common::AssertGPUSupport();
}
#endif

void CopyGradient(Context const* ctx, linalg::Matrix<GradientPair> const* in_gpair,
                  bst_group_t group_id, linalg::Matrix<GradientPair>* out_gpair) {
  out_gpair->SetDevice(ctx->Device());
  out_gpair->Reshape(in_gpair->Shape(0), 1);
  if (ctx->IsCUDA()) {
    GPUCopyGradient(ctx, in_gpair, group_id, out_gpair);
  } else {
    auto const& in = *in_gpair;
    auto h_tmp = out_gpair->HostView();
    auto h_in = in.HostView().Slice(linalg::All(), group_id);
    CHECK_EQ(h_tmp.Size(), h_in.Size());
    common::ParallelFor(h_in.Size(), ctx->Threads(), [&](auto i) { h_tmp(i) = h_in(i); });
  }
}

void GPUScalePrediction(common::Span<float>, float)
#if defined(XGBOOST_USE_CUDA)
    ;  // NOLINT
#else
{
  common::AssertGPUSupport();
}
#endif

void ScalePrediction(Context const* ctx, HostDeviceVector<float>* predictions, float scale) {
  if (ctx->IsCUDA()) {
    predictions->SetDevice(ctx->Device());
    GPUScalePrediction(predictions->DeviceSpan(), scale);
    return;
  }
  auto& values = predictions->HostVector();
  common::ParallelFor(values.size(), ctx->Threads(), [&](auto i) { values[i] *= scale; });
}

void GBTree::DoBoost(std::shared_ptr<DMatrix> p_fmat, GradientContainer* in_gpair,
                     ObjFunction const*) {
  auto predt = prediction_cache_.Cache(p_fmat, ctx_->Device());
  if (model_.learner_model_param->IsVectorLeaf()) {
    CHECK(tparam_.tree_method == TreeMethod::kHist || tparam_.tree_method == TreeMethod::kAuto)
        << "Only the hist tree method is supported for building multi-target trees with vector "
           "leaf.";
  }
  if (in_gpair->HasValueGrad()) {
    CHECK(model_.learner_model_param->IsVectorLeaf())
        << "Reduced gradient must be used with vector leaf trees";
    CHECK(!tree_param_.HasMonotone())
        << "Monotonic constraints are not supported with reduced gradients.";
  }

  TreesOneIter new_trees;
  bst_target_t const n_groups = model_.learner_model_param->OutputLength();
  monitor_.Start("BoostNewTrees");

  // Define the categories.
  if (this->model_.Cats()->Empty() && !p_fmat->Cats()->Empty()) {
    auto in_cats = p_fmat->Cats();
    this->model_.Cats()->Copy(this->ctx_, *in_cats);
    this->model_.Cats()->Sort(this->ctx_);
  } else {
    CHECK_EQ(this->model_.Cats()->NumCatsTotal(), p_fmat->Cats()->NumCatsTotal())
        << "A new dataset with different categorical features is used for training an existing "
           "model.";
  }

  predt->predictions.SetDevice(ctx_->Device());
  auto const& predictor = this->GetPredictor(false, &predt->predictions, p_fmat.get());
  if (predt->predictions.Size() == 0 && p_fmat->Info().num_row_ != 0) {
    CHECK_EQ(predt->version, 0);
    predictor->InitOutPredictions(p_fmat->Info(), &predt->predictions, model_);
  }
  auto out = linalg::MakeTensorView(ctx_, &predt->predictions, p_fmat->Info().num_row_,
                                    model_.learner_model_param->OutputLength());
  CHECK_NE(n_groups, 0);

  // The node position for each row, 1 HDV for each tree in the forest.  Note that the
  // position is negated if the row is sampled out.
  std::vector<HostDeviceVector<bst_node_t>> node_position;
  auto predict_from_node_positions = [&](std::vector<HostDeviceVector<bst_node_t>>& positions,
                                         TreesOneGroup const& trees,
                                         linalg::MatrixView<float> out_preds) {
    CHECK_EQ(positions.size(), trees.size());
    for (auto& position : positions) {
      if (out_preds.Shape(0) != 0 && position.Size() != out_preds.Shape(0)) {
        return false;
      }
      position.SetDevice(predt->predictions.Device());
    }
    std::vector<RegTree const*> tree_ptrs;
    tree_ptrs.reserve(trees.size());
    for (auto const& tree : trees) {
      tree_ptrs.push_back(tree.get());
    }
    predictor->PredictFromLeafIds(common::Span{positions}, common::Span{tree_ptrs}, out_preds);
    return true;
  };

  if (model_.learner_model_param->IsVectorLeaf() ||
      model_.learner_model_param->OutputLength() == 1u) {
    TreesOneGroup ret;
    BoostNewTrees(in_gpair, p_fmat.get(), 0, &node_position, &ret);
    if (predict_from_node_positions(node_position, ret, out)) {
      predt->Update(1);
    }
    new_trees.push_back(std::move(ret));
  } else {
    // Multi-target, scalar leaf
    CHECK_EQ(in_gpair->gpair.Size() % n_groups, 0U)
        << "Must have exactly n_groups * n_samples gpairs.";
    GradientContainer tmp;
    tmp.gpair = linalg::Matrix<GradientPair>{
        {in_gpair->gpair.Shape(0), static_cast<std::size_t>(1ul)}, ctx_->Device()};
    bool cache_updated{true};
    for (bst_target_t gid = 0; gid < n_groups; ++gid) {
      node_position.clear();
      CopyGradient(ctx_, &in_gpair->gpair, gid, &tmp.gpair);
      TreesOneGroup ret;
      BoostNewTrees(&tmp, p_fmat.get(), gid, &node_position, &ret);
      auto v_predt = out.Slice(linalg::All(), linalg::Range(gid, gid + 1));
      cache_updated = predict_from_node_positions(node_position, ret, v_predt) && cache_updated;
      new_trees.push_back(std::move(ret));
    }
    if (cache_updated) {
      predt->Update(1);
    }
  }

  monitor_.Stop("BoostNewTrees");
  model_.CommitModel(std::move(new_trees));
  if (dparam_.HasDropout()) {
    // The cache contains the sampled training margin plus the new tree. It is not a valid
    // prefix of the fixed-weight inference model and must not be reused.
    predt->Reset();
  }
}

std::vector<RegTree*> GBTree::InitNewTrees(bst_target_t bst_group, TreesOneGroup* ret) {
  std::vector<RegTree*> new_trees;
  ret->clear();
  // create the trees
  for (int i = 0; i < model_.param.num_parallel_tree; ++i) {
    if (tparam_.process_type == TreeProcessType::kDefault) {
      CHECK(!updaters_.empty());
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
  return new_trees;
}

void GBTree::BoostNewTrees(GradientContainer* gpair, DMatrix* p_fmat, int bst_group,
                           std::vector<HostDeviceVector<bst_node_t>>* out_position,
                           TreesOneGroup* ret) {
  std::vector<RegTree*> new_trees = this->InitNewTrees(bst_group, ret);

  // update the trees
  auto n_out = model_.learner_model_param->OutputLength() * p_fmat->Info().num_row_;
  StringView msg{
      "Mismatching size between number of rows from input data and size of gradient vector."};
  if (!model_.learner_model_param->IsVectorLeaf() && p_fmat->Info().num_row_ != 0) {
    CHECK_EQ(n_out % gpair->gpair.Size(), 0) << msg;
  } else if (model_.learner_model_param->IsVectorLeaf()) {
    // vector leaf
    if (!gpair->HasValueGrad()) {
      CHECK_EQ(gpair->gpair.Size(), n_out) << msg;
    }
  }

  out_position->resize(new_trees.size());

  // Rescale learning rate according to the number of trees
  auto lr = tree_param_.learning_rate;
  tree_param_.learning_rate /= static_cast<float>(new_trees.size());
  for (auto& up : updaters_) {
    up->Update(&tree_param_, gpair, p_fmat,
               common::Span<HostDeviceVector<bst_node_t>>{*out_position}, new_trees);
  }
  tree_param_.learning_rate = lr;
}

void GBTree::LoadConfig(Json const& in) {
  auto name = get<String const>(in["name"]);
  CHECK(name == "gbtree" || name == "dart")
      << "Unknown booster name in model JSON: `" << name
      << "`. Only `gbtree` or legacy `dart` boosters are accepted here.";
  auto const& config = name == "dart" ? in["gbtree"] : in;
  FromJson(config["gbtree_train_param"], &tparam_);
  FromJson(config["tree_train_param"], &tree_param_);
  auto const& obj = get<Object const>(config);
  auto it = obj.find("dart_train_param");
  bool has_dropout_rate{false};
  if (it != obj.cend()) {
    auto const& dart_config = get<Object const>(it->second);
    has_dropout_rate = dart_config.find("dropout_rate") != dart_config.cend();
    FromJson(it->second, &dparam_);
  } else if (name == "dart") {
    auto const& dart_config = get<Object const>(in["dart_train_param"]);
    has_dropout_rate = dart_config.find("dropout_rate") != dart_config.cend();
    FromJson(in["dart_train_param"], &dparam_);
  } else {
    dparam_ = {};
  }
  if (!has_dropout_rate && dparam_.skip_drop != 0.0f) {
    dparam_.dropout_rate = dparam_.skip_drop;
    LOG(WARNING) << "`skip_drop` has been removed and is interpreted as `dropout_rate`.";
  }
  CHECK_LT(dparam_.dropout_rate, 1.0f)
      << "`dropout_rate` must be less than 1 so retained predictions can be scaled.";

  // Process type cannot be kUpdate from loaded model
  // This would cause all trees to be pushed to trees_to_update
  // e.g. updating a model, then saving and loading it would result in an empty model
  tparam_.process_type = TreeProcessType::kDefault;
  std::int32_t const n_gpus = curt::AllVisibleGPUs();

  std::vector<Json> updater_seq;
  if (IsA<Object>(config["updater"])) {
    // before 2.0
    error::WarnOldSerialization();
    for (auto const& kv : get<Object const>(config["updater"])) {
      auto name = kv.first;
      auto config = kv.second;
      config["name"] = name;
      updater_seq.push_back(config);
    }
  } else {
    // after 2.0
    auto const& j_updaters = get<Array const>(config["updater"]);
    updater_seq = j_updaters;
  }

  updaters_.clear();

  for (auto const& config : updater_seq) {
    auto name = get<String>(config["name"]);
    if (n_gpus == 0 && name == "grow_gpu_hist") {
      name = "grow_quantile_histmaker";
      LOG(WARNING) << "Changing updater from `grow_gpu_hist` to `grow_quantile_histmaker`.";
    }
    updaters_.emplace_back(TreeUpdater::Create(name, ctx_, &model_.learner_model_param->task));
    updaters_.back()->LoadConfig(config);
  }

  specified_updater_ = get<Boolean>(config["specified_updater"]);
}

void GBTree::SaveConfig(Json* p_out) const {
  auto& out = *p_out;
  out["name"] = String("gbtree");
  out["gbtree_train_param"] = ToJson(tparam_);
  out["tree_train_param"] = ToJson(tree_param_);
  out["dart_train_param"] = ToJson(dparam_);

  // Process type cannot be kUpdate from loaded model
  // This would cause all trees to be pushed to trees_to_update
  // e.g. updating a model, then saving and loading it would result in an empty
  // model
  out["gbtree_train_param"]["process_type"] = String("default");
  // Duplicated from SaveModel so that user can get `num_parallel_tree` without parsing
  // the model. We might remove this once we can deprecate `best_ntree_limit` so that the
  // language binding doesn't need to know about the forest size.
  out["gbtree_model_param"] = ToJson(model_.param);

  out["updater"] = Array{};
  auto& j_updaters = get<Array>(out["updater"]);

  for (auto const& up : this->updaters_) {
    Json up_config{Object{}};
    up_config["name"] = String{up->Name()};
    up->SaveConfig(&up_config);
    j_updaters.emplace_back(up_config);
  }
  out["specified_updater"] = Boolean{specified_updater_};
}

void GBTree::LoadModel(Json const& in) {
  auto name = get<String const>(in["name"]);
  CHECK(name == "gbtree" || name == "dart");
  auto const& model = name == "dart" ? in["gbtree"] : in;
  model_.LoadModel(model["model"]);
  auto const& obj = get<Object const>(name == "dart" ? in : model);
  // Compatibility for older models that stored DART weights beside the tree model.
  auto it = obj.find("weight_drop");
  if (it != obj.cend()) {
    auto const& j_weight_drop = get<Array const>(it->second);
    model_.weight_drop.resize(j_weight_drop.size());
    for (size_t i = 0; i < model_.weight_drop.size(); ++i) {
      model_.weight_drop[i] = get<Number const>(j_weight_drop[i]);
    }
  }
  CHECK_LE(model_.weight_drop.size(), model_.trees.size());
  FoldTreeWeights(&model_);
}

void GBTree::SaveModel(Json* p_out) const {
  auto& out = *p_out;
  out["name"] = String("gbtree");
  out["model"] = Object();
  model_.SaveModel(&out["model"]);
}

std::vector<std::uint8_t> GBTree::DropoutMask(bool is_training) {
  if (!is_training || !dparam_.HasDropout() || model_.trees.empty()) {
    return {};
  }
  std::uniform_real_distribution<> runif(0.0, 1.0);
  auto& rnd = ctx_->Rng();
  std::vector<std::uint8_t> dropout_mask(model_.trees.size());
  for (std::size_t i = 0; i < dropout_mask.size(); ++i) {
    dropout_mask[i] = runif(rnd) >= dparam_.dropout_rate;
  }
  return dropout_mask;
}

void GBTree::Slice(bst_layer_t begin, bst_layer_t end, bst_layer_t step, GradientBooster* out,
                   bool* out_of_bound) const {
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
  auto& out_tree_info = out_model.tree_info.HostVector();

  auto const& in_tree_info = this->model_.tree_info.ConstHostVector();

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
        std::unique_ptr<RegTree> new_tree{this->model_.trees.at(in_tree_idx)->Copy()};
        out_trees.emplace_back(std::move(new_tree));

        bst_group_t group = in_tree_info[in_tree_idx];
        out_tree_info.push_back(group);

        out_model.iteration_indptr[out_l + 1]++;
      });

  std::partial_sum(out_indptr.cbegin(), out_indptr.cend(), out_indptr.begin());
  CHECK_EQ(out_model.iteration_indptr.front(), 0);

  out_model.param.num_trees = out_model.trees.size();
  out_model.param.num_parallel_tree = model_.param.num_parallel_tree;

  p_gbtree->dparam_ = this->dparam_;
}

void GBTree::PredictBatch(std::shared_ptr<DMatrix> p_fmat, HostDeviceVector<float>* out_preds,
                          bool is_training, bst_layer_t layer_begin, bst_layer_t layer_end) {
  auto cache = prediction_cache_.Cache(p_fmat, ctx_->Device());
  auto const* tree_mask = static_cast<std::vector<std::uint8_t> const*>(nullptr);
  auto dropout_mask = this->DropoutMask(is_training);
  auto apply_dropout = !dropout_mask.empty();
  if (apply_dropout) {
    tree_mask = &dropout_mask;
  }

  // An ordinary prediction can reuse a cached prefix of the model output. A randomly masked
  // training prediction cannot participate in this cache.
  if (layer_end == 0) {
    layer_end = this->BoostedRounds();
  }

  auto cache_version = cache->version;
  // We can preserve the cache only when:
  // - prediction is not randomly masked
  // - prediction starts from iteration 0, so the result is a cacheable prefix
  auto preserve_cache = tree_mask == nullptr && model_.TreeWeights() == nullptr &&
                        p_fmat->Info().base_margin_.Empty() && layer_begin == 0;
  // We can reuse the existing cached prefix only when:
  // - the result itself is cacheable
  // - the requested range does not move backwards past the cached version
  auto reuse_cache = preserve_cache && layer_end >= static_cast<bst_layer_t>(cache_version);
  // Initialize output when:
  // - the cached prefix cannot be reused, or
  // - the cache is valid but still empty
  auto initialize_output = !reuse_cache || cache_version == 0;
  auto prediction_begin = reuse_cache ? cache_version : layer_begin;

  if (!reuse_cache) {
    cache->version = 0;
    cache_version = 0;
  }

  if (cache->predictions.Size() == 0 && p_fmat->Info().num_row_ != 0) {
    CHECK_EQ(cache->version, 0);
  }

  auto const& predictor = GetPredictor(is_training, &cache->predictions, p_fmat.get());
  if (initialize_output) {
    // cache->Size() can be non-zero as it's initialized here before any
    // tree is built at the 0^th iterator.
    predictor->InitOutPredictions(p_fmat->Info(), &cache->predictions, model_);
  }

  if (apply_dropout) {
    // Protect the base score or base margin from the normalization applied below.
    ScalePrediction(ctx_, &cache->predictions, 1.0f - dparam_.dropout_rate);
  }

  auto [tree_begin, tree_end] = detail::LayerToTree(model_, prediction_begin, layer_end);
  CHECK_LE(tree_end, model_.trees.size()) << "Invalid number of trees.";
  if (tree_end > tree_begin) {
    predictor->PredictBatch(p_fmat.get(), &cache->predictions, model_, tree_begin, tree_end,
                            tree_mask);
  }
  if (apply_dropout) {
    ScalePrediction(ctx_, &cache->predictions, detail::DropoutScale(dparam_.dropout_rate));
  }

  if (!preserve_cache) {
    cache->version = 0;
  } else {
    cache->Update(layer_end - cache_version);
  }

  out_preds->SetDevice(ctx_->Device());
  out_preds->Resize(cache->predictions.Size());
  out_preds->Copy(cache->predictions);
}

void GBTree::InplacePredict(std::shared_ptr<DMatrix> p_m, float missing,
                            HostDeviceVector<float>* out_preds, bst_layer_t layer_begin,
                            bst_layer_t layer_end) const {
  auto [tree_begin, tree_end] = detail::LayerToTree(model_, layer_begin, layer_end);
  CHECK_LE(tree_end, model_.trees.size()) << "Invalid number of trees.";
  if (p_m->Ctx()->Device() != this->ctx_->Device()) {
    error::MismatchedDevices(this->ctx_, p_m->Ctx());
    auto proxy = std::dynamic_pointer_cast<data::DMatrixProxy>(p_m);
    CHECK(proxy) << error::InplacePredictProxy();
    auto p_fmat = data::CreateDMatrixFromProxy(ctx_, proxy, missing);
    auto const& predictor = GetPredictor(false, out_preds, p_fmat.get());
    predictor->InitOutPredictions(p_fmat->Info(), out_preds, model_);
    if (tree_end > tree_begin) {
      predictor->PredictBatch(p_fmat.get(), out_preds, model_, tree_begin, tree_end);
    }
    return;
  }

  bool known_type = this->ctx_->DispatchDevice(
      [&, begin = tree_begin, end = tree_end] {
        return this->cpu_predictor_->InplacePredict(p_m, model_, missing, out_preds, begin, end);
      },
      [&, begin = tree_begin, end = tree_end] {
        return this->gpu_predictor_->InplacePredict(p_m, model_, missing, out_preds, begin, end);
#if defined(XGBOOST_USE_SYCL)
      },
      [&, begin = tree_begin, end = tree_end] {
        return this->sycl_predictor_->InplacePredict(p_m, model_, missing, out_preds, begin, end);
#endif  // defined(XGBOOST_USE_SYCL)
      });
  if (!known_type) {
    auto proxy = std::dynamic_pointer_cast<data::DMatrixProxy>(p_m);
    CHECK(proxy) << error::InplacePredictProxy();
    LOG(FATAL) << "Unknown data type for inplace prediction:" << proxy->Adapter().type().name();
  }
}

[[nodiscard]] std::unique_ptr<Predictor> const& GBTree::GetPredictor(
    bool is_training, HostDeviceVector<float> const* out_pred, DMatrix* f_dmat) const {
  // Data comes from SparsePageDMatrix. Since we are loading data in pages, no need to
  // prevent data copy.
  if (f_dmat && !f_dmat->SingleColBlock()) {
    if (ctx_->IsCPU()) {
      return cpu_predictor_;
    } else if (ctx_->IsCUDA()) {
      common::AssertGPUSupport();
      CHECK(gpu_predictor_);
      return gpu_predictor_;
    } else {
#if defined(XGBOOST_USE_SYCL)
      common::AssertSYCLSupport();
      CHECK(sycl_predictor_);
      return sycl_predictor_;
#endif  // defined(XGBOOST_USE_SYCL)
    }
  }

  // Data comes from Device DMatrix.
  auto is_ellpack =
      f_dmat && f_dmat->PageExists<EllpackPage>() && !f_dmat->PageExists<SparsePage>();
  // Data comes from device memory, like CuDF or CuPy.
  auto is_from_device = f_dmat && f_dmat->PageExists<SparsePage>() &&
                        (*(f_dmat->GetBatches<SparsePage>().begin())).data.DeviceCanRead();
  auto on_device = is_ellpack || is_from_device;

  // Use GPU Predictor if data is already on device and gpu_id is set.
  if (on_device && ctx_->IsCUDA()) {
    common::AssertGPUSupport();
    CHECK(gpu_predictor_);
    return gpu_predictor_;
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
      !on_device && is_training) {
    CHECK(cpu_predictor_);
    return cpu_predictor_;
  }

  if (ctx_->IsCPU()) {
    return cpu_predictor_;
  } else if (ctx_->IsCUDA()) {
    common::AssertGPUSupport();
    CHECK(gpu_predictor_);
    return gpu_predictor_;
  } else {
#if defined(XGBOOST_USE_SYCL)
    common::AssertSYCLSupport();
    CHECK(sycl_predictor_);
    return sycl_predictor_;
#endif  // defined(XGBOOST_USE_SYCL)
  }

  return cpu_predictor_;
}

// register the objective functions
DMLC_REGISTER_PARAMETER(GBTreeModelParam);
DMLC_REGISTER_PARAMETER(GBTreeTrainParam);
DMLC_REGISTER_PARAMETER(DartTrainParam);

XGBOOST_REGISTER_GBM(GBTree, "gbtree")
    .describe("Tree booster, gradient boosted trees.")
    .set_body([](LearnerModelParam const* booster_config, Context const* ctx) {
      auto* p = new GBTree{booster_config, ctx};
      return p;
    });
}  // namespace xgboost::gbm
