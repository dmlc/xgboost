/**
 * Copyright 2026, XGBoost contributors
 *
 * @brief Implementation of gradient booster for Tree-G (Graph-Aware Decision Trees)
 */
#include "booster.h"

#include <xgboost/base.h>                // for bst_float, bst_layer_t, bst_target_t, GradientPair
#include <xgboost/context.h>             // for Context
#include <xgboost/data.h>                // for DMatrix, MetaInfo
#include <xgboost/feature_map.h>         // for FeatureMap
#include <xgboost/gbm.h>                 // for GradientBooster, XGBOOST_REGISTER_GBM
#include <xgboost/host_device_vector.h>  // for HostDeviceVector
#include <xgboost/json.h>                // for Json, Object, Array, Integer, String, get
#include <xgboost/learner.h>             // for LearnerModelState
#include <xgboost/linalg.h>              // for All
#include <xgboost/logging.h>             // for CHECK, LOG
#include <xgboost/objective.h>           // for ObjFunction
#include <xgboost/span.h>                // for Span
#include <xgboost/string_view.h>         // for StringView

#include <algorithm>  // for transform
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t, uint32_t
#include <memory>     // for make_unique, shared_ptr
#include <numeric>    // for iota
#include <set>        // for set
#include <string>     // for string
#include <utility>    // for move
#include <vector>     // for vector

#include "../../src/collective/communicator-inl.h"  // for IsDistributed
#include "../../src/common/threading_utils.h"       // for ParallelFor

namespace xgboost::treeg {
namespace {
// User-facing error messages. Kept here rather than in src/common/error_msg.h because that
// header belongs to the core library.
constexpr StringView kNoDistributed = "treeg does not support distributed training.";
constexpr StringView kCpuOnlyTraining =
    "Tree-G currently only supports CPU training. GPU support is planned for future releases.";
constexpr StringView kCpuOnlyPrediction = "Tree-G prediction is CPU-only.";
constexpr StringView kOnlyDefaultProcessType = "treeg supports only process_type=default";
constexpr StringView kNoMultiOutputTree =
    "Tree-G does not support multi_strategy=multi_output_tree. "
    "Use the default one-vs-all strategy for multi-class problems.";
constexpr StringView kOnlySingleParallelTree = "Tree-G currently only supports num_parallel_tree=1";
constexpr StringView kNoPredictLeaf =
    "Leaf prediction is not yet implemented for Tree-G. "
    "This requires proper graph-aware tree traversal.";
constexpr StringView kNoPredictContribution =
    "Feature contributions (SHAP) are not yet implemented for Tree-G. "
    "This feature requires graph-aware contribution computation.";
constexpr StringView kNoPredictInteraction =
    "Interaction contributions are not yet implemented for Tree-G. "
    "This feature requires graph-aware contribution computation.";

/**
 * @brief Seed the prediction buffer with the base margin.
 *
 * Mirrors `xgboost::predictor::InitOutPredictions`, which cannot be reused here
 * because it is typed on `gbm::GBTreeModel`. The base margin supplied through
 * `MetaInfo` takes precedence over the learner intercept.
 *
 * @param info Meta information of the matrix being predicted on.
 * @param state Learner model state holding the intercept.
 * @param out_preds Output buffer, resized to `n_rows * OutputLength()`.
 */
void InitOutPredictions(MetaInfo const& info, LearnerModelState const* state,
                        HostDeviceVector<float>* out_preds) {
  CHECK_NE(state->num_output_group, 0);
  auto const n_groups = state->OutputLength();
  out_preds->Resize(static_cast<std::size_t>(n_groups) * info.num_row_);

  HostDeviceVector<float> const* base_margin = info.base_margin_.Data();
  if (!base_margin->Empty()) {
    CHECK_EQ(base_margin->Size(), out_preds->Size())
        << "Invalid shape of base_margin. Expected (" << info.num_row_ << ", " << n_groups << ").";
    out_preds->Copy(*base_margin);
    return;
  }

  auto base_score = state->BaseScore(DeviceOrd::CPU());
  if (base_score.Size() == 1) {
    out_preds->Fill(base_score(0));
    return;
  }

  // Multi-output model with a per-target intercept.
  CHECK_EQ(base_score.Size(), n_groups);
  auto& h_preds = out_preds->HostVector();
  for (std::size_t i = 0; i < info.num_row_; ++i) {
    for (bst_target_t g = 0; g < n_groups; ++g) {
      h_preds[i * n_groups + g] = base_score(g);
    }
  }
}

/**
 * @brief Copy the graph-shape parameters (the ones persisted with the trees) between the
 *        training parameters and the model parameters.
 */
template <typename From, typename To>
void CopyGraphShape(From const& from, To* to) {
  to->max_walk_len = from.max_walk_len;
  to->max_ancestor_distance = from.max_ancestor_distance;
  to->num_features = from.num_features;
  to->num_vertices = from.num_vertices;
  to->intercept = from.intercept;
}
}  // anonymous namespace

// Register parameters
DMLC_REGISTER_PARAMETER(TreeGBoosterModelParam);
DMLC_REGISTER_PARAMETER(TreeGBoosterTrainParam);

// ========== TreeGBoosterModel Implementation ==========

void TreeGBoosterModel::SaveModel(Json* p_out) const {
  auto& out = *p_out;
  CHECK_EQ(param.num_trees, static_cast<std::int32_t>(trees.size()));

  out["treeg_booster_model_param"] = ToJson(param);

  std::vector<Json> trees_json(trees.size());
  common::ParallelFor(trees.size(), ctx_->Threads(), [&](auto t) {
    Json jtree{Object{}};
    trees[t]->SaveModel(&jtree);
    jtree["id"] = Integer{static_cast<Integer::Int>(t)};
    trees_json[t] = std::move(jtree);
  });

  out["trees"] = Array(std::move(trees_json));

  // Save tree_info (group index per tree)
  std::vector<Json> jtree_info(tree_info.size());
  std::transform(tree_info.cbegin(), tree_info.cend(), jtree_info.begin(),
                 [](bst_target_t g) { return Integer{static_cast<Integer::Int>(g)}; });
  out["tree_info"] = Array{std::move(jtree_info)};

  std::vector<Json> jiteration_indptr(iteration_indptr.size());
  std::transform(iteration_indptr.cbegin(), iteration_indptr.cend(), jiteration_indptr.begin(),
                 [](bst_tree_t i) { return Integer{i}; });
  out["iteration_indptr"] = Array{std::move(jiteration_indptr)};
}

void TreeGBoosterModel::LoadModel(Json const& in) {
  FromJson(in["treeg_booster_model_param"], &param);

  trees.clear();

  auto const& jmodel = get<Object const>(in);
  auto const& trees_json = get<Array const>(jmodel.at("trees"));
  CHECK_EQ(trees_json.size(), static_cast<std::size_t>(param.num_trees));
  trees.resize(param.num_trees);

  common::ParallelFor(param.num_trees, ctx_->Threads(), [&](auto t) {
    auto tree_id = get<Integer const>(trees_json[t]["id"]);
    trees.at(tree_id) = std::make_unique<RegTreeG>();
    trees[tree_id]->LoadModel(trees_json[t]);
  });

  // Load tree_info
  tree_info.clear();
  auto ti_it = jmodel.find("tree_info");
  if (ti_it != jmodel.cend()) {
    auto const& ti_vec = get<Array const>(ti_it->second);
    tree_info.resize(ti_vec.size());
    std::transform(ti_vec.cbegin(), ti_vec.cend(), tree_info.begin(),
                   [](Json const& v) { return static_cast<bst_target_t>(get<Integer const>(v)); });
  } else {
    // Old model without tree_info: all trees belong to group 0
    tree_info.resize(param.num_trees, 0);
  }

  auto indptr_it = jmodel.find("iteration_indptr");
  iteration_indptr.clear();
  if (indptr_it != jmodel.cend()) {
    auto const& vec = get<Array const>(indptr_it->second);
    iteration_indptr.resize(vec.size());
    std::transform(vec.cbegin(), vec.cend(), iteration_indptr.begin(),
                   [](Json const& v) { return get<Integer const>(v); });
  } else {
    // Old model without iteration_indptr
    iteration_indptr.resize(param.num_trees + 1);
    for (std::int32_t i = 0; i <= param.num_trees; ++i) {
      iteration_indptr[i] = i;
    }
  }
}

std::vector<std::string> TreeGBoosterModel::DumpModel(FeatureMap const& fmap, bool with_stats,
                                                      std::int32_t /*n_threads*/,
                                                      std::string const& format) const {
  std::vector<std::string> dump(trees.size());
  for (std::size_t i = 0; i < trees.size(); ++i) {
    dump[i] = trees[i]->DumpModel(fmap, with_stats, format);
  }
  return dump;
}

bst_tree_t TreeGBoosterModel::CommitModel(TreeGOneIter&& new_trees) {
  CHECK(!iteration_indptr.empty());
  CHECK_EQ(iteration_indptr.back(), param.num_trees);

  bst_tree_t n_new_trees = 0;
  bst_target_t n_groups = learner_model_state->OutputLength();
  CHECK_EQ(new_trees.size(), static_cast<std::size_t>(n_groups))
      << "TreeGOneIter must have exactly n_groups (" << n_groups << ") groups, got "
      << new_trees.size();

  for (bst_target_t gidx = 0; gidx < n_groups; ++gidx) {
    n_new_trees += static_cast<bst_tree_t>(new_trees[gidx].size());
    CommitModelGroup(std::move(new_trees[gidx]), gidx);
  }

  iteration_indptr.push_back(n_new_trees + iteration_indptr.back());

  CHECK_EQ(param.num_trees, static_cast<std::int32_t>(trees.size()));
  return n_new_trees;
}

void TreeGBoosterModel::CommitModelGroup(TreeGOneGroup&& new_trees, bst_target_t group_idx) {
  for (auto& new_tree : new_trees) {
    trees.push_back(std::move(new_tree));
    tree_info.push_back(group_idx);
  }
  param.num_trees += static_cast<std::int32_t>(new_trees.size());
}

// ========== TreeGBooster Implementation ==========

TreeGBooster::TreeGBooster(LearnerModelState const* booster_config, Context const* ctx)
    : GradientBooster{ctx}, model_{booster_config, ctx_} {}

std::set<std::string> TreeGBooster::Configure(Args const& cfg) {
  // Histograms and attention subsets are computed over local rows only and never allreduced,
  // so distributed training would silently produce independent per-worker trees. Every rank
  // runs Configure and fails here symmetrically, so no rank is left waiting in a collective.
  CHECK(!collective::IsDistributed()) << kNoDistributed;

  // Graph shape depends on the parameters below; drop graphs built under the old ones.
  cache_.Clear();

  auto used = UpdateAndGetUsedParameters(&bparam_, cfg);
  used.merge(UpdateAndGetUsedParameters(&tparam_, cfg));
  used.merge(model_.Configure(cfg));

  // The graph shape is stored twice: in tparam_ (read by the updater) and in model_.param
  // (persisted with the trees). A trained or loaded model is authoritative, because its trees
  // were built for that shape. This also undoes XGBoostParameter resetting absent keys to
  // their defaults when the Learner re-configures after load_model with no Tree-G keys.
  // Before training, the user configuration is authoritative.
  if (!model_.trees.empty()) {
    CopyGraphShape(model_.param, &tparam_);
  } else {
    CopyGraphShape(tparam_, &model_.param);
  }

  CHECK(ctx_->IsCPU()) << kCpuOnlyTraining;

  // process_type is kept for gbtree compatibility. A correct "update" (refresh/prune) would
  // need to re-route every vertex through the stored attention subsets; it is not implemented.
  CHECK_EQ(bparam_.process_type, "default") << kOnlyDefaultProcessType;

  // Initialize updater if not yet created
  if (!updater_) {
    updater_ = std::make_unique<TreeGUpdater>(ctx_);
  }

  // Pass the parsed struct directly: a string round-trip would truncate floats.
  updater_->Configure(tparam_);

  return used;
}

void TreeGBooster::DoBoost(std::shared_ptr<DMatrix> p_fmat, GradientContainer* in_gpair,
                           ObjFunction const* /*obj*/) {
  CHECK(p_fmat);
  CHECK_NOTNULL(in_gpair);
  CHECK_NOTNULL(updater_);

  // Tree-G does not support multi-output-tree (vector leaf) strategy.
  // Use default one-vs-all (scalar leaf) for multi-class problems.
  CHECK(!model_.learner_model_state->IsVectorLeaf()) << kNoMultiOutputTree;

  // Validate input dimensions
  auto num_rows = p_fmat->Info().num_row_;
  CHECK_EQ(num_rows, tparam_.num_vertices)
      << "DMatrix has " << num_rows << " rows, but num_vertices is " << tparam_.num_vertices;

  bst_target_t n_groups = model_.learner_model_state->OutputLength();
  CHECK_NE(n_groups, 0);

  // Build graph data once per training DMatrix, shared across all rounds/groups/trees.
  auto p_entry = cache_.CacheItem(p_fmat);
  if (!p_entry->graph_built) {
    p_entry->graph =
        GraphData::Build(p_fmat.get(), model_.param.num_vertices, model_.param.num_features,
                         model_.param.max_walk_len, model_.param.intercept);
    p_entry->graph_built = true;
    ++num_graph_builds_;
  }
  GraphData const& graph_data = p_entry->graph;

  // The cached margin may only be advanced if it covers exactly the committed rounds;
  // otherwise drop it and let the next PredictBatch rebuild it from scratch. A base margin is
  // never cached (PredictBatch does not reuse it, and it may be changed by the user).
  auto& cached_preds = p_entry->predictions;
  bool update_cache = p_fmat->Info().base_margin_.Empty() &&
                      p_entry->version == static_cast<std::uint32_t>(model_.BoostedRounds());
  if (update_cache && cached_preds.Empty()) {
    InitOutPredictions(p_fmat->Info(), model_.learner_model_state, &cached_preds);
  }
  if (!update_cache) {
    cached_preds.Resize(0);
    p_entry->version = 0;
  }
  std::vector<bst_node_t> position;

  // Build one tree per group (class) per boosting iteration.
  TreeGOneIter new_trees;

  for (bst_target_t gid = 0; gid < n_groups; ++gid) {
    // Extract per-group gradient pairs.
    // The gradient matrix has shape [n_samples, n_groups].  Slice column gid.
    std::vector<GradientPair> gpairs(num_rows);
    auto h_view = in_gpair->gpair.HostView();
    auto group_view = h_view.Slice(linalg::All(), gid);
    for (std::size_t i = 0; i < num_rows; ++i) {
      gpairs[i] = group_view(i);
    }

    // Create and train tree(s) for this group
    TreeGOneGroup ret;
    std::vector<RegTreeG*> ptrs = InitNewTrees(&ret);
    for (auto* tree_ptr : ptrs) {
      updater_->Update(gpairs, graph_data, tree_ptr, update_cache ? &position : nullptr);
      if (update_cache) {
        // One add per vertex per tree, in tree order: same sum order as a fresh prediction.
        auto& h_preds = cached_preds.HostVector();
        RegTreeG const& tree = *tree_ptr;
        common::ParallelFor(num_rows, ctx_->Threads(), [&](std::size_t v) {
          h_preds[v * n_groups + gid] += tree[position[v]].LeafValue();
        });
      }
    }
    new_trees.push_back(std::move(ret));
  }

  // Commit trees to model (tags each tree with its group index)
  CommitModel(std::move(new_trees));
  if (update_cache) {
    p_entry->version += 1;
  }
}

std::vector<RegTreeG*> TreeGBooster::InitNewTrees(TreeGOneGroup* ret) {
  std::vector<RegTreeG*> ptrs;
  ret->clear();

  // Currently only support num_parallel_tree = 1
  CHECK_EQ(model_.param.num_parallel_tree, 1) << kOnlySingleParallelTree;

  auto tree =
      std::make_unique<RegTreeG>(tparam_.num_vertices, tparam_.num_features, tparam_.max_walk_len,
                                 tparam_.max_ancestor_distance, tparam_.intercept);
  ptrs.push_back(tree.get());
  ret->push_back(std::move(tree));

  return ptrs;
}

void TreeGBooster::CommitModel(TreeGOneIter&& new_trees) {
  model_.CommitModel(std::forward<TreeGOneIter>(new_trees));
}

void TreeGBooster::PredictBatchImpl(std::shared_ptr<DMatrix> const& p_fmat,
                                    HostDeviceVector<float>* out_preds, bst_layer_t layer_begin,
                                    bst_layer_t layer_end) const {
  CHECK_NOTNULL(p_fmat);
  CHECK_NOTNULL(out_preds);

  auto const n = static_cast<VertexIdx>(p_fmat->Info().num_row_);
  CHECK_EQ(n, model_.param.num_vertices) << "DMatrix has " << n << " rows, but model expects "
                                         << model_.param.num_vertices << " vertices.";

  bst_target_t n_groups = model_.learner_model_state->OutputLength();

  // Resolve layer range → tree range
  if (layer_end == 0) {
    layer_end = model_.BoostedRounds();
  }
  CHECK_GE(layer_begin, 0);
  CHECK_LE(layer_end, model_.BoostedRounds());
  CHECK_LE(layer_begin, layer_end);

  bst_tree_t tree_begin = model_.iteration_indptr[layer_begin];
  bst_tree_t tree_end = model_.iteration_indptr[layer_end];
  CHECK_LE(tree_end, static_cast<bst_tree_t>(model_.trees.size()));

  // Initialize output: interleaved layout [n_vertices, n_groups]
  // preds[v * n_groups + g] is the raw prediction for vertex v, group g.
  // `InitOutPredictions` seeds the buffer with the base margin, falling back to
  // the intercept when `base_margin_` is absent from the MetaInfo.
  InitOutPredictions(p_fmat->Info(), model_.learner_model_state, out_preds);
  auto& h_preds = out_preds->HostVector();

  // Reuse the training graph if this DMatrix is cached; otherwise build a transient one.
  // `intercept` only affects a warning in Build, so the cached graph is identical.
  GraphData local_graph;
  GraphData const* p_graph = nullptr;
  if (auto const* entry = cache_.Find(p_fmat.get()); entry != nullptr && entry->graph_built) {
    p_graph = &entry->graph;
  } else {
    local_graph = GraphData::Build(p_fmat.get(), model_.param.num_vertices,
                                   model_.param.num_features, model_.param.max_walk_len);
    p_graph = &local_graph;
  }
  GraphData const& graph_data = *p_graph;

  // Accumulate predictions from each tree, routed by tree_info group.
  for (bst_tree_t t = tree_begin; t < tree_end; ++t) {
    bst_target_t gid = model_.tree_info[t];
    PredictSingleTree(*model_.trees[t], graph_data, &h_preds, n_groups, gid);
  }
}

void TreeGBooster::PredictBatch(std::shared_ptr<DMatrix> dmat, HostDeviceVector<float>* out_preds,
                                bool /*training*/, bst_layer_t layer_begin, bst_layer_t layer_end) {
  CHECK(ctx_->IsCPU()) << kCpuOnlyPrediction;

  // If no trees yet, the output is just the base margin.
  if (model_.trees.empty()) {
    InitOutPredictions(dmat->Info(), model_.learner_model_state, out_preds);
    return;
  }

  // gbtree-style prediction cache (training DMatrices only; prediction never inserts).
  auto* entry = cache_.Find(dmat.get());
  bool const preserve_cache =
      entry != nullptr && layer_begin == 0 && dmat->Info().base_margin_.Empty();
  if (!preserve_cache) {
    // Layer slices and base-margin requests are computed from scratch; the cache stays valid.
    PredictBatchImpl(dmat, out_preds, layer_begin, layer_end);
    return;
  }

  if (layer_end == 0) {
    layer_end = model_.BoostedRounds();
  }
  CHECK_LE(layer_end, model_.BoostedRounds());
  auto const n = static_cast<VertexIdx>(dmat->Info().num_row_);
  CHECK_EQ(n, model_.param.num_vertices) << "DMatrix has " << n << " rows, but model expects "
                                         << model_.param.num_vertices << " vertices.";
  CHECK(entry->graph_built);

  auto version = static_cast<bst_layer_t>(entry->version);
  bool const reuse_cache = !entry->predictions.Empty() && layer_end >= version;
  if (!reuse_cache) {
    InitOutPredictions(dmat->Info(), model_.learner_model_state, &entry->predictions);
    version = 0;
  }

  bst_target_t const n_groups = model_.learner_model_state->OutputLength();
  auto& h_preds = entry->predictions.HostVector();
  for (bst_tree_t t = model_.iteration_indptr[version]; t < model_.iteration_indptr[layer_end];
       ++t) {
    PredictSingleTree(*model_.trees[t], entry->graph, &h_preds, n_groups, model_.tree_info[t]);
  }
  entry->version = static_cast<std::uint32_t>(layer_end);

  out_preds->Resize(entry->predictions.Size());
  out_preds->Copy(entry->predictions);
}

// ========== Two-Phase Prediction Helpers ==========

void TreeGBooster::PredictSingleTree(const RegTreeG& tree, const GraphData& graph_data,
                                     std::vector<float>* preds, bst_target_t n_groups,
                                     bst_target_t group_id) const {
  CHECK_NOTNULL(preds);
  VertexIdx n = model_.param.num_vertices;
  CHECK_EQ(static_cast<std::size_t>(preds->size()), static_cast<std::size_t>(n) * n_groups);

  // Handle trivial case: single root leaf
  if (tree[RegTreeG::kRoot].IsLeaf()) {
    float leaf_val = tree[RegTreeG::kRoot].LeafValue();
    for (VertexIdx v = 0; v < n; ++v) {
      (*preds)[static_cast<std::size_t>(v) * n_groups + group_id] += leaf_val;
    }
    return;
  }

  // =========== Phase 1: Reconstruct the attention subsets ===========
  // Rebuilds S_{u,+}/S_{u,-} and phi_u for every split-node, sharing its definition
  // with training so that both route vertices identically.
  std::vector<GeneratedSubsets> subsets;
  std::vector<std::vector<float>> split_values;
  ReconstructSubsets(tree, graph_data, &subsets, &split_values, ctx_->Threads());

  // =========== Phase 2: Per-Vertex Tree Traversal ===========
  // Each vertex independently walks root→leaf using cached split_values.
  // Predictions are written to the interleaved buffer at [v * n_groups + group_id];
  // each v writes only its own slot, so the parallel loop is race-free and bit-identical.
  common::ParallelFor(n, ctx_->Threads(), [&](VertexIdx v) {
    bst_node_t nid = RegTreeG::kRoot;
    while (!tree[nid].IsLeaf()) {
      float threshold = tree[nid].SplitCond();

      if (split_values[nid][v] <= threshold) {
        nid = tree[nid].LeftChild();
      } else {
        nid = tree[nid].RightChild();
      }
    }
    (*preds)[static_cast<std::size_t>(v) * n_groups + group_id] += tree[nid].LeafValue();
  });
}

void TreeGBooster::PredictLeaf(DMatrix*, HostDeviceVector<bst_float>*, bst_layer_t, bst_layer_t,
                               bool) {
  LOG(FATAL) << kNoPredictLeaf;
}

void TreeGBooster::PredictContribution(DMatrix*, HostDeviceVector<float>*, bst_layer_t, bst_layer_t,
                                       bool) {
  LOG(FATAL) << kNoPredictContribution;
}

void TreeGBooster::PredictInteractionContributions(DMatrix*, HostDeviceVector<float>*, bst_layer_t,
                                                   bst_layer_t, bool) {
  LOG(FATAL) << kNoPredictInteraction;
}

std::vector<std::string> TreeGBooster::DumpModel(FeatureMap const& fmap, bool with_stats,
                                                 std::string format) const {
  return model_.DumpModel(fmap, with_stats, ctx_->Threads(), format);
}

void TreeGBooster::FeatureScore(std::string const& importance_type,
                                common::Span<std::int32_t const> trees,
                                std::vector<bst_feature_t>* features,
                                std::vector<float>* scores) const {
  bool const is_weight = importance_type == "weight";
  bool const is_gain = importance_type == "gain" || importance_type == "total_gain";
  bool const is_cover = importance_type == "cover" || importance_type == "total_cover";
  if (!(is_weight || is_gain || is_cover)) {
    LOG(FATAL) << "Unknown feature importance type, expected one of: "
               << R"({"weight", "total_gain", "total_cover", "gain", "cover"}, got: )"
               << importance_type;
  }

  // Only real DMatrix columns [0, num_features) are scored. The virtual intercept
  // (split index == num_features) has no column, so it is skipped.
  bst_feature_t const n_features = model_.param.num_features;
  std::vector<std::size_t> split_counts(n_features, 0);
  std::vector<double> gain_map(n_features, 0.0);

  std::vector<std::int32_t> tree_idx;
  if (trees.empty()) {
    tree_idx.resize(model_.trees.size());
    std::iota(tree_idx.begin(), tree_idx.end(), 0);
    trees = common::Span<std::int32_t const>(tree_idx);
  }

  auto total_n_trees = model_.trees.size();

  for (auto idx : trees) {
    CHECK_GE(idx, 0) << "Invalid tree index: " << idx;
    CHECK_LT(idx, static_cast<std::int32_t>(total_n_trees)) << "Invalid tree index: " << idx;
    auto const& tree = *model_.trees[idx];

    for (bst_node_t nidx = 0; nidx < tree.NumNodes(); ++nidx) {
      auto const& node = tree[nidx];
      if (node.IsLeaf()) {
        continue;
      }
      auto const split_idx = node.SplitIndex();
      if (split_idx >= n_features) {
        continue;  // intercept split
      }
      split_counts[split_idx]++;
      if (is_gain) {
        gain_map[split_idx] += tree.Stat(nidx).loss_chg;
      } else if (is_cover) {
        gain_map[split_idx] += tree.Stat(nidx).sum_hess;
      }
    }
  }

  bool const is_mean = importance_type == "gain" || importance_type == "cover";
  features->clear();
  scores->clear();
  for (bst_feature_t i = 0; i < n_features; ++i) {
    if (split_counts[i] == 0) {
      continue;
    }
    double score = gain_map[i];
    if (is_weight) {
      score = static_cast<double>(split_counts[i]);
    } else if (is_mean) {
      score /= static_cast<double>(split_counts[i]);
    }
    features->push_back(i);
    scores->push_back(static_cast<float>(score));
  }
}

void TreeGBooster::LoadConfig(Json const& in) {
  CHECK_EQ(get<String>(in["name"]), "treeg");
  FromJson(in["treeg_booster_train_param"], &bparam_);
  FromJson(in["treeg_train_param"], &tparam_);

  // Only process_type=default is supported.
  bparam_.process_type = "default";
}

void TreeGBooster::SaveConfig(Json* p_out) const {
  auto& out = *p_out;
  out["name"] = String("treeg");
  out["treeg_booster_train_param"] = ToJson(bparam_);
  out["treeg_train_param"] = ToJson(tparam_);
}

void TreeGBooster::SaveModel(Json* p_out) const {
  auto& out = *p_out;
  out["name"] = String("treeg");
  out["model"] = Object{};
  model_.SaveModel(&out["model"]);
}

void TreeGBooster::LoadModel(Json const& in) {
  CHECK_EQ(get<String>(in["name"]), "treeg");
  model_.LoadModel(in["model"]);
  cache_.Clear();
  CopyGraphShape(model_.param, &tparam_);
}

}  // namespace xgboost::treeg

// Register Tree-G booster
XGBOOST_REGISTER_GBM(TreeGBooster, "treeg")
    .describe("Tree-G booster for graph-aware decision trees.")
    .set_body([](xgboost::LearnerModelState const* booster_config, xgboost::Context const* ctx) {
      return new xgboost::treeg::TreeGBooster{booster_config, ctx};
    });
