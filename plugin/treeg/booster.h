/**
 * Copyright 2026, XGBoost contributors
 *
 * @brief Gradient booster for Tree-G (Graph-Aware Decision Trees)
 */
#ifndef PLUGIN_TREEG_BOOSTER_H_
#define PLUGIN_TREEG_BOOSTER_H_

#include <xgboost/base.h>   // for bst_float, bst_node_t, bst_feature_t, bst_target_t, bst_tree_t
#include <xgboost/cache.h>  // for DMatrixCache
#include <xgboost/data.h>   // for DMatrix
#include <xgboost/feature_map.h>         // for FeatureMap
#include <xgboost/gbm.h>                 // for GradientBooster
#include <xgboost/host_device_vector.h>  // for HostDeviceVector
#include <xgboost/learner.h>             // for LearnerModelState
#include <xgboost/logging.h>             // for CHECK, LOG
#include <xgboost/model.h>               // for Model
#include <xgboost/parameter.h>           // for XGBoostParameter

#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t, uint32_t
#include <memory>   // for unique_ptr, shared_ptr
#include <set>      // for set
#include <string>   // for string
#include <thread>   // for this_thread
#include <vector>   // for vector

#include "model.h"    // for RegTreeG
#include "updater.h"  // for TreeGUpdater, TreeGTrainParam

namespace xgboost {

class Json;
struct Context;
struct LearnerModelState;

namespace treeg {

/**
 * @brief Per-DMatrix state cached across boosting rounds (training matrices only).
 *
 * `predictions` holds the raw margin of the first `version` boosting rounds (gbtree's
 * `PredictionCacheEntry` layout). It is advanced by `DoBoost` from the updater's leaf
 * positions, so training-time prediction does not re-run any tree.
 */
struct TreeGCacheEntry {
  GraphData graph;
  bool graph_built{false};
  HostDeviceVector<float> predictions;
  std::uint32_t version{0};
};

/**
 * @brief DMatrix-keyed cache, sized like gbtree's `PredictionContainer`.
 *
 * Not thread-safe for concurrent training and prediction on the same booster, which
 * XGBoost does not support anyway.
 */
class TreeGCache : public DMatrixCache<TreeGCacheEntry> {
  static constexpr std::size_t DefaultSize() { return 64; }

 public:
  TreeGCache() : DMatrixCache<TreeGCacheEntry>{DefaultSize()} {}

  /** @brief Return the live entry for `m` on this thread, or nullptr. Never inserts. */
  [[nodiscard]] TreeGCacheEntry const* Find(DMatrix const* m) const {
    auto it = container_.find(Key{m, std::this_thread::get_id()});
    if (it == container_.cend() || it->second.ref.expired()) {
      return nullptr;
    }
    return it->second.value.get();
  }

  /** @brief Mutable overload of Find(). Never inserts. */
  [[nodiscard]] TreeGCacheEntry* Find(DMatrix const* m) {
    auto it = container_.find(Key{m, std::this_thread::get_id()});
    if (it == container_.end() || it->second.ref.expired()) {
      return nullptr;
    }
    return it->second.value.get();
  }

  /** @brief Drop every entry. */
  void Clear() {
    container_.clear();
    queue_ = {};
  }
};

/**
 * @brief Container for Tree-G trees built for one group (class) in an iteration.
 *
 * For single-target, each iteration has one group with one tree.
 * For multi-class one-vs-all (num_parallel_tree=1), each group has one tree.
 */
using TreeGOneGroup = std::vector<std::unique_ptr<RegTreeG>>;

/**
 * @brief Container for all Tree-G trees built in one boosting iteration.
 *
 * Indexed by group (class) index. For single-target, size is 1.
 * For multi-class one-vs-all (e.g., multi:softmax), size equals num_classes.
 */
using TreeGOneIter = std::vector<TreeGOneGroup>;

/**
 * @brief Model parameters for TreeGBooster.
 *
 * Manages model-level configuration including tree counts, graph topology,
 * and neighborhood aggregation settings.
 */
struct TreeGBoosterModelParam : public dmlc::Parameter<TreeGBoosterModelParam> {
  /** @brief Total number of trees in the model */
  std::int32_t num_trees{0};

  /** @brief Number of parallel trees per iteration (currently 1 for Tree-G) */
  std::int32_t num_parallel_tree{1};

  /** @brief Maximum walk length for neighborhood aggregation (d_max) */
  WalkLen max_walk_len{2};

  /** @brief Maximum ancestor distance for attention mechanism (a_max) */
  WalkLen max_ancestor_distance{2};

  /** @brief Number of original node features (k) in the input */
  bst_feature_t num_features{0};

  /** @brief Number of vertices in the graph */
  VertexIdx num_vertices{0};

  /** @brief Whether a virtual constant-1 intercept feature is enabled */
  bool intercept{false};

  TreeGBoosterModelParam() = default;

  DMLC_DECLARE_PARAMETER(TreeGBoosterModelParam) {
    DMLC_DECLARE_FIELD(num_trees).set_lower_bound(0).set_default(0).describe(
        "Total number of trees in the model.");
    DMLC_DECLARE_FIELD(num_parallel_tree)
        .set_lower_bound(1)
        .set_default(1)
        .describe("Number of parallel trees per iteration (currently 1 for Tree-G).");
    DMLC_DECLARE_FIELD(max_walk_len)
        .set_lower_bound(0)
        .set_default(2)
        .describe("Maximum walk length for neighborhood aggregation.");
    DMLC_DECLARE_FIELD(max_ancestor_distance)
        .set_lower_bound(0)
        .set_default(2)
        .describe("Maximum ancestor distance for attention mechanism.");
    DMLC_DECLARE_FIELD(num_features)
        .set_lower_bound(0)
        .set_default(0)
        .describe("Number of original node features in the input.");
    DMLC_DECLARE_FIELD(num_vertices)
        .set_lower_bound(0)
        .set_default(0)
        .describe("Number of vertices in the graph.");
    DMLC_DECLARE_FIELD(intercept).set_default(false).describe(
        "Add a virtual constant-1 intercept feature for graph-aware splits.");
  }
};

/**
 * @brief Model container for Tree-G gradient booster.
 *
 * Manages the collection of RegTreeG trees and model metadata.
 * Separates model state from training logic for clean serialization.
 */
struct TreeGBoosterModel : public Model {
  /**
   * @brief Construct a TreeGBooster model.
   * @param learner_model Global learner model parameters
   * @param ctx Execution context
   */
  explicit TreeGBoosterModel(LearnerModelState const* learner_model, Context const* ctx)
      : learner_model_state{learner_model}, ctx_{ctx} {}

  /**
   * @brief Configure the model with parameters.
   * @param cfg Configuration arguments
   * @return The set of parameter keys consumed by this model.
   */
  std::set<std::string> Configure(Args const& cfg) {
    if (trees.empty()) {
      return UpdateAndGetUsedParameters(&param, cfg);
    }
    return {};
  }

  /**
   * @brief Save model to JSON.
   * @param p_out Output JSON object
   */
  void SaveModel(Json* p_out) const override;

  /**
   * @brief Load model from JSON.
   * @param in Input JSON object
   */
  void LoadModel(Json const& in) override;

  /**
   * @brief Dump model to human-readable format.
   * @param fmap Feature map for feature name resolution
   * @param with_stats Whether to include statistics in the dump
   * @param n_threads Number of threads for parallel dumping
   * @param format Output format ("text" or "json")
   * @return Vector of string representations for each tree
   */
  [[nodiscard]] std::vector<std::string> DumpModel(FeatureMap const& fmap, bool with_stats,
                                                   std::int32_t n_threads,
                                                   std::string const& format) const;

  /**
   * @brief Add trees to the model after training iteration (multi-group aware).
   *
   * For single-target, new_trees has one group. For multi-class one-vs-all,
   * new_trees has one group per class. Each tree is tagged with its group index
   * in tree_info for correct prediction routing.
   *
   * @param new_trees Trees to add, indexed by group
   * @return Number of trees added
   */
  bst_tree_t CommitModel(TreeGOneIter&& new_trees);

  /**
   * @brief Add trees for a specific group/class.
   * @param new_trees Trees to add
   * @param group_idx Group/class index (0 for single-target)
   */
  void CommitModelGroup(TreeGOneGroup&& new_trees, bst_target_t group_idx);

  /**
   * @brief Get the number of boosting rounds (iterations).
   * @return Number of completed boosting rounds
   */
  [[nodiscard]] std::int32_t BoostedRounds() const {
    if (trees.empty()) {
      CHECK_EQ(iteration_indptr.size(), 1);
    }
    return static_cast<std::int32_t>(iteration_indptr.size() - 1);
  }

  /** @brief Global learner model parameters */
  LearnerModelState const* learner_model_state;

  /** @brief Tree-G model parameters */
  TreeGBoosterModelParam param;

  /** @brief Collection of trained Tree-G trees */
  std::vector<std::unique_ptr<RegTreeG>> trees;

  /**
   * @brief Group index for each tree (maps tree → class for multi-class).
   *
   * tree_info[i] is the class/group that tree i predicts for.
   * For single-target, all entries are 0.
   */
  std::vector<bst_target_t> tree_info;

  /**
   * @brief Iteration boundaries.
   *
   * iteration_indptr[i] is the starting tree index for iteration i.
   * Size is BoostedRounds() + 1.
   */
  std::vector<bst_tree_t> iteration_indptr{0};

  /** @brief Execution context */
  [[nodiscard]] Context const* Ctx() const { return ctx_; }

 private:
  Context const* ctx_;
};

/**
 * @brief Booster-level parameters that are not passed to the updater.
 *
 * Mirrors `gbm::GBTreeTrainParam`. All tree-growth and graph parameters live in
 * `TreeGTrainParam`; the trained shape (`num_parallel_tree`, graph dimensions) lives in
 * `TreeGBoosterModelParam`.
 */
struct TreeGBoosterTrainParam : public XGBoostParameter<TreeGBoosterTrainParam> {
  /** @brief Process type. Kept for gbtree compatibility; only "default" is supported. */
  std::string process_type{"default"};

  DMLC_DECLARE_PARAMETER(TreeGBoosterTrainParam) {
    DMLC_DECLARE_FIELD(process_type)
        .set_default("default")
        .describe("Boosting process type. Tree-G supports only 'default'.");
  }
};

/**
 * @brief Gradient booster for Tree-G (Graph-Aware Decision Trees).
 *
 * TreeGBooster implements ensemble learning for graph-structured data using
 * the Tree-G algorithm. It manages a collection of RegTreeG trees that
 * incorporate graph topology through:
 * - Neighborhood aggregation features (A, A², ..., A^d)
 * - Graph-aware split functions with attention mechanism
 * - Active set propagation for efficient node masking
 *
 * Key differences from GBTree:
 * - Uses RegTreeG instead of RegTree
 * - Custom prediction that evaluates graph-aware split functions
 * - Single updater type (TreeGUpdater) using exact splits
 * - Assumes transductive learning (full graph visible during training)
 *
 * Expected input format: [X | A | A² | ... | A^d]
 * - X: Original node features (n × k)
 * - A^i: i-th power of adjacency matrix (n × n flattened)
 * - Labels: (n,) with dummy values for test nodes
 * - Weights: 0 for test nodes, 1 for training nodes
 */
class TreeGBooster : public GradientBooster {
 public:
  /**
   * @brief Construct a TreeGBooster.
   * @param booster_config Global learner model parameters
   * @param ctx Execution context (CPU only for Tree-G)
   */
  explicit TreeGBooster(LearnerModelState const* booster_config, Context const* ctx);

  /**
   * @brief Configure the booster with training parameters.
   * @param cfg Configuration arguments
   * @return The set of parameter keys consumed by this booster and its updater.
   */
  std::set<std::string> Configure(Args const& cfg) override;

  /**
   * @brief Perform one boosting iteration.
   *
   * Trains a new RegTreeG tree using TreeGUpdater and adds it to the model.
   *
   * @param p_fmat Training data matrix with format [X | A | A² | ... | A^d]
   * @param in_gpair Gradient statistics (gradient and hessian)
   * @param obj Objective function
   */
  void DoBoost(std::shared_ptr<DMatrix> p_fmat, GradientContainer* in_gpair,
               ObjFunction const* obj) override;

  /**
   * @brief Generate predictions for given data.
   *
   * Implements graph-aware prediction by:
   * 1. Traversing each RegTreeG tree
   * 2. Dynamically evaluating active sets at split nodes
   * 3. Computing graph-aware split values using adjacency matrices
   *
   * Training DMatrices keep a gbtree-style prediction cache: `DoBoost` adds each new
   * tree's leaf values from the updater's leaf positions, so during training this call
   * predicts no tree at all. When `layer_begin == 0` and there is no `base_margin`, only
   * rounds `[version, layer_end)` are predicted on top of the cached margin (the cache is
   * rebuilt if `layer_end < version`). Other requests, and DMatrices that were never
   * trained on, recompute from the base margin and leave the cache untouched.
   *
   * @param dmat Input data matrix with format [X | A | A² | ... | A^d]
   * @param out_preds Output predictions
   * @param training Whether prediction is for training (unused in Tree-G)
   * @param layer_begin Starting boosting round
   * @param layer_end Ending boosting round (0 = all rounds)
   */
  void PredictBatch(std::shared_ptr<DMatrix> dmat, HostDeviceVector<float>* out_preds,
                    bool training, bst_layer_t layer_begin, bst_layer_t layer_end) override;

  /**
   * @brief Leaf prediction. Not implemented for Tree-G; always raises an error.
   */
  void PredictLeaf(DMatrix* dmat, HostDeviceVector<bst_float>* out_preds, bst_layer_t layer_begin,
                   bst_layer_t layer_end, bool strict_shape) override;

  /**
   * @brief Compute feature contributions (SHAP values).
   *
   * Note: Tree-G specific SHAP computation is not yet implemented.
   *
   * @param dmat Input data matrix
   * @param out_contribs Output contributions
   * @param layer_begin Starting boosting round
   * @param layer_end Ending boosting round
   * @param approximate Use approximate SHAP computation
   */
  void PredictContribution(DMatrix* dmat, HostDeviceVector<float>* out_contribs,
                           bst_layer_t layer_begin, bst_layer_t layer_end,
                           bool approximate) override;

  /**
   * @brief Compute interaction contributions.
   *
   * Note: Not yet implemented for Tree-G.
   *
   * @param dmat Input data matrix
   * @param out_contribs Output contributions
   * @param layer_begin Starting boosting round
   * @param layer_end Ending boosting round
   * @param approximate Use approximate computation
   */
  void PredictInteractionContributions(DMatrix* dmat, HostDeviceVector<float>* out_contribs,
                                       bst_layer_t layer_begin, bst_layer_t layer_end,
                                       bool approximate) override;

  /**
   * @brief Dump model in human-readable format.
   *
   * @param fmap Feature map for feature name resolution
   * @param with_stats Include split statistics in output
   * @param format Output format ("text" or "json")
   * @return Vector of string representations for each tree
   */
  [[nodiscard]] std::vector<std::string> DumpModel(FeatureMap const& fmap, bool with_stats,
                                                   std::string format) const override;

  /**
   * @brief Compute feature importance scores aggregated over graph neighbourhoods.
   *
   * A Tree-G split on base feature k tests (A^d_[S] x[:,k])_v <= t, i.e. feature k aggregated
   * over walks of length d (restriction type r) into the attention subset S. For
   * k in [0, num_features), with N_k the split nodes whose split index is k (summed over every
   * d, r and attention source):
   *  - weight = |N_k|;
   *  - total_gain = sum of loss_chg over N_k, gain = its mean;
   *  - total_cover = sum of sum_hess over N_k (hessian of the labelled vertices reaching the
   *    node, not of the vertices in S that supply the value), cover = its mean.
   *
   * Splits on the virtual intercept (split index == num_features) are excluded: the intercept
   * has no DMatrix column, so every returned index is a real column in [0, num_features).
   *
   * @param importance_type Type of importance: "weight", "gain", "cover", "total_gain", "total_cover"
   * @param trees Indices of trees to include (empty = all trees)
   * @param features Output feature indices
   * @param scores Output importance scores
   */
  void FeatureScore(std::string const& importance_type, common::Span<std::int32_t const> trees,
                    std::vector<bst_feature_t>* features,
                    std::vector<float>* scores) const override;

  /**
   * @brief Load configuration from JSON.
   * @param in Input JSON object
   */
  void LoadConfig(Json const& in) override;

  /**
   * @brief Save configuration to JSON.
   * @param p_out Output JSON object
   */
  void SaveConfig(Json* p_out) const override;

  /**
   * @brief Save model to JSON.
   * @param p_out Output JSON object
   */
  void SaveModel(Json* p_out) const override;

  /**
   * @brief Load model from JSON.
   * @param in Input JSON object
   */
  void LoadModel(Json const& in) override;

  /**
   * @brief Get number of boosted rounds.
   * @return Number of completed boosting rounds
   */
  [[nodiscard]] std::int32_t BoostedRounds() const override { return model_.BoostedRounds(); }

  /**
   * @brief Get training parameters.
   * @return Current training parameters
   */
  [[nodiscard]] TreeGTrainParam const& GetTrainParam() const { return tparam_; }

  /**
   * @brief Access the trained ensemble. Test-only.
   *
   * Exposed so tests can assert that predictions agree with the trees that produced them,
   * which is not observable through the GradientBooster interface alone.
   *
   * @return The booster's model.
   */
  [[nodiscard]] TreeGBoosterModel const& ModelForTesting() const { return model_; }

  /**
   * @brief Access the configured updater. Test-only.
   * @return The updater, or nullptr before the first Configure().
   */
  [[nodiscard]] TreeGUpdater const* UpdaterForTesting() const { return updater_.get(); }

 protected:
  /**
   * @brief Core prediction implementation.
   *
   * Evaluates graph-aware splits for each vertex by:
   * 1. Computing split values using adjacency matrices
   * 2. Following tree paths based on active sets
   * 3. Accumulating leaf weights
   *
   * @param p_fmat Input data matrix
   * @param out_preds Output predictions
   * @param layer_begin Starting boosting round
   * @param layer_end Ending boosting round
   */
  void PredictBatchImpl(std::shared_ptr<DMatrix> const& p_fmat, HostDeviceVector<float>* out_preds,
                        bst_layer_t layer_begin, bst_layer_t layer_end) const;

  /**
   * @brief Predict all vertex leaf values for a single tree (two-phase).
   *
   * Phase 1 (subset reconstruction):
   *   - `ReconstructSubsets` walks the split-nodes in BFS order, resolving each
   *     node's used subset S_{*u,rho_u} from its ancestor pointer, computing
   *     phi_u for every vertex, and partitioning S into its attention subsets.  Training
   *     uses the same functions, so both agree by construction.
   *
   * Phase 2 (per-vertex tree traversal):
   *   - Each vertex independently walks from root to leaf using the cached
   *     split values to decide left/right at every split node.
   *
   * @param tree       The trained RegTreeG tree
   * @param graph_data Graph the tree is applied to
   * @param[in,out] preds  Prediction vector (length n * n_groups); leaf values
   *                       are *added* at positions [v * n_groups + group_id].
   * @param n_groups  Number of output groups (1 for single-target)
   * @param group_id  Group/class this tree predicts for
   */
  void PredictSingleTree(const RegTreeG& tree, const GraphData& graph_data,
                         std::vector<float>* preds, bst_target_t n_groups,
                         bst_target_t group_id) const;

  /**
   * @brief Initialize new trees for one group in a boosting iteration.
   *
   * Creates num_parallel_tree trees and places them into ret.
   * Returns raw pointers for the updater to work with.
   *
   * @param[out] ret Output group to receive the new trees
   * @return Vector of raw pointers to the new trees
   */
  [[nodiscard]] std::vector<RegTreeG*> InitNewTrees(TreeGOneGroup* ret);

  /**
   * @brief Commit new trees to the model after training.
   * @param new_trees Trees to commit
   */
  virtual void CommitModel(TreeGOneIter&& new_trees);

  // --- Data members ---

  /** @brief Tree-G model containing trained trees */
  TreeGBoosterModel model_;

  /** @brief Booster-level parameters (process_type). */
  TreeGBoosterTrainParam bparam_;

  /** @brief Tree-growth and graph parameters, shared with the updater. */
  TreeGTrainParam tparam_;

  /** @brief Tree updater for building RegTreeG trees */
  std::unique_ptr<TreeGUpdater> updater_;

  /** @brief Per-training-DMatrix cache (graph built once per matrix). */
  mutable TreeGCache cache_;

  /** @brief Number of training-graph builds; observable for tests. */
  std::size_t num_graph_builds_{0};

 public:
  /** @brief Number of times DoBoost built a GraphData (for tests). */
  [[nodiscard]] std::size_t NumGraphBuilds() const { return num_graph_builds_; }

  /**
   * @brief Number of boosting rounds held in the prediction cache of `m`. Test-only.
   * @return The cached `version`, or 0 if `m` has no cache entry.
   */
  [[nodiscard]] std::uint32_t CachedVersionForTesting(DMatrix const* m) const {
    auto const* entry = cache_.Find(m);
    return entry == nullptr ? 0 : entry->version;
  }
};

}  // namespace treeg
}  // namespace xgboost

#endif  // PLUGIN_TREEG_BOOSTER_H_
