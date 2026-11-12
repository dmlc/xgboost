/**
 * Copyright 2026, XGBoost contributors
 *
 * @brief Tree updater for Tree-G (Graph-Aware Decision Trees)
 */
#ifndef PLUGIN_TREEG_UPDATER_H_
#define PLUGIN_TREEG_UPDATER_H_

#include <xgboost/base.h>       // for bst_node_t, bst_feature_t, GradientPair
#include <xgboost/logging.h>    // for CHECK, LOG
#include <xgboost/parameter.h>  // for XGBoostParameter
#include <xgboost/span.h>       // for Span

#include <cstddef>  // for size_t
#include <random>   // for mt19937, bernoulli_distribution
#include <utility>  // for move, pair
#include <vector>   // for vector

#include "model.h"  // for RegTreeG, ActiveSet, RestrictionType

namespace xgboost {

class Json;
struct Context;

namespace treeg {

/**
 * @brief User-facing training parameters of the `treeg` booster.
 *
 * Parsed once by `TreeGBooster` and passed to `TreeGUpdater`. The graph-shape fields
 * (`num_vertices`, `num_features`, `max_walk_len`, `max_ancestor_distance`, `intercept`) are
 * also persisted in `TreeGBoosterModelParam`; the booster keeps the two in sync.
 */
struct TreeGTrainParam : public XGBoostParameter<TreeGTrainParam> {
  /** @brief Learning rate (step size) for weight updates */
  float learning_rate{0.3f};

  /** @brief Minimum loss reduction required for a split */
  float min_split_loss{0.0f};

  /** @brief Maximum number of leaves in the tree */
  bst_node_t max_leaves{32};

  /** @brief Minimum sum of hessian needed in a child */
  float min_child_weight{1.0f};

  /** @brief L2 regularization on leaf weights */
  float reg_lambda{1.0f};

  /** @brief L1 regularization on leaf weights */
  float reg_alpha{0.0f};

  /** @brief Maximum delta step for weight estimation (0 = no constraint) */
  float max_delta_step{0.0f};

  /** @brief Maximum walk length for neighborhood aggregation (d_max) */
  WalkLen max_walk_len{2};

  /** @brief Maximum ancestor distance for attention mechanism (a_max) */
  WalkLen max_ancestor_distance{2};

  /** @brief Number of original node features (k) */
  bst_feature_t num_features{0};

  /** @brief Number of vertices in the graph */
  VertexIdx num_vertices{0};

  /** @brief Whether to add a virtual constant-1 intercept feature.
   *  At d>=1 this becomes a graph-connectivity feature: sum_j A^d[i,j]*mask[j]. */
  bool intercept{false};

  /** @brief Probability of including each restriction type in the split search.
   *  Each restriction type (kSource, kCycle) is independently included with this
   *  probability via Bernoulli sampling. If the sampled subset is empty, falls back
   *  to r=kSource with attention depth forced to 0. Default 1.0 (all types). */
  float attention_type_sample_probability{1.0f};

  /** @brief Fraction of original node features (K) to sample per tree.
   *  Sampling occurs once per tree, applied only to X features [0, K).
   *  The intercept feature (if enabled) is always included.
   *  Default 1.0 (no sampling). */
  float colsample_bytree{1.0f};

  /** @brief Memory cap (bytes) for the per-tree cache of node-independent split values.
   *  0 disables the cache. Affects speed only, never the model. */
  std::size_t max_cached_split_bytes{static_cast<std::size_t>(1) << 30};

  DMLC_DECLARE_PARAMETER(TreeGTrainParam) {
    DMLC_DECLARE_FIELD(learning_rate)
        .set_lower_bound(0.0f)
        .set_default(0.3f)
        .describe("Learning rate (step size) of update.");
    DMLC_DECLARE_FIELD(min_split_loss)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe("Minimum loss reduction required to make a split.");
    DMLC_DECLARE_FIELD(max_leaves)
        .set_lower_bound(1)
        .set_default(32)
        .describe("Maximum number of leaves in the tree.");
    DMLC_DECLARE_FIELD(min_child_weight)
        .set_lower_bound(0.0f)
        .set_default(1.0f)
        .describe("Minimum sum of hessian needed in a child.");
    DMLC_DECLARE_FIELD(reg_lambda)
        .set_lower_bound(0.0f)
        .set_default(1.0f)
        .describe("L2 regularization on leaf weights.");
    DMLC_DECLARE_FIELD(reg_alpha).set_lower_bound(0.0f).set_default(0.0f).describe(
        "L1 regularization on leaf weights.");
    DMLC_DECLARE_FIELD(max_delta_step)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe("Maximum delta step for weight estimation.");
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
        .describe("Number of original node features. Required (0 means unset).");
    DMLC_DECLARE_FIELD(num_vertices)
        .set_lower_bound(0)
        .set_default(0)
        .describe("Number of vertices in the graph. Required (0 means unset).");
    DMLC_DECLARE_FIELD(intercept).set_default(false).describe(
        "Add a virtual constant-1 intercept feature for graph-aware splits.");
    DMLC_DECLARE_FIELD(attention_type_sample_probability)
        .set_range(0.0f, 1.0f)
        .set_default(1.0f)
        .describe(
            "Probability of including each restriction type (kSource, kCycle) in the "
            "split search. Each type is sampled independently via Bernoulli trial. "
            "If the sampled set is empty, falls back to kSource with attention depth 0.");
    DMLC_DECLARE_FIELD(colsample_bytree)
        .set_range(0.0f, 1.0f)
        .set_default(1.0f)
        .describe(
            "Fraction of original node features (K) to sample per tree. "
            "Only X features are sampled; adjacency powers and intercept are kept. "
            "Default 1.0 (no sampling).");
    DMLC_DECLARE_FIELD(max_cached_split_bytes)
        .set_lower_bound(0)
        .set_default(static_cast<std::size_t>(1) << 30)
        .describe(
            "Memory cap in bytes for caching node-independent split values within one tree. "
            "0 disables the cache. Changes speed only, never the model.");
    // Same aliases as tree::TrainParam (src/tree/param.h).
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_alpha, alpha);
    DMLC_DECLARE_ALIAS(min_split_loss, gamma);
    DMLC_DECLARE_ALIAS(learning_rate, eta);
  }
};

/**
 * @brief Gradient statistics for a set of vertices.
 *
 * Accumulates first and second order gradient statistics, weighted by instance weights.
 */
struct GradStats {
  /** @brief Sum of gradient values (first-order) */
  double sum_grad{0.0};

  /** @brief Sum of hessian values (second-order) */
  double sum_hess{0.0};

  GradStats() = default;
  GradStats(double grad, double hess) : sum_grad{grad}, sum_hess{hess} {}

  /** @brief Add another GradStats to this one */
  GradStats& operator+=(const GradStats& rhs) {
    sum_grad += rhs.sum_grad;
    sum_hess += rhs.sum_hess;
    return *this;
  }

  /** @brief Subtract another GradStats from this one */
  GradStats& operator-=(const GradStats& rhs) {
    sum_grad -= rhs.sum_grad;
    sum_hess -= rhs.sum_hess;
    return *this;
  }

  /** @brief Add two GradStats */
  friend GradStats operator+(GradStats lhs, const GradStats& rhs) {
    lhs += rhs;
    return lhs;
  }

  /** @brief Test if this node has sufficient hessian weight */
  [[nodiscard]] bool HasSufficientWeight(float min_child_weight) const {
    return sum_hess >= min_child_weight;
  }

  /** @brief Check if statistics are empty */
  [[nodiscard]] bool IsEmpty() const { return sum_hess == 0.0; }
};

/**
 * @brief Candidate split for a Tree-G node.
 *
 * Stores all information needed to evaluate and apply a split.
 */
struct SplitEntry {
  /** @brief Feature index for the split */
  bst_feature_t split_feature{0};

  /** @brief Threshold value for the split */
  float threshold{0.0f};

  /** @brief Walk length (d) for graph aggregation */
  WalkLen walk_len{0};

  /** @brief Restriction type (Source or Cycle) */
  RestrictionType restriction_type{RestrictionType::kSource};

  /** @brief Ancestor node index for attention mechanism */
  bst_node_t ancestor_idx{0};

  /** @brief Ancestor sign (true = use S_plus, false = use S_minus) */
  bool ancestor_sign{true};

  /** @brief Vertices of the reached set routed right (phi > threshold), i.e. D_2.
   *  This is the routing partition used for gradient statistics and for the right
   *  child's reached set. It is NOT the attention subset S_{u,+}. */
  ActiveSet reached_gte;

  /** @brief Vertices of the reached set routed left (phi <= threshold), i.e. D_1. */
  ActiveSet reached_lte;

  /** @brief Gradient statistics for left child (<=) */
  GradStats left_stats;

  /** @brief Gradient statistics for right child (>) */
  GradStats right_stats;

  /** @brief Loss change (gain) from this split */
  float loss_chg{0.0f};

  SplitEntry() = default;

  /** @brief Check if this split is valid */
  [[nodiscard]] bool IsValid() const { return loss_chg > 0.0f; }

  /** @brief Update split if the new one is better */
  bool Update(const SplitEntry& candidate) {
    if (candidate.loss_chg > this->loss_chg) {
      *this = candidate;
      return true;
    }
    return false;
  }
};

/**
 * @brief Information about a leaf node that can potentially be split.
 */
struct LeafCandidate {
  /** @brief Node ID of the leaf */
  bst_node_t node_id{0};

  /** @brief Best split found for this leaf */
  SplitEntry best_split;

  /** @brief Vertices that were routed to this leaf by ancestor splits.
   *  This is the set of vertices whose gradients should be used for gain
   *  computation — NOT the attention set from the pointer mechanism. */
  ActiveSet reached_set;

  LeafCandidate() = default;
  LeafCandidate(bst_node_t nid, SplitEntry split) : node_id{nid}, best_split{std::move(split)} {}
  LeafCandidate(bst_node_t nid, SplitEntry split, ActiveSet reached)
      : node_id{nid}, best_split{std::move(split)}, reached_set{std::move(reached)} {}

  /** @brief Comparison operator for priority queue (max-heap by gain) */
  bool operator<(const LeafCandidate& other) const {
    return best_split.loss_chg < other.best_split.loss_chg;
  }
};

/**
 * @brief Result of sampling restriction types for a node's split search.
 *
 * When the sampled set is empty (all types rejected), the fallback sets
 * restriction_types = {kSource} and forces effective_max_ancestor_distance = 0,
 * reducing the node to a plain split without graph attention.
 */
struct SampledRestrictionConfig {
  /** @brief Restriction types to search over for this node */
  std::vector<RestrictionType> restriction_types;

  /** @brief Effective maximum ancestor distance (may be 0 on fallback) */
  WalkLen effective_max_ancestor_distance;
};

/**
 * @brief Tree updater for Tree-G algorithm.
 *
 * Implements greedy best-first tree growth with graph-aware split evaluation.
 * Uses exact enumeration of split candidates and loss-guided leaf expansion.
 *
 * The split search runs on `ctx->Threads()` threads (parallel over sampled features,
 * deterministic tie-break on the lowest candidate), so the grown tree does not depend on
 * `nthread`. Node-independent split values and their sorted orders are cached per tree
 * under `max_cached_split_bytes`; the cache changes speed only, never the tree.
 */
class TreeGUpdater {
 public:
  explicit TreeGUpdater(const Context* ctx) : ctx_{ctx} {}

  /**
   * @brief Configure the updater with parameters.
   * @param args Parameter key-value pairs
   */
  void Configure(const Args& args);

  /**
   * @brief Configure the updater from an already-parsed parameter struct.
   *
   * Used by the booster so that floating-point parameters are passed without a lossy
   * string round-trip.
   *
   * @param param Training parameters; validated before use.
   */
  void Configure(TreeGTrainParam const& param);

  /**
   * @brief Access the active training parameters. Test-only.
   * @return The parameters set by the last Configure().
   */
  [[nodiscard]] TreeGTrainParam const& ParamForTesting() const { return param_; }

  /**
   * @brief Loss change of a split, as in gbtree: gain(left) + gain(right) - gain(parent).
   *
   * Node gains use `tree::CalcGain` from src/tree/param.h, so `reg_alpha`, `reg_lambda` and
   * `max_delta_step` have exactly the gbtree meaning.
   *
   * @param parent_stats Parent node statistics
   * @param left_stats Left child statistics
   * @param right_stats Right child statistics
   * @return Loss change, in double precision.
   */
  [[nodiscard]] double CalcGain(const GradStats& parent_stats, const GradStats& left_stats,
                                const GradStats& right_stats) const;

  /**
   * @brief Optimal leaf weight, as in gbtree (`tree::CalcWeight`), before the learning rate.
   * @param stats Gradient statistics
   * @return Optimal weight, in double precision.
   */
  [[nodiscard]] double CalcWeight(const GradStats& stats) const;

  /**
   * @brief Whether a split with this loss change may be applied.
   *
   * Mirrors gbtree's expand-entry validation: the loss change must exceed `kRtEps` and must
   * not be below `min_split_loss`.
   *
   * @param loss_chg Loss change of the candidate split.
   * @return True if the split is accepted.
   */
  [[nodiscard]] bool IsSplitAccepted(float loss_chg) const;

  /**
   * @brief Update a Tree-G model with gradient statistics.
   * @param gpairs Gradient pairs (first and second order gradients) for each vertex.
   *               Must already include sample-weight scaling (applied by the objective).
   * @param graph_data Dense pre-extracted graph data (features + adjacency powers)
   * @param tree Tree to update (will be modified)
   * @param out_position Optional output, resized to `num_vertices`: the leaf id each vertex is
   *                     routed to in the finished tree. Used by the booster's prediction cache.
   */
  void Update(const std::vector<GradientPair>& gpairs, const GraphData& graph_data, RegTreeG* tree,
              std::vector<bst_node_t>* out_position = nullptr);

  /** @brief Get the name of this updater */
  [[nodiscard]] static char const* Name() { return "TreeGUpdater"; }

  /**
   * @brief Generated subsets S_{u,+}, S_{u,-} of the most recently grown tree.
   *
   * Indexed by node id and populated for split-nodes only. Exposed so that tests can
   * compare training against ReconstructSubsets; not part of the model.
   */
  [[nodiscard]] std::vector<GeneratedSubsets> const& GeneratedSubsetsForTesting() const {
    return subsets_;
  }

 private:
  /**
   * @brief Initialize a new tree with root node statistics.
   * @param gpairs Gradient pairs (already weight-scaled by the objective)
   * @param tree Tree to initialize
   */
  void InitRoot(const std::vector<GradientPair>& gpairs, RegTreeG* tree);

  /**
   * @brief Find the best split for a leaf node.
   * @param node_id Node ID of the leaf
   * @param tree Current tree
   * @param graph_data Dense pre-extracted graph data (features + adjacency powers)
   * @param gpairs Gradient pairs (already weight-scaled by the objective)
   * @return Best split found (or invalid split if none found)
   */
  [[nodiscard]] SplitEntry FindBestSplit(bst_node_t node_id, const ActiveSet& reached_set,
                                         const RegTreeG& tree, const GraphData& graph_data,
                                         const std::vector<GradientPair>& gpairs);

  /**
   * @brief Scalar result of scanning one split-value column (no ActiveSets).
   *
   * `loss_chg` is the float-rounded best gain (0 when no valid threshold exists).
   */
  struct ThresholdScan {
    float loss_chg{0.0f};
    float threshold{0.0f};
    GradStats left_stats;
    GradStats right_stats;
  };

  /** @brief One scan context per (d, r, attention set) block, shared by all its columns. */
  struct SplitBlock {
    const std::vector<float>* matrix;
    bst_feature_t stride;
    WalkLen walk_len;
    RestrictionType restriction_type;
    bst_node_t ancestor_idx;
    bool ancestor_sign;
    /** @brief Number of local columns scanned (column c maps to a matrix column and feature). */
    bst_feature_t n_columns;
    /** @brief True at d=0: the matrix is graph_data.features, column = original feature. */
    bool raw_features;
    /** @brief Pre-sorted vertex order, column-major (n_columns × N, local column c), or
     *  nullptr / empty when no order is cached for this block. */
    const std::vector<VertexIdx>* presorted{nullptr};
  };

  /**
   * @brief Scan one column of a split-value matrix for the threshold with highest gain.
   *
   * Extracts column @p column_idx from @p result_matrix (N × stride, row-major) for the
   * reached vertices, sorts them by value and performs a linear scan. Returns scalars only;
   * the routing partition is built by MaterialiseReached for the final winner.
   *
   * @param scratch Reusable per-thread buffer for (value, vertex) pairs.
   * @param presorted All N vertex ids of this column sorted by (value, vertex), or empty.
   *        When set and n·log2(n) > N (n = reached count), the sorted pairs are produced by
   *        a filtered walk over it instead of std::sort; both give the same sequence.
   */
  [[nodiscard]] ThresholdScan ScanThreshold(const std::vector<float>& result_matrix,
                                            bst_feature_t stride, bst_feature_t column_idx,
                                            const ActiveSet& reached_set,
                                            const std::vector<GradientPair>& gpairs,
                                            std::vector<std::pair<float, VertexIdx>>* scratch,
                                            common::Span<VertexIdx const> presorted) const;

  /**
   * @brief Build reached_lte / reached_gte of @p split: vertex v of @p reached_set goes to
   *        lte iff its value <= threshold. Equals the sorted prefix chosen by ScanThreshold,
   *        because the threshold is always the last value of a run of equal values.
   */
  void MaterialiseReached(const std::vector<float>& result_matrix, bst_feature_t stride,
                          bst_feature_t column_idx, const ActiveSet& reached_set,
                          SplitEntry* split) const;

  /**
   * @brief Search every column of one block in parallel and fold the winner into
   *        @p best_split.
   *
   * Each thread keeps one scalar best (highest float gain, then lowest column). Threads are
   * merged with the same rule, and the block winner replaces @p best_split only on a strictly
   * greater gain, so ties resolve to the earliest candidate in d -> r -> set -> k order,
   * exactly as the serial search. Reached sets are materialised once per block winner.
   */
  void SearchBlock(const SplitBlock& block, const ActiveSet& reached_set,
                   const std::vector<GradientPair>& gpairs, SplitEntry* best_split);

  /**
   * @brief Search the (d, r) block whose used subset is all vertices.
   *
   * Uses the per-tree cached matrix and pre-sorted order when present.
   */
  void SearchAllVerticesBlock(WalkLen d, RestrictionType r, const ActiveSet& reached_set,
                              const RegTreeG& tree, const GraphData& graph_data,
                              const std::vector<GradientPair>& gpairs, SplitEntry* best_split);

  /**
   * @brief Search the (d, r) blocks whose used subset is an ancestor's attention subset.
   *
   * Visits @p ancestors in the given order (nearest first), skipping those farther than
   * @p max_distance, and for each one searches S_{a,+} before S_{a,-}. This order decides
   * ties and must not change.
   */
  void SearchAncestorBlocks(WalkLen d, RestrictionType r, bst_node_t node_id,
                            const std::vector<std::pair<bst_node_t, WalkLen>>& ancestors,
                            WalkLen max_distance, const ActiveSet& reached_set,
                            const RegTreeG& tree, const GraphData& graph_data,
                            const std::vector<GradientPair>& gpairs, SplitEntry* best_split);

  /**
   * @brief Get ancestors of a node up to max distance.
   * @param node_id Node ID
   * @param tree Tree structure
   * @return List of (ancestor_id, distance) pairs
   */
  [[nodiscard]] std::vector<std::pair<bst_node_t, WalkLen>> GetAncestors(bst_node_t node_id,
                                                                         const RegTreeG& tree);

  /**
   * @brief Validate param_ and reseed the PRNG. Shared by both Configure overloads.
   */
  void Validate();

  /**
   * @brief Apply a split to expand a leaf node.
   *
   * Expands the node, then generates its subsets S_{u,+}, S_{u,-} by
   * resolving the used subset S_{*u,rho_u} and partitioning it with the same split-value
   * kernel prediction uses. The subsets are stored in subsets_, not in the tree, because
   * they are graph-dependent derived data.
   *
   * Requires subsets_ to be pre-sized by Update(); it is never resized here.
   *
   * @param leaf Leaf candidate with split information
   * @param graph_data Graph the tree is being trained on
   * @param tree Tree to modify
   */
  void ApplySplit(const LeafCandidate& leaf, const GraphData& graph_data, RegTreeG* tree);

  /**
   * @brief Compute gradient statistics for an active set.
   *
   * Sums gradient pairs over active vertices.  The gpairs must already include
   * sample-weight scaling (applied by the XGBoost objective); this function
   * does **not** apply weights itself.
   *
   * @param active_set Active vertices
   * @param gpairs Gradient pairs (already weight-scaled)
   * @return Gradient statistics
   */
  [[nodiscard]] GradStats ComputeGradStats(const ActiveSet& active_set,
                                           const std::vector<GradientPair>& gpairs) const;

  /**
   * @brief Sample which restriction types to search for a node's split.
   *
   * Each restriction type (kSource, kCycle) is independently included with
   * probability attention_type_sample_probability via a Bernoulli trial.
   * If the sampled set is empty, falls back to {kSource} with
   * effective_max_ancestor_distance forced to 0.
   *
   * @return Sampled restriction configuration for the node
   */
  [[nodiscard]] SampledRestrictionConfig SampleRestrictionTypes();

  /** @brief Runtime context */
  const Context* ctx_;

  /** @brief Training parameters */
  TreeGTrainParam param_;

  /** @brief PRNG for stochastic restriction-type sampling, seeded from ctx_->seed */
  std::mt19937 rng_;

  /** @brief Sampled original-feature indices for current tree, subset of [0, K).
   *  Sorted in ascending order for cache-friendly access. */
  std::vector<bst_feature_t> sampled_features_;

  /** @brief Attention subsets per node for the tree currently being grown.
   *  Scratch state: sized to the tree's maximum node count and cleared at the start of every
   *  Update(), then populated for split-nodes only. Never resized while the tree grows, so
   *  references into it remain valid. Descendants resolve their used subset from here via
   *  ResolveUsedSubset. */
  std::vector<GeneratedSubsets> subsets_;

  /** @brief An ActiveSet with every vertex active; the used subset of self-pointing nodes. */
  ActiveSet all_vertices_;

  /** @brief Per-tree cache of "all vertices" split-value matrices for d >= 1.
   *  Indexed by AllVerticesCacheIndex(d, r). An empty entry means "not cached"; the
   *  caller then recomputes per node. Filled under max_cached_split_bytes and freed at the
   *  end of Update(), so it never lives across trees. */
  std::vector<std::vector<float>> all_vertices_cache_;

  /** @brief Whether the "cache budget too small" message was already logged. */
  bool logged_cache_skip_{false};

  /** @brief Per-thread (value, vertex) buffers for ScanThreshold, indexed by OpenMP thread. */
  std::vector<std::vector<std::pair<float, VertexIdx>>> scan_scratch_;

  /** @brief Slot of (d, r) in all_vertices_cache_, for d >= 1. */
  [[nodiscard]] static std::size_t AllVerticesCacheIndex(WalkLen d, RestrictionType r) {
    return static_cast<std::size_t>(d - 1) * 2 + (r == RestrictionType::kCycle ? 1 : 0);
  }

  /**
   * @brief Fill all_vertices_cache_ in the order (1, Source), (1, Cycle), (2, Source), ...
   *        stopping at the first matrix that does not fit in max_cached_split_bytes.
   */
  void FillAllVerticesCache(const RegTreeG& tree, const GraphData& graph_data);

  /** @brief Free all_vertices_cache_ and sorted_order_. */
  void ReleaseAllVerticesCache();

  /**
   * @brief Move the vertices of a split node to its children.
   *
   * CHECKs that each moved vertex sits at `parent` and lands in exactly one child.
   *
   * @param parent Node being split.
   * @param left_reached Vertices routed to the left child (value <= threshold).
   * @param left_child Left child id.
   * @param right_reached Vertices routed to the right child.
   * @param right_child Right child id.
   * @param[in,out] position Leaf id of every vertex.
   */
  void UpdatePosition(bst_node_t parent, const ActiveSet& left_reached, bst_node_t left_child,
                      const ActiveSet& right_reached, bst_node_t right_child,
                      std::vector<bst_node_t>* position) const;

  /** @brief Per-tree pre-sorted vertex orders Slot 0 is d=0 (S sampled columns);
   *  slot 1 + AllVerticesCacheIndex(d, r) is the cached all-vertices matrix of (d, r).
   *  Each entry is column-major: column c holds all N vertex ids sorted by (value, vertex).
   *  An empty entry means "not cached". Admitted only with the budget left after the
   *  matrices, and freed at the end of Update(). */
  std::vector<std::vector<VertexIdx>> sorted_order_;

  /**
   * @brief Build sorted_order_ for d=0 and every cached (d, r), in that order, using the
   *        part of max_cached_split_bytes not used by the matrices.
   */
  void FillSortedOrders(const GraphData& graph_data, std::size_t budget_left);

  /**
   * @brief Sample features for a new tree.
   *
   * When colsample_bytree < 1.0, randomly selects max(1, floor(K * colsample_bytree))
   * features from [0, K) via partial Fisher-Yates shuffle.
   * When colsample_bytree >= 1.0, uses all K features (zero overhead).
   * Called once per tree at the start of Update().
   */
  void SampleFeatures();
};

}  // namespace treeg
}  // namespace xgboost

#endif  // PLUGIN_TREEG_UPDATER_H_
