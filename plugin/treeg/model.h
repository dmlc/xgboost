/**
 * Copyright 2024-2026, XGBoost contributors
 *
 * @brief Model structure for Tree-G (Graph-Aware Decision Trees)
 */
#ifndef PLUGIN_TREEG_MODEL_H_
#define PLUGIN_TREEG_MODEL_H_

#include <xgboost/base.h>         // for bst_node_t, bst_feature_t
#include <xgboost/feature_map.h>  // for FeatureMap
#include <xgboost/logging.h>      // for CHECK, LOG
#include <xgboost/model.h>        // for Model
#include <xgboost/span.h>         // for Span

#include <algorithm>  // for fill
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t, int64_t, uint8_t
#include <limits>     // for numeric_limits
#include <sstream>    // for stringstream
#include <string>     // for string
#include <vector>     // for vector

namespace xgboost {

class Json;
class DMatrix;

namespace treeg {

/** @brief Index of a graph vertex (a DMatrix row). Not a tree-node index (`bst_node_t`). */
using VertexIdx = std::int32_t;
/** @brief Walk length or hop distance in the graph (the d in A^d). */
using WalkLen = std::int32_t;

/**
 * @brief Sparse CSR (Compressed Sparse Row) matrix in SoA layout.
 *
 * Stores the standard three-array CSR representation with separate
 * col_idx and values arrays (Structure of Arrays) for compatibility
 * with sparse BLAS libraries and efficient SpMV/SpMM kernels.
 */
struct CsrMatrix {
  /**
   * @brief Row pointers, size nrows+1.
   *
   * 64-bit because nnz(A^d) grows quickly with d and can exceed INT32_MAX even when the
   * vertex count fits in an `int`.
   */
  std::vector<std::int64_t> row_ptr;
  std::vector<int> col_idx;   ///< Column indices, size nnz. Fits: N <= INT32_MAX is checked.
  std::vector<float> values;  ///< Non-zero values, size nnz.
  int nrows{0};               ///< Number of rows.
  int ncols{0};               ///< Number of columns.

  /** @brief Number of stored non-zeros. */
  [[nodiscard]] std::int64_t Nnz() const { return row_ptr.empty() ? 0 : row_ptr.back(); }
};

/**
 * @brief Pre-extracted graph data for efficient split computation.
 *
 * Converts a DMatrix with layout [X | A^1 | A^2 | ... | A^d_max] from its
 * internal SparsePage CSR representation into:
 * - A dense feature matrix X (features are genuinely dense).
 * - Sparse CSR adjacency power matrices A^d (graphs are sparse).
 * - Precomputed diagonal vectors for Cycle restriction (O(1) per vertex).
 *
 * Storing A^d as CSR reduces memory from O(N^2) to O(nnz) per power and
 * enables SpMV/SpMM at O(nnz) instead of O(N^2) FLOPs.
 *
 * Memory layout:
 * - X: row-major (num_vertices × num_features), X[i * num_features + k].
 * - adj_powers[d-1]: CSR sparse matrix for A^d, shape (N, N).
 * - adj_diag[d-1]: dense vector of length N, adj_diag[d-1][i] = A^d[i,i].
 */
struct GraphData {
  /** @brief Number of vertices (N). */
  VertexIdx num_vertices{0};
  /** @brief Number of original node features (K). */
  bst_feature_t num_features{0};
  /** @brief Maximum walk length (d_max). */
  WalkLen max_walk_len{0};

  /** @brief Dense feature matrix X, shape (N, K), row-major. */
  std::vector<float> features;

  /**
   * @brief Sparse adjacency power matrices A^1 … A^d_max in CSR format.
   *
   * adj_powers[d-1] is the CSR representation of A^d, shape (N, N).
   */
  std::vector<CsrMatrix> adj_powers;

  /**
   * @brief Precomputed diagonal vectors diag(A^1) … diag(A^d_max).
   *
   * adj_diag[d-1][i] = A^d[i, i].  Used by the Cycle restriction branch
   * for O(N) element-wise computation without touching the full CSR matrix.
   */
  std::vector<std::vector<float>> adj_diag;

  /**
   * @brief Build GraphData from a DMatrix.
   *
   * Performs a single pass over the SparsePage CSR data, dispatching each entry
   * to X or the appropriate adj_powers[d-1] CSR based on its column index.
   * Diagonal elements are extracted into adj_diag[d-1] as well.
   *
   * Size limits (checked before any allocation): `num_vertices <= INT32_MAX`, and the
   * column count `num_features + num_vertices * max_walk_len` must fit in `bst_feature_t`.
   *
   * @param data       DMatrix with column layout [X | A^1 | A^2 | … | A^d_max].
   * @param num_vertices  Number of vertices (N); determines row count and A^d block sizes.
   * @param num_features  Number of original node features (K).
   * @param max_walk_len  Maximum walk length (d_max); number of adjacency power blocks.
   * @param intercept     Whether the virtual intercept feature is enabled. Used only to warn
   *                      when it has no effect (max_walk_len == 0).
   * @return Populated GraphData.
   */
  [[nodiscard]] static GraphData Build(DMatrix* data, VertexIdx num_vertices,
                                       bst_feature_t num_features, WalkLen max_walk_len,
                                       bool intercept = false);
};

/**
 * @brief Restriction type for Tree-G split computation.
 *
 * Defines how neighborhood aggregation is computed:
 * - kSource (r=1): Aggregates features from neighbors via A^d[:, subset] @ X[subset, k] - only consider walks starting in the vertices in subset.
 * - kCycle (r=2): Aggregates features via diagonal of A^d multiplied by X[:, k] - only consider walks that start and end in the same vertex in subset.
 */
enum class RestrictionType : uint8_t { kSource = 1, kCycle = 2 };

/**
 * @brief Active vertex set representation for Tree-G.
 *
 * Stores sets of vertex indices using a mask vector.
 * For a graph with n vertices, uses n * sizeof(float) bytes.
 * Provides O(1) membership testing and efficient mask operations for vectorized computation.
 *
 * The mask holds `float` values 0 or 1, the diagonal of diag(v) in the paper's notation. The
 * split-value kernels only test entries against 0; they do not multiply by them.
 */
class ActiveSet {
 public:
  /**
   * @brief Construct an empty active set.
   */
  ActiveSet() : num_vertices_{0} {}

  /**
   * @brief Construct an active set for a graph with n vertices.
   * @param n Number of vertices in the graph
   * @param init_all If true, initialize all vertices as active
   */
  explicit ActiveSet(VertexIdx n, bool init_all = false);

  /**
   * @brief Test if a vertex is in the active set.
   * @param vertex_idx Index of the vertex to test
   * @return true if vertex is active, false otherwise
   */
  [[nodiscard]] bool Test(VertexIdx vertex_idx) const {
    CHECK_LT(vertex_idx, num_vertices_);
    return mask_[vertex_idx] != 0.0f;
  }

  /**
   * @brief Set or clear a vertex in the active set.
   * @param vertex_idx Index of the vertex
   * @param value true to add vertex, false to remove
   */
  void Set(VertexIdx vertex_idx, bool value) {
    CHECK_LT(vertex_idx, num_vertices_);
    mask_[vertex_idx] = value ? 1.0f : 0.0f;
  }

  /**
   * @brief Clear all vertices from the active set.
   */
  void Clear() { std::fill(mask_.begin(), mask_.end(), 0.0f); }

  /**
   * @brief Apply a function to each active vertex.
   * @param func Callable accepting a VertexIdx
   */
  template <typename Func>
  void ForEach(Func&& func) const {
    for (VertexIdx i = 0; i < num_vertices_; ++i) {
      if (mask_[i] != 0.0f) {
        func(i);
      }
    }
  }

  /**
   * @brief Count the number of active vertices.
   * @return Number of vertices in the active set
   */
  [[nodiscard]] VertexIdx Count() const;

  /**
   * @brief Get the total number of vertices this set can hold.
   * @return Number of vertices
   */
  [[nodiscard]] VertexIdx NumVertices() const { return num_vertices_; }

  /**
   * @brief Get the mask vector for this active set.
   *
   * Returns an indicator vector v of length n where v[i] = 1.0 if vertex i
   * is in the active set, else v[i] = 0.0.
   * This represents the diagonal of the mask matrix Mask = diag(v).
   *
   * @return Const reference to the mask vector
   */
  [[nodiscard]] const std::vector<float>& Mask() const { return mask_; }

  bool operator==(const ActiveSet& other) const {
    return num_vertices_ == other.num_vertices_ && mask_ == other.mask_;
  }

 private:
  VertexIdx num_vertices_;
  std::vector<float> mask_;
};

/**
 * @brief Node statistics for Tree-G regression tree nodes.
 *
 * Stores statistics computed during tree construction, similar to RTreeNodeStat.
 */
struct TreeGNodeStat {
  /** @brief loss change caused by current split */
  float loss_chg;
  /** @brief sum of hessian values, used to measure coverage of data */
  float sum_hess;
  /** @brief weight of current node */
  float base_weight;

  TreeGNodeStat() = default;
  TreeGNodeStat(float loss_chg, float sum_hess, float weight)
      : loss_chg{loss_chg}, sum_hess{sum_hess}, base_weight{weight} {}

  bool operator==(const TreeGNodeStat& b) const {
    return loss_chg == b.loss_chg && sum_hess == b.sum_hess && base_weight == b.base_weight;
  }
};

/**
 * @brief Parameters for Tree-G model.
 */
struct TreeGParam {
  /** @brief The number of nodes */
  bst_node_t num_nodes{1};
  /** @brief The number of original node features (k) */
  bst_feature_t num_features{0};
  /** @brief Maximum walk length for neighborhood aggregation (d_max) */
  WalkLen max_walk_len{2};
  /** @brief Maximum ancestor distance for attention mechanism (a_max) */
  WalkLen max_ancestor_distance{2};
  /** @brief Number of vertices in the graph */
  VertexIdx num_vertices{0};
  /** @brief Whether a virtual constant-1 intercept feature is enabled */
  bool intercept{false};

  bool operator==(const TreeGParam& b) const {
    return num_nodes == b.num_nodes && num_features == b.num_features &&
           max_walk_len == b.max_walk_len && max_ancestor_distance == b.max_ancestor_distance &&
           num_vertices == b.num_vertices && intercept == b.intercept;
  }

  void FromJson(Json const& in);
  void ToJson(Json* p_out) const;
};

/**
 * @brief Tree-G model for graph-aware decision trees.
 *
 * Extends the standard regression tree with graph topology awareness through:
 * - Pointer mechanism for tracking active vertex subsets
 * - Neighborhood aggregation with configurable depth
 * - Attention-based ancestor referencing
 */
class RegTreeG : public Model {
 public:
  using SplitCondT = float;
  /** @brief Invalid/missing node ID sentinel value.
   * Must be compatible with the parent_ encoding which uses bit 31 for is_left_child flag.
   * Using INT_MAX (0x7FFFFFFF) instead of -1 to avoid masking issues with Parent() getter.
   */
  static constexpr bst_node_t kInvalidNodeId = std::numeric_limits<bst_node_t>::max();
  static constexpr bst_node_t kRoot{0};

  /**
   * @brief Tree node for Tree-G with graph-aware extensions.
   *
   * In addition to standard tree node fields (parent, children, split condition),
   * stores the pointer mechanism data needed for graph-aware splitting:
   * - Active vertex sets (S_plus, S_minus)
   * - Ancestor pointer information
   * - Graph aggregation parameters (walk length, restriction type)
   */
  class TreeGNode {
   public:
    TreeGNode() = default;

    /** @brief index of left child */
    [[nodiscard]] bst_node_t LeftChild() const { return cleft_; }

    /** @brief index of right child */
    [[nodiscard]] bst_node_t RightChild() const { return cright_; }

    /** @brief feature index of split condition */
    [[nodiscard]] bst_feature_t SplitIndex() const { return sindex_; }

    /** @brief whether current node is leaf node */
    [[nodiscard]] bool IsLeaf() const { return cleft_ == kInvalidNodeId; }

    /** @return get leaf value of leaf node */
    [[nodiscard]] float LeafValue() const { return value_; }

    /** @return get split condition of the node */
    [[nodiscard]] SplitCondT SplitCond() const { return value_; }

    /** @brief get parent of the node */
    [[nodiscard]] bst_node_t Parent() const { return parent_ & ((1U << 31) - 1); }

    /** @brief whether current node is left child */
    [[nodiscard]] bool IsLeftChild() const { return (parent_ & (1U << 31)) != 0; }

    /** @brief whether current node is root */
    [[nodiscard]] bool IsRoot() const { return parent_ == kInvalidNodeId; }

    /**
     * @brief Set the left child
     * @param nid node id to left child
     */
    void SetLeftChild(bst_node_t nid) { cleft_ = nid; }

    /**
     * @brief Set the right child
     * @param nid node id to right child
     */
    void SetRightChild(bst_node_t nid) { cright_ = nid; }

    /**
     * @brief Set split condition of current node
     * @param split_index feature index to split
     * @param split_cond  split condition
     */
    void SetSplit(bst_feature_t split_index, SplitCondT split_cond) {
      sindex_ = split_index;
      value_ = split_cond;
    }

    /**
     * @brief Set the leaf value of the node
     * @param value leaf value
     * @param right right index, could be used to store additional information
     */
    void SetLeaf(float value, bst_node_t right = kInvalidNodeId) {
      value_ = value;
      cleft_ = kInvalidNodeId;
      cright_ = right;
    }

    /**
     * @brief Set parent of the node
     * @param pidx parent node index
     * @param is_left_child whether this node is left child of parent
     */
    void SetParent(bst_node_t pidx, bool is_left_child = true) {
      if (is_left_child) pidx |= (1U << 31);
      parent_ = pidx;
    }

    // ========== Tree-G specific getters ==========

    /** @brief Get ancestor node index for pointer mechanism */
    [[nodiscard]] bst_node_t AncestorIndex() const { return ancestor_index_; }

    /** @brief Get ancestor sign (true = use S_plus, false = use S_minus) */
    [[nodiscard]] bool AncestorSign() const { return ancestor_sign_; }

    /** @brief Get restriction type (Source or Cycle) */
    [[nodiscard]] RestrictionType RestrictionTypeValue() const { return restriction_type_; }

    /** @brief Get walk length (neighborhood depth d) */
    [[nodiscard]] treeg::WalkLen WalkLen() const { return walk_len_; }

    // ========== Tree-G specific setters ==========

    /** @brief Set ancestor pointer information */
    void SetAncestorPointer(bst_node_t ancestor_idx, bool use_ancestor_gte) {
      ancestor_index_ = ancestor_idx;
      ancestor_sign_ = use_ancestor_gte;
    }

    /** @brief Set restriction type (Source or Cycle) */
    void SetRestrictionType(RestrictionType r_type) { restriction_type_ = r_type; }

    /** @brief Set walk length (neighborhood depth d) */
    void SetWalkLen(treeg::WalkLen d) { walk_len_ = d; }

    bool operator==(const TreeGNode& b) const {
      return parent_ == b.parent_ && cleft_ == b.cleft_ && cright_ == b.cright_ &&
             sindex_ == b.sindex_ && value_ == b.value_ && ancestor_index_ == b.ancestor_index_ &&
             ancestor_sign_ == b.ancestor_sign_ && restriction_type_ == b.restriction_type_ &&
             walk_len_ == b.walk_len_;
    }

   private:
    // ========== Base node fields ==========
    /** @brief pointer to parent, highest bit indicates left child */
    bst_node_t parent_{kInvalidNodeId};
    /** @brief pointer to left child */
    bst_node_t cleft_{kInvalidNodeId};
    /** @brief pointer to right child */
    bst_node_t cright_{kInvalidNodeId};
    /** @brief split feature index */
    bst_feature_t sindex_{0};
    /** @brief leaf value for a leaf, split condition for a split node */
    float value_{0.0f};

    // ========== Tree-G specific fields ==========
    /** @brief Index of ancestor node for pointer mechanism */
    bst_node_t ancestor_index_{kInvalidNodeId};
    /** @brief Which ancestor set to use (true = S_plus, false = S_minus) */
    bool ancestor_sign_{true};
    /** @brief Restriction type (Source or Cycle) */
    RestrictionType restriction_type_{RestrictionType::kSource};
    /** @brief Walk length (neighborhood depth d) for this split */
    treeg::WalkLen walk_len_{0};
  };

  // ========== Constructors ==========

  /**
   * @brief Default constructor - creates a single root leaf with value 0.
   *
   * The root state comes entirely from default member initializers (`num_nodes{1}`, one
   * value-initialized node and stat), so the constructor itself is defaulted.
   */
  RegTreeG() = default;

  /**
   * @brief Constructor with graph and feature dimensions.
   * @param n_vertices Number of vertices in the graph
   * @param n_features Number of original node features
   * @param max_walk_len Maximum neighborhood depth
   * @param max_ancestor_dist Maximum ancestor distance for attention
   * @param intercept Whether to enable virtual intercept feature
   */
  RegTreeG(VertexIdx n_vertices, bst_feature_t n_features, WalkLen max_walk_len = 2,
           WalkLen max_ancestor_dist = 2, bool intercept = false);

  // ========== Accessors ==========

  /** @brief Get node given node id */
  TreeGNode& operator[](bst_node_t nidx) { return nodes_[nidx]; }

  /** @brief Get const node given node id */
  [[nodiscard]] const TreeGNode& operator[](bst_node_t nidx) const { return nodes_[nidx]; }

  /** @brief Get node statistics given node id */
  TreeGNodeStat& Stat(bst_node_t nid) { return stats_[nid]; }

  /** @brief Get const node statistics given node id */
  [[nodiscard]] const TreeGNodeStat& Stat(bst_node_t nid) const { return stats_[nid]; }

  /** @brief Get tree parameters */
  [[nodiscard]] const TreeGParam& GetParam() const { return param_; }

  /** @brief Get number of nodes */
  [[nodiscard]] bst_node_t NumNodes() const { return param_.num_nodes; }

  /** @brief Get number of original node features */
  [[nodiscard]] bst_feature_t NumFeatures() const { return param_.num_features; }

  /** @brief Get number of vertices in the graph */
  [[nodiscard]] VertexIdx NumVertices() const { return param_.num_vertices; }

  /** @brief Get maximum walk length */
  [[nodiscard]] WalkLen MaxWalkLen() const { return param_.max_walk_len; }

  /** @brief Get maximum ancestor distance */
  [[nodiscard]] WalkLen MaxAncestorDistance() const { return param_.max_ancestor_distance; }

  /** @brief Whether the virtual intercept feature is enabled */
  [[nodiscard]] bool Intercept() const { return param_.intercept; }

  // ========== Tree modification methods ==========

  /**
   * @brief Expand a leaf node into two child nodes.
   * @param nid Node id to expand
   * @param split_index Feature index for the split
   * @param split_value Split threshold
   * @param base_weight Base weight before learning rate
   * @param left_leaf_weight Left leaf weight (after learning rate)
   * @param right_leaf_weight Right leaf weight (after learning rate)
   * @param loss_change Loss reduction from this split
   * @param sum_hess Sum of hessians for this node
   * @param left_sum Sum of hessians for left child
   * @param right_sum Sum of hessians for right child
   */
  void ExpandNode(bst_node_t nid, bst_feature_t split_index, float split_value, float base_weight,
                  float left_leaf_weight, float right_leaf_weight, float loss_change,
                  float sum_hess, float left_sum, float right_sum);

  /**
   * @brief Set Tree-G specific split parameters for a node after expansion.
   *
   * Stores only the learned parameters (k, d, r, *, rho). The subsets S_{u,+}, S_{u,-}
   * generated by the node are graph-dependent and are rebuilt on demand by
   * ReconstructSubsets during inference, and by the updater during training.
   *
   * @param nid Node id
   * @param walk_len Neighborhood depth for this split
   * @param restriction_type Source or Cycle restriction
   * @param ancestor_idx Ancestor node index, or kInvalidNodeId for a self pointer
   * @param ancestor_sign Which ancestor subset to use (true = S_plus)
   */
  void SetNodeTreeGParams(bst_node_t nid, WalkLen walk_len, RestrictionType restriction_type,
                          bst_node_t ancestor_idx, bool ancestor_sign);

  // ========== Model interface ==========

  void LoadModel(Json const& in) override;
  void SaveModel(Json* out) const override;

  /**
   * @brief Dump the tree model to a string.
   *
   * @param fmap Feature map for feature name resolution
   * @param with_stats Whether to include statistics in the dump
   * @param format Output format ("text", "json", or "dot")
   * @return String representation of the tree
   */
  [[nodiscard]] std::string DumpModel(FeatureMap const& fmap, bool with_stats,
                                      std::string const& format) const;

  bool operator==(const RegTreeG& b) const {
    return nodes_ == b.nodes_ && stats_ == b.stats_ && param_ == b.param_;
  }

  // ========== Graph-Aware Split Computation ==========
  //
  // All split values phi_{k,d,r}(S) = ((A^d o M_r(S)) f_k) are computed by the single kernel
  // ComputeSplitValues(). Training thresholds are compared against values recomputed at
  // prediction time, so there must be exactly one implementation. For every output entry
  // (i, k) the summation runs over the CSR row of A^d in stored order, so the result is the
  // same for any column set and any thread count.

  /**
   * @brief Compute split values for every vertex and a set of features.
   *
   * - d = 0: result[i, c] = X[i, features[c]] (the subset is ignored).
   * - Source (d >= 1): result = A^d (diag(mask) X[:, features]), a sparse SpMM.
   * - Cycle (d >= 1): result[i, c] = diag(A^d)[i] * mask[i] * X[i, features[c]].
   *
   * When @p with_intercept is true, a trailing column is appended for the virtual
   * constant-1 feature; it requires the tree to have the intercept enabled.
   *
   * @param walk_len         Walk length d.
   * @param restriction_type Source or Cycle; ignored when d = 0.
   * @param subset           Used subset S defining the mask; ignored when d = 0.
   * @param graph_data       Graph the tree is applied to.
   * @param features         Indices of the original features, each in [0, num_features).
   * @param with_intercept   Append the intercept column.
   * @param n_threads        Threads for the row loop; does not change the result.
   * @return Row-major N x F matrix, F = |features| + (with_intercept ? 1 : 0).
   */
  [[nodiscard]] std::vector<float> ComputeSplitValues(
      WalkLen walk_len, RestrictionType restriction_type, ActiveSet const& subset,
      GraphData const& graph_data, common::Span<bst_feature_t const> features, bool with_intercept,
      std::int32_t n_threads = 1) const;

  /**
   * @brief Split values of one split feature for every vertex.
   *
   * A single-column call of ComputeSplitValues(). `feature_idx == num_features` selects the
   * virtual intercept, which is only valid when the tree has the intercept enabled.
   *
   * @return Vector of N split values.
   */
  [[nodiscard]] std::vector<float> ComputeFeatureSplitValues(
      bst_feature_t feature_idx, WalkLen walk_len, RestrictionType restriction_type,
      ActiveSet const& subset, GraphData const& graph_data, std::int32_t n_threads = 1) const;

 private:
  /** @brief Tree parameters */
  TreeGParam param_;
  /** @brief Vector of tree nodes */
  std::vector<TreeGNode> nodes_ = std::vector<TreeGNode>(1);
  /** @brief Vector of node statistics */
  std::vector<TreeGNodeStat> stats_ = std::vector<TreeGNodeStat>(1);

  /**
   * @brief Append a new node.
   * @return Node id of the allocated node
   */
  [[nodiscard]] bst_node_t AllocNode();

  /**
   * @brief Validate one node of a freshly loaded model.
   *
   * Checks parent/child links, split index, walk length and that the ancestor pointer refers
   * to an ancestor within max_ancestor_distance. Nodes must be validated in ascending id order.
   *
   * @param nid Node id
   * @throws dmlc::Error if the node is inconsistent.
   */
  void ValidateNode(bst_node_t nid) const;

  /**
   * @brief Display name of a split feature.
   *
   * Returns "intercept" for the virtual intercept (split_index == num_features when enabled),
   * otherwise the feature-map name, falling back to "f<index>".
   */
  [[nodiscard]] std::string SplitFeatureName(FeatureMap const& fmap,
                                             bst_feature_t split_index) const;

  /**
   * @brief Recursive helper for DumpModel.
   */
  void DumpModelImpl(std::stringstream& ss, bst_node_t nid, int depth, FeatureMap const& fmap,
                     bool with_stats) const;
};

/**
 * @brief Attention subsets S_{u,+}, S_{u,-} generated by a split-node u.
 *
 * Given the used subset S of node u (see ResolveUsedSubset):
 *  - lte = S_{u,-} = { j in S : phi_u(j) <= theta_u }
 *  - gte = S_{u,+} = { j in S : phi_u(j) >  theta_u }
 *
 * The subsets partition S and are computed over *all* vertices of the graph, including
 * zero-weight (test) vertices. They are graph-dependent derived data and are therefore never
 * stored in the model; training and prediction rebuild them with the same functions.
 */
struct GeneratedSubsets {
  /** @brief S_{u,-}: vertices of the used subset with phi_u <= theta_u. */
  ActiveSet lte;
  /** @brief S_{u,+}: vertices of the used subset with phi_u > theta_u. */
  ActiveSet gte;
  /**
   * @brief True once PartitionSubset has filled lte and gte.
   *
   * Callers index subset containers by node id, so entries for leaves, deleted nodes and
   * not-yet-processed split-nodes are default-constructed.  This flag states that invariant
   * directly instead of inferring it from an empty ActiveSet, which is also a legal value
   * for a split-node whose used subset happens to be empty.
   */
  bool populated{false};
};

/**
 * @brief Resolve the used subset S_{*u,rho_u} of a split-node.
 *
 * A node whose ancestor pointer is RegTreeG::kInvalidNodeId points to itself and uses all
 * vertices. Otherwise the subset is S_{*u,+} or S_{*u,-} of the referenced ancestor,
 * selected by the node's ancestor sign.
 *
 * @param node         The split-node.
 * @param subsets      Generated subsets indexed by node id; the referenced ancestor must
 *                     already have `populated == true`.
 * @param all_vertices An ActiveSet with every vertex active (returned for self-pointers).
 * @return Reference into either @p subsets or @p all_vertices.
 * @throws dmlc::Error if the ancestor pointer is out of range or not yet populated.
 */
[[nodiscard]] ActiveSet const& ResolveUsedSubset(RegTreeG::TreeGNode const& node,
                                                 std::vector<GeneratedSubsets> const& subsets,
                                                 ActiveSet const& all_vertices);

/**
 * @brief Partition a used subset into attention subsets by comparing split values to a threshold.
 *
 * Uses the same comparison as routing: phi <= threshold goes to lte, otherwise gte.
 *
 * @param split_values Split value phi(j) for every vertex j, length N.
 * @param used         The used subset S to partition.
 * @param threshold    Split threshold theta.
 * @return The pair (S_{u,-}, S_{u,+}).
 */
[[nodiscard]] GeneratedSubsets PartitionSubset(std::vector<float> const& split_values,
                                               ActiveSet const& used, float threshold);

/**
 * @brief Rebuild the generated subsets and split values of every split-node of a tree.
 *
 * Traverses split-nodes in BFS order so that every ancestor is processed before its
 * descendants. For each split-node u: resolves S_{*u,rho_u}, computes phi_u for all vertices
 * via RegTreeG::ComputeFeatureSplitValues, and partitions S with PartitionSubset.
 *
 * @param tree         A trained tree.
 * @param graph_data   Graph the tree is applied to.
 * @param subsets      Output, resized to tree.NumNodes(); populated for split-nodes only.
 * @param split_values Output, resized to tree.NumNodes(); split_values[u] has length N for
 *                     split-nodes and is empty otherwise.
 * @param n_threads    Threads for computing each node's split values.
 */
void ReconstructSubsets(RegTreeG const& tree, GraphData const& graph_data,
                        std::vector<GeneratedSubsets>* subsets,
                        std::vector<std::vector<float>>* split_values, std::int32_t n_threads = 1);

}  // namespace treeg
}  // namespace xgboost

#endif  // PLUGIN_TREEG_MODEL_H_
