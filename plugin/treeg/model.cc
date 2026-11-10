/**
 * Copyright 2026, XGBoost contributors
 *
 * @brief Implementation of Tree-G model for graph-aware decision trees
 */
#include "model.h"

#include <xgboost/base.h>         // for bst_node_t, bst_feature_t
#include <xgboost/data.h>         // for DMatrix, SparsePage
#include <xgboost/feature_map.h>  // for FeatureMap
#include <xgboost/json.h>         // for Json, Object, Array, Integer, Number, Boolean, get
#include <xgboost/logging.h>      // for CHECK, LOG
#include <xgboost/span.h>         // for Span
#include <xgboost/string_view.h>  // for StringView

#include <array>    // for array
#include <atomic>   // for atomic
#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t, int64_t, uint64_t
#include <limits>   // for numeric_limits
#include <queue>    // for queue
#include <sstream>  // for stringstream
#include <string>   // for string, stoi, stoul, to_string
#include <utility>  // for move
#include <vector>   // for vector

#include "../../src/common/threading_utils.h"  // for ParallelFor

namespace xgboost::treeg {

// ========== GraphData Implementation ==========

GraphData GraphData::Build(DMatrix* data, VertexIdx num_vertices, bst_feature_t num_features,
                           WalkLen max_walk_len, bool intercept) {
  CHECK_NOTNULL(data);
  CHECK_GE(num_vertices, 0);
  CHECK_GE(num_features, 0);
  CHECK_GE(max_walk_len, 0);

  // With d = 0 the model sees only X: no adjacency blocks, so no graph data. The virtual
  // intercept is constant 1.0 at d = 0 and can never split, so it is silently ignored.
  if (intercept && max_walk_len == 0) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
      LOG(WARNING) << "Tree-G: intercept=true with max_walk_len=0. The model has no access to "
                   << "graph data, and the intercept is ignored (it is constant at d=0).";
    }
  }

  // The column layout is the only thing that tells X apart from the adjacency blocks, so a
  // mis-shaped matrix would silently train on garbage. Require it exactly.
  auto const expected_cols =
      static_cast<std::uint64_t>(num_features) +
      static_cast<std::uint64_t>(num_vertices) * static_cast<std::uint64_t>(max_walk_len);
  // Size limits. Vertex ids are stored as `int` in the CSR column index, and DMatrix column ids
  // are `bst_feature_t`, which XGBoost fixes at 32 bits.
  CHECK_LE(static_cast<std::int64_t>(num_vertices),
           static_cast<std::int64_t>(std::numeric_limits<int>::max()))
      << "Tree-G graph too large: num_vertices = " << num_vertices
      << " exceeds the limit INT32_MAX = " << std::numeric_limits<int>::max() << ".";
  CHECK_LE(expected_cols, static_cast<std::uint64_t>(std::numeric_limits<bst_feature_t>::max()))
      << "Tree-G graph too large: num_features + num_vertices * max_walk_len = " << num_features
      << " + " << num_vertices << " * " << max_walk_len << " = " << expected_cols
      << " columns exceeds the DMatrix column limit (bst_feature_t max = "
      << std::numeric_limits<bst_feature_t>::max() << "). Reduce max_walk_len or the graph size.";
  CHECK_EQ(data->Info().num_col_, expected_cols)
      << "Tree-G expects the DMatrix column layout [X | A | A^2 | ... | A^d] with "
      << "num_features + num_vertices * max_walk_len = " << num_features << " + " << num_vertices
      << " * " << max_walk_len << " = " << expected_cols << " columns, but the DMatrix has "
      << data->Info().num_col_ << ".";
  // Transductive setting: exactly one row per vertex.
  CHECK_EQ(data->Info().num_row_, static_cast<std::uint64_t>(num_vertices))
      << "Tree-G expects one DMatrix row per vertex: num_vertices = " << num_vertices
      << ", but the DMatrix has " << data->Info().num_row_ << " rows.";

  GraphData result;
  result.num_vertices = num_vertices;
  result.num_features = num_features;
  result.max_walk_len = max_walk_len;

  auto N = static_cast<std::size_t>(num_vertices);
  auto K = static_cast<std::size_t>(num_features);

  // Allocate dense feature matrix (zero-initialized).
  result.features.resize(N * K, 0.0f);

  // Allocate CSR adjacency powers and diagonal vectors.
  result.adj_powers.resize(max_walk_len);
  result.adj_diag.resize(max_walk_len);
  for (WalkLen d = 0; d < max_walk_len; ++d) {
    // Initialize row_ptr with N+1 zeros for CSR construction.
    result.adj_powers[d].nrows = static_cast<int>(N);
    result.adj_powers[d].ncols = static_cast<int>(N);
    result.adj_powers[d].row_ptr.resize(N + 1, 0);
    result.adj_diag[d].resize(N, 0.0f);
  }

  if (num_vertices == 0) {
    return result;
  }

  // ---------- Two-pass CSR construction ----------
  // Tree-G needs the whole graph resident at once: a split aggregates over an
  // arbitrary vertex subset, so no page-at-a-time decomposition exists. Reject
  // multi-page input rather than silently building the graph from page 0.
  CHECK(data->SingleColBlock()) << "Tree-G requires a single in-memory batch, but the input "
                                   "DMatrix has "
                                << data->NumBatches()
                                << " pages. External memory is not supported.";

  // Pass 1: count non-zeros per row for each adjacency power.
  auto batch_iter = data->GetBatches<SparsePage>();
  auto batch = batch_iter.begin();
  CHECK(batch != batch_iter.end()) << "Empty data batch";

  const auto& page = *batch;
  auto view = page.GetView();
  CHECK_EQ(N, view.Size()) << "SparsePage row count does not match num_vertices";

  // Every column index is < num_features + N * max_walk_len (checked above), so every
  // adjacency entry maps to a valid power d_idx < max_walk_len.
  for (VertexIdx i = 0; i < num_vertices; ++i) {
    auto row_i = view[i];
    for (const auto& entry : row_i) {
      auto col = static_cast<bst_feature_t>(entry.index);
      if (col >= num_features && entry.fvalue != 0.0f) {
        bst_feature_t adj_offset = col - num_features;
        auto d_idx = static_cast<WalkLen>(adj_offset / N);
        result.adj_powers[d_idx].row_ptr[i + 1]++;
      }
    }
  }

  // Convert counts to prefix sums and allocate col_idx/values.
  for (WalkLen d = 0; d < max_walk_len; ++d) {
    auto& csr = result.adj_powers[d];
    for (std::size_t i = 0; i < N; ++i) {
      csr.row_ptr[i + 1] += csr.row_ptr[i];
    }
    std::int64_t const nnz = csr.row_ptr.back();
    csr.col_idx.resize(static_cast<std::size_t>(nnz));
    csr.values.resize(static_cast<std::size_t>(nnz));
  }

  // Pass 2: populate X, col_idx, values, and diag.
  // Use a temporary write-position array per CSR to track insertion offsets.
  std::vector<std::vector<std::int64_t>> write_pos(max_walk_len);
  for (WalkLen d = 0; d < max_walk_len; ++d) {
    write_pos[d].resize(N, 0);
  }

  for (VertexIdx i = 0; i < num_vertices; ++i) {
    auto row_i = view[i];
    for (const auto& entry : row_i) {
      auto col = static_cast<bst_feature_t>(entry.index);
      if (col < num_features) {
        // Feature column: X[i, col]
        result.features[static_cast<std::size_t>(i) * K + col] = entry.fvalue;
      } else if (entry.fvalue != 0.0f) {
        // Adjacency block: determine which power d and column j.
        // Skip explicit zeros to keep CSR truly sparse.
        bst_feature_t adj_offset = col - num_features;
        auto d_idx = static_cast<WalkLen>(adj_offset / N);
        auto j = static_cast<VertexIdx>(adj_offset % N);
        auto& csr = result.adj_powers[d_idx];
        std::int64_t const pos = csr.row_ptr[i] + write_pos[d_idx][i];
        csr.col_idx[pos] = static_cast<int>(j);
        csr.values[pos] = entry.fvalue;
        write_pos[d_idx][i]++;

        // Extract diagonal.
        if (j == i) {
          result.adj_diag[d_idx][i] = entry.fvalue;
        }
      }
    }
  }

  return result;
}

// ========== ActiveSet Implementation ==========

ActiveSet::ActiveSet(VertexIdx n, bool init_all)
    : num_vertices_{n}, mask_(n, init_all ? 1.0f : 0.0f) {}

VertexIdx ActiveSet::Count() const {
  VertexIdx count = 0;
  for (float val : mask_) {
    if (val != 0.0f) {
      ++count;
    }
  }
  return count;
}

// ========== TreeGParam Implementation ==========

void TreeGParam::FromJson(Json const& in) {
  auto const& obj = get<Object const>(in);

  this->num_nodes = std::stoi(get<String const>(obj.at("num_nodes")));
  this->num_features = std::stoul(get<String const>(obj.at("num_features")));
  this->max_walk_len = std::stoi(get<String const>(obj.at("max_walk_len")));
  this->max_ancestor_distance = std::stoi(get<String const>(obj.at("max_ancestor_distance")));
  this->num_vertices = std::stoi(get<String const>(obj.at("num_vertices")));

  auto const intercept_it = obj.find(StringView{"intercept"});
  CHECK(intercept_it != obj.cend()) << "Invalid tree model: missing `intercept` in tree_param.";
  this->intercept = std::stoi(get<String const>(intercept_it->second)) != 0;
}

void TreeGParam::ToJson(Json* p_out) const {
  auto& out = *p_out;
  out["num_nodes"] = std::to_string(this->num_nodes);
  out["num_features"] = std::to_string(this->num_features);
  out["max_walk_len"] = std::to_string(this->max_walk_len);
  out["max_ancestor_distance"] = std::to_string(this->max_ancestor_distance);
  out["num_vertices"] = std::to_string(this->num_vertices);
  out["intercept"] = std::to_string(this->intercept ? 1 : 0);
}

// ========== RegTreeG Implementation ==========

RegTreeG::RegTreeG(VertexIdx n_vertices, bst_feature_t n_features, WalkLen max_walk_len,
                   WalkLen max_ancestor_dist, bool intercept)
    : RegTreeG() {
  param_.num_vertices = n_vertices;
  param_.num_features = n_features;
  param_.max_walk_len = max_walk_len;
  param_.max_ancestor_distance = max_ancestor_dist;
  param_.intercept = intercept;
}

bst_node_t RegTreeG::AllocNode() {
  bst_node_t nd = param_.num_nodes++;
  CHECK_LT(param_.num_nodes, std::numeric_limits<bst_node_t>::max())
      << "number of nodes in the tree exceed limit";

  nodes_.resize(param_.num_nodes);
  stats_.resize(param_.num_nodes);

  return nd;
}

void RegTreeG::ExpandNode(bst_node_t nid, bst_feature_t split_index, float split_value,
                          float base_weight, float left_leaf_weight, float right_leaf_weight,
                          float loss_change, float sum_hess, float left_sum, float right_sum) {
  CHECK_LT(nid, param_.num_nodes) << "Invalid node index";
  CHECK(nodes_[nid].IsLeaf()) << "Can only expand leaf nodes";

  bst_node_t pleft = this->AllocNode();
  bst_node_t pright = this->AllocNode();

  auto& node = nodes_[nid];

  // Set up parent-child relationships
  node.SetLeftChild(pleft);
  node.SetRightChild(pright);
  nodes_[pleft].SetParent(nid, true);
  nodes_[pright].SetParent(nid, false);

  // Set split condition
  node.SetSplit(split_index, split_value);

  // Initialize children as leaves
  nodes_[pleft].SetLeaf(left_leaf_weight);
  nodes_[pright].SetLeaf(right_leaf_weight);

  // Set statistics
  stats_[nid] = TreeGNodeStat{loss_change, sum_hess, base_weight};
  stats_[pleft] = TreeGNodeStat{0.0f, left_sum, left_leaf_weight};
  stats_[pright] = TreeGNodeStat{0.0f, right_sum, right_leaf_weight};
}

void RegTreeG::SetNodeTreeGParams(bst_node_t nid, WalkLen walk_len,
                                  RestrictionType restriction_type, bst_node_t ancestor_idx,
                                  bool ancestor_sign) {
  CHECK_LT(nid, param_.num_nodes) << "Invalid node index";
  CHECK_LE(walk_len, param_.max_walk_len) << "Walk length exceeds maximum";

  auto& node = nodes_[nid];
  node.SetWalkLen(walk_len);
  node.SetRestrictionType(restriction_type);
  node.SetAncestorPointer(ancestor_idx, ancestor_sign);
}

void RegTreeG::LoadModel(Json const& in) {
  auto const& obj = get<Object const>(in);

  // Load parameters
  param_.FromJson(obj.at("tree_param"));
  CHECK_GE(param_.num_nodes, 1) << "Invalid tree model: num_nodes must be positive.";
  CHECK_GE(param_.max_walk_len, 0) << "Invalid tree model: negative max_walk_len.";
  CHECK_GE(param_.max_ancestor_distance, 0)
      << "Invalid tree model: negative max_ancestor_distance.";

  // Load nodes
  auto const& nodes_arr = get<Array const>(obj.at("nodes"));
  CHECK_EQ(nodes_arr.size(), param_.num_nodes) << "Node array size mismatch";

  // Start from default-constructed nodes so no state survives from a previous model.
  nodes_.assign(param_.num_nodes, TreeGNode{});
  stats_.assign(param_.num_nodes, TreeGNodeStat{});

  for (size_t i = 0; i < nodes_arr.size(); ++i) {
    auto const& node_obj = get<Object const>(nodes_arr[i]);
    auto& node = nodes_[i];

    // Load basic node fields
    bst_node_t parent = get<Integer const>(node_obj.at("parent"));
    bst_node_t left = get<Integer const>(node_obj.at("left_child"));
    bst_node_t right = get<Integer const>(node_obj.at("right_child"));
    auto const sindex = static_cast<uint32_t>(get<Integer const>(node_obj.at("split_index")));

    node.SetParent(parent & ((1U << 31) - 1), (parent & (1U << 31)) != 0);
    node.SetLeftChild(left);
    node.SetRightChild(right);

    // Check if leaf or split
    if (left == kInvalidNodeId) {
      float leaf_value = get<Number const>(node_obj.at("leaf_value"));
      node.SetLeaf(leaf_value);
    } else {
      // Split node
      float split_cond = get<Number const>(node_obj.at("split_cond"));
      node.SetSplit(sindex, split_cond);

      // Load Tree-G specific fields. The restriction type is validated before the cast:
      // an out-of-range integer cast to the enum would otherwise be accepted silently.
      WalkLen walk_len = get<Integer const>(node_obj.at("walk_len"));
      auto const restriction_type_int = get<Integer const>(node_obj.at("restriction_type"));
      CHECK(restriction_type_int == static_cast<Integer::Int>(RestrictionType::kSource) ||
            restriction_type_int == static_cast<Integer::Int>(RestrictionType::kCycle))
          << "Invalid tree model: node " << i << " has unknown restriction_type "
          << restriction_type_int << ".";
      bst_node_t ancestor_idx = get<Integer const>(node_obj.at("ancestor_index"));
      bool ancestor_sign = get<Boolean const>(node_obj.at("ancestor_sign"));

      node.SetWalkLen(walk_len);
      node.SetRestrictionType(static_cast<RestrictionType>(restriction_type_int));
      node.SetAncestorPointer(ancestor_idx, ancestor_sign);
    }

    // Load statistics
    auto const& stat_obj = get<Object const>(node_obj.at("stats"));
    stats_[i].loss_chg = get<Number const>(stat_obj.at("loss_chg"));
    stats_[i].sum_hess = get<Number const>(stat_obj.at("sum_hess"));
    stats_[i].base_weight = get<Number const>(stat_obj.at("base_weight"));
  }

  // Validate in ascending id order: ValidateNode walks the parent chain, which only visits
  // smaller ids that have therefore already been validated.
  for (bst_node_t nid = 0; nid < param_.num_nodes; ++nid) {
    ValidateNode(nid);
  }
}

void RegTreeG::ValidateNode(bst_node_t nid) const {
  auto const& node = nodes_[nid];

  // Parent: the root has none; every other node has a split-node parent with a smaller id
  // that points back to it.
  if (nid == kRoot) {
    CHECK_EQ(node.Parent(), kInvalidNodeId) << "Invalid tree model: the root has a parent.";
  } else {
    CHECK_LT(node.Parent(), nid) << "Invalid tree model: node " << nid << " has parent "
                                 << node.Parent() << ", expected a smaller node id.";
    auto const& parent = nodes_[node.Parent()];
    CHECK(!parent.IsLeaf()) << "Invalid tree model: parent " << node.Parent() << " of node " << nid
                            << " is not a split node.";
    CHECK_EQ(node.IsLeftChild() ? parent.LeftChild() : parent.RightChild(), nid)
        << "Invalid tree model: parent " << node.Parent() << " does not point back to node " << nid
        << ".";
  }

  if (node.IsLeaf()) {
    return;
  }

  for (bst_node_t const child : {node.LeftChild(), node.RightChild()}) {
    CHECK(child > nid && child < param_.num_nodes)
        << "Invalid tree model: node " << nid << " has child " << child
        << ", expected a node id in (" << nid << ", " << param_.num_nodes << ").";
  }
  CHECK_NE(node.LeftChild(), node.RightChild())
      << "Invalid tree model: node " << nid << " has identical children.";

  bst_feature_t const n_split_features = param_.num_features + (param_.intercept ? 1 : 0);
  CHECK_LT(node.SplitIndex(), n_split_features)
      << "Invalid tree model: node " << nid << " splits on feature " << node.SplitIndex()
      << ", but the tree has " << n_split_features << " split features.";

  CHECK(node.WalkLen() >= 0 && node.WalkLen() <= param_.max_walk_len)
      << "Invalid tree model: node " << nid << " has walk_len " << node.WalkLen()
      << ", expected a value in [0, " << param_.max_walk_len << "].";

  // Ancestor attention: the used subset comes from the node itself (all vertices) or from an
  // ancestor on the root path at most max_ancestor_distance levels up.
  bst_node_t const ancestor = node.AncestorIndex();
  if (ancestor == kInvalidNodeId) {
    return;
  }
  CHECK_LT(ancestor, nid) << "Invalid tree model: node " << nid << " points to ancestor "
                          << ancestor << ", expected a smaller node id.";
  bst_node_t current = nid;
  bool found = false;
  for (WalkLen distance = 1; distance <= param_.max_ancestor_distance; ++distance) {
    current = nodes_[current].Parent();
    if (current == kInvalidNodeId) {
      break;
    }
    if (current == ancestor) {
      found = true;
      break;
    }
  }
  CHECK(found) << "Invalid tree model: node " << nid << " points to node " << ancestor
               << ", which is not an ancestor within max_ancestor_distance = "
               << param_.max_ancestor_distance << ".";
}

void RegTreeG::SaveModel(Json* p_out) const {
  auto& out = *p_out;

  // Save parameters
  Json tree_param{Object()};
  param_.ToJson(&tree_param);
  out["tree_param"] = std::move(tree_param);

  // Save nodes
  Array nodes_arr;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    auto const& node = nodes_[i];
    Json node_obj{Object()};

    // Save basic node fields
    node_obj["parent"] = Integer(node.Parent() | (node.IsLeftChild() ? (1U << 31) : 0));
    node_obj["left_child"] = Integer(node.LeftChild());
    node_obj["right_child"] = Integer(node.RightChild());

    if (node.IsLeaf()) {
      node_obj["leaf_value"] = Number(node.LeafValue());
      node_obj["split_index"] = Integer(0);
      node_obj["split_cond"] = Number(0.0f);
      // Save default Tree-G fields for leaves
      node_obj["walk_len"] = Integer(0);
      node_obj["restriction_type"] = Integer(static_cast<int>(RestrictionType::kSource));
      node_obj["ancestor_index"] = Integer(kInvalidNodeId);
      node_obj["ancestor_sign"] = Boolean{false};
    } else {
      node_obj["split_cond"] = Number(node.SplitCond());
      node_obj["split_index"] = Integer(node.SplitIndex());
      node_obj["leaf_value"] = Number(0.0f);

      // Save Tree-G specific fields
      node_obj["walk_len"] = Integer(node.WalkLen());
      node_obj["restriction_type"] = Integer(static_cast<int>(node.RestrictionTypeValue()));
      node_obj["ancestor_index"] = Integer(node.AncestorIndex());
      node_obj["ancestor_sign"] = Boolean{node.AncestorSign()};
    }

    // Save statistics
    Json stat_obj{Object()};
    stat_obj["loss_chg"] = Number(stats_[i].loss_chg);
    stat_obj["sum_hess"] = Number(stats_[i].sum_hess);
    stat_obj["base_weight"] = Number(stats_[i].base_weight);
    node_obj["stats"] = std::move(stat_obj);

    nodes_arr.GetArray().push_back(std::move(node_obj));
  }
  out["nodes"] = std::move(nodes_arr);
}

std::string RegTreeG::DumpModel(FeatureMap const& fmap, bool with_stats,
                                std::string const& format) const {
  std::stringstream ss;

  if (format == "json") {
    LOG(FATAL) << "JSON format for RegTreeG dump is not yet implemented";
  } else if (format == "dot") {
    LOG(FATAL) << "Graphviz DOT format for RegTreeG dump is not yet implemented";
  } else {
    // Default to text format
    DumpModelImpl(ss, kRoot, 0, fmap, with_stats);
  }

  return ss.str();
}

std::string RegTreeG::SplitFeatureName(FeatureMap const& fmap, bst_feature_t split_index) const {
  // The intercept's index equals num_features, which is also the DMatrix column of A[0,0];
  // the feature-map name of that column must not be used for it.
  if (param_.intercept && split_index == param_.num_features) {
    return "intercept";
  }
  if (split_index < static_cast<bst_feature_t>(fmap.Size())) {
    return fmap.Name(split_index);
  }
  return "f" + std::to_string(split_index);
}

void RegTreeG::DumpModelImpl(std::stringstream& ss, bst_node_t nid, int depth,
                             FeatureMap const& fmap, bool with_stats) const {
  if (nid >= param_.num_nodes) {
    return;
  }

  auto const& node = nodes_[nid];
  std::string tabs(depth, '\t');

  if (node.IsLeaf()) {
    // Leaf node
    ss << tabs << nid << ":leaf=" << node.LeafValue();
    if (with_stats) {
      ss << ",cover=" << stats_[nid].sum_hess;
    }
  } else {
    // Split node. A vertex goes left iff its aggregated value <= threshold, and the threshold
    // is an observed value (not a midpoint), so the comparison is printed as "<=".
    std::string const fname = SplitFeatureName(fmap, node.SplitIndex());

    ss << tabs << nid << ":[" << fname << "<=" << node.SplitCond();

    // Add Tree-G specific information
    ss << ",d=" << node.WalkLen();
    ss << ",r=" << (node.RestrictionTypeValue() == RestrictionType::kSource ? "src" : "cyc");
    ss << ",anc=" << node.AncestorIndex() << (node.AncestorSign() ? "+" : "-");

    ss << "] yes=" << node.LeftChild() << ",no=" << node.RightChild();

    if (with_stats) {
      ss << ",gain=" << stats_[nid].loss_chg;
      ss << ",cover=" << stats_[nid].sum_hess;
    }
  }

  ss << "\n";

  // Recursively dump children
  if (!node.IsLeaf()) {
    DumpModelImpl(ss, node.LeftChild(), depth + 1, fmap, with_stats);
    DumpModelImpl(ss, node.RightChild(), depth + 1, fmap, with_stats);
  }
}

// ========== Graph-Aware Split Computation ==========

std::vector<float> RegTreeG::ComputeSplitValues(WalkLen walk_len, RestrictionType restriction_type,
                                                ActiveSet const& subset,
                                                GraphData const& graph_data,
                                                common::Span<bst_feature_t const> features,
                                                bool with_intercept, std::int32_t n_threads) const {
  CHECK_LE(walk_len, param_.max_walk_len);
  CHECK(!with_intercept || param_.intercept) << "The intercept is not enabled for this tree.";

  auto const n = static_cast<std::size_t>(param_.num_vertices);
  auto const k = static_cast<std::size_t>(param_.num_features);
  auto const s = features.size();
  std::size_t const n_cols = with_intercept ? s + 1 : s;
  for (auto f : features) {
    CHECK_LT(f, param_.num_features);
  }

  // Row i of the gathered feature matrix: X[i, features] (+ 1.0 for the intercept).
  auto gather_row = [&](std::size_t i, float* out) {
    for (std::size_t c = 0; c < s; ++c) {
      out[c] = graph_data.features[i * k + features[c]];
    }
    if (with_intercept) {
      out[s] = 1.0f;
    }
  };

  std::vector<float> result(n * n_cols, 0.0f);

  // d = 0: raw features; the subset is not applied.
  if (walk_len == 0) {
    common::ParallelFor(n, n_threads, [&](std::size_t i) { gather_row(i, &result[i * n_cols]); });
    return result;
  }

  auto const& mask = subset.Mask();

  if (restriction_type == RestrictionType::kSource) {
    // Only walks starting in S: zero the rows of X outside S, then SpMM with A^d.
    std::vector<float> x_masked(n * n_cols, 0.0f);
    common::ParallelFor(n, n_threads, [&](std::size_t j) {
      if (mask[j] != 0.0f) {
        gather_row(j, &x_masked[j * n_cols]);
      }
    });
    auto const& csr = graph_data.adj_powers[walk_len - 1];
    common::ParallelFor(n, n_threads, [&](std::size_t i) {
      for (std::int64_t p = csr.row_ptr[i]; p < csr.row_ptr[i + 1]; ++p) {
        float const a = csr.values[p];
        auto const j = static_cast<std::size_t>(csr.col_idx[p]);
        for (std::size_t c = 0; c < n_cols; ++c) {
          result[i * n_cols + c] += a * x_masked[j * n_cols + c];
        }
      }
    });
    return result;
  }

  // Cycle: only closed walks at vertices of S, i.e. the diagonal of A^d.
  auto const& diag = graph_data.adj_diag[walk_len - 1];
  common::ParallelFor(n, n_threads, [&](std::size_t i) {
    if (mask[i] == 0.0f) {
      return;
    }
    float* row = &result[i * n_cols];
    gather_row(i, row);
    for (std::size_t c = 0; c < n_cols; ++c) {
      row[c] *= diag[i];
    }
  });
  return result;
}

std::vector<float> RegTreeG::ComputeFeatureSplitValues(bst_feature_t feature_idx, WalkLen walk_len,
                                                       RestrictionType restriction_type,
                                                       ActiveSet const& subset,
                                                       GraphData const& graph_data,
                                                       std::int32_t n_threads) const {
  bool const is_intercept = feature_idx == param_.num_features;
  CHECK(feature_idx < param_.num_features || (param_.intercept && is_intercept))
      << "Invalid split feature " << feature_idx << ".";
  std::array<bst_feature_t, 1> const feature{feature_idx};
  auto features = is_intercept ? common::Span<bst_feature_t const>{}
                               : common::Span<bst_feature_t const>{feature};
  return ComputeSplitValues(walk_len, restriction_type, subset, graph_data, features, is_intercept,
                            n_threads);
}

// ========== Attention Subsets ==========

ActiveSet const& ResolveUsedSubset(RegTreeG::TreeGNode const& node,
                                   std::vector<GeneratedSubsets> const& subsets,
                                   ActiveSet const& all_vertices) {
  bst_node_t const ancestor = node.AncestorIndex();
  if (ancestor == RegTreeG::kInvalidNodeId) {
    return all_vertices;
  }
  CHECK_LT(static_cast<std::size_t>(ancestor), subsets.size())
      << "Ancestor pointer " << ancestor << " is out of range.";
  auto const& generated = subsets[ancestor];
  CHECK(generated.populated) << "Ancestor " << ancestor
                             << " has no generated subsets; it must be a split-node "
                             << "processed before its descendants.";
  CHECK_EQ(generated.lte.NumVertices(), all_vertices.NumVertices())
      << "Ancestor " << ancestor << " was generated over a different vertex count.";
  return node.AncestorSign() ? generated.gte : generated.lte;
}

GeneratedSubsets PartitionSubset(std::vector<float> const& split_values, ActiveSet const& used,
                                 float threshold) {
  VertexIdx const n = used.NumVertices();
  CHECK_EQ(split_values.size(), static_cast<std::size_t>(n));

  GeneratedSubsets out{ActiveSet{n, false}, ActiveSet{n, false}, /*populated=*/true};
  used.ForEach([&](VertexIdx j) {
    if (split_values[j] <= threshold) {
      out.lte.Set(j, true);
    } else {
      out.gte.Set(j, true);
    }
  });
  return out;
}

void ReconstructSubsets(RegTreeG const& tree, GraphData const& graph_data,
                        std::vector<GeneratedSubsets>* subsets,
                        std::vector<std::vector<float>>* split_values, std::int32_t n_threads) {
  CHECK(subsets);
  CHECK(split_values);
  CHECK_EQ(graph_data.num_vertices, tree.NumVertices());

  auto const n_nodes = static_cast<std::size_t>(tree.NumNodes());
  subsets->assign(n_nodes, GeneratedSubsets{});
  split_values->assign(n_nodes, std::vector<float>{});

  ActiveSet const all_vertices{tree.NumVertices(), true};
  std::queue<bst_node_t> bfs;
  bfs.push(RegTreeG::kRoot);
  while (!bfs.empty()) {
    bst_node_t const nid = bfs.front();
    bfs.pop();
    auto const& node = tree[nid];
    if (node.IsLeaf()) {
      continue;
    }
    ActiveSet const& used = ResolveUsedSubset(node, *subsets, all_vertices);
    auto phi =
        tree.ComputeFeatureSplitValues(node.SplitIndex(), node.WalkLen(),
                                       node.RestrictionTypeValue(), used, graph_data, n_threads);
    (*subsets)[nid] = PartitionSubset(phi, used, node.SplitCond());
    (*split_values)[nid] = std::move(phi);

    bfs.push(node.LeftChild());
    bfs.push(node.RightChild());
  }
}

}  // namespace xgboost::treeg
