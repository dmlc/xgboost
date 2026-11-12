/**
 * Copyright 2026, XGBoost contributors
 *
 * @brief Implementation of Tree-G updater for graph-aware decision trees
 */
#include "updater.h"

#include <dmlc/omp.h>           // for omp_get_thread_num
#include <xgboost/base.h>       // for kRtEps, GradientPair, bst_node_t, bst_feature_t
#include <xgboost/context.h>    // for Context
#include <xgboost/logging.h>    // for CHECK, LOG
#include <xgboost/parameter.h>  // for DMLC_REGISTER_PARAMETER
#include <xgboost/span.h>       // for Span

#include <algorithm>  // for max, sort
#include <cmath>      // for floor
#include <cstddef>    // for size_t, ptrdiff_t
#include <limits>     // for numeric_limits
#include <numeric>    // for iota
#include <queue>      // for priority_queue
#include <random>     // for mt19937, bernoulli_distribution, uniform_int_distribution
#include <utility>    // for move, pair, swap
#include <vector>     // for vector

#include "../../src/common/threading_utils.h"  // for ParallelFor
#include "../../src/tree/param.h"              // for CalcGain, CalcWeight, IsValidExpandEntry

namespace xgboost::treeg {

// Register parameters
DMLC_REGISTER_PARAMETER(TreeGTrainParam);

void TreeGUpdater::Configure(const Args& args) {
  param_.UpdateAllowUnknown(args);
  Validate();
}

void TreeGUpdater::Configure(TreeGTrainParam const& param) {
  param_ = param;
  Validate();
}

void TreeGUpdater::Validate() {
  CHECK_GT(param_.num_vertices, 0) << "num_vertices must be set (> 0) for booster=treeg";
  CHECK_GT(param_.num_features, 0) << "num_features must be set (> 0) for booster=treeg";
  CHECK_GT(param_.max_leaves, 0) << "max_leaves must be positive";
  // set_range(0, 1) admits 0, which would sample no feature at all.
  CHECK_GT(param_.colsample_bytree, 0.0f) << "colsample_bytree must be positive";

  // Seed the PRNG from the context seed for reproducible restriction-type sampling.
  rng_.seed(static_cast<std::mt19937::result_type>(ctx_->seed));
}

void TreeGUpdater::InitRoot(const std::vector<GradientPair>& gpairs, RegTreeG* tree) {
  CHECK_EQ(gpairs.size(), param_.num_vertices);

  // Initialize tree with graph parameters
  *tree = RegTreeG(param_.num_vertices, param_.num_features, param_.max_walk_len,
                   param_.max_ancestor_distance, param_.intercept);

  // Compute root statistics (all vertices active initially)
  ActiveSet all_vertices(param_.num_vertices, true);
  GradStats root_stats = ComputeGradStats(all_vertices, gpairs);

  // Set root node weight
  auto const root_weight = static_cast<float>(CalcWeight(root_stats));
  (*tree)[0].SetLeaf(root_weight * param_.learning_rate);

  // Set root node statistics
  tree->Stat(0).sum_hess = static_cast<float>(root_stats.sum_hess);
  tree->Stat(0).base_weight = root_weight;
  tree->Stat(0).loss_chg = 0.0f;
}

void TreeGUpdater::Update(const std::vector<GradientPair>& gpairs, const GraphData& graph_data,
                          RegTreeG* tree, std::vector<bst_node_t>* out_position) {
  CHECK_NOTNULL(tree);
  CHECK_EQ(gpairs.size(), param_.num_vertices);

  // Every vertex starts at the root, which reaches all vertices.
  if (out_position != nullptr) {
    out_position->assign(static_cast<std::size_t>(param_.num_vertices), RegTreeG::kRoot);
  }

  // Sample features for this tree (once per tree).
  SampleFeatures();

  // Reset the per-tree attention-subset scratch state.  Sized once, up front,
  // to the maximum node count the growth loop below can reach: it starts with a single root
  // and performs at most (max_leaves - 1) splits, each adding two nodes.  Sizing here means
  // ApplySplit never reallocates, so the ActiveSet references it holds into this buffer
  // cannot dangle.
  all_vertices_ = ActiveSet(param_.num_vertices, /*init_all=*/true);
  subsets_.assign(static_cast<std::size_t>(2 * param_.max_leaves - 1), GeneratedSubsets{});

  // Initialize root node
  InitRoot(gpairs, tree);

  // Node-independent split values for d >= 1, computed once per tree (bounded by budget).
  FillAllVerticesCache(*tree, graph_data);

  // Root's reached set = all vertices
  ActiveSet root_reached(param_.num_vertices, /*init_all=*/true);

  // Priority queue for leaf candidates (max-heap by gain)
  std::priority_queue<LeafCandidate> leaf_queue;

  // Find initial split for root
  SplitEntry root_split = FindBestSplit(0, root_reached, *tree, graph_data, gpairs);
  if (root_split.IsValid() && IsSplitAccepted(root_split.loss_chg)) {
    leaf_queue.emplace(0, std::move(root_split), std::move(root_reached));
  }

  // Greedy best-first expansion
  bst_node_t num_leaves = 1;
  while (!leaf_queue.empty() && num_leaves < param_.max_leaves) {
    // Get leaf with highest gain
    LeafCandidate best_leaf = std::move(const_cast<LeafCandidate&>(leaf_queue.top()));
    leaf_queue.pop();

    // Skip if gain is below threshold
    if (!IsSplitAccepted(best_leaf.best_split.loss_chg)) {
      break;
    }

    // Children's reached sets are the routing partition (D_1 / D_2) of the parent's
    // reached set.  These are distinct from the attention subsets,
    // which partition the parent's *used* subset and are produced by ApplySplit.
    ActiveSet left_reached(best_leaf.best_split.reached_lte);
    ActiveSet right_reached(best_leaf.best_split.reached_gte);

    // Apply the split
    ApplySplit(best_leaf, graph_data, tree);
    num_leaves += 1;  // We now have one more leaf (split creates 2, removes 1)

    // Find splits for the two new children
    bst_node_t left_child = (*tree)[best_leaf.node_id].LeftChild();
    bst_node_t right_child = (*tree)[best_leaf.node_id].RightChild();

    if (out_position != nullptr) {
      UpdatePosition(best_leaf.node_id, left_reached, left_child, right_reached, right_child,
                     out_position);
    }

    SplitEntry left_split = FindBestSplit(left_child, left_reached, *tree, graph_data, gpairs);
    if (left_split.IsValid() && IsSplitAccepted(left_split.loss_chg)) {
      leaf_queue.emplace(left_child, std::move(left_split), std::move(left_reached));
    }

    SplitEntry right_split = FindBestSplit(right_child, right_reached, *tree, graph_data, gpairs);
    if (right_split.IsValid() && IsSplitAccepted(right_split.loss_chg)) {
      leaf_queue.emplace(right_child, std::move(right_split), std::move(right_reached));
    }
  }

  ReleaseAllVerticesCache();

  if (out_position != nullptr) {
    // Each split moves exactly the parent's vertices, so every vertex must end on a leaf.
    auto const& h_pos = *out_position;
    common::ParallelFor(h_pos.size(), ctx_->Threads(), [&](std::size_t v) {
      CHECK((*tree)[h_pos[v]].IsLeaf()) << "Vertex " << v << " is not routed to a leaf.";
    });
  }
}

void TreeGUpdater::UpdatePosition(bst_node_t parent, const ActiveSet& left_reached,
                                  bst_node_t left_child, const ActiveSet& right_reached,
                                  bst_node_t right_child, std::vector<bst_node_t>* position) const {
  auto& h_pos = *position;
  common::ParallelFor(h_pos.size(), ctx_->Threads(), [&](std::size_t i) {
    auto const v = static_cast<VertexIdx>(i);
    bool const in_left = left_reached.Test(v);
    bool const in_right = right_reached.Test(v);
    if (!in_left && !in_right) {
      return;
    }
    // A vertex of the split node goes to exactly one child; no other vertex moves.
    CHECK(in_left != in_right) << "Vertex " << v << " routed to both children of " << parent;
    CHECK_EQ(h_pos[i], parent) << "Vertex " << v << " is not at the node being split.";
    h_pos[i] = in_left ? left_child : right_child;
  });
}

void TreeGUpdater::FillAllVerticesCache(const RegTreeG& tree, const GraphData& graph_data) {
  all_vertices_cache_.clear();
  if (param_.max_walk_len <= 0) {
    FillSortedOrders(graph_data, param_.max_cached_split_bytes);
    return;
  }
  all_vertices_cache_.resize(static_cast<std::size_t>(param_.max_walk_len) * 2);

  std::size_t const s_total = sampled_features_.size() + (param_.intercept ? 1 : 0);
  std::size_t const bytes_per_matrix =
      static_cast<std::size_t>(param_.num_vertices) * s_total * sizeof(float);
  std::size_t used = 0;
  std::size_t n_cached = 0;
  for (WalkLen d = 1; d <= param_.max_walk_len; ++d) {
    for (auto r : {RestrictionType::kSource, RestrictionType::kCycle}) {
      if (n_cached != AllVerticesCacheIndex(d, r) ||
          bytes_per_matrix > param_.max_cached_split_bytes - used) {
        continue;  // Admission stops at the first matrix that does not fit.
      }
      all_vertices_cache_[AllVerticesCacheIndex(d, r)] = tree.ComputeSplitValues(
          d, r, all_vertices_, graph_data, sampled_features_, param_.intercept, ctx_->Threads());
      used += bytes_per_matrix;
      ++n_cached;
    }
  }

  if (n_cached < all_vertices_cache_.size() && !logged_cache_skip_) {
    LOG(INFO) << "Tree-G: cached " << n_cached << "/" << all_vertices_cache_.size()
              << " split-value matrices; raise max_cached_split_bytes for speed.";
    logged_cache_skip_ = true;
  }

  // Matrices take priority (a missing matrix costs an SpMM, a missing order only a sort).
  FillSortedOrders(graph_data, param_.max_cached_split_bytes - used);
}

void TreeGUpdater::FillSortedOrders(const GraphData& graph_data, std::size_t budget_left) {
  sorted_order_.clear();
  sorted_order_.resize(1 + all_vertices_cache_.size());
  auto const N = static_cast<std::size_t>(param_.num_vertices);
  auto const S = static_cast<bst_feature_t>(sampled_features_.size());

  // Sort the N vertex ids of each column by (value, vertex): the same order as std::sort on
  // (value, vertex) pairs in ScanThreshold.
  auto build = [&](std::vector<float> const& matrix, std::size_t stride, bst_feature_t n_cols,
                   auto&& matrix_column) {
    std::vector<VertexIdx> order(N * n_cols);
    common::ParallelFor(n_cols, ctx_->Threads(), [&](bst_feature_t c) {
      auto const col = static_cast<std::size_t>(matrix_column(c));
      auto begin = order.begin() + static_cast<std::ptrdiff_t>(c * N);
      auto end = begin + static_cast<std::ptrdiff_t>(N);
      std::iota(begin, end, 0);
      std::sort(begin, end, [&](VertexIdx a, VertexIdx b) {
        float const va = matrix[static_cast<std::size_t>(a) * stride + col];
        float const vb = matrix[static_cast<std::size_t>(b) * stride + col];
        return va < vb || (!(vb < va) && a < b);
      });
    });
    return order;
  };

  std::size_t used = 0;
  auto admit = [&](bst_feature_t n_cols) {
    std::size_t const bytes = N * n_cols * sizeof(VertexIdx);
    if (n_cols == 0 || bytes > budget_left - used) {
      return false;
    }
    used += bytes;
    return true;
  };

  if (admit(S)) {
    sorted_order_[0] = build(graph_data.features, param_.num_features, S,
                             [&](bst_feature_t c) { return sampled_features_[c]; });
  }
  for (std::size_t i = 0; i < all_vertices_cache_.size(); ++i) {
    auto const& matrix = all_vertices_cache_[i];
    if (matrix.empty()) {
      continue;
    }
    auto const s_total = static_cast<bst_feature_t>(matrix.size() / N);
    if (!admit(s_total)) {
      break;
    }
    sorted_order_[1 + i] = build(matrix, s_total, s_total, [](bst_feature_t c) { return c; });
  }
}

void TreeGUpdater::ReleaseAllVerticesCache() {
  all_vertices_cache_.clear();
  all_vertices_cache_.shrink_to_fit();
  sorted_order_.clear();
  sorted_order_.shrink_to_fit();
}

SplitEntry TreeGUpdater::FindBestSplit(bst_node_t node_id, const ActiveSet& reached_set,
                                       const RegTreeG& tree, const GraphData& graph_data,
                                       const std::vector<GradientPair>& gpairs) {
  SplitEntry best_split;
  const auto& node = tree[node_id];

  CHECK(node.IsLeaf()) << "Can only find splits for leaf nodes";

  // Sample which restriction types to search for this node
  SampledRestrictionConfig restriction_config = SampleRestrictionTypes();

  // Get ancestors for attention mechanism
  auto ancestors = GetAncestors(node_id, tree);

  // Number of sampled features (S). Blocks with d >= 1 add the intercept column.
  bst_feature_t S = static_cast<bst_feature_t>(sampled_features_.size());

  // === d=0: split values = raw X[i,k], mask-independent ===
  // Read graph_data.features directly (stride K, column original_k); no copy needed.
  // The virtual intercept is constant 1.0 at d=0 (no threshold variation), so only the S
  // sampled X columns are searched.
  SearchBlock(SplitBlock{&graph_data.features, param_.num_features, /*walk_len=*/0,
                         RestrictionType::kSource, RegTreeG::kInvalidNodeId,
                         /*ancestor_sign=*/true, S, /*raw_features=*/true,
                         sorted_order_.empty() ? nullptr : &sorted_order_[0]},
              reached_set, gpairs, &best_split);

  // === d >= 1: graph-aware splits ===
  // Loop order: d (outermost) -> r -> used subset -> k (innermost); it decides ties.
  // RegTreeG::ComputeSplitValues returns the N x S_total matrix for all sampled features of
  // one (d, r, subset) block: a CSR SpMM for Source, a diagonal scaling for Cycle.
  for (WalkLen d = 1; d <= param_.max_walk_len; ++d) {
    for (auto r : restriction_config.restriction_types) {
      SearchAllVerticesBlock(d, r, reached_set, tree, graph_data, gpairs, &best_split);
      if (restriction_config.effective_max_ancestor_distance > 0) {
        SearchAncestorBlocks(d, r, node_id, ancestors,
                             restriction_config.effective_max_ancestor_distance, reached_set, tree,
                             graph_data, gpairs, &best_split);
      }
    }
  }

  return best_split;
}

void TreeGUpdater::SearchAllVerticesBlock(WalkLen d, RestrictionType r,
                                          const ActiveSet& reached_set, const RegTreeG& tree,
                                          const GraphData& graph_data,
                                          const std::vector<GradientPair>& gpairs,
                                          SplitEntry* best_split) {
  auto const S_total = static_cast<bst_feature_t>(sampled_features_.size()) +
                       static_cast<bst_feature_t>(param_.intercept ? 1 : 0);
  // Node-independent: use the per-tree cache when present, else recompute.
  std::vector<float> computed;
  auto const& cached =
      all_vertices_cache_.empty() ? computed : all_vertices_cache_[AllVerticesCacheIndex(d, r)];
  if (cached.empty()) {
    computed = tree.ComputeSplitValues(d, r, all_vertices_, graph_data, sampled_features_,
                                       param_.intercept, ctx_->Threads());
  }
  auto const& result_matrix = cached.empty() ? computed : cached;
  // An order exists only for a cached matrix, so it always matches result_matrix.
  const std::vector<VertexIdx>* presorted =
      cached.empty() ? nullptr : &sorted_order_[1 + AllVerticesCacheIndex(d, r)];

  SearchBlock(SplitBlock{&result_matrix, S_total, d, r, RegTreeG::kInvalidNodeId,
                         /*ancestor_sign=*/true, S_total, /*raw_features=*/false, presorted},
              reached_set, gpairs, best_split);
}

void TreeGUpdater::SearchAncestorBlocks(
    WalkLen d, RestrictionType r, bst_node_t node_id,
    const std::vector<std::pair<bst_node_t, WalkLen>>& ancestors, WalkLen max_distance,
    const ActiveSet& reached_set, const RegTreeG& tree, const GraphData& graph_data,
    const std::vector<GradientPair>& gpairs, SplitEntry* best_split) {
  auto const S_total = static_cast<bst_feature_t>(sampled_features_.size()) +
                       static_cast<bst_feature_t>(param_.intercept ? 1 : 0);
  for (const auto& [ancestor_id, distance] : ancestors) {
    if (distance > max_distance) continue;

    // Candidate used subsets are the ancestor's attention subsets S_{a,+}, S_{a,-}.
    // They are held in the updater, not in the tree.
    CHECK_LT(static_cast<std::size_t>(ancestor_id), subsets_.size());
    const auto& ancestor_subsets = subsets_[ancestor_id];
    CHECK(ancestor_subsets.populated)
        << "Ancestor " << ancestor_id << " of node " << node_id
        << " has no generated subsets; ancestors must be split before their descendants.";

    for (bool use_gte : {true, false}) {
      const ActiveSet& attention_set = use_gte ? ancestor_subsets.gte : ancestor_subsets.lte;

      if (attention_set.Count() == 0) continue;

      auto result_matrix = tree.ComputeSplitValues(
          d, r, attention_set, graph_data, sampled_features_, param_.intercept, ctx_->Threads());

      SearchBlock(SplitBlock{&result_matrix, S_total, d, r, ancestor_id, use_gte, S_total,
                             /*raw_features=*/false},
                  reached_set, gpairs, best_split);
    }
  }
}

void TreeGUpdater::SampleFeatures() {
  bst_feature_t K = param_.num_features;

  if (param_.colsample_bytree >= 1.0f) {
    // No sampling — use all features.
    sampled_features_.resize(K);
    std::iota(sampled_features_.begin(), sampled_features_.end(), 0);
    return;
  }

  auto num_sampled = std::max(static_cast<bst_feature_t>(1),
                              static_cast<bst_feature_t>(std::floor(K * param_.colsample_bytree)));

  // Build full index array, then partial Fisher-Yates shuffle.
  sampled_features_.resize(K);
  std::iota(sampled_features_.begin(), sampled_features_.end(), 0);

  for (bst_feature_t i = 0; i < num_sampled; ++i) {
    std::uniform_int_distribution<bst_feature_t> dist(i, K - 1);
    std::swap(sampled_features_[i], sampled_features_[dist(rng_)]);
  }

  sampled_features_.resize(num_sampled);
  std::sort(sampled_features_.begin(), sampled_features_.end());
}

SampledRestrictionConfig TreeGUpdater::SampleRestrictionTypes() {
  std::vector<RestrictionType> sampled_types;
  std::bernoulli_distribution dist(param_.attention_type_sample_probability);

  // Sample kSource (r=1)
  if (dist(rng_)) {
    sampled_types.push_back(RestrictionType::kSource);
  }

  // Sample kCycle (r=2)
  if (dist(rng_)) {
    sampled_types.push_back(RestrictionType::kCycle);
  }

  SampledRestrictionConfig config;
  if (sampled_types.empty()) {
    // Fallback: use kSource only and disable ancestor attention mechanism
    config.restriction_types = {RestrictionType::kSource};
    config.effective_max_ancestor_distance = 0;
  } else {
    config.restriction_types = std::move(sampled_types);
    config.effective_max_ancestor_distance = param_.max_ancestor_distance;
  }

  return config;
}
TreeGUpdater::ThresholdScan TreeGUpdater::ScanThreshold(
    const std::vector<float>& result_matrix, bst_feature_t stride, bst_feature_t column_idx,
    const ActiveSet& reached_set, const std::vector<GradientPair>& gpairs,
    std::vector<std::pair<float, VertexIdx>>* scratch,
    common::Span<VertexIdx const> presorted) const {
  ThresholdScan out;

  // Extract split values for this feature from the pre-computed result matrix.
  // result_matrix is N × stride, row-major: vertex i, column j →
  //   result_matrix[i * stride + j].
  auto& split_values = *scratch;
  split_values.clear();
  auto const n = static_cast<double>(reached_set.Count());
  split_values.reserve(reached_set.Count());

  auto value_of = [&](VertexIdx v) {
    return result_matrix[static_cast<std::size_t>(v) * stride + column_idx];
  };
  if (!presorted.empty() && n > 1 && n * std::log2(n) > static_cast<double>(presorted.size())) {
    // Filtered walk over the (value, vertex) order: O(N), same sequence as the sort below.
    for (auto v : presorted) {
      if (reached_set.Test(v)) {
        split_values.emplace_back(value_of(v), v);
      }
    }
  } else {
    reached_set.ForEach([&](VertexIdx v) { split_values.emplace_back(value_of(v), v); });
    // Sort by split value (ascending order), ties by vertex id.
    std::sort(split_values.begin(), split_values.end());
  }

  // Skip if too few vertices
  if (split_values.size() < 2) {
    return out;
  }

  // Compute parent statistics (sum over all vertices in reached set)
  GradStats parent_stats;
  for (const auto& [value, vertex_idx] : split_values) {
    const auto& gpair = gpairs[vertex_idx];
    parent_stats.sum_grad += gpair.GetGrad();
    parent_stats.sum_hess += gpair.GetHess();
  }

  // Single linear scan with incremental statistics accumulation.
  // As we scan left-to-right through sorted values:
  // - left_stats accumulates vertices seen so far (will go to lte side)
  // - right_stats = parent_stats - left_stats (will go to gt side)
  GradStats left_stats;
  float last_value = split_values[0].first;
  double local_best_gain = 0.0;

  for (const auto& [value, vertex_idx] : split_values) {
    // When we encounter a new value, evaluate split at the boundary.
    if (value != last_value && !left_stats.IsEmpty()) {
      GradStats right_stats = parent_stats;
      right_stats -= left_stats;

      if (left_stats.HasSufficientWeight(param_.min_child_weight) &&
          right_stats.HasSufficientWeight(param_.min_child_weight)) {
        double const gain = CalcGain(parent_stats, left_stats, right_stats);

        if (gain > local_best_gain) {
          local_best_gain = gain;
          out.left_stats = left_stats;
          out.right_stats = right_stats;
          out.threshold = last_value;
        }
      }
    }

    // Accumulate current vertex into left statistics
    const auto& gpair = gpairs[vertex_idx];
    left_stats.sum_grad += gpair.GetGrad();
    left_stats.sum_hess += gpair.GetHess();
    last_value = value;
  }

  // loss_chg is stored as float (as in gbtree's SplitEntry); compare at that precision.
  out.loss_chg = static_cast<float>(local_best_gain);
  return out;
}

void TreeGUpdater::MaterialiseReached(const std::vector<float>& result_matrix, bst_feature_t stride,
                                      bst_feature_t column_idx, const ActiveSet& reached_set,
                                      SplitEntry* split) const {
  // Build the routing partition (D_1 / D_2 of Algorithm 2) only for the winning
  // candidate.  This partitions the *reached* set and drives the children's reached
  // sets; the attention subsets are produced later by ApplySplit.
  ActiveSet reached_lte(param_.num_vertices, false);
  ActiveSet reached_gte(param_.num_vertices, false);
  reached_set.ForEach([&](VertexIdx v) {
    float val = result_matrix[static_cast<std::size_t>(v) * stride + column_idx];
    if (val <= split->threshold) {
      reached_lte.Set(v, true);
    } else {
      reached_gte.Set(v, true);
    }
  });
  split->reached_lte = std::move(reached_lte);
  split->reached_gte = std::move(reached_gte);
}

void TreeGUpdater::SearchBlock(const SplitBlock& block, const ActiveSet& reached_set,
                               const std::vector<GradientPair>& gpairs, SplitEntry* best_split) {
  CHECK_NOTNULL(best_split);
  auto const S = static_cast<bst_feature_t>(sampled_features_.size());
  auto const n_threads = ctx_->Threads();
  if (scan_scratch_.size() < static_cast<std::size_t>(n_threads)) {
    scan_scratch_.resize(n_threads);
  }

  // Column c of the block -> (matrix column, original feature index).
  auto matrix_column = [&](bst_feature_t c) {
    return block.raw_features ? sampled_features_[c] : c;
  };
  auto feature_of = [&](bst_feature_t c) {
    return c < S ? sampled_features_[c] : param_.num_features;
  };

  // One scalar best per thread: highest float gain, then lowest column.
  struct Best {
    ThresholdScan scan;
    bst_feature_t column{std::numeric_limits<bst_feature_t>::max()};
  };
  auto better = [](ThresholdScan const& s, bst_feature_t c, Best const& b) {
    return s.loss_chg > b.scan.loss_chg || (s.loss_chg == b.scan.loss_chg && c < b.column);
  };
  std::vector<Best> thread_best(n_threads);

  auto const N = static_cast<std::size_t>(param_.num_vertices);
  auto presorted_of = [&](bst_feature_t c) {
    if (block.presorted == nullptr || block.presorted->empty()) {
      return common::Span<VertexIdx const>{};
    }
    return common::Span<VertexIdx const>{block.presorted->data() + c * N, N};
  };

  common::ParallelFor(block.n_columns, n_threads, [&](bst_feature_t c) {
    auto const tid = omp_get_thread_num();
    auto scan = ScanThreshold(*block.matrix, block.stride, matrix_column(c), reached_set, gpairs,
                              &scan_scratch_[tid], presorted_of(c));
    if (scan.loss_chg > 0.0f && better(scan, c, thread_best[tid])) {
      thread_best[tid] = Best{scan, c};
    }
  });

  Best winner;
  for (auto const& b : thread_best) {
    if (b.scan.loss_chg > 0.0f && better(b.scan, b.column, winner)) {
      winner = b;
    }
  }

  // Earlier blocks win ties: replace only on a strictly greater gain.
  if (winner.scan.loss_chg <= 0.0f || winner.scan.loss_chg <= best_split->loss_chg) {
    return;
  }

  best_split->split_feature = feature_of(winner.column);
  best_split->threshold = winner.scan.threshold;
  best_split->walk_len = block.walk_len;
  best_split->restriction_type = block.restriction_type;
  best_split->ancestor_idx = block.ancestor_idx;
  best_split->ancestor_sign = block.ancestor_sign;
  best_split->left_stats = winner.scan.left_stats;
  best_split->right_stats = winner.scan.right_stats;
  best_split->loss_chg = winner.scan.loss_chg;
  MaterialiseReached(*block.matrix, block.stride, matrix_column(winner.column), reached_set,
                     best_split);
}

std::vector<std::pair<bst_node_t, WalkLen>> TreeGUpdater::GetAncestors(bst_node_t node_id,
                                                                       const RegTreeG& tree) {
  std::vector<std::pair<bst_node_t, WalkLen>> ancestors;

  bst_node_t current = node_id;
  WalkLen distance = 0;

  while (current != RegTreeG::kInvalidNodeId) {
    bst_node_t parent = tree[current].Parent();
    if (parent == RegTreeG::kInvalidNodeId) break;

    distance++;
    ancestors.emplace_back(parent, distance);
    current = parent;

    if (distance >= param_.max_ancestor_distance) break;
  }

  return ancestors;
}

double TreeGUpdater::CalcGain(const GradStats& parent_stats, const GradStats& left_stats,
                              const GradStats& right_stats) const {
  // Same arithmetic as gbtree: tree::CalcGain is instantiated with TreeGTrainParam, which
  // exposes reg_alpha, reg_lambda and max_delta_step (compile-time proof of compatibility).
  // No 0.5 factor, so min_split_loss has the gbtree scale.
  auto node_gain = [this](GradStats const& s) {
    return tree::CalcGain(param_, s.sum_grad, s.sum_hess);
  };
  return node_gain(left_stats) + node_gain(right_stats) - node_gain(parent_stats);
}

double TreeGUpdater::CalcWeight(const GradStats& stats) const {
  return tree::CalcWeight(param_, stats.sum_grad, stats.sum_hess);
}

bool TreeGUpdater::IsSplitAccepted(float loss_chg) const {
  // Mirrors tree::IsValidExpandEntry (src/tree/driver.h).
  return loss_chg > kRtEps && !(loss_chg < param_.min_split_loss);
}

void TreeGUpdater::ApplySplit(const LeafCandidate& leaf, const GraphData& graph_data,
                              RegTreeG* tree) {
  const auto& split = leaf.best_split;

  // Calculate leaf weights and apply the learning rate.
  auto const left_weight = static_cast<float>(CalcWeight(split.left_stats) * param_.learning_rate);
  auto const right_weight =
      static_cast<float>(CalcWeight(split.right_stats) * param_.learning_rate);

  // Calculate parent weight for statistics
  GradStats parent_stats = split.left_stats + split.right_stats;
  auto const parent_weight = static_cast<float>(CalcWeight(parent_stats));

  // Expand the node in the tree
  tree->ExpandNode(leaf.node_id, split.split_feature, split.threshold, parent_weight, left_weight,
                   right_weight, split.loss_chg, static_cast<float>(parent_stats.sum_hess),
                   static_cast<float>(split.left_stats.sum_hess),
                   static_cast<float>(split.right_stats.sum_hess));

  // Store the learned split parameters.  Only these are persisted; the generated
  // subsets below are graph-dependent and therefore stay in the updater.
  tree->SetNodeTreeGParams(leaf.node_id, split.walk_len, split.restriction_type, split.ancestor_idx,
                           split.ancestor_sign);

  // ---- Generate the node's subsets S_{u,+}, S_{u,-} ----
  // The subsets partition the node's *used* subset S_{*u,rho_u} over all vertices of the
  // graph, including zero-weight (test) vertices.  They are deliberately not derived from
  // the reached set: a vertex may belong to S_{*u,rho_u} without ever reaching this node.
  //
  // subsets_ was sized for the whole tree in Update(); it is never resized here, so the
  // reference `used` below stays valid for the duration of this function.
  CHECK_LT(static_cast<std::size_t>(leaf.node_id), subsets_.size())
      << "Generated-subset buffer is too small for node " << leaf.node_id << ".";

  const ActiveSet& used = ResolveUsedSubset((*tree)[leaf.node_id], subsets_, all_vertices_);
  auto phi =
      tree->ComputeFeatureSplitValues(split.split_feature, split.walk_len, split.restriction_type,
                                      used, graph_data, ctx_->Threads());
  subsets_[leaf.node_id] = PartitionSubset(phi, used, split.threshold);

  // Invariant: recomputing the chosen split from its parameters routes the reached set
  // exactly as the split search did. A mismatch would mean the stored parameters do not
  // describe the scored candidate, so prediction would disagree with training.
  leaf.reached_set.ForEach([&](VertexIdx v) {
    bool const goes_left = phi[v] <= split.threshold;
    CHECK_EQ(goes_left, split.reached_lte.Test(v))
        << "Recomputed split disagrees with the split search for vertex " << v << " at node "
        << leaf.node_id << ": training and prediction would route it differently.";
  });
}

GradStats TreeGUpdater::ComputeGradStats(const ActiveSet& active_set,
                                         const std::vector<GradientPair>& gpairs) const {
  GradStats stats;

  active_set.ForEach([&](VertexIdx vertex_idx) {
    CHECK_LT(vertex_idx, gpairs.size());

    const auto& gpair = gpairs[vertex_idx];

    // Gradients already include sample-weight scaling (applied by the
    // XGBoost objective), so we accumulate them directly — matching
    // the convention used by XGBoost's built-in tree updaters.
    stats.sum_grad += gpair.GetGrad();
    stats.sum_hess += gpair.GetHess();
  });

  return stats;
}

}  // namespace xgboost::treeg
