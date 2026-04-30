/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cuda/std/utility>  // for swap
#include <cuda/std/variant>  // for variant
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../common/categorical.h"
#include "../../common/common.h"
#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../common/cuda_rt_utils.h"   // for SetDevice
#include "../../common/device_helpers.cuh"
#include "../../common/math.h"
#include "../../common/nvtx_utils.h"
#include "../../common/optional_weight.h"
#include "../../data/batch_utils.h"      // for StaticBatch
#include "../../data/cat_container.cuh"  // for EncPolicy, MakeCatAccessor
#include "../../data/cat_container.h"    // for NoOpAccessor
#include "../../data/ellpack_page.cuh"
#include "../../gbm/gbtree_model.h"
#include "../../tree/tree_view.h"
#include "../gbtree_view.h"
#include "../gpu_data_accessor.cuh"
#include "../predict_fn.h"  // for GetTreeLimit
#include "quadrature.h"
#include "shap.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/linalg.h"  // for UnravelIndex
#include "xgboost/logging.h"
#include "xgboost/multi_target_tree_model.h"  // for MTNotImplemented

namespace xgboost::interpretability::cuda_impl {
namespace {
using predictor::EllpackLoader;
using predictor::GBTreeModelView;
using predictor::SparsePageLoaderNoShared;
using predictor::SparsePageView;
using ::xgboost::cuda_impl::StaticBatch;

using TreeViewVar = cuda::std::variant<tree::ScalarTreeView, tree::MultiTargetTreeView>;

constexpr std::size_t kGpuQuadraturePoints = 8;
constexpr std::size_t kMaxGpuQuadratureDepth = 64;
constexpr std::size_t kGpuQuadratureRowsPerWarp = 4;
constexpr std::size_t kGpuQuadratureTreeBlockThreads = 64;
constexpr std::array<std::size_t, 3> kGpuQuadratureDepthBuckets{{16, 32, 64}};
using QuadratureRule = detail::QuadratureTreeShapRule;
std::vector<float> MakeGroupRootMeanSums(gbm::GBTreeModel const& model, bst_tree_t tree_end,
                                         std::vector<float> const* tree_weights) {
  auto h_tree_groups = model.TreeGroups(DeviceOrd::CPU());
  auto n_groups = model.learner_model_param->num_output_group;
  std::vector<double> h_group_root_mean_sums(n_groups, 0.0);
  for (bst_tree_t tree_idx = 0; tree_idx < tree_end; ++tree_idx) {
    auto const weight = tree_weights == nullptr ? 1.0f : (*tree_weights)[tree_idx];
    auto const tree = model.trees.at(tree_idx)->HostScView();
    h_group_root_mean_sums[h_tree_groups[tree_idx]] +=
        detail::FillRootMeanValue(tree, RegTree::kRoot) * weight;
  }

  std::vector<float> out(h_group_root_mean_sums.size());
  std::transform(h_group_root_mean_sums.cbegin(), h_group_root_mean_sums.cend(), out.begin(),
                 [](double v) { return static_cast<float>(v); });
  return out;
}

struct CompressedNode {
  bst_node_t left{RegTree::kInvalidNodeId};
  bst_node_t right{RegTree::kInvalidNodeId};
  bst_feature_t split_global{0};
  float split_cond{0};
  float leaf_value{0};
  float left_weight{0};
  float right_weight{0};
  std::uint32_t cat_begin{0};
  std::uint32_t cat_size{0};
  std::uint8_t default_left{0};
  std::uint8_t is_leaf{0};
  std::uint8_t is_categorical{0};
  std::uint8_t prev_same_offset_plus1{0};
};

struct CompressedTree {
  std::uint32_t node_begin{0};
  bst_target_t group{0};
};

struct CompressedModel {
  dh::device_vector<CompressedTree> trees;
  dh::device_vector<CompressedNode> nodes;
  dh::device_vector<std::uint32_t> categories;
};

std::size_t DepthBucketIndex(std::size_t path_depth) {
  for (std::size_t i = 0; i < kGpuQuadratureDepthBuckets.size(); ++i) {
    if (path_depth <= kGpuQuadratureDepthBuckets[i]) {
      return i;
    }
  }
  LOG(FATAL) << "GPU QuadratureSHAP currently supports trees of depth up to "
             << (kMaxGpuQuadratureDepth - 1) << ".";
  return kGpuQuadratureDepthBuckets.size() - 1;
}

CompressedModel MakeCompressedModel(Context const* ctx, gbm::GBTreeModel const& model,
                                    std::vector<bst_tree_t> const& tree_indices,
                                    std::vector<float> const* tree_weights) {
  std::vector<CompressedTree> h_trees;
  std::vector<CompressedNode> h_nodes;
  std::vector<std::uint32_t> h_categories;
  auto h_tree_groups = model.TreeGroups(DeviceOrd::CPU());
  static_cast<void>(ctx);

  h_trees.reserve(tree_indices.size());
  for (auto tree_idx : tree_indices) {
    auto const weight = tree_weights == nullptr ? 1.0f : (*tree_weights)[tree_idx];
    auto const tree = model.trees.at(tree_idx)->HostScView();

    auto node_begin = h_nodes.size();
    h_nodes.resize(node_begin + tree.Size());

    for (bst_node_t nidx = 0; nidx < tree.Size(); ++nidx) {
      auto& out = h_nodes[node_begin + nidx];
      if (tree.IsLeaf(nidx)) {
        out.is_leaf = 1;
        out.leaf_value = tree.LeafValue(nidx) * weight;
        continue;
      }

      auto left = tree.LeftChild(nidx);
      auto right = tree.RightChild(nidx);
      auto parent_cover = static_cast<double>(tree.SumHess(nidx));
      CHECK_GT(parent_cover, 0.0);

      out.left = left;
      out.right = right;
      out.split_global = tree.SplitIndex(nidx);
      out.split_cond = tree.SplitCond(nidx);
      out.left_weight = detail::GuardChildWeight(tree.SumHess(left), tree.SumHess(nidx));
      out.right_weight = detail::GuardChildWeight(tree.SumHess(right), tree.SumHess(nidx));
      if (common::IsCat(tree.cats.split_type, nidx)) {
        auto node_cats = tree.NodeCats(nidx);
        CHECK_LE(node_cats.size(),
                 static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()));
        CHECK_LE(h_categories.size(),
                 static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()));
        out.cat_begin = static_cast<std::uint32_t>(h_categories.size());
        out.cat_size = static_cast<std::uint32_t>(node_cats.size());
        out.is_categorical = 1;
        h_categories.insert(h_categories.end(), node_cats.begin(), node_cats.end());
      }
      out.default_left = tree.DefaultLeft(nidx);
      out.is_leaf = 0;
    }

    for (bst_node_t nidx = 0; nidx < tree.Size(); ++nidx) {
      auto& out = h_nodes[node_begin + nidx];
      if (out.is_leaf) {
        continue;
      }

      std::uint8_t prev_same_offset_plus1 = 0;
      std::uint16_t distance = 0;
      auto ancestor = nidx;
      while (!tree.IsRoot(ancestor)) {
        ancestor = tree.Parent(ancestor);
        ++distance;
        auto const& ancestor_node = h_nodes[node_begin + ancestor];
        if (!ancestor_node.is_leaf && ancestor_node.split_global == out.split_global) {
          prev_same_offset_plus1 = static_cast<std::uint8_t>(distance + 1);
          break;
        }
      }
      out.prev_same_offset_plus1 = prev_same_offset_plus1;
    }

    h_trees.push_back(
        CompressedTree{static_cast<std::uint32_t>(node_begin), h_tree_groups[tree_idx]});
  }

  CompressedModel out;
  out.trees = dh::device_vector<CompressedTree>(h_trees.cbegin(), h_trees.cend());
  out.nodes = dh::device_vector<CompressedNode>(h_nodes.cbegin(), h_nodes.cend());
  out.categories = dh::device_vector<std::uint32_t>(h_categories.cbegin(), h_categories.cend());
  return out;
}

template <int RowsPerWarp>
XGBOOST_DEVICE constexpr unsigned ActiveSubgroupMask(int row_slot) {
  constexpr int kSegmentWidth = dh::WarpThreads() / RowsPerWarp;
  static_assert(kSegmentWidth >= static_cast<int>(kGpuQuadraturePoints));
  return ((1u << kGpuQuadraturePoints) - 1u) << (row_slot * kSegmentWidth);
}

XGBOOST_DEVICE inline float PreviousPathProbability(std::uint8_t prev_same_offset_plus1, int depth,
                                                    float const* q_vals_row) {
  if (prev_same_offset_plus1 == 0) {
    return 1.0f;
  }
  auto prev_depth = depth - static_cast<int>(prev_same_offset_plus1) + 1;
  return q_vals_row[prev_depth];
}

XGBOOST_DEVICE inline float ExtractQuadratureEdgeDeltaLocal(float quad_node, float quad_weight,
                                                            float ret_val, float p_enter,
                                                            float q_prev) {
  auto weighted_ret = ret_val * quad_weight;
  float edge_delta = 0.0f;
  if (p_enter != 1.0f) {
    auto alpha_enter = p_enter - 1.0f;
    edge_delta += alpha_enter / (1.0f + alpha_enter * quad_node);
  }
  if (q_prev != 1.0f) {
    auto alpha_exit = q_prev - 1.0f;
    edge_delta -= alpha_exit / (1.0f + alpha_exit * quad_node);
  }
  return weighted_ret * edge_delta;
}

XGBOOST_DEVICE inline float ExtractQuadratureInteractionDeltaLocal(float quad_node,
                                                                   float edge_delta_local,
                                                                   float q_partner) {
  if (q_partner == 1.0f) {
    return 0.0f;
  }
  auto alpha_partner = q_partner - 1.0f;
  return alpha_partner * edge_delta_local / (1.0f + alpha_partner * quad_node);
}

template <typename Loader>
struct IsSparsePageLoaderNoShared : std::false_type {};

template <typename EncAccessor>
struct IsSparsePageLoaderNoShared<SparsePageLoaderNoShared<EncAccessor>> : std::true_type {};

// Encapsulate the tail-tile versus full-tile differences so the traversal code can focus on
// probability updates instead of mask plumbing.
template <bool kHasRowMask, int RowsPerWarp, int MaxPoints>
struct SubgroupOps {
  static constexpr int kRowsPerWarpValue = RowsPerWarp;
  static constexpr int kSegmentWidth = dh::WarpThreads() / RowsPerWarp;
  static constexpr unsigned kFullMask = 0xffffffffu;

  int row_slot;
  int point;
  unsigned subgroup_mask;
  unsigned warp_mask;
  bool row_valid;
  bool is_leader;
  bool is_warp_leader;

  XGBOOST_DEV_INLINE SubgroupOps(int lane, bst_idx_t valid_rows_in_tail)
      : row_slot{lane / kSegmentWidth},
        point{lane % kSegmentWidth},
        subgroup_mask{kFullMask},
        warp_mask{kFullMask},
        row_valid{true},
        is_leader{point == 0},
        is_warp_leader{lane == 0} {
    if constexpr (kHasRowMask) {
      subgroup_mask = ActiveSubgroupMask<RowsPerWarp>(row_slot);
      warp_mask = __activemask();
      row_valid = static_cast<bst_idx_t>(row_slot) < valid_rows_in_tail;
    }
  }

  [[nodiscard]] XGBOOST_DEV_INLINE bool Participates() const { return point < MaxPoints; }

  [[nodiscard]] XGBOOST_DEV_INLINE bool RowActive() const {
    if constexpr (kHasRowMask) {
      return row_valid;
    } else {
      return true;
    }
  }

  [[nodiscard]] XGBOOST_DEV_INLINE bool ShouldWrite() const {
    return is_leader && this->RowActive();
  }

  template <typename T>
  [[nodiscard]] XGBOOST_DEV_INLINE T Broadcast(T value) const {
    // Each row uses an independent kGpuQuadraturePoints-wide subgroup inside the warp.
    if constexpr (kHasRowMask) {
      return __shfl_sync(subgroup_mask, value, 0, MaxPoints);
    } else {
      return __shfl_sync(kFullMask, value, 0, MaxPoints);
    }
  }

  template <typename T>
  [[nodiscard]] XGBOOST_DEV_INLINE T Sum(T value) const {
    for (int offset = MaxPoints / 2; offset > 0; offset /= 2) {
      if constexpr (kHasRowMask) {
        value += __shfl_down_sync(subgroup_mask, value, offset, MaxPoints);
      } else {
        value += __shfl_down_sync(kFullMask, value, offset, MaxPoints);
      }
    }
    return value;
  }

  XGBOOST_DEV_INLINE void Sync() const {
    if constexpr (kHasRowMask) {
      __syncwarp(warp_mask);
    } else {
      __syncwarp();
    }
  }
};

// Wrap the shared-memory layout in semantic accessors so the task runner talks in terms of path
// state instead of raw multidimensional indexing.
template <int MaxPoints, int RowsPerWarp, int DepthCap, int kWarpsPerBlock, bool kUseQPrevCache>
struct QuadratureSharedState {
  bst_node_t (&nodes)[kWarpsPerBlock][DepthCap];
  std::uint8_t (&stages)[kWarpsPerBlock][DepthCap];
  std::uint8_t (&goes_left)[kWarpsPerBlock][RowsPerWarp][DepthCap];
  // q_d(t): path probability at depth d for one row-slot evaluated at quadrature point t.
  float (&path_prob)[kWarpsPerBlock][RowsPerWarp][DepthCap];
  // G_d(t): multiplicative basis carried down the path before the leaf value is applied.
  float (&basis)[kWarpsPerBlock][RowsPerWarp][DepthCap][MaxPoints];
  float (&q_prev_cache)[kUseQPrevCache ? kWarpsPerBlock : 1][kUseQPrevCache ? RowsPerWarp : 1]
                       [kUseQPrevCache ? DepthCap : 1];

  [[nodiscard]] XGBOOST_DEV_INLINE bst_node_t& Node(int warp, int depth) {
    return nodes[warp][depth];
  }

  [[nodiscard]] XGBOOST_DEV_INLINE bst_node_t const& Node(int warp, int depth) const {
    return nodes[warp][depth];
  }

  [[nodiscard]] XGBOOST_DEV_INLINE std::uint8_t& Stage(int warp, int depth) {
    return stages[warp][depth];
  }

  [[nodiscard]] XGBOOST_DEV_INLINE std::uint8_t const& Stage(int warp, int depth) const {
    return stages[warp][depth];
  }

  [[nodiscard]] XGBOOST_DEV_INLINE bool GoesLeft(int warp, int row_slot, int depth) const {
    return static_cast<bool>(goes_left[warp][row_slot][depth]);
  }

  XGBOOST_DEV_INLINE void SetGoesLeft(int warp, int row_slot, int depth, bool value) {
    goes_left[warp][row_slot][depth] = static_cast<std::uint8_t>(value);
  }

  [[nodiscard]] XGBOOST_DEV_INLINE float& PathProbability(int warp, int row_slot, int depth) {
    return path_prob[warp][row_slot][depth];
  }

  [[nodiscard]] XGBOOST_DEV_INLINE float const* PathProbabilityRow(int warp, int row_slot) const {
    return path_prob[warp][row_slot];
  }

  [[nodiscard]] XGBOOST_DEV_INLINE float& Basis(int warp, int row_slot, int depth, int point) {
    return basis[warp][row_slot][depth][point];
  }

  [[nodiscard]] XGBOOST_DEV_INLINE float LoadQPrev(int warp, int row_slot, int depth,
                                                   std::uint8_t prev_same_offset_plus1) const {
    if constexpr (kUseQPrevCache) {
      return q_prev_cache[warp][row_slot][depth];
    } else {
      return PreviousPathProbability(prev_same_offset_plus1, depth,
                                     this->PathProbabilityRow(warp, row_slot));
    }
  }

  XGBOOST_DEV_INLINE void StoreQPrev(int warp, int row_slot, int depth, float q_prev) {
    if constexpr (kUseQPrevCache) {
      q_prev_cache[warp][row_slot][depth] = q_prev;
    }
  }
};

template <typename Loader, typename SubgroupT, typename SharedT>
struct QuadratureShapTaskRunner {
  Loader loader;
  SubgroupT subgroup;
  SharedT shared;
  CompressedTree const* trees;
  CompressedNode const* nodes;
  std::uint32_t const* categories;
  float* phis;
  bst_idx_t base_rowid;
  bst_target_t n_groups;
  bst_feature_t n_columns;
  std::size_t row_tile_begin;
  std::size_t row_tiles;
  int warp;
  float quad_node;
  float quad_weight;

  [[nodiscard]] XGBOOST_DEV_INLINE bool EvaluateGoesLeft(bst_idx_t ridx,
                                                         CompressedNode const& node) const {
    auto fvalue = loader.GetElement(ridx, node.split_global);
    if (common::CheckNAN(fvalue)) {
      return static_cast<bool>(node.default_left);
    }
    if (node.is_categorical) {
      auto cats = common::Span<std::uint32_t const>{categories + node.cat_begin, node.cat_size};
      return common::Decision(cats, fvalue);
    }
    return fvalue < node.split_cond;
  }

  XGBOOST_DEV_INLINE void AddContribution(bst_idx_t row_idx, bst_target_t tree_group,
                                          bst_feature_t split_global, float contrib) const {
    if (!subgroup.ShouldWrite()) {
      return;
    }
    auto out_row = phis + (row_idx * n_groups + tree_group) * n_columns;
    atomicAdd(out_row + split_global, contrib);
  }

  XGBOOST_DEV_INLINE void InitializeTask() {
    if (subgroup.is_warp_leader) {
      shared.Node(warp, 0) = RegTree::kRoot;
      shared.Stage(warp, 0) = 0;
    }
    // Start each row with G_0(t) = 1 at every quadrature node.
    shared.Basis(warp, subgroup.row_slot, 0, subgroup.point) = 1.0f;
    subgroup.Sync();
  }

  XGBOOST_DEV_INLINE bool HandleReturn(bst_idx_t row_idx, bst_target_t tree_group,
                                       CompressedNode const* nodes_for_tree, int* stack_size,
                                       bool* have_return, float* ret_val) {
    if (*stack_size == 0) {
      return false;
    }

    int parent_depth = *stack_size - 1;
    auto const& node = nodes_for_tree[shared.Node(warp, parent_depth)];
    int child_idx = static_cast<int>(shared.Stage(warp, parent_depth)) - 1;

    float p_enter = 0.0f;
    float q_prev = 1.0f;
    if (subgroup.is_leader && subgroup.RowActive()) {
      p_enter = shared.PathProbability(warp, subgroup.row_slot, parent_depth);
      q_prev = shared.LoadQPrev(warp, subgroup.row_slot, parent_depth, node.prev_same_offset_plus1);
    }
    p_enter = subgroup.Broadcast(p_enter);
    q_prev = subgroup.Broadcast(q_prev);

    // Extraction uses
    //   H * w(t) * ret_val *
    //   [ (p_enter - 1) / (1 + (p_enter - 1) t)
    //   - (q_prev  - 1) / (1 + (q_prev  - 1) t) ].
    // The two rational terms are the "enter current feature" and "rewind to previous same
    // feature" adjustments from the quadrature recurrence.
    float contrib =
        ExtractQuadratureEdgeDeltaLocal(quad_node, quad_weight, *ret_val, p_enter, q_prev);
    contrib = subgroup.Sum(contrib);
    this->AddContribution(row_idx, tree_group, node.split_global, contrib);

    if (child_idx == 0) {
      auto child_weight = node.right_weight;
      auto child_node = node.right;
      float p_e = 0.0f;
      if (subgroup.is_leader) {
        if (subgroup.RowActive()) {
          auto goes_left = shared.GoesLeft(warp, subgroup.row_slot, parent_depth);
          p_e = goes_left ? 0.0f : q_prev / child_weight;
        }
        shared.PathProbability(warp, subgroup.row_slot, parent_depth) = p_e;
      }
      p_e = subgroup.Broadcast(p_e);

      if (subgroup.is_warp_leader) {
        shared.Node(warp, *stack_size) = child_node;
        shared.Stage(warp, *stack_size) = 0;
        shared.Stage(warp, parent_depth) = 2;
      }
      // Push the sibling subtree with
      //   G_child(t) = G_parent(t) * child_weight *
      //                (1 + (p_e   - 1) t) / (1 + (q_prev - 1) t).
      // This preserves the basis after swapping the active feature state from q_prev to p_e.
      auto alpha_e = p_e - 1.0f;
      auto v = shared.Basis(warp, subgroup.row_slot, parent_depth, subgroup.point) * child_weight *
               (1.0f + alpha_e * quad_node);
      if (q_prev != 1.0f) {
        auto alpha_old = q_prev - 1.0f;
        v /= 1.0f + alpha_old * quad_node;
      }
      shared.Basis(warp, subgroup.row_slot, *stack_size, subgroup.point) = v;
      subgroup.Sync();
      shared.Basis(warp, subgroup.row_slot, parent_depth, subgroup.point) = *ret_val;
      (*stack_size)++;
      *have_return = false;
    } else {
      *ret_val += shared.Basis(warp, subgroup.row_slot, parent_depth, subgroup.point);
      (*stack_size)--;
      *have_return = true;
    }

    return true;
  }

  XGBOOST_DEV_INLINE void Descend(CompressedNode const* nodes_for_tree, bst_idx_t ridx,
                                  int* stack_size, bool* have_return, float* ret_val) {
    int depth = *stack_size - 1;
    auto const& node = nodes_for_tree[shared.Node(warp, depth)];
    if (node.is_leaf) {
      *ret_val = shared.Basis(warp, subgroup.row_slot, depth, subgroup.point) * node.leaf_value;
      (*stack_size)--;
      *have_return = true;
      return;
    }

    // stage == 0 explores the left child first. After the return path updates the parent state,
    // the second visit uses the cached go-left decision to push the right child.
    int child = static_cast<int>(shared.Stage(warp, depth) != 0);
    if (child == 0) {
      if (subgroup.is_warp_leader) {
        shared.Stage(warp, depth) = 1;
      }
      subgroup.Sync();
    }

    auto child_weight = child == 0 ? node.left_weight : node.right_weight;
    auto child_node = child == 0 ? node.left : node.right;
    float q_prev = 1.0f;
    if (subgroup.is_leader) {
      if (subgroup.RowActive()) {
        q_prev = PreviousPathProbability(node.prev_same_offset_plus1, depth,
                                         shared.PathProbabilityRow(warp, subgroup.row_slot));
      }
      shared.StoreQPrev(warp, subgroup.row_slot, depth, q_prev);
    }
    q_prev = subgroup.Broadcast(q_prev);

    float p_e = 0.0f;
    if (subgroup.is_leader) {
      bool goes_left = false;
      if (subgroup.RowActive()) {
        goes_left = this->EvaluateGoesLeft(ridx, node);
        // p_e is the path probability after taking the chosen child for this row.
        p_e = (child == 0 ? goes_left : !goes_left) ? q_prev / child_weight : 0.0f;
      }
      shared.SetGoesLeft(warp, subgroup.row_slot, depth, goes_left);
      shared.PathProbability(warp, subgroup.row_slot, depth) = p_e;
    }
    p_e = subgroup.Broadcast(p_e);

    if (subgroup.is_warp_leader) {
      shared.Node(warp, *stack_size) = child_node;
      shared.Stage(warp, *stack_size) = 0;
    }
    // Same recurrence as the sibling push above: reweight G_d(t) by the child weight and replace
    // q_prev with the new path probability p_e at this depth.
    auto alpha_e = p_e - 1.0f;
    auto v = shared.Basis(warp, subgroup.row_slot, depth, subgroup.point) * child_weight *
             (1.0f + alpha_e * quad_node);
    if (q_prev != 1.0f) {
      auto alpha_old = q_prev - 1.0f;
      v /= 1.0f + alpha_old * quad_node;
    }
    shared.Basis(warp, subgroup.row_slot, *stack_size, subgroup.point) = v;
    subgroup.Sync();
    (*stack_size)++;
  }

  XGBOOST_DEV_INLINE void RunTask(std::size_t task) {
    auto tree_idx = task / row_tiles;
    auto row_tile = task % row_tiles;
    auto ridx = (row_tile_begin + row_tile) * SubgroupT::kRowsPerWarpValue + subgroup.row_slot;
    auto row_idx = base_rowid + static_cast<bst_idx_t>(ridx);
    auto tree = trees[tree_idx];
    auto nodes_for_tree = nodes + tree.node_begin;

    this->InitializeTask();

    int stack_size = 1;
    bool have_return = false;
    float ret_val = 0.0f;
    while (stack_size > 0 || have_return) {
      if (have_return) {
        if (!this->HandleReturn(row_idx, tree.group, nodes_for_tree, &stack_size, &have_return,
                                &ret_val)) {
          break;
        }
        continue;
      }
      this->Descend(nodes_for_tree, ridx, &stack_size, &have_return, &ret_val);
    }
  }
};

template <typename Loader, typename SubgroupT, typename SharedT>
struct QuadratureShapInteractionTaskRunner {
  Loader loader;
  SubgroupT subgroup;
  SharedT shared;
  CompressedTree const* trees;
  CompressedNode const* nodes;
  std::uint32_t const* categories;
  float* phis;
  bst_idx_t base_rowid;
  bst_target_t n_groups;
  bst_feature_t n_columns;
  std::size_t row_tile_begin;
  std::size_t row_tiles;
  int warp;
  float quad_node;
  float quad_weight;

  [[nodiscard]] XGBOOST_DEV_INLINE bool EvaluateGoesLeft(bst_idx_t ridx,
                                                         CompressedNode const& node) const {
    auto fvalue = loader.GetElement(ridx, node.split_global);
    if (common::CheckNAN(fvalue)) {
      return static_cast<bool>(node.default_left);
    }
    if (node.is_categorical) {
      auto cats = common::Span<std::uint32_t const>{categories + node.cat_begin, node.cat_size};
      return common::Decision(cats, fvalue);
    }
    return fvalue < node.split_cond;
  }

  XGBOOST_DEV_INLINE void AddDiagonalContribution(bst_idx_t row_idx, bst_target_t tree_group,
                                                  bst_feature_t split_global, float contrib) const {
    if (!subgroup.ShouldWrite()) {
      return;
    }
    auto out_idx =
        (static_cast<std::size_t>(row_idx) * n_groups + tree_group) * n_columns * n_columns +
        static_cast<std::size_t>(split_global) * n_columns + split_global;
    atomicAdd(phis + out_idx, contrib);
  }

  XGBOOST_DEV_INLINE void AddPairContribution(bst_idx_t row_idx, bst_target_t tree_group,
                                              bst_feature_t split_i, bst_feature_t split_j,
                                              float contrib) const {
    if (!subgroup.ShouldWrite()) {
      return;
    }
    auto out_idx =
        (static_cast<std::size_t>(row_idx) * n_groups + tree_group) * n_columns * n_columns +
        static_cast<std::size_t>(split_i) * n_columns + split_j;
    atomicAdd(phis + out_idx, contrib);
  }

  template <typename Fn>
  XGBOOST_DEV_INLINE void ForEachUniquePartner(CompressedNode const* nodes_for_tree,
                                               int current_depth, bst_feature_t current_split,
                                               Fn&& fn) const {
    bool skipped_current = false;
    for (int depth = current_depth; depth >= 0; --depth) {
      auto const& candidate = nodes_for_tree[shared.Node(warp, depth)];
      if (candidate.is_leaf) {
        continue;
      }
      auto split = candidate.split_global;
      if (!skipped_current && split == current_split) {
        skipped_current = true;
        continue;
      }
      bool shadowed = false;
      for (int newer = current_depth; newer > depth; --newer) {
        auto const& newer_node = nodes_for_tree[shared.Node(warp, newer)];
        if (!newer_node.is_leaf && newer_node.split_global == split) {
          shadowed = true;
          break;
        }
      }
      if (!shadowed) {
        fn(depth, split);
      }
    }
  }

  XGBOOST_DEV_INLINE void InitializeTask() {
    if (subgroup.is_warp_leader) {
      shared.Node(warp, 0) = RegTree::kRoot;
      shared.Stage(warp, 0) = 0;
    }
    shared.Basis(warp, subgroup.row_slot, 0, subgroup.point) = 1.0f;
    subgroup.Sync();
  }

  XGBOOST_DEV_INLINE bool HandleReturn(bst_idx_t row_idx, bst_target_t tree_group,
                                       CompressedNode const* nodes_for_tree, int* stack_size,
                                       bool* have_return, float* ret_val) {
    if (*stack_size == 0) {
      return false;
    }

    int parent_depth = *stack_size - 1;
    auto const& node = nodes_for_tree[shared.Node(warp, parent_depth)];
    int child_idx = static_cast<int>(shared.Stage(warp, parent_depth)) - 1;

    float p_enter = 0.0f;
    float q_prev = 1.0f;
    if (subgroup.is_leader && subgroup.RowActive()) {
      p_enter = shared.PathProbability(warp, subgroup.row_slot, parent_depth);
      q_prev = shared.LoadQPrev(warp, subgroup.row_slot, parent_depth, node.prev_same_offset_plus1);
    }
    p_enter = subgroup.Broadcast(p_enter);
    q_prev = subgroup.Broadcast(q_prev);

    auto edge_delta_local =
        ExtractQuadratureEdgeDeltaLocal(quad_node, quad_weight, *ret_val, p_enter, q_prev);
    auto diag_contrib = subgroup.Sum(edge_delta_local);
    this->AddDiagonalContribution(row_idx, tree_group, node.split_global, diag_contrib);

    this->ForEachUniquePartner(
        nodes_for_tree, parent_depth, node.split_global,
        [&](int partner_depth, bst_feature_t partner_split) {
          float q_partner = 1.0f;
          if (subgroup.is_leader && subgroup.RowActive()) {
            q_partner = shared.PathProbability(warp, subgroup.row_slot, partner_depth);
          }
          q_partner = subgroup.Broadcast(q_partner);
          auto pair_delta_local =
              ExtractQuadratureInteractionDeltaLocal(quad_node, edge_delta_local, q_partner);
          auto pair_contrib = subgroup.Sum(pair_delta_local);
          this->AddPairContribution(row_idx, tree_group, node.split_global, partner_split,
                                    pair_contrib);
        });

    if (child_idx == 0) {
      auto child_weight = node.right_weight;
      auto child_node = node.right;
      float p_e = 0.0f;
      if (subgroup.is_leader) {
        if (subgroup.RowActive()) {
          auto goes_left = shared.GoesLeft(warp, subgroup.row_slot, parent_depth);
          p_e = goes_left ? 0.0f : q_prev / child_weight;
        }
        shared.PathProbability(warp, subgroup.row_slot, parent_depth) = p_e;
      }
      p_e = subgroup.Broadcast(p_e);

      if (subgroup.is_warp_leader) {
        shared.Node(warp, *stack_size) = child_node;
        shared.Stage(warp, *stack_size) = 0;
        shared.Stage(warp, parent_depth) = 2;
      }
      auto alpha_e = p_e - 1.0f;
      auto v = shared.Basis(warp, subgroup.row_slot, parent_depth, subgroup.point) * child_weight *
               (1.0f + alpha_e * quad_node);
      if (q_prev != 1.0f) {
        auto alpha_old = q_prev - 1.0f;
        v /= 1.0f + alpha_old * quad_node;
      }
      shared.Basis(warp, subgroup.row_slot, *stack_size, subgroup.point) = v;
      subgroup.Sync();
      shared.Basis(warp, subgroup.row_slot, parent_depth, subgroup.point) = *ret_val;
      (*stack_size)++;
      *have_return = false;
    } else {
      *ret_val += shared.Basis(warp, subgroup.row_slot, parent_depth, subgroup.point);
      (*stack_size)--;
      *have_return = true;
    }

    return true;
  }

  XGBOOST_DEV_INLINE void Descend(CompressedNode const* nodes_for_tree, bst_idx_t ridx,
                                  int* stack_size, bool* have_return, float* ret_val) {
    int depth = *stack_size - 1;
    auto const& node = nodes_for_tree[shared.Node(warp, depth)];
    if (node.is_leaf) {
      *ret_val = shared.Basis(warp, subgroup.row_slot, depth, subgroup.point) * node.leaf_value;
      (*stack_size)--;
      *have_return = true;
      return;
    }

    int child = static_cast<int>(shared.Stage(warp, depth) != 0);
    if (child == 0) {
      if (subgroup.is_warp_leader) {
        shared.Stage(warp, depth) = 1;
      }
      subgroup.Sync();
    }

    auto child_weight = child == 0 ? node.left_weight : node.right_weight;
    auto child_node = child == 0 ? node.left : node.right;
    float q_prev = 1.0f;
    if (subgroup.is_leader) {
      if (subgroup.RowActive()) {
        q_prev = PreviousPathProbability(node.prev_same_offset_plus1, depth,
                                         shared.PathProbabilityRow(warp, subgroup.row_slot));
      }
      shared.StoreQPrev(warp, subgroup.row_slot, depth, q_prev);
    }
    q_prev = subgroup.Broadcast(q_prev);

    float p_e = 0.0f;
    if (subgroup.is_leader) {
      bool goes_left = false;
      if (subgroup.RowActive()) {
        goes_left = this->EvaluateGoesLeft(ridx, node);
        p_e = (child == 0 ? goes_left : !goes_left) ? q_prev / child_weight : 0.0f;
      }
      shared.SetGoesLeft(warp, subgroup.row_slot, depth, goes_left);
      shared.PathProbability(warp, subgroup.row_slot, depth) = p_e;
    }
    p_e = subgroup.Broadcast(p_e);

    if (subgroup.is_warp_leader) {
      shared.Node(warp, *stack_size) = child_node;
      shared.Stage(warp, *stack_size) = 0;
    }
    auto alpha_e = p_e - 1.0f;
    auto v = shared.Basis(warp, subgroup.row_slot, depth, subgroup.point) * child_weight *
             (1.0f + alpha_e * quad_node);
    if (q_prev != 1.0f) {
      auto alpha_old = q_prev - 1.0f;
      v /= 1.0f + alpha_old * quad_node;
    }
    shared.Basis(warp, subgroup.row_slot, *stack_size, subgroup.point) = v;
    subgroup.Sync();
    (*stack_size)++;
  }

  XGBOOST_DEV_INLINE void RunTask(std::size_t task) {
    auto tree_idx = task / row_tiles;
    auto row_tile = task % row_tiles;
    auto ridx = (row_tile_begin + row_tile) * SubgroupT::kRowsPerWarpValue + subgroup.row_slot;
    auto row_idx = base_rowid + static_cast<bst_idx_t>(ridx);
    auto tree = trees[tree_idx];
    auto nodes_for_tree = nodes + tree.node_begin;

    this->InitializeTask();

    int stack_size = 1;
    bool have_return = false;
    float ret_val = 0.0f;
    while (stack_size > 0 || have_return) {
      if (have_return) {
        if (!this->HandleReturn(row_idx, tree.group, nodes_for_tree, &stack_size, &have_return,
                                &ret_val)) {
          break;
        }
        continue;
      }
      this->Descend(nodes_for_tree, ridx, &stack_size, &have_return, &ret_val);
    }
  }
};

template <int MaxPoints, int RowsPerWarp, int BlockThreads, int DepthCap, bool kHasRowMask,
          typename Loader>
__global__ void __launch_bounds__(BlockThreads, 9)
    QuadratureShapTaskKernel(Loader loader, bst_idx_t base_rowid, bst_target_t n_groups,
                             bst_feature_t n_columns, std::size_t row_tile_begin,
                             std::size_t row_tiles, bst_idx_t valid_rows_in_tail,
                             std::size_t n_trees, CompressedTree const* __restrict__ trees,
                             CompressedNode const* __restrict__ nodes,
                             std::uint32_t const* __restrict__ categories,
                             float const* __restrict__ quad_nodes,
                             float const* __restrict__ quad_weights, float* __restrict__ phis) {
  static_assert(MaxPoints == kGpuQuadraturePoints);
  static_assert(DepthCap <= static_cast<int>(kMaxGpuQuadratureDepth));
  static_assert(dh::WarpThreads() % RowsPerWarp == 0);
  static_assert(BlockThreads % dh::WarpThreads() == 0);
  using SubgroupT = SubgroupOps<kHasRowMask, RowsPerWarp, MaxPoints>;
  constexpr int kSegmentWidth = SubgroupT::kSegmentWidth;
  if constexpr (!kHasRowMask) {
    static_assert(kSegmentWidth == MaxPoints,
                  "Full-tile specialization assumes every warp lane participates.");
  }
  constexpr int kWarpsPerBlock = BlockThreads / dh::WarpThreads();
  constexpr bool kUseQPrevCache = IsSparsePageLoaderNoShared<Loader>::value;
  using SharedT =
      QuadratureSharedState<MaxPoints, RowsPerWarp, DepthCap, kWarpsPerBlock, kUseQPrevCache>;

  __shared__ bst_node_t s_node[kWarpsPerBlock][DepthCap];
  __shared__ std::uint8_t s_stage[kWarpsPerBlock][DepthCap];
  __shared__ std::uint8_t s_goes_left[kWarpsPerBlock][RowsPerWarp][DepthCap];
  __shared__ float s_path_p[kWarpsPerBlock][RowsPerWarp][DepthCap];
  __shared__ float s_c_vals[kWarpsPerBlock][RowsPerWarp][DepthCap][MaxPoints];
  __shared__ float s_q_prev[kUseQPrevCache ? kWarpsPerBlock : 1][kUseQPrevCache ? RowsPerWarp : 1]
                           [kUseQPrevCache ? DepthCap : 1];

  int warp = static_cast<int>(threadIdx.x) / dh::WarpThreads();
  int lane = static_cast<int>(threadIdx.x) % dh::WarpThreads();
  auto subgroup = SubgroupT{lane, valid_rows_in_tail};
  if (!subgroup.Participates()) {
    return;
  }

  auto shared = SharedT{s_node, s_stage, s_goes_left, s_path_p, s_c_vals, s_q_prev};
  auto global_warp =
      (static_cast<std::size_t>(blockIdx.x) * BlockThreads + threadIdx.x) / dh::WarpThreads();
  auto warp_stride = (static_cast<std::size_t>(gridDim.x) * BlockThreads) / dh::WarpThreads();
  auto n_tasks = n_trees * row_tiles;

  auto runner = QuadratureShapTaskRunner<Loader, SubgroupT, SharedT>{loader,
                                                                     subgroup,
                                                                     shared,
                                                                     trees,
                                                                     nodes,
                                                                     categories,
                                                                     phis,
                                                                     base_rowid,
                                                                     n_groups,
                                                                     n_columns,
                                                                     row_tile_begin,
                                                                     row_tiles,
                                                                     warp,
                                                                     quad_nodes[subgroup.point],
                                                                     quad_weights[subgroup.point]};

  for (std::size_t task = global_warp; task < n_tasks; task += warp_stride) {
    runner.RunTask(task);
  }
}

template <int MaxPoints, int RowsPerWarp, int BlockThreads, int DepthCap, bool kHasRowMask,
          typename Loader>
void LaunchQuadratureShapTasks(Context const* ctx, Loader loader, bst_idx_t base_rowid,
                               bst_target_t n_groups, bst_feature_t n_columns,
                               std::size_t row_tile_begin, std::size_t row_tiles,
                               bst_idx_t valid_rows_in_tail, CompressedModel const& compressed,
                               common::Span<float const> quad_nodes,
                               common::Span<float const> quad_weights,
                               HostDeviceVector<float>* out_contribs) {
  static_assert(BlockThreads % dh::WarpThreads() == 0);
  constexpr int kWarpsPerBlock = BlockThreads / dh::WarpThreads();
  if (compressed.trees.empty() || row_tiles == 0) {
    return;
  }
  auto trees = thrust::raw_pointer_cast(compressed.trees.data());
  auto nodes = thrust::raw_pointer_cast(compressed.nodes.data());
  auto categories = thrust::raw_pointer_cast(compressed.categories.data());
  auto d_quad_nodes = quad_nodes.data();
  auto d_quad_weights = quad_weights.data();
  auto phis = out_contribs->DeviceSpan().data();
  auto n_tasks = compressed.trees.size() * row_tiles;
  auto grids = common::DivRoundUp(n_tasks, static_cast<std::size_t>(kWarpsPerBlock));
  QuadratureShapTaskKernel<MaxPoints, RowsPerWarp, BlockThreads, DepthCap, kHasRowMask>
      <<<static_cast<uint32_t>(grids), static_cast<uint32_t>(BlockThreads), 0,
         ctx->CUDACtx()->Stream()>>>(loader, base_rowid, n_groups, n_columns, row_tile_begin,
                                     row_tiles, valid_rows_in_tail, compressed.trees.size(), trees,
                                     nodes, categories, d_quad_nodes, d_quad_weights, phis);
  dh::safe_cuda(cudaGetLastError());
}

template <int MaxPoints, int RowsPerWarp, int BlockThreads, int DepthCap, typename Loader>
void LaunchQuadratureShapBuckets(Context const* ctx, Loader loader, bst_idx_t base_rowid,
                                 bst_target_t n_groups, bst_feature_t n_columns,
                                 CompressedModel const& compressed,
                                 common::Span<float const> quad_nodes,
                                 common::Span<float const> quad_weights,
                                 HostDeviceVector<float>* out_contribs) {
  auto full_row_tiles = static_cast<std::size_t>(loader.NumRows() / RowsPerWarp);
  auto tail_rows = static_cast<bst_idx_t>(loader.NumRows() % RowsPerWarp);
  LaunchQuadratureShapTasks<MaxPoints, RowsPerWarp, BlockThreads, DepthCap, false>(
      ctx, loader, base_rowid, n_groups, n_columns, /*row_tile_begin=*/0, full_row_tiles,
      /*valid_rows_in_tail=*/RowsPerWarp, compressed, quad_nodes, quad_weights, out_contribs);
  if (tail_rows != 0) {
    LaunchQuadratureShapTasks<MaxPoints, RowsPerWarp, BlockThreads, DepthCap, true>(
        ctx, loader, base_rowid, n_groups, n_columns, /*row_tile_begin=*/full_row_tiles,
        /*row_tiles=*/1, tail_rows, compressed, quad_nodes, quad_weights, out_contribs);
  }
}

template <int MaxPoints, int RowsPerWarp, int BlockThreads, int DepthCap, bool kHasRowMask,
          typename Loader>
__global__ void __launch_bounds__(BlockThreads, 9) QuadratureShapInteractionTaskKernel(
    Loader loader, bst_idx_t base_rowid, bst_target_t n_groups, bst_feature_t n_columns,
    std::size_t row_tile_begin, std::size_t row_tiles, bst_idx_t valid_rows_in_tail,
    std::size_t n_trees, CompressedTree const* __restrict__ trees,
    CompressedNode const* __restrict__ nodes, std::uint32_t const* __restrict__ categories,
    float const* __restrict__ quad_nodes, float const* __restrict__ quad_weights,
    float* __restrict__ phis) {
  static_assert(MaxPoints == kGpuQuadraturePoints);
  static_assert(DepthCap <= static_cast<int>(kMaxGpuQuadratureDepth));
  static_assert(dh::WarpThreads() % RowsPerWarp == 0);
  static_assert(BlockThreads % dh::WarpThreads() == 0);
  using SubgroupT = SubgroupOps<kHasRowMask, RowsPerWarp, MaxPoints>;
  constexpr int kSegmentWidth = SubgroupT::kSegmentWidth;
  if constexpr (!kHasRowMask) {
    static_assert(kSegmentWidth == MaxPoints,
                  "Full-tile specialization assumes every warp lane participates.");
  }
  constexpr int kWarpsPerBlock = BlockThreads / dh::WarpThreads();
  constexpr bool kUseQPrevCache = IsSparsePageLoaderNoShared<Loader>::value;
  using SharedT =
      QuadratureSharedState<MaxPoints, RowsPerWarp, DepthCap, kWarpsPerBlock, kUseQPrevCache>;

  __shared__ bst_node_t s_node[kWarpsPerBlock][DepthCap];
  __shared__ std::uint8_t s_stage[kWarpsPerBlock][DepthCap];
  __shared__ std::uint8_t s_goes_left[kWarpsPerBlock][RowsPerWarp][DepthCap];
  __shared__ float s_path_p[kWarpsPerBlock][RowsPerWarp][DepthCap];
  __shared__ float s_c_vals[kWarpsPerBlock][RowsPerWarp][DepthCap][MaxPoints];
  __shared__ float s_q_prev[kUseQPrevCache ? kWarpsPerBlock : 1][kUseQPrevCache ? RowsPerWarp : 1]
                           [kUseQPrevCache ? DepthCap : 1];

  int warp = static_cast<int>(threadIdx.x) / dh::WarpThreads();
  int lane = static_cast<int>(threadIdx.x) % dh::WarpThreads();
  auto subgroup = SubgroupT{lane, valid_rows_in_tail};
  if (!subgroup.Participates()) {
    return;
  }

  auto shared = SharedT{s_node, s_stage, s_goes_left, s_path_p, s_c_vals, s_q_prev};
  auto global_warp =
      (static_cast<std::size_t>(blockIdx.x) * BlockThreads + threadIdx.x) / dh::WarpThreads();
  auto warp_stride = (static_cast<std::size_t>(gridDim.x) * BlockThreads) / dh::WarpThreads();
  auto n_tasks = n_trees * row_tiles;

  auto runner =
      QuadratureShapInteractionTaskRunner<Loader, SubgroupT, SharedT>{loader,
                                                                      subgroup,
                                                                      shared,
                                                                      trees,
                                                                      nodes,
                                                                      categories,
                                                                      phis,
                                                                      base_rowid,
                                                                      n_groups,
                                                                      n_columns,
                                                                      row_tile_begin,
                                                                      row_tiles,
                                                                      warp,
                                                                      quad_nodes[subgroup.point],
                                                                      quad_weights[subgroup.point]};

  for (std::size_t task = global_warp; task < n_tasks; task += warp_stride) {
    runner.RunTask(task);
  }
}

template <int MaxPoints, int RowsPerWarp, int BlockThreads, int DepthCap, bool kHasRowMask,
          typename Loader>
void LaunchQuadratureShapInteractionTasks(Context const* ctx, Loader loader, bst_idx_t base_rowid,
                                          bst_target_t n_groups, bst_feature_t n_columns,
                                          std::size_t row_tile_begin, std::size_t row_tiles,
                                          bst_idx_t valid_rows_in_tail,
                                          CompressedModel const& compressed,
                                          common::Span<float const> quad_nodes,
                                          common::Span<float const> quad_weights,
                                          HostDeviceVector<float>* out_contribs) {
  static_assert(BlockThreads % dh::WarpThreads() == 0);
  constexpr int kWarpsPerBlock = BlockThreads / dh::WarpThreads();
  if (compressed.trees.empty() || row_tiles == 0) {
    return;
  }
  auto trees = thrust::raw_pointer_cast(compressed.trees.data());
  auto nodes = thrust::raw_pointer_cast(compressed.nodes.data());
  auto categories = thrust::raw_pointer_cast(compressed.categories.data());
  auto d_quad_nodes = quad_nodes.data();
  auto d_quad_weights = quad_weights.data();
  auto phis = out_contribs->DeviceSpan().data();
  auto n_tasks = compressed.trees.size() * row_tiles;
  auto grids = common::DivRoundUp(n_tasks, static_cast<std::size_t>(kWarpsPerBlock));
  QuadratureShapInteractionTaskKernel<MaxPoints, RowsPerWarp, BlockThreads, DepthCap, kHasRowMask>
      <<<static_cast<uint32_t>(grids), static_cast<uint32_t>(BlockThreads), 0,
         ctx->CUDACtx()->Stream()>>>(loader, base_rowid, n_groups, n_columns, row_tile_begin,
                                     row_tiles, valid_rows_in_tail, compressed.trees.size(), trees,
                                     nodes, categories, d_quad_nodes, d_quad_weights, phis);
  dh::safe_cuda(cudaGetLastError());
}

template <int MaxPoints, int RowsPerWarp, int BlockThreads, int DepthCap, typename Loader>
void LaunchQuadratureShapInteractionBuckets(Context const* ctx, Loader loader, bst_idx_t base_rowid,
                                            bst_target_t n_groups, bst_feature_t n_columns,
                                            CompressedModel const& compressed,
                                            common::Span<float const> quad_nodes,
                                            common::Span<float const> quad_weights,
                                            HostDeviceVector<float>* out_contribs) {
  auto full_row_tiles = static_cast<std::size_t>(loader.NumRows() / RowsPerWarp);
  auto tail_rows = static_cast<bst_idx_t>(loader.NumRows() % RowsPerWarp);
  LaunchQuadratureShapInteractionTasks<MaxPoints, RowsPerWarp, BlockThreads, DepthCap, false>(
      ctx, loader, base_rowid, n_groups, n_columns, /*row_tile_begin=*/0, full_row_tiles,
      /*valid_rows_in_tail=*/RowsPerWarp, compressed, quad_nodes, quad_weights, out_contribs);
  if (tail_rows != 0) {
    LaunchQuadratureShapInteractionTasks<MaxPoints, RowsPerWarp, BlockThreads, DepthCap, true>(
        ctx, loader, base_rowid, n_groups, n_columns, /*row_tile_begin=*/full_row_tiles,
        /*row_tiles=*/1, tail_rows, compressed, quad_nodes, quad_weights, out_contribs);
  }
}

template <typename EncAccessor, typename Fn>
void DispatchByBatchLoader(Context const* ctx, DMatrix* p_fmat, bst_feature_t n_features,
                           EncAccessor acc, Fn&& fn) {
  using AccT = std::decay_t<EncAccessor>;
  if (p_fmat->PageExists<SparsePage>()) {
    for (auto& page : p_fmat->GetBatches<SparsePage>()) {
      SparsePageView batch{ctx, page, n_features};
      auto loader = SparsePageLoaderNoShared<AccT>{batch, acc};
      fn(std::move(loader), page.base_rowid);
    }
  } else {
    p_fmat->Info().feature_types.SetDevice(ctx->Device());
    auto feature_types = p_fmat->Info().feature_types.ConstDeviceSpan();

    for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx, StaticBatch(true))) {
      page.Impl()->Visit(ctx, feature_types, [&](auto&& batch) {
        using BatchT = std::remove_reference_t<decltype(batch)>;
        auto loader = EllpackLoader<BatchT, AccT>{batch,
                                                  /*use_shared=*/false,
                                                  n_features,
                                                  batch.NumRows(),
                                                  std::numeric_limits<float>::quiet_NaN(),
                                                  AccT{acc}};
        fn(std::move(loader), batch.base_rowid);
      });
    }
  }
}

template <typename Fn>
void LaunchShap(Context const* ctx, DMatrix* p_fmat, enc::DeviceColumnsView const& new_enc,
                gbm::GBTreeModel const& model, Fn&& fn) {
  auto n_features = model.learner_model_param->num_feature;
  if (model.Cats() && model.Cats()->HasCategorical() && new_enc.HasCategorical()) {
    auto [acc, mapping] = ::xgboost::cuda_impl::MakeCatAccessor(ctx, new_enc, model.Cats());
    DispatchByBatchLoader(ctx, p_fmat, n_features, std::move(acc), fn);
  } else {
    DispatchByBatchLoader(ctx, p_fmat, n_features, NoOpAccessor{}, fn);
  }
}
}  // namespace
void ShapValues(Context const* ctx, DMatrix* p_fmat, HostDeviceVector<float>* out_contribs,
                gbm::GBTreeModel const& model, bst_tree_t tree_end,
                std::vector<float> const* tree_weights, int condition, unsigned condition_feature) {
  xgboost_NVTX_FN_RANGE();
  CHECK_EQ(condition, 0) << "GPU QuadratureTreeSHAP does not support conditional SHAP.";
  CHECK_EQ(condition_feature, 0) << "GPU QuadratureTreeSHAP does not support conditional SHAP.";
  CHECK(!model.learner_model_param->IsVectorLeaf()) << "Predict contribution" << MTNotImplemented();
  CHECK(!p_fmat->Info().IsColumnSplit())
      << "Predict contribution support for column-wise data split is not yet implemented.";

  tree_end = predictor::GetTreeLimit(model.trees, tree_end);
  auto const ngroup = model.learner_model_param->num_output_group;
  CHECK_NE(ngroup, 0);
  auto const ncolumns = model.learner_model_param->num_feature + 1;
  auto dim_size = ncolumns * ngroup;
  out_contribs->SetDevice(ctx->Device());
  out_contribs->Resize(p_fmat->Info().num_row_ * dim_size);
  out_contribs->Fill(0.0f);

  bst_node_t max_depth = 0;
  std::array<std::vector<bst_tree_t>, kGpuQuadratureDepthBuckets.size()> tree_buckets;
  for (bst_tree_t tree_idx = 0; tree_idx < tree_end; ++tree_idx) {
    CHECK(!model.trees[tree_idx]->IsMultiTarget()) << "Predict contribution" << MTNotImplemented();
    detail::ValidateQuadratureTreeShapCovers(model.trees[tree_idx]->HostScView(), RegTree::kRoot,
                                             "GPU");
    auto tree_depth = model.trees[tree_idx]->MaxDepth();
    max_depth = std::max(max_depth, tree_depth);
    auto path_depth = static_cast<std::size_t>(tree_depth) + 1;
    auto bucket_idx = DepthBucketIndex(path_depth);
    tree_buckets[bucket_idx].push_back(tree_idx);
  }
  CHECK_LE(max_depth + 1, static_cast<bst_node_t>(kMaxGpuQuadratureDepth))
      << "GPU QuadratureSHAP currently supports trees of depth up to "
      << (kMaxGpuQuadratureDepth - 1) << ".";
  auto h_group_root_mean_sums = MakeGroupRootMeanSums(model, tree_end, tree_weights);

  auto const& rule = detail::GetQuadratureTreeShapRule();
  std::array<float, kGpuQuadraturePoints> h_quad_nodes{};
  std::array<float, kGpuQuadraturePoints> h_quad_weights{};
  for (std::size_t i = 0; i < kGpuQuadraturePoints; ++i) {
    h_quad_nodes[i] = rule.nodes[i];
    h_quad_weights[i] = rule.weights[i];
  }
  dh::device_vector<float> d_quad_nodes(h_quad_nodes.cbegin(), h_quad_nodes.cend());
  dh::device_vector<float> d_quad_weights(h_quad_weights.cbegin(), h_quad_weights.cend());
  dh::device_vector<float> d_group_root_mean_sums(h_group_root_mean_sums.cbegin(),
                                                  h_group_root_mean_sums.cend());
  auto compressed_16 = MakeCompressedModel(ctx, model, tree_buckets[0], tree_weights);
  auto compressed_32 = MakeCompressedModel(ctx, model, tree_buckets[1], tree_weights);
  auto compressed_64 = MakeCompressedModel(ctx, model, tree_buckets[2], tree_weights);

  auto new_enc =
      p_fmat->Cats()->NeedRecode() ? p_fmat->Cats()->DeviceView(ctx) : enc::DeviceColumnsView{};
  auto quad_nodes =
      common::Span<float const>{thrust::raw_pointer_cast(d_quad_nodes.data()), d_quad_nodes.size()};
  auto quad_weights = common::Span<float const>{thrust::raw_pointer_cast(d_quad_weights.data()),
                                                d_quad_weights.size()};
  auto group_root_mean_sums = common::Span<float const>{
      thrust::raw_pointer_cast(d_group_root_mean_sums.data()), d_group_root_mean_sums.size()};

  LaunchShap(ctx, p_fmat, new_enc, model, [&](auto&& loader, bst_idx_t base_rowid) {
    LaunchQuadratureShapBuckets<kGpuQuadraturePoints, kGpuQuadratureRowsPerWarp,
                                kGpuQuadratureTreeBlockThreads, 16>(
        ctx, loader, base_rowid, ngroup, ncolumns, compressed_16, quad_nodes, quad_weights,
        out_contribs);
    LaunchQuadratureShapBuckets<kGpuQuadraturePoints, kGpuQuadratureRowsPerWarp,
                                kGpuQuadratureTreeBlockThreads, 32>(
        ctx, loader, base_rowid, ngroup, ncolumns, compressed_32, quad_nodes, quad_weights,
        out_contribs);
    LaunchQuadratureShapBuckets<kGpuQuadraturePoints, kGpuQuadratureRowsPerWarp,
                                kGpuQuadratureTreeBlockThreads, 64>(
        ctx, loader, base_rowid, ngroup, ncolumns, compressed_64, quad_nodes, quad_weights,
        out_contribs);
  });

  p_fmat->Info().base_margin_.SetDevice(ctx->Device());
  auto margin = p_fmat->Info().base_margin_.Data()->ConstDeviceSpan();
  auto base_score = model.learner_model_param->BaseScore(ctx);
  auto phis = out_contribs->DeviceSpan();
  auto n_samples = p_fmat->Info().num_row_;
  dh::LaunchN(n_samples * ngroup, ctx->CUDACtx()->Stream(), [=] __device__(std::size_t idx) {
    auto [_, gid] = linalg::UnravelIndex(idx, n_samples, ngroup);
    phis[(idx + 1) * ncolumns - 1] +=
        group_root_mean_sums[gid] + (margin.empty() ? base_score(gid) : margin[idx]);
  });
}
void ShapInteractionValues(Context const* ctx, DMatrix* p_fmat,
                           HostDeviceVector<float>* out_contribs, gbm::GBTreeModel const& model,
                           bst_tree_t tree_end, std::vector<float> const* tree_weights,
                           bool approximate) {
  xgboost_NVTX_FN_RANGE();
  if (approximate) {
    LOG(FATAL) << "Approximated contribution is not implemented in GPU predictor, use CPU instead.";
  }
  CHECK(!model.learner_model_param->IsVectorLeaf())
      << "Predict interaction contribution" << MTNotImplemented();
  CHECK(!p_fmat->Info().IsColumnSplit()) << "Predict interaction contribution support for "
                                            "column-wise data split is not yet implemented.";

  tree_end = predictor::GetTreeLimit(model.trees, tree_end);
  auto const ngroup = model.learner_model_param->num_output_group;
  CHECK_NE(ngroup, 0);
  auto const ncolumns = model.learner_model_param->num_feature + 1;
  auto dim_size = ncolumns * ncolumns * ngroup;
  out_contribs->SetDevice(ctx->Device());
  out_contribs->Resize(p_fmat->Info().num_row_ * dim_size);
  out_contribs->Fill(0.0f);

  bst_node_t max_depth = 0;
  std::array<std::vector<bst_tree_t>, kGpuQuadratureDepthBuckets.size()> tree_buckets;
  for (bst_tree_t tree_idx = 0; tree_idx < tree_end; ++tree_idx) {
    CHECK(!model.trees[tree_idx]->IsMultiTarget())
        << "Predict interaction contribution" << MTNotImplemented();
    detail::ValidateQuadratureTreeShapCovers(model.trees[tree_idx]->HostScView(), RegTree::kRoot,
                                             "GPU");
    auto tree_depth = model.trees[tree_idx]->MaxDepth();
    max_depth = std::max(max_depth, tree_depth);
    auto path_depth = static_cast<std::size_t>(tree_depth) + 1;
    auto bucket_idx = DepthBucketIndex(path_depth);
    tree_buckets[bucket_idx].push_back(tree_idx);
  }
  CHECK_LE(max_depth + 1, static_cast<bst_node_t>(kMaxGpuQuadratureDepth))
      << "GPU QuadratureSHAP currently supports trees of depth up to "
      << (kMaxGpuQuadratureDepth - 1) << ".";

  auto h_group_root_mean_sums = MakeGroupRootMeanSums(model, tree_end, tree_weights);
  auto const& rule = detail::GetQuadratureTreeShapRule();
  std::array<float, kGpuQuadraturePoints> h_quad_nodes{};
  std::array<float, kGpuQuadraturePoints> h_quad_weights{};
  for (std::size_t i = 0; i < kGpuQuadraturePoints; ++i) {
    h_quad_nodes[i] = rule.nodes[i];
    h_quad_weights[i] = rule.weights[i];
  }
  dh::device_vector<float> d_quad_nodes(h_quad_nodes.cbegin(), h_quad_nodes.cend());
  dh::device_vector<float> d_quad_weights(h_quad_weights.cbegin(), h_quad_weights.cend());
  dh::device_vector<float> d_group_root_mean_sums(h_group_root_mean_sums.cbegin(),
                                                  h_group_root_mean_sums.cend());
  auto compressed_16 = MakeCompressedModel(ctx, model, tree_buckets[0], tree_weights);
  auto compressed_32 = MakeCompressedModel(ctx, model, tree_buckets[1], tree_weights);
  auto compressed_64 = MakeCompressedModel(ctx, model, tree_buckets[2], tree_weights);

  auto new_enc =
      p_fmat->Cats()->NeedRecode() ? p_fmat->Cats()->DeviceView(ctx) : enc::DeviceColumnsView{};
  auto quad_nodes =
      common::Span<float const>{thrust::raw_pointer_cast(d_quad_nodes.data()), d_quad_nodes.size()};
  auto quad_weights = common::Span<float const>{thrust::raw_pointer_cast(d_quad_weights.data()),
                                                d_quad_weights.size()};
  auto group_root_mean_sums = common::Span<float const>{
      thrust::raw_pointer_cast(d_group_root_mean_sums.data()), d_group_root_mean_sums.size()};

  LaunchShap(ctx, p_fmat, new_enc, model, [&](auto&& loader, bst_idx_t base_rowid) {
    LaunchQuadratureShapInteractionBuckets<kGpuQuadraturePoints, kGpuQuadratureRowsPerWarp,
                                           kGpuQuadratureTreeBlockThreads, 16>(
        ctx, loader, base_rowid, ngroup, ncolumns, compressed_16, quad_nodes, quad_weights,
        out_contribs);
    LaunchQuadratureShapInteractionBuckets<kGpuQuadraturePoints, kGpuQuadratureRowsPerWarp,
                                           kGpuQuadratureTreeBlockThreads, 32>(
        ctx, loader, base_rowid, ngroup, ncolumns, compressed_32, quad_nodes, quad_weights,
        out_contribs);
    LaunchQuadratureShapInteractionBuckets<kGpuQuadraturePoints, kGpuQuadratureRowsPerWarp,
                                           kGpuQuadratureTreeBlockThreads, 64>(
        ctx, loader, base_rowid, ngroup, ncolumns, compressed_64, quad_nodes, quad_weights,
        out_contribs);
  });

  p_fmat->Info().base_margin_.SetDevice(ctx->Device());
  auto margin = p_fmat->Info().base_margin_.Data()->ConstDeviceSpan();
  auto base_score = model.learner_model_param->BaseScore(ctx);
  auto phis = out_contribs->DeviceSpan();
  auto n_samples = p_fmat->Info().num_row_;
  dh::LaunchN(n_samples * ngroup, ctx->CUDACtx()->Stream(), [=] __device__(std::size_t idx) {
    auto [ridx, gid] = linalg::UnravelIndex(idx, n_samples, ngroup);
    auto matrix_offset = (static_cast<std::size_t>(ridx) * ngroup + gid) * ncolumns * ncolumns;
    auto matrix = phis.subspan(matrix_offset, ncolumns * ncolumns);
    matrix[(ncolumns - 1) * ncolumns + (ncolumns - 1)] +=
        group_root_mean_sums[gid] + (margin.empty() ? base_score(gid) : margin[idx]);
    for (bst_feature_t r = 0; r < ncolumns; ++r) {
      for (bst_feature_t c = r + 1; c < ncolumns; ++c) {
        auto sym = 0.5f * (matrix[r * ncolumns + c] + matrix[c * ncolumns + r]);
        matrix[r * ncolumns + c] = sym;
        matrix[c * ncolumns + r] = sym;
      }
    }
    for (bst_feature_t r = 0; r < ncolumns; ++r) {
      float value = matrix[r * ncolumns + r];
      for (bst_feature_t c = 0; c < ncolumns; ++c) {
        if (c != r) {
          value -= matrix[r * ncolumns + c];
        }
      }
      matrix[r * ncolumns + r] = value;
    }
  });
}

void ApproxFeatureImportance(Context const*, DMatrix*, HostDeviceVector<float>*,
                             gbm::GBTreeModel const&, bst_tree_t, std::vector<float> const*) {
  StringView not_implemented{
      "contribution is not implemented in the GPU predictor, use CPU instead."};
  LOG(FATAL) << "Approximated " << not_implemented;
}
}  // namespace xgboost::interpretability::cuda_impl
