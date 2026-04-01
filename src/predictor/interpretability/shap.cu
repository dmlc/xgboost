/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#include <GPUTreeShap/gpu_treeshap.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cuda/functional>   // for proclaim_return_type
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
constexpr double kQuadratureShapQeps = 1e-15;
using QuadratureRule = detail::EndpointQuadratureRule<kGpuQuadraturePoints>;

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
  std::uint8_t default_left{0};
  std::uint8_t is_leaf{0};
  std::uint8_t prev_same_offset_plus1{0};
};

struct CompressedTree {
  std::uint32_t node_begin{0};
  bst_target_t group{0};
};

struct CompressedModel {
  dh::device_vector<CompressedTree> trees;
  dh::device_vector<CompressedNode> nodes;
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
  auto h_tree_groups = model.TreeGroups(DeviceOrd::CPU());
  static_cast<void>(ctx);

  h_trees.reserve(tree_indices.size());
  for (auto tree_idx : tree_indices) {
    auto const weight = tree_weights == nullptr ? 1.0f : (*tree_weights)[tree_idx];
    auto const tree = model.trees.at(tree_idx)->HostScView();
    CHECK(!tree.HasCategoricalSplit())
        << "GPU QuadratureSHAP prototype does not support categorical splits.";

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
      out.left_weight = static_cast<float>(static_cast<double>(tree.SumHess(left)) / parent_cover);
      out.right_weight =
          static_cast<float>(static_cast<double>(tree.SumHess(right)) / parent_cover);
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
    return common::CheckNAN(fvalue) ? static_cast<bool>(node.default_left)
                                    : fvalue < node.split_cond;
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
    return common::CheckNAN(fvalue) ? static_cast<bool>(node.default_left)
                                    : fvalue < node.split_cond;
  }

  XGBOOST_DEV_INLINE void AddDiagonalContribution(bst_idx_t row_idx, bst_target_t tree_group,
                                                  bst_feature_t split_global, float contrib) const {
    if (!subgroup.ShouldWrite()) {
      return;
    }
    auto out_idx = gpu_treeshap::IndexPhiInteractions(row_idx, n_groups, tree_group, n_columns - 1,
                                                      split_global, split_global);
    atomicAdd(phis + out_idx, contrib);
  }

  XGBOOST_DEV_INLINE void AddPairContribution(bst_idx_t row_idx, bst_target_t tree_group,
                                              bst_feature_t split_i, bst_feature_t split_j,
                                              float contrib) const {
    if (!subgroup.ShouldWrite()) {
      return;
    }
    auto out_idx = gpu_treeshap::IndexPhiInteractions(row_idx, n_groups, tree_group, n_columns - 1,
                                                      split_i, split_j);
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
  auto d_quad_nodes = quad_nodes.data();
  auto d_quad_weights = quad_weights.data();
  auto phis = out_contribs->DeviceSpan().data();
  auto n_tasks = compressed.trees.size() * row_tiles;
  auto grids = common::DivRoundUp(n_tasks, static_cast<std::size_t>(kWarpsPerBlock));
  QuadratureShapTaskKernel<MaxPoints, RowsPerWarp, BlockThreads, DepthCap, kHasRowMask>
      <<<static_cast<uint32_t>(grids), static_cast<uint32_t>(BlockThreads), 0,
         ctx->CUDACtx()->Stream()>>>(loader, base_rowid, n_groups, n_columns, row_tile_begin,
                                     row_tiles, valid_rows_in_tail, compressed.trees.size(), trees,
                                     nodes, d_quad_nodes, d_quad_weights, phis);
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
    CompressedNode const* __restrict__ nodes, float const* __restrict__ quad_nodes,
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

  auto runner =
      QuadratureShapInteractionTaskRunner<Loader, SubgroupT, SharedT>{loader,
                                                                      subgroup,
                                                                      shared,
                                                                      trees,
                                                                      nodes,
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
  auto d_quad_nodes = quad_nodes.data();
  auto d_quad_weights = quad_weights.data();
  auto phis = out_contribs->DeviceSpan().data();
  auto n_tasks = compressed.trees.size() * row_tiles;
  auto grids = common::DivRoundUp(n_tasks, static_cast<std::size_t>(kWarpsPerBlock));
  QuadratureShapInteractionTaskKernel<MaxPoints, RowsPerWarp, BlockThreads, DepthCap, kHasRowMask>
      <<<static_cast<uint32_t>(grids), static_cast<uint32_t>(BlockThreads), 0,
         ctx->CUDACtx()->Stream()>>>(loader, base_rowid, n_groups, n_columns, row_tile_begin,
                                     row_tiles, valid_rows_in_tail, compressed.trees.size(), trees,
                                     nodes, d_quad_nodes, d_quad_weights, phis);
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

struct CopyViews {
  Context const* ctx;
  explicit CopyViews(Context const* ctx) : ctx{ctx} {}

  void operator()(dh::DeviceUVector<TreeViewVar>* p_dst, std::vector<TreeViewVar>&& src) {
    xgboost_NVTX_FN_RANGE();
    p_dst->resize(src.size());
    auto d_dst = dh::ToSpan(*p_dst);
    dh::safe_cuda(cudaMemcpyAsync(d_dst.data(), src.data(), d_dst.size_bytes(), cudaMemcpyDefault,
                                  ctx->CUDACtx()->Stream()));
  }
};

using DeviceModel = GBTreeModelView<dh::DeviceUVector, TreeViewVar, CopyViews>;

struct ShapSplitCondition {
  ShapSplitCondition() = default;
  XGBOOST_DEVICE
  ShapSplitCondition(float feature_lower_bound, float feature_upper_bound, bool is_missing_branch,
                     common::CatBitField cats)
      : feature_lower_bound(feature_lower_bound),
        feature_upper_bound(feature_upper_bound),
        is_missing_branch(is_missing_branch),
        categories{std::move(cats)} {
    assert(feature_lower_bound <= feature_upper_bound);
  }

  float feature_lower_bound;
  float feature_upper_bound;
  common::CatBitField categories;
  bool is_missing_branch;

  [[nodiscard]] XGBOOST_DEVICE bool EvaluateSplit(float x) const {
    if (isnan(x)) {
      return is_missing_branch;
    }
    if (categories.Capacity() != 0) {
      auto cat = static_cast<uint32_t>(x);
      return categories.Check(cat);
    } else {
      return x >= feature_lower_bound && x < feature_upper_bound;
    }
  }

  XGBOOST_DEVICE static common::CatBitField Intersect(common::CatBitField l,
                                                      common::CatBitField r) {
    if (l.Data() == r.Data()) {
      return l;
    }
    if (l.Capacity() > r.Capacity()) {
      cuda::std::swap(l, r);
    }
    auto l_bits = l.Bits();
    auto r_bits = r.Bits();
    auto n_bits = l_bits.size() < r_bits.size() ? l_bits.size() : r_bits.size();
    for (size_t i = 0; i < n_bits; ++i) {
      l_bits[i] &= r_bits[i];
    }
    return l;
  }

  XGBOOST_DEVICE void Merge(ShapSplitCondition other) {
    if (categories.Capacity() != 0 || other.categories.Capacity() != 0) {
      categories = Intersect(categories, other.categories);
    } else {
      feature_lower_bound = max(feature_lower_bound, other.feature_lower_bound);
      feature_upper_bound = min(feature_upper_bound, other.feature_upper_bound);
    }
    is_missing_branch = is_missing_branch && other.is_missing_branch;
  }
};

struct PathInfo {
  std::size_t length;
  bst_node_t nidx;
  bst_tree_t tree_idx;

  [[nodiscard]] XGBOOST_DEVICE bool IsLeaf() const { return nidx != -1; }
};
static_assert(sizeof(PathInfo) == 16);

auto MakeTreeSegments(Context const* ctx, bst_tree_t tree_begin, bst_tree_t tree_end,
                      gbm::GBTreeModel const& model) {
  auto tree_segments = HostDeviceVector<size_t>({}, ctx->Device());
  auto& h_tree_segments = tree_segments.HostVector();
  h_tree_segments.reserve((tree_end - tree_begin) + 1);
  std::size_t sum = 0;
  h_tree_segments.push_back(sum);
  for (auto tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
    auto const& p_tree = model.trees.at(tree_idx);
    CHECK(!p_tree->IsMultiTarget()) << " SHAP " << MTNotImplemented();
    sum += p_tree->Size();
    h_tree_segments.push_back(sum);
  }
  return tree_segments;
}

void ExtractPaths(Context const* ctx,
                  dh::device_vector<gpu_treeshap::PathElement<ShapSplitCondition>>* paths,
                  gbm::GBTreeModel const& h_model, DeviceModel const& d_model,
                  dh::device_vector<uint32_t>* path_categories,
                  common::OptionalWeights tree_weights) {
  curt::SetDevice(ctx->Ordinal());

  dh::caching_device_vector<PathInfo> info(d_model.n_nodes);
  auto d_trees = d_model.Trees();
  auto tree_segments = MakeTreeSegments(ctx, d_model.tree_begin, d_model.tree_end, h_model);
  CHECK_EQ(tree_segments.ConstHostVector().back(), d_model.n_nodes);
  auto d_tree_segments = tree_segments.ConstDeviceSpan();

  auto path_it = dh::MakeIndexTransformIter(
      cuda::proclaim_return_type<PathInfo>([=] __device__(size_t idx) -> PathInfo {
        bst_tree_t const tree_idx = dh::SegmentId(d_tree_segments, idx);
        bst_node_t const nidx = idx - d_tree_segments[tree_idx];
        auto const& tree = cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx]);
        if (!tree.IsLeaf(nidx) || tree.IsDeleted(nidx)) {
          return PathInfo{0, -1, 0};
        }
        std::size_t path_length = 1;
        auto iter_nidx = nidx;
        while (!tree.IsRoot(iter_nidx)) {
          iter_nidx = tree.Parent(iter_nidx);
          path_length++;
        }
        return PathInfo{path_length, nidx, tree_idx};
      }));
  auto end = thrust::copy_if(
      ctx->CUDACtx()->CTP(), path_it, path_it + d_model.n_nodes, info.begin(),
      cuda::proclaim_return_type<bool>([=] __device__(PathInfo const& e) { return e.IsLeaf(); }));

  info.resize(end - info.begin());
  using LenT = decltype(std::declval<PathInfo>().length);
  auto length_iterator = dh::MakeTransformIterator<LenT>(
      info.begin(), cuda::proclaim_return_type<LenT>(
                        [=] __device__(PathInfo const& info) { return info.length; }));
  dh::caching_device_vector<size_t> path_segments(info.size() + 1);
  thrust::fill_n(ctx->CUDACtx()->CTP(), path_segments.begin(), 1, std::size_t{0});
  thrust::inclusive_scan(ctx->CUDACtx()->CTP(), length_iterator, length_iterator + info.size(),
                         path_segments.begin() + 1);

  paths->resize(path_segments.back());

  auto d_paths = dh::ToSpan(*paths);
  auto d_info = info.data().get();
  auto d_tree_groups = d_model.tree_groups;
  auto d_path_segments = path_segments.data().get();

  std::size_t max_cat = 0;
  if (std::any_of(h_model.trees.cbegin(), h_model.trees.cend(),
                  [](auto const& p_tree) { return p_tree->HasCategoricalSplit(); })) {
    auto max_elem_it = dh::MakeIndexTransformIter([=] __device__(std::size_t i) -> std::size_t {
      auto tree_idx = dh::SegmentId(d_tree_segments, i);
      auto nidx = i - d_tree_segments[tree_idx];
      return cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx])
          .GetCategoriesMatrix()
          .node_ptr[nidx]
          .size;
    });
    auto max_cat_it =
        thrust::max_element(ctx->CUDACtx()->CTP(), max_elem_it, max_elem_it + d_model.n_nodes);
    dh::CachingDeviceUVector<std::size_t> d_max_cat(1);
    auto s_max_cat = dh::ToSpan(d_max_cat);
    dh::LaunchN(1, ctx->CUDACtx()->Stream(),
                [=] __device__(std::size_t) { s_max_cat[0] = *max_cat_it; });
    dh::safe_cuda(
        cudaMemcpy(&max_cat, s_max_cat.data(), s_max_cat.size_bytes(), cudaMemcpyDeviceToHost));
    CHECK_GE(max_cat, 1);
    path_categories->resize(max_cat * paths->size());
  }

  common::Span<uint32_t> d_path_categories = dh::ToSpan(*path_categories);

  dh::LaunchN(info.size(), ctx->CUDACtx()->Stream(), [=] __device__(size_t idx) {
    auto path_info = d_info[idx];
    auto tree = cuda::std::get<tree::ScalarTreeView>(d_trees[path_info.tree_idx]);
    std::int32_t group = d_tree_groups[path_info.tree_idx];
    auto child_nidx = path_info.nidx;

    // TreeSHAP is linear in the leaf outputs, so DART weights can be applied by
    // scaling each tree's leaf value before it enters the path representation.
    float v = tree.LeafValue(child_nidx) * tree_weights[path_info.tree_idx];
    const float inf = std::numeric_limits<float>::infinity();
    size_t output_position = d_path_segments[idx + 1] - 1;

    while (!tree.IsRoot(child_nidx)) {
      auto parent_nidx = tree.Parent(child_nidx);
      double child_cover = tree.SumHess(child_nidx);
      double parent_cover = tree.SumHess(parent_nidx);
      double zero_fraction = child_cover / parent_cover;

      bool is_left_path = tree.LeftChild(parent_nidx) == child_nidx;
      bool is_missing_path = (!tree.DefaultLeft(parent_nidx) && !is_left_path) ||
                             (tree.DefaultLeft(parent_nidx) && is_left_path);

      float lower_bound = -inf;
      float upper_bound = inf;
      common::CatBitField bits;
      if (common::IsCat(tree.cats.split_type, tree.Parent(child_nidx))) {
        auto path_cats = d_path_categories.subspan(max_cat * output_position, max_cat);
        auto node_cats = tree.NodeCats(tree.Parent(child_nidx));
        SPAN_CHECK(path_cats.size() >= node_cats.size());
        for (size_t i = 0; i < node_cats.size(); ++i) {
          path_cats[i] = is_left_path ? ~node_cats[i] : node_cats[i];
        }
        bits = common::CatBitField{path_cats};
      } else {
        lower_bound = is_left_path ? -inf : tree.SplitCond(parent_nidx);
        upper_bound = is_left_path ? tree.SplitCond(parent_nidx) : inf;
      }
      d_paths[output_position--] = gpu_treeshap::PathElement<ShapSplitCondition>{
          idx,           tree.SplitIndex(parent_nidx),
          group,         ShapSplitCondition{lower_bound, upper_bound, is_missing_path, bits},
          zero_fraction, v};

      child_nidx = parent_nidx;
    }
    d_paths[output_position] = {idx, -1, group, ShapSplitCondition{-inf, inf, false, {}}, 1.0, v};
  });
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
                std::vector<float> const* tree_weights, int, unsigned) {
  xgboost_NVTX_FN_RANGE();
  StringView not_implemented{
      "contribution is not implemented in the GPU predictor, use CPU instead."};
  CHECK(!p_fmat->Info().IsColumnSplit())
      << "Predict contribution support for column-wise data split is not yet implemented.";
  dh::safe_cuda(cudaSetDevice(ctx->Ordinal()));
  out_contribs->SetDevice(ctx->Device());
  tree_end = predictor::GetTreeLimit(model.trees, tree_end);

  const int ngroup = model.learner_model_param->num_output_group;
  CHECK_NE(ngroup, 0);
  size_t contributions_columns = model.learner_model_param->num_feature + 1;
  auto dim_size = contributions_columns * model.learner_model_param->num_output_group;
  out_contribs->Resize(p_fmat->Info().num_row_ * dim_size);
  out_contribs->Fill(0.0f);
  auto phis = out_contribs->DeviceSpan();

  dh::device_vector<gpu_treeshap::PathElement<ShapSplitCondition>> device_paths;
  DeviceModel d_model{ctx->Device(), model, true, 0, tree_end, CopyViews{ctx}};
  dh::device_vector<float> d_tree_weights;
  auto weights = common::OptionalWeights{1.0f};
  if (tree_weights != nullptr) {
    // GPU TreeSHAP consumes device-resident path data, so materialize the optional
    // tree weights on device before extracting the weighted leaf outputs.
    d_tree_weights.assign(tree_weights->cbegin(), tree_weights->cbegin() + tree_end);
    weights = common::OptionalWeights{common::Span<float const>{
        thrust::raw_pointer_cast(d_tree_weights.data()), d_tree_weights.size()}};
  }

  auto new_enc =
      p_fmat->Cats()->NeedRecode() ? p_fmat->Cats()->DeviceView(ctx) : enc::DeviceColumnsView{};

  dh::device_vector<uint32_t> categories;
  ExtractPaths(ctx, &device_paths, model, d_model, &categories, weights);

  LaunchShap(ctx, p_fmat, new_enc, model, [&](auto&& loader, bst_idx_t base_rowid) {
    auto begin = dh::tbegin(phis) + base_rowid * dim_size;
    gpu_treeshap::GPUTreeShap<dh::XGBDeviceAllocator<int>>(
        loader, device_paths.begin(), device_paths.end(), ngroup, begin, dh::tend(phis));
  });

  p_fmat->Info().base_margin_.SetDevice(ctx->Device());
  const auto margin = p_fmat->Info().base_margin_.Data()->ConstDeviceSpan();

  auto base_score = model.learner_model_param->BaseScore(ctx);
  bst_idx_t n_samples = p_fmat->Info().num_row_;
  dh::LaunchN(n_samples * ngroup, ctx->CUDACtx()->Stream(), [=] __device__(std::size_t idx) {
    auto [_, gid] = linalg::UnravelIndex(idx, n_samples, ngroup);
    phis[(idx + 1) * contributions_columns - 1] += margin.empty() ? base_score(gid) : margin[idx];
  });
}

void QuadratureShapValues(Context const* ctx, DMatrix* p_fmat,
                          HostDeviceVector<float>* out_contribs, gbm::GBTreeModel const& model,
                          bst_tree_t tree_end, std::vector<float> const* tree_weights,
                          std::size_t quadrature_points) {
  xgboost_NVTX_FN_RANGE();
  CHECK(!model.learner_model_param->IsVectorLeaf()) << "Predict contribution" << MTNotImplemented();
  CHECK(!p_fmat->Info().IsColumnSplit())
      << "Predict contribution support for column-wise data split is not yet implemented.";
  CHECK_EQ(quadrature_points, kGpuQuadraturePoints)
      << "GPU QuadratureSHAP currently uses a fixed quadrature size of " << kGpuQuadraturePoints
      << ".";

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

  auto rule = detail::MakeEndpointQuadrature<kGpuQuadraturePoints>(kQuadratureShapQeps);
  std::array<float, kGpuQuadraturePoints> h_quad_nodes{};
  std::array<float, kGpuQuadraturePoints> h_quad_weights{};
  for (std::size_t i = 0; i < kGpuQuadraturePoints; ++i) {
    h_quad_nodes[i] = static_cast<float>(rule.nodes[i]);
    h_quad_weights[i] = static_cast<float>(rule.weights[i]);
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

void QuadratureShapInteractionValues(Context const* ctx, DMatrix* p_fmat,
                                     HostDeviceVector<float>* out_contribs,
                                     gbm::GBTreeModel const& model, bst_tree_t tree_end,
                                     std::vector<float> const* tree_weights,
                                     std::size_t quadrature_points) {
  xgboost_NVTX_FN_RANGE();
  CHECK(!model.learner_model_param->IsVectorLeaf())
      << "Predict interaction contribution" << MTNotImplemented();
  CHECK(!p_fmat->Info().IsColumnSplit()) << "Predict interaction contribution support for "
                                            "column-wise data split is not yet implemented.";
  CHECK_EQ(quadrature_points, kGpuQuadraturePoints)
      << "GPU QuadratureSHAP currently uses a fixed quadrature size of " << kGpuQuadraturePoints
      << ".";

  tree_end = predictor::GetTreeLimit(model.trees, tree_end);
  auto const ngroup = model.learner_model_param->num_output_group;
  CHECK_NE(ngroup, 0);
  auto const ncolumns = model.learner_model_param->num_feature + 1;
  auto const n_features = model.learner_model_param->num_feature;
  auto dim_size = ncolumns * ncolumns * ngroup;
  out_contribs->SetDevice(ctx->Device());
  out_contribs->Resize(p_fmat->Info().num_row_ * dim_size);
  out_contribs->Fill(0.0f);

  bst_node_t max_depth = 0;
  std::array<std::vector<bst_tree_t>, kGpuQuadratureDepthBuckets.size()> tree_buckets;
  for (bst_tree_t tree_idx = 0; tree_idx < tree_end; ++tree_idx) {
    CHECK(!model.trees[tree_idx]->IsMultiTarget())
        << "Predict interaction contribution" << MTNotImplemented();
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
  auto rule = detail::MakeEndpointQuadrature<kGpuQuadraturePoints>(kQuadratureShapQeps);
  std::array<float, kGpuQuadraturePoints> h_quad_nodes{};
  std::array<float, kGpuQuadraturePoints> h_quad_weights{};
  for (std::size_t i = 0; i < kGpuQuadraturePoints; ++i) {
    h_quad_nodes[i] = static_cast<float>(rule.nodes[i]);
    h_quad_weights[i] = static_cast<float>(rule.weights[i]);
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
    auto bias_idx =
        gpu_treeshap::IndexPhiInteractions(ridx, ngroup, gid, n_features, n_features, n_features);
    phis[bias_idx] += group_root_mean_sums[gid] + (margin.empty() ? base_score(gid) : margin[idx]);

    auto matrix_offset = gpu_treeshap::IndexPhiInteractions(ridx, ngroup, gid, n_features, 0, 0);
    auto matrix = phis.subspan(matrix_offset, ncolumns * ncolumns);
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

void ShapInteractionValues(Context const* ctx, DMatrix* p_fmat,
                           HostDeviceVector<float>* out_contribs, gbm::GBTreeModel const& model,
                           bst_tree_t tree_end, std::vector<float> const* tree_weights,
                           bool approximate) {
  xgboost_NVTX_FN_RANGE();
  std::string not_implemented{"contribution is not implemented in GPU predictor, use cpu instead."};
  if (approximate) {
    LOG(FATAL) << "Approximated " << not_implemented;
  }
  dh::safe_cuda(cudaSetDevice(ctx->Ordinal()));
  out_contribs->SetDevice(ctx->Device());
  tree_end = predictor::GetTreeLimit(model.trees, tree_end);

  const int ngroup = model.learner_model_param->num_output_group;
  CHECK_NE(ngroup, 0);
  size_t contributions_columns = model.learner_model_param->num_feature + 1;
  auto dim_size =
      contributions_columns * contributions_columns * model.learner_model_param->num_output_group;
  out_contribs->Resize(p_fmat->Info().num_row_ * dim_size);
  out_contribs->Fill(0.0f);
  auto phis = out_contribs->DeviceSpan();

  dh::device_vector<gpu_treeshap::PathElement<ShapSplitCondition>> device_paths;
  DeviceModel d_model{ctx->Device(), model, true, 0, tree_end, CopyViews{ctx}};
  dh::device_vector<float> d_tree_weights;
  auto weights = common::OptionalWeights{1.0f};
  if (tree_weights != nullptr) {
    // GPU TreeSHAP consumes device-resident path data, so materialize the optional
    // tree weights on device before extracting the weighted leaf outputs.
    d_tree_weights.assign(tree_weights->cbegin(), tree_weights->cbegin() + tree_end);
    weights = common::OptionalWeights{common::Span<float const>{
        thrust::raw_pointer_cast(d_tree_weights.data()), d_tree_weights.size()}};
  }

  dh::device_vector<uint32_t> categories;
  ExtractPaths(ctx, &device_paths, model, d_model, &categories, weights);
  auto new_enc =
      p_fmat->Cats()->NeedRecode() ? p_fmat->Cats()->DeviceView(ctx) : enc::DeviceColumnsView{};

  LaunchShap(ctx, p_fmat, new_enc, model, [&](auto&& loader, bst_idx_t base_rowid) {
    auto begin = dh::tbegin(phis) + base_rowid * dim_size;
    gpu_treeshap::GPUTreeShapInteractions<dh::XGBDeviceAllocator<int>>(
        loader, device_paths.begin(), device_paths.end(), ngroup, begin, dh::tend(phis));
  });

  p_fmat->Info().base_margin_.SetDevice(ctx->Device());
  const auto margin = p_fmat->Info().base_margin_.Data()->ConstDeviceSpan();

  auto base_score = model.learner_model_param->BaseScore(ctx);
  size_t n_features = model.learner_model_param->num_feature;
  bst_idx_t n_samples = p_fmat->Info().num_row_;
  dh::LaunchN(n_samples * ngroup, ctx->CUDACtx()->Stream(), [=] __device__(size_t idx) {
    auto [ridx, gidx] = linalg::UnravelIndex(idx, n_samples, ngroup);
    phis[gpu_treeshap::IndexPhiInteractions(ridx, ngroup, gidx, n_features, n_features,
                                            n_features)] +=
        margin.empty() ? base_score(gidx) : margin[idx];
  });
}

void ApproxFeatureImportance(Context const*, DMatrix*, HostDeviceVector<float>*,
                             gbm::GBTreeModel const&, bst_tree_t, std::vector<float> const*) {
  StringView not_implemented{
      "contribution is not implemented in the GPU predictor, use CPU instead."};
  LOG(FATAL) << "Approximated " << not_implemented;
}
}  // namespace xgboost::interpretability::cuda_impl
