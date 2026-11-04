/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cuda/std/utility>  // for swap
#include <cuda/std/variant>  // for variant
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../common/categorical.h"
#include "../../common/common.h"
#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../common/cuda_rt_utils.h"   // for SetDevice
#include "../../common/device_helpers.cuh"
#include "../../common/kernel.h"  // for KernelRegistration
#include "../../common/math.h"
#include "../../common/nvtx_utils.h"
#include "../../data/batch_utils.h"      // for StaticBatch
#include "../../data/cat_container.cuh"  // for EncPolicy, MakeCatAccessor
#include "../../data/cat_container.h"    // for NoOpAccessor
#include "../../data/ellpack_page.cuh"
#include "../../gbm/gbtree_model.h"
#include "../../tree/tree_view.h"
#include "../gbtree_view.h"
#include "../gpu_data_accessor.cuh"
#include "../predict_fn.h"         // for GetTreeLimit
#include "../prediction_kernel.h"  // for prediction kernels
#include "dmlc/registry.h"         // for DMLC_REGISTRY_FILE_TAG
#include "quadrature.h"
#include "shap.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/linalg.h"  // for UnravelIndex
#include "xgboost/logging.h"

namespace xgboost::interpretability::cuda_impl {
DMLC_REGISTRY_FILE_TAG(shap_cuda);

namespace {
common::KernelRegistration<predictor::PredictContributionKernel> const kPredictContributionCUDA{
    DeviceOrd::kCUDA, &ShapValues};
common::KernelRegistration<predictor::PredictApproxContributionKernel> const
    kPredictApproxContributionCUDA{DeviceOrd::kCUDA, &ApproxFeatureImportance};
common::KernelRegistration<predictor::PredictInteractionContributionsKernel> const
    kPredictInteractionCUDA{DeviceOrd::kCUDA, &ShapInteractionValues};

using predictor::EllpackLoader;
using predictor::GBTreeModelView;
using predictor::SparsePageLoaderNoShared;
using predictor::SparsePageView;
using ::xgboost::cuda_impl::StaticBatch;

using TreeViewVar = cuda::std::variant<tree::ScalarTreeView, tree::MultiTargetTreeView>;

constexpr int kGpuQuadraturePoints = static_cast<int>(detail::kQuadratureTreeShapPoints);

// One warp evaluates multiple rows for the same tree. Give each row one lane per quadrature point
// so the quadrature loop maps directly onto warp lanes, with the remaining warp-level parallelism
// used for independent rows.
constexpr int kGpuQuadratureRowsPerWarp = dh::WarpThreads() / kGpuQuadraturePoints;
constexpr int kGpuQuadratureSegmentWidth = dh::WarpThreads() / kGpuQuadratureRowsPerWarp;
constexpr unsigned kFullWarpMask = 0xffffffffu;
// The traversal state is stored in shared memory per warp. Keeping blocks to two warps avoids
// exhausting shared memory when launching the deepest bucket while still providing block-level
// occupancy for shallower buckets.
constexpr int kGpuQuadratureTreeBlockThreads = 64;
constexpr int kGpuQuadratureWarpsPerBlock = kGpuQuadratureTreeBlockThreads / dh::WarpThreads();

// The traversal stack lives in shared memory and is sized at compile time. Group trees by depth so
// shallow models use smaller stack/basis arrays, while deeper trees still get a bounded fallback
// specialization instead of forcing every model through the largest shared-memory footprint.
constexpr std::array<std::size_t, 4> kGpuQuadratureDepthBuckets{{16, 32, 64, 128}};
constexpr std::size_t kMaxGpuQuadratureDepth = kGpuQuadratureDepthBuckets.back();
using QuadratureRule = detail::QuadratureRule;
// Leaf payload is interpreted by CompressedTree::is_vector_leaf: scalar trees keep the weighted
// leaf value inline, while vector-leaf trees store an offset into CompressedModel::leaf_values.
union LeafPayload {
  float value;
  std::uint32_t begin;

  XGBOOST_DEVICE constexpr LeafPayload() : value{0.0f} {}
  explicit constexpr LeafPayload(float value) : value{value} {}
  explicit constexpr LeafPayload(std::uint32_t begin) : begin{begin} {}
};

struct CompressedNode {
  bst_node_t left{RegTree::kInvalidNodeId};
  bst_node_t right{RegTree::kInvalidNodeId};
  bst_feature_t split_global{0};
  float split_cond{0};
  LeafPayload leaf{};
  float left_weight{0};
  float right_weight{0};
  std::uint32_t cat_begin{0};
  std::uint32_t cat_size{0};
  std::uint8_t default_left{0};
  std::uint8_t is_leaf{0};
  std::uint8_t is_categorical{0};
  // Distance to the nearest ancestor that split on the same feature, plus one. Zero means the
  // feature has not appeared earlier on this path. Storing plus-one keeps the sentinel cheap and
  // lets traversal recover q_prev without rescanning ancestors in the hot loop.
  std::uint8_t prev_same_offset_plus1{0};
};

struct CompressedTree {
  std::uint32_t node_begin{0};
  bst_target_t group_idx{0};
  bst_target_t target_idx{0};
  std::uint8_t is_vector_leaf{0};
};

struct CompressedModel {
  dh::device_vector<CompressedTree> trees;
  dh::device_vector<CompressedNode> nodes;
  dh::device_vector<float> leaf_values;
  dh::device_vector<std::uint32_t> categories;
};

struct GpuQuadratureModelData {
  std::array<CompressedModel, kGpuQuadratureDepthBuckets.size()> compressed;
  QuadratureRule rule;
  dh::device_vector<float> group_root_mean_sums;

  [[nodiscard]] common::Span<float const> GroupRootMeanSums() const {
    return {thrust::raw_pointer_cast(group_root_mean_sums.data()), group_root_mean_sums.size()};
  }
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

float LeafValue(tree::ScalarTreeView const& tree, bst_node_t nidx, bst_target_t target_idx) {
  CHECK_EQ(target_idx, 0);
  return tree.LeafValue(nidx);
}

float LeafValue(tree::MultiTargetTreeView const& tree, bst_node_t nidx, bst_target_t target_idx) {
  auto leaf_value = tree.LeafValue(nidx);
  CHECK_LT(target_idx, leaf_value.Size());
  return leaf_value(target_idx);
}

template <typename Tree>
void CompressTree(Tree const& tree, std::vector<CompressedNode>* p_nodes,
                  std::vector<float>* p_leaf_values, std::vector<std::uint32_t>* p_categories,
                  float weight, std::uint32_t* p_node_begin) {
  auto& h_nodes = *p_nodes;
  auto& h_leaf_values = *p_leaf_values;
  auto& h_categories = *p_categories;

  auto node_begin = h_nodes.size();
  CHECK_LE(node_begin, static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()));
  *p_node_begin = static_cast<std::uint32_t>(node_begin);
  h_nodes.resize(node_begin + tree.Size());

  for (bst_node_t nidx = 0; nidx < tree.Size(); ++nidx) {
    auto& out = h_nodes[node_begin + nidx];
    if (tree.IsLeaf(nidx)) {
      out.is_leaf = 1;
      if (tree.NumTargets() == 1) {
        out.leaf = LeafPayload{LeafValue(tree, nidx, 0) * weight};
      } else {
        auto const n_targets = static_cast<std::size_t>(tree.NumTargets());
        auto constexpr kMaxOffset =
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
        CHECK_LE(n_targets, kMaxOffset);
        CHECK_LE(h_leaf_values.size(), kMaxOffset - n_targets)
            << "Compressed GPU SHAP leaf value offsets exceed uint32_t range.";
        out.leaf = LeafPayload{static_cast<std::uint32_t>(h_leaf_values.size())};
        for (bst_target_t target_idx = 0; target_idx < tree.NumTargets(); ++target_idx) {
          h_leaf_values.push_back(LeafValue(tree, nidx, target_idx) * weight);
        }
      }
      continue;
    }

    auto left = tree.LeftChild(nidx);
    auto right = tree.RightChild(nidx);
    auto parent_cover = tree.SumHess(nidx);
    CHECK_GE(parent_cover, 0.0f);
    CHECK_GE(tree.SumHess(left), 0.0f);
    CHECK_GE(tree.SumHess(right), 0.0f);

    out.left = left;
    out.right = right;
    out.split_global = tree.SplitIndex(nidx);
    out.split_cond = tree.SplitCond(nidx);
    out.left_weight = detail::BranchWeight(tree.SumHess(left), parent_cover);
    out.right_weight = detail::BranchWeight(tree.SumHess(right), parent_cover);
    if (common::IsCat(tree.cats.split_type, nidx)) {
      auto node_cats = tree.NodeCats(nidx);
      auto constexpr kMaxOffset =
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
      CHECK_LE(node_cats.size(), kMaxOffset)
          << "Compressed GPU SHAP category segment exceeds uint32_t range.";
      CHECK_LE(h_categories.size(), kMaxOffset - node_cats.size())
          << "Compressed GPU SHAP category offsets exceed uint32_t range.";
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

    // Repeated feature splits need q_prev from the previous occurrence of the same feature,
    // not the default probability 1.0. Compute that ancestor distance once during compression
    // and keep the GPU traversal stack purely array-index based.
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
}

CompressedModel CompressTreeBucket(gbm::GBTreeModel const& model,
                                   std::vector<bst_tree_t> const& tree_indices,
                                   common::Span<bst_target_t const> h_tree_groups,
                                   bst_target_t n_groups, std::vector<float> const* tree_weights) {
  std::vector<CompressedTree> h_trees;
  std::vector<CompressedNode> h_nodes;
  std::vector<float> h_leaf_values;
  std::vector<std::uint32_t> h_categories;

  h_trees.reserve(tree_indices.size() * std::max<bst_target_t>(n_groups, 1));
  for (auto tree_idx : tree_indices) {
    auto const weight = tree_weights == nullptr ? 1.0f : (*tree_weights)[tree_idx];
    std::uint32_t node_begin{0};
    if (model.trees.at(tree_idx)->IsMultiTarget()) {
      auto const tree = model.trees.at(tree_idx)->HostMtView();
      auto const n_targets = tree.NumTargets();
      CHECK_EQ(n_targets, n_groups);
      CompressTree(tree, &h_nodes, &h_leaf_values, &h_categories, weight, &node_begin);
      for (bst_target_t target_idx = 0; target_idx < n_targets; ++target_idx) {
        h_trees.push_back(CompressedTree{node_begin, target_idx, target_idx, true});
      }
    } else {
      auto const tree = model.trees.at(tree_idx)->HostScView();
      CompressTree(tree, &h_nodes, &h_leaf_values, &h_categories, weight, &node_begin);
      h_trees.push_back(CompressedTree{node_begin, h_tree_groups[tree_idx], 0, false});
    }
  }

  CompressedModel out;
  out.trees = dh::device_vector<CompressedTree>(h_trees.cbegin(), h_trees.cend());
  out.nodes = dh::device_vector<CompressedNode>(h_nodes.cbegin(), h_nodes.cend());
  out.leaf_values = dh::device_vector<float>(h_leaf_values.cbegin(), h_leaf_values.cend());
  out.categories = dh::device_vector<std::uint32_t>(h_categories.cbegin(), h_categories.cend());
  return out;
}

GpuQuadratureModelData PrepareGpuQuadratureModel(gbm::GBTreeModel const& model, bst_tree_t tree_end,
                                                 bst_target_t n_groups,
                                                 std::vector<float> const* tree_weights,
                                                 char const* prediction_kind) {
  if (tree_weights != nullptr) {
    CHECK_GE(tree_weights->size(), static_cast<std::size_t>(tree_end));
  }
  auto const h_tree_groups = model.TreeGroups(DeviceOrd::CPU());
  bst_node_t max_depth = 0;
  std::array<std::vector<bst_tree_t>, kGpuQuadratureDepthBuckets.size()> tree_buckets;
  std::vector<double> group_root_mean_sums(n_groups, 0.0);

  for (bst_tree_t tree_idx = 0; tree_idx < tree_end; ++tree_idx) {
    auto tree_depth = model.trees[tree_idx]->MaxDepth();
    max_depth = std::max(max_depth, tree_depth);
    // MaxDepth counts edges, while the iterative traversal stack stores nodes along the active
    // path. Account for the root node when selecting the smallest fitting depth specialization.
    auto path_depth = static_cast<std::size_t>(tree_depth) + 1;
    auto bucket_idx = DepthBucketIndex(path_depth);
    tree_buckets[bucket_idx].push_back(tree_idx);
    auto const weight = tree_weights == nullptr ? 1.0f : (*tree_weights)[tree_idx];
    if (model.trees[tree_idx]->IsMultiTarget()) {
      auto const tree = model.trees[tree_idx]->HostMtView();
      auto const n_targets = tree.NumTargets();
      CHECK_EQ(n_targets, n_groups)
          << prediction_kind << " expects one vector-leaf target per output group.";
      std::vector<double> root_means(n_targets, 0.0);
      detail::FillRootMeanValues(tree, RegTree::kRoot, 1.0, &root_means);
      for (bst_target_t target_idx = 0; target_idx < n_targets; ++target_idx) {
        group_root_mean_sums[target_idx] += root_means[target_idx] * weight;
      }
    } else {
      auto const tree = model.trees[tree_idx]->HostScView();
      group_root_mean_sums[h_tree_groups[tree_idx]] +=
          detail::FillRootMeanValue(tree, RegTree::kRoot) * weight;
    }
  }
  CHECK_LE(max_depth + 1, static_cast<bst_node_t>(kMaxGpuQuadratureDepth))
      << "GPU QuadratureSHAP currently supports trees of depth up to "
      << (kMaxGpuQuadratureDepth - 1) << ".";

  std::vector<float> h_group_root_mean_sums(group_root_mean_sums.size());
  std::transform(group_root_mean_sums.cbegin(), group_root_mean_sums.cend(),
                 h_group_root_mean_sums.begin(), [](double v) { return static_cast<float>(v); });

  GpuQuadratureModelData out;
  out.rule = detail::GetQuadratureRule();
  out.group_root_mean_sums =
      dh::device_vector<float>(h_group_root_mean_sums.cbegin(), h_group_root_mean_sums.cend());
  for (std::size_t i = 0; i < tree_buckets.size(); ++i) {
    out.compressed[i] =
        CompressTreeBucket(model, tree_buckets[i], h_tree_groups, n_groups, tree_weights);
  }
  return out;
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

struct SubgroupOps {
  static_assert(kGpuQuadratureSegmentWidth == kGpuQuadraturePoints);
  int row_slot;
  int point;
  // False for the padding slots of the last row tile: they repeat the last row and never write.
  bool row_valid;
  bool is_leader;
  bool is_warp_leader;

  XGBOOST_DEV_INLINE explicit SubgroupOps(int lane)
      : row_slot{lane / kGpuQuadratureSegmentWidth},
        point{lane % kGpuQuadratureSegmentWidth},
        row_valid{true},
        is_leader{point == 0},
        is_warp_leader{lane == 0} {}

  [[nodiscard]] XGBOOST_DEV_INLINE bool ShouldWrite() const { return is_leader && row_valid; }

  template <typename T>
  [[nodiscard]] XGBOOST_DEV_INLINE T Sum(T value) const {
    for (int offset = kGpuQuadraturePoints / 2; offset > 0; offset /= 2) {
      value += __shfl_down_sync(kFullWarpMask, value, offset, kGpuQuadraturePoints);
    }
    return value;
  }

  XGBOOST_DEV_INLINE void Sync() const { __syncwarp(); }
};

// Wrap the shared-memory layout in semantic accessors so the task runner talks in terms of path
// state instead of raw multidimensional indexing.
template <int DepthCap>
struct QuadratureSharedState {
  bst_node_t nodes[kGpuQuadratureWarpsPerBlock][DepthCap];
  std::uint8_t stages[kGpuQuadratureWarpsPerBlock][DepthCap];
  std::uint8_t goes_left[kGpuQuadratureWarpsPerBlock][kGpuQuadratureRowsPerWarp][DepthCap];
  // q_d(t): path probability at depth d for one row-slot evaluated at quadrature point t.
  float path_prob[kGpuQuadratureWarpsPerBlock][kGpuQuadratureRowsPerWarp][DepthCap];
  // G_d(t): multiplicative basis carried down the path before the leaf value is applied.
  float basis[kGpuQuadratureWarpsPerBlock][kGpuQuadratureRowsPerWarp][DepthCap]
             [kGpuQuadraturePoints];

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
};

// Interactions also enumerate the features on the active path. All rows of a warp share the path,
// so keep each depth's split feature and whether a deeper split on the same feature shadows it.
template <int DepthCap>
struct QuadratureInteractionSharedState : public QuadratureSharedState<DepthCap> {
  bst_feature_t splits[kGpuQuadratureWarpsPerBlock][DepthCap];
  std::uint8_t shadowed[kGpuQuadratureWarpsPerBlock][DepthCap];

  [[nodiscard]] XGBOOST_DEV_INLINE bst_feature_t& Split(int warp, int depth) {
    return splits[warp][depth];
  }

  [[nodiscard]] XGBOOST_DEV_INLINE bool Shadowed(int warp, int depth) const {
    return static_cast<bool>(shadowed[warp][depth]);
  }

  XGBOOST_DEV_INLINE void SetShadowed(int warp, int depth, bool value) {
    shadowed[warp][depth] = static_cast<std::uint8_t>(value);
  }
};

template <typename Loader, typename SharedT>
struct QuadratureShapTaskRunner {
  Loader loader;
  SubgroupOps subgroup;
  SharedT& shared;
  CompressedTree const* trees;
  CompressedNode const* nodes;
  float const* leaf_values;
  std::uint32_t const* categories;
  float* phis;
  bst_idx_t base_rowid;
  bst_target_t n_groups;
  bst_feature_t n_columns;
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

    auto p_enter = shared.PathProbability(warp, subgroup.row_slot, parent_depth);
    auto q_prev = PreviousPathProbability(node.prev_same_offset_plus1, parent_depth,
                                          shared.PathProbabilityRow(warp, subgroup.row_slot));

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

    // All lanes must finish reading the current path before the warp leader reuses the child node
    // and stage slots.
    subgroup.Sync();
    if (child_idx == 0) {
      auto child_weight = node.right_weight;
      auto child_node = node.right;
      auto p_e =
          shared.GoesLeft(warp, subgroup.row_slot, parent_depth) ? 0.0f : q_prev / child_weight;
      if (subgroup.is_leader) {
        shared.PathProbability(warp, subgroup.row_slot, parent_depth) = p_e;
      }

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

  XGBOOST_DEV_INLINE void Descend(CompressedTree const& tree, CompressedNode const* nodes_for_tree,
                                  bst_idx_t ridx, int* stack_size, bool* have_return,
                                  float* ret_val) {
    int depth = *stack_size - 1;
    auto const& node = nodes_for_tree[shared.Node(warp, depth)];
    if (node.is_leaf) {
      auto leaf_value =
          tree.is_vector_leaf ? leaf_values[node.leaf.begin + tree.target_idx] : node.leaf.value;
      *ret_val = shared.Basis(warp, subgroup.row_slot, depth, subgroup.point) * leaf_value;
      (*stack_size)--;
      *have_return = true;
      return;
    }

    // stage == 0 explores the left child first. After the return path updates the parent state,
    // the second visit uses the cached go-left decision to push the right child.
    int child = static_cast<int>(shared.Stage(warp, depth) != 0);
    // All lanes must read the old stage before the warp leader changes it.
    subgroup.Sync();
    if (child == 0) {
      if (subgroup.is_warp_leader) {
        shared.Stage(warp, depth) = 1;
      }
      subgroup.Sync();
    }

    auto child_weight = child == 0 ? node.left_weight : node.right_weight;
    auto child_node = child == 0 ? node.left : node.right;
    // For repeated feature splits, replace the default q_prev with the path probability from
    // the nearest previous split on the same feature.
    auto q_prev = PreviousPathProbability(node.prev_same_offset_plus1, depth,
                                          shared.PathProbabilityRow(warp, subgroup.row_slot));
    bool goes_left = this->EvaluateGoesLeft(ridx, node);
    // p_e is the path probability after taking the chosen child for this row.
    auto p_e = (child == 0 ? goes_left : !goes_left) ? q_prev / child_weight : 0.0f;
    if (subgroup.is_leader) {
      shared.SetGoesLeft(warp, subgroup.row_slot, depth, goes_left);
      shared.PathProbability(warp, subgroup.row_slot, depth) = p_e;
    }

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
    auto ridx = row_tile * kGpuQuadratureRowsPerWarp + subgroup.row_slot;
    // The last tile can have empty row slots. Pad them with the last row and drop their output.
    subgroup.row_valid = ridx < loader.NumRows();
    ridx = subgroup.row_valid ? ridx : loader.NumRows() - 1;
    auto row_idx = base_rowid + static_cast<bst_idx_t>(ridx);
    auto tree = trees[tree_idx];
    auto nodes_for_tree = nodes + tree.node_begin;

    this->InitializeTask();

    int stack_size = 1;
    bool have_return = false;
    float ret_val = 0.0f;
    while (stack_size > 0 || have_return) {
      if (have_return) {
        if (!this->HandleReturn(row_idx, tree.group_idx, nodes_for_tree, &stack_size, &have_return,
                                &ret_val)) {
          break;
        }
        continue;
      }
      this->Descend(tree, nodes_for_tree, ridx, &stack_size, &have_return, &ret_val);
    }
  }
};

template <typename Loader, typename SharedT>
struct QuadratureShapInteractionTaskRunner {
  Loader loader;
  SubgroupOps subgroup;
  SharedT& shared;
  CompressedTree const* trees;
  CompressedNode const* nodes;
  float const* leaf_values;
  std::uint32_t const* categories;
  float* phis;
  bst_idx_t base_rowid;
  bst_target_t n_groups;
  bst_feature_t n_columns;
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

  XGBOOST_DEV_INLINE void EnterPathNode(CompressedNode const& node, int depth) {
    shared.Split(warp, depth) = node.split_global;
    shared.SetShadowed(warp, depth, false);
    if (node.prev_same_offset_plus1 != 0) {
      shared.SetShadowed(warp, depth - static_cast<int>(node.prev_same_offset_plus1) + 1, true);
    }
  }

  XGBOOST_DEV_INLINE void ExitPathNode(CompressedNode const& node, int depth) {
    if (node.prev_same_offset_plus1 == 0) {
      return;
    }
    subgroup.Sync();
    if (subgroup.is_warp_leader) {
      shared.SetShadowed(warp, depth - static_cast<int>(node.prev_same_offset_plus1) + 1, false);
    }
    subgroup.Sync();
  }

  template <typename Fn>
  XGBOOST_DEV_INLINE void ForEachUniquePartner(int current_depth, Fn&& fn) const {
    for (int depth = current_depth - 1; depth >= 0; --depth) {
      if (!shared.Shadowed(warp, depth)) {
        fn(depth, shared.Split(warp, depth));
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

    auto p_enter = shared.PathProbability(warp, subgroup.row_slot, parent_depth);
    auto q_prev = PreviousPathProbability(node.prev_same_offset_plus1, parent_depth,
                                          shared.PathProbabilityRow(warp, subgroup.row_slot));

    auto edge_delta_local =
        ExtractQuadratureEdgeDeltaLocal(quad_node, quad_weight, *ret_val, p_enter, q_prev);
    auto diag_contrib = subgroup.Sum(edge_delta_local);
    this->AddDiagonalContribution(row_idx, tree_group, node.split_global, diag_contrib);

    this->ForEachUniquePartner(parent_depth, [&](int partner_depth, bst_feature_t partner_split) {
      auto q_partner = shared.PathProbability(warp, subgroup.row_slot, partner_depth);
      auto pair_delta_local =
          ExtractQuadratureInteractionDeltaLocal(quad_node, edge_delta_local, q_partner);
      auto pair_contrib = subgroup.Sum(pair_delta_local);
      this->AddPairContribution(row_idx, tree_group, node.split_global, partner_split,
                                pair_contrib);
    });

    // All lanes must finish reading the current path before the warp leader reuses the child node
    // and stage slots.
    subgroup.Sync();
    if (child_idx == 0) {
      auto child_weight = node.right_weight;
      auto child_node = node.right;
      auto p_e =
          shared.GoesLeft(warp, subgroup.row_slot, parent_depth) ? 0.0f : q_prev / child_weight;
      if (subgroup.is_leader) {
        shared.PathProbability(warp, subgroup.row_slot, parent_depth) = p_e;
      }

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
      this->ExitPathNode(node, parent_depth);
      *ret_val += shared.Basis(warp, subgroup.row_slot, parent_depth, subgroup.point);
      (*stack_size)--;
      *have_return = true;
    }

    return true;
  }

  XGBOOST_DEV_INLINE void Descend(CompressedTree const& tree, CompressedNode const* nodes_for_tree,
                                  bst_idx_t ridx, int* stack_size, bool* have_return,
                                  float* ret_val) {
    int depth = *stack_size - 1;
    auto const& node = nodes_for_tree[shared.Node(warp, depth)];
    if (node.is_leaf) {
      auto leaf_value =
          tree.is_vector_leaf ? leaf_values[node.leaf.begin + tree.target_idx] : node.leaf.value;
      *ret_val = shared.Basis(warp, subgroup.row_slot, depth, subgroup.point) * leaf_value;
      (*stack_size)--;
      *have_return = true;
      return;
    }

    int child = static_cast<int>(shared.Stage(warp, depth) != 0);
    // All lanes must read the old stage before the warp leader changes it.
    subgroup.Sync();
    if (child == 0) {
      if (subgroup.is_warp_leader) {
        shared.Stage(warp, depth) = 1;
        this->EnterPathNode(node, depth);
      }
      subgroup.Sync();
    }

    auto child_weight = child == 0 ? node.left_weight : node.right_weight;
    auto child_node = child == 0 ? node.left : node.right;
    auto q_prev = PreviousPathProbability(node.prev_same_offset_plus1, depth,
                                          shared.PathProbabilityRow(warp, subgroup.row_slot));
    bool goes_left = this->EvaluateGoesLeft(ridx, node);
    auto p_e = (child == 0 ? goes_left : !goes_left) ? q_prev / child_weight : 0.0f;
    if (subgroup.is_leader) {
      shared.SetGoesLeft(warp, subgroup.row_slot, depth, goes_left);
      shared.PathProbability(warp, subgroup.row_slot, depth) = p_e;
    }

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
    auto ridx = row_tile * kGpuQuadratureRowsPerWarp + subgroup.row_slot;
    // The last tile can have empty row slots. Pad them with the last row and drop their output.
    subgroup.row_valid = ridx < loader.NumRows();
    ridx = subgroup.row_valid ? ridx : loader.NumRows() - 1;
    auto row_idx = base_rowid + static_cast<bst_idx_t>(ridx);
    auto tree = trees[tree_idx];
    auto nodes_for_tree = nodes + tree.node_begin;

    this->InitializeTask();

    int stack_size = 1;
    bool have_return = false;
    float ret_val = 0.0f;
    while (stack_size > 0 || have_return) {
      if (have_return) {
        if (!this->HandleReturn(row_idx, tree.group_idx, nodes_for_tree, &stack_size, &have_return,
                                &ret_val)) {
          break;
        }
        continue;
      }
      this->Descend(tree, nodes_for_tree, ridx, &stack_size, &have_return, &ret_val);
    }
  }
};

template <int DepthCap, typename Loader>
__global__ void __launch_bounds__(kGpuQuadratureTreeBlockThreads, 9)
    QuadratureShapTaskKernel(Loader loader, bst_idx_t base_rowid, bst_target_t n_groups,
                             bst_feature_t n_columns, std::size_t row_tiles, std::size_t n_trees,
                             CompressedTree const* __restrict__ trees,
                             CompressedNode const* __restrict__ nodes,
                             float const* __restrict__ leaf_values,
                             std::uint32_t const* __restrict__ categories, QuadratureRule rule,
                             float* __restrict__ phis) {
  static_assert(DepthCap <= static_cast<int>(kMaxGpuQuadratureDepth));
  static_assert(dh::WarpThreads() % kGpuQuadratureRowsPerWarp == 0);
  static_assert(kGpuQuadratureTreeBlockThreads % dh::WarpThreads() == 0);
  using SharedT = QuadratureSharedState<DepthCap>;

  __shared__ SharedT shared;

  int warp = static_cast<int>(threadIdx.x) / dh::WarpThreads();
  int lane = static_cast<int>(threadIdx.x) % dh::WarpThreads();
  auto subgroup = SubgroupOps{lane};

  auto global_warp =
      (static_cast<std::size_t>(blockIdx.x) * kGpuQuadratureTreeBlockThreads + threadIdx.x) /
      dh::WarpThreads();
  auto warp_stride =
      (static_cast<std::size_t>(gridDim.x) * kGpuQuadratureTreeBlockThreads) / dh::WarpThreads();
  auto n_tasks = n_trees * row_tiles;

  auto runner = QuadratureShapTaskRunner<Loader, SharedT>{loader,
                                                          subgroup,
                                                          shared,
                                                          trees,
                                                          nodes,
                                                          leaf_values,
                                                          categories,
                                                          phis,
                                                          base_rowid,
                                                          n_groups,
                                                          n_columns,
                                                          row_tiles,
                                                          warp,
                                                          rule.nodes[subgroup.point],
                                                          rule.weights[subgroup.point]};

  for (std::size_t task = global_warp; task < n_tasks; task += warp_stride) {
    runner.RunTask(task);
  }
}

template <int DepthCap, typename Loader>
void LaunchQuadratureShapBuckets(Context const* ctx, Loader loader, bst_idx_t base_rowid,
                                 bst_target_t n_groups, bst_feature_t n_columns,
                                 CompressedModel const& compressed, QuadratureRule rule,
                                 HostDeviceVector<float>* out_contribs) {
  static_assert(kGpuQuadratureTreeBlockThreads % dh::WarpThreads() == 0);
  auto row_tiles =
      static_cast<std::size_t>(common::DivRoundUp(loader.NumRows(), kGpuQuadratureRowsPerWarp));
  if (compressed.trees.empty() || row_tiles == 0) {
    return;
  }
  auto trees = thrust::raw_pointer_cast(compressed.trees.data());
  auto nodes = thrust::raw_pointer_cast(compressed.nodes.data());
  auto leaf_values = thrust::raw_pointer_cast(compressed.leaf_values.data());
  auto categories = thrust::raw_pointer_cast(compressed.categories.data());
  auto phis = out_contribs->DeviceSpan().data();
  auto n_tasks = compressed.trees.size() * row_tiles;
  auto grids = common::DivRoundUp(n_tasks, static_cast<std::size_t>(kGpuQuadratureWarpsPerBlock));
  QuadratureShapTaskKernel<DepthCap>
      <<<static_cast<uint32_t>(grids), static_cast<uint32_t>(kGpuQuadratureTreeBlockThreads), 0,
         ctx->CUDACtx()->Stream()>>>(loader, base_rowid, n_groups, n_columns, row_tiles,
                                     compressed.trees.size(), trees, nodes, leaf_values, categories,
                                     rule, phis);
  dh::safe_cuda(cudaGetLastError());
}

template <int DepthCap, typename Loader>
__global__ void __launch_bounds__(kGpuQuadratureTreeBlockThreads, 9)
    QuadratureShapInteractionTaskKernel(Loader loader, bst_idx_t base_rowid, bst_target_t n_groups,
                                        bst_feature_t n_columns, std::size_t row_tiles,
                                        std::size_t n_trees,
                                        CompressedTree const* __restrict__ trees,
                                        CompressedNode const* __restrict__ nodes,
                                        float const* __restrict__ leaf_values,
                                        std::uint32_t const* __restrict__ categories,
                                        QuadratureRule rule, float* __restrict__ phis) {
  static_assert(DepthCap <= static_cast<int>(kMaxGpuQuadratureDepth));
  static_assert(dh::WarpThreads() % kGpuQuadratureRowsPerWarp == 0);
  static_assert(kGpuQuadratureTreeBlockThreads % dh::WarpThreads() == 0);
  using SharedT = QuadratureInteractionSharedState<DepthCap>;

  __shared__ SharedT shared;

  int warp = static_cast<int>(threadIdx.x) / dh::WarpThreads();
  int lane = static_cast<int>(threadIdx.x) % dh::WarpThreads();
  auto subgroup = SubgroupOps{lane};

  auto global_warp =
      (static_cast<std::size_t>(blockIdx.x) * kGpuQuadratureTreeBlockThreads + threadIdx.x) /
      dh::WarpThreads();
  auto warp_stride =
      (static_cast<std::size_t>(gridDim.x) * kGpuQuadratureTreeBlockThreads) / dh::WarpThreads();
  auto n_tasks = n_trees * row_tiles;

  auto runner = QuadratureShapInteractionTaskRunner<Loader, SharedT>{loader,
                                                                     subgroup,
                                                                     shared,
                                                                     trees,
                                                                     nodes,
                                                                     leaf_values,
                                                                     categories,
                                                                     phis,
                                                                     base_rowid,
                                                                     n_groups,
                                                                     n_columns,
                                                                     row_tiles,
                                                                     warp,
                                                                     rule.nodes[subgroup.point],
                                                                     rule.weights[subgroup.point]};

  for (std::size_t task = global_warp; task < n_tasks; task += warp_stride) {
    runner.RunTask(task);
  }
}

template <int DepthCap, typename Loader>
void LaunchQuadratureShapInteractionBuckets(Context const* ctx, Loader loader, bst_idx_t base_rowid,
                                            bst_target_t n_groups, bst_feature_t n_columns,
                                            CompressedModel const& compressed, QuadratureRule rule,
                                            HostDeviceVector<float>* out_contribs) {
  static_assert(kGpuQuadratureTreeBlockThreads % dh::WarpThreads() == 0);
  auto row_tiles =
      static_cast<std::size_t>(common::DivRoundUp(loader.NumRows(), kGpuQuadratureRowsPerWarp));
  if (compressed.trees.empty() || row_tiles == 0) {
    return;
  }
  auto trees = thrust::raw_pointer_cast(compressed.trees.data());
  auto nodes = thrust::raw_pointer_cast(compressed.nodes.data());
  auto leaf_values = thrust::raw_pointer_cast(compressed.leaf_values.data());
  auto categories = thrust::raw_pointer_cast(compressed.categories.data());
  auto phis = out_contribs->DeviceSpan().data();
  auto n_tasks = compressed.trees.size() * row_tiles;
  auto grids = common::DivRoundUp(n_tasks, static_cast<std::size_t>(kGpuQuadratureWarpsPerBlock));
  QuadratureShapInteractionTaskKernel<DepthCap>
      <<<static_cast<uint32_t>(grids), static_cast<uint32_t>(kGpuQuadratureTreeBlockThreads), 0,
         ctx->CUDACtx()->Stream()>>>(loader, base_rowid, n_groups, n_columns, row_tiles,
                                     compressed.trees.size(), trees, nodes, leaf_values, categories,
                                     rule, phis);
  dh::safe_cuda(cudaGetLastError());
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
  auto n_features = model.learner_model_state->num_feature;
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
                std::vector<float> const* tree_weights) {
  xgboost_NVTX_FN_RANGE();
  curt::SetDevice(ctx->Ordinal());

  tree_end = predictor::GetTreeLimit(model.trees, tree_end);
  auto const ngroup = model.learner_model_state->num_output_group;
  CHECK_NE(ngroup, 0);
  auto const ncolumns = model.learner_model_state->num_feature + 1;
  auto dim_size = ncolumns * ngroup;
  out_contribs->SetDevice(ctx->Device());
  out_contribs->Resize(p_fmat->Info().num_row_ * dim_size);
  out_contribs->Fill(0.0f);

  auto prepared =
      PrepareGpuQuadratureModel(model, tree_end, ngroup, tree_weights, "Predict contribution");

  auto new_enc =
      p_fmat->Cats()->NeedRecode() ? p_fmat->Cats()->DeviceView(ctx) : enc::DeviceColumnsView{};
  auto group_root_mean_sums = prepared.GroupRootMeanSums();

  LaunchShap(ctx, p_fmat, new_enc, model, [&](auto&& loader, bst_idx_t base_rowid) {
    LaunchQuadratureShapBuckets<16>(ctx, loader, base_rowid, ngroup, ncolumns,
                                    prepared.compressed[0], prepared.rule, out_contribs);
    LaunchQuadratureShapBuckets<32>(ctx, loader, base_rowid, ngroup, ncolumns,
                                    prepared.compressed[1], prepared.rule, out_contribs);
    LaunchQuadratureShapBuckets<64>(ctx, loader, base_rowid, ngroup, ncolumns,
                                    prepared.compressed[2], prepared.rule, out_contribs);
    LaunchQuadratureShapBuckets<128>(ctx, loader, base_rowid, ngroup, ncolumns,
                                     prepared.compressed[3], prepared.rule, out_contribs);
  });

  p_fmat->Info().base_margin_.SetDevice(ctx->Device());
  auto margin = p_fmat->Info().base_margin_.Data()->ConstDeviceSpan();
  auto base_score = model.learner_model_state->BaseScore(ctx);
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

  curt::SetDevice(ctx->Ordinal());
  tree_end = predictor::GetTreeLimit(model.trees, tree_end);
  auto const ngroup = model.learner_model_state->num_output_group;
  CHECK_NE(ngroup, 0);
  auto const ncolumns = model.learner_model_state->num_feature + 1;
  auto dim_size = ncolumns * ncolumns * ngroup;
  out_contribs->SetDevice(ctx->Device());
  out_contribs->Resize(p_fmat->Info().num_row_ * dim_size);
  out_contribs->Fill(0.0f);

  auto prepared = PrepareGpuQuadratureModel(model, tree_end, ngroup, tree_weights,
                                            "Predict interaction contribution");

  auto new_enc =
      p_fmat->Cats()->NeedRecode() ? p_fmat->Cats()->DeviceView(ctx) : enc::DeviceColumnsView{};
  auto group_root_mean_sums = prepared.GroupRootMeanSums();

  LaunchShap(ctx, p_fmat, new_enc, model, [&](auto&& loader, bst_idx_t base_rowid) {
    LaunchQuadratureShapInteractionBuckets<16>(ctx, loader, base_rowid, ngroup, ncolumns,
                                               prepared.compressed[0], prepared.rule, out_contribs);
    LaunchQuadratureShapInteractionBuckets<32>(ctx, loader, base_rowid, ngroup, ncolumns,
                                               prepared.compressed[1], prepared.rule, out_contribs);
    LaunchQuadratureShapInteractionBuckets<64>(ctx, loader, base_rowid, ngroup, ncolumns,
                                               prepared.compressed[2], prepared.rule, out_contribs);
    LaunchQuadratureShapInteractionBuckets<128>(ctx, loader, base_rowid, ngroup, ncolumns,
                                                prepared.compressed[3], prepared.rule,
                                                out_contribs);
  });

  p_fmat->Info().base_margin_.SetDevice(ctx->Device());
  auto margin = p_fmat->Info().base_margin_.Data()->ConstDeviceSpan();
  auto base_score = model.learner_model_state->BaseScore(ctx);
  auto phis = out_contribs->DeviceSpan();
  auto n_matrices = p_fmat->Info().num_row_ * ngroup;
  // Average the two directional estimates of each pair, one thread per pair.
  dh::LaunchN(n_matrices * ncolumns * ncolumns, ctx->CUDACtx()->Stream(),
              [=] __device__(std::size_t idx) {
                auto [m, r, c] = linalg::UnravelIndex(idx, n_matrices, ncolumns, ncolumns);
                if (r < c) {
                  auto mirror = (m * ncolumns + c) * ncolumns + r;
                  auto sym = 0.5f * (phis[idx] + phis[mirror]);
                  phis[idx] = sym;
                  phis[mirror] = sym;
                }
              });
  // Each diagonal entry is the SHAP value minus the row's interactions. Since the matrix is
  // symmetric, subtract the column instead: neighbouring threads then read neighbouring addresses.
  dh::LaunchN(n_matrices * ncolumns, ctx->CUDACtx()->Stream(), [=] __device__(std::size_t idx) {
    auto [m, c] = linalg::UnravelIndex(idx, n_matrices, ncolumns);
    auto matrix = phis.subspan(m * ncolumns * ncolumns, ncolumns * ncolumns);
    float value = matrix[c * ncolumns + c];
    if (c == ncolumns - 1) {
      auto gid = m % ngroup;
      value += group_root_mean_sums[gid] + (margin.empty() ? base_score(gid) : margin[m]);
    }
    for (bst_feature_t r = 0; r < ncolumns; ++r) {
      if (r != c) {
        value -= matrix[r * ncolumns + c];
      }
    }
    matrix[c * ncolumns + c] = value;
  });
}

void ApproxFeatureImportance(Context const*, DMatrix*, HostDeviceVector<float>*,
                             gbm::GBTreeModel const&, bst_tree_t, std::vector<float> const*) {
  StringView not_implemented{
      "contribution is not implemented in the GPU predictor, use CPU instead."};
  LOG(FATAL) << "Approximated " << not_implemented;
}
}  // namespace xgboost::interpretability::cuda_impl
