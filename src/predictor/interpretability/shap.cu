/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <algorithm>
#include <cuda/std/variant>  // for variant
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../common/cuda_rt_utils.h"   // for SetDevice
#include "../../common/device_helpers.cuh"
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

constexpr std::size_t kQuadratureTreeShapPoints = 8;
constexpr double kQuadratureTreeShapBuildQeps = 1e-15;
constexpr float kQuadratureTreeShapUnseen = -999.0f;

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

struct QuadratureRule {
  float nodes[kQuadratureTreeShapPoints];
  float weights[kQuadratureTreeShapPoints];
};

struct QuadraturePathElement {
  bst_feature_t split_index;
  float p_child;
};

struct GpuQuadratureTreeShapModelData {
  HostDeviceVector<float> group_root_mean_sums;
  bst_node_t max_depth{0};
};

QuadratureRule MakeQuadratureRule() {
  auto const rule_d =
      detail::MakeEndpointQuadrature<kQuadratureTreeShapPoints>(kQuadratureTreeShapBuildQeps);
  QuadratureRule out;
  for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
    out.nodes[i] = static_cast<float>(rule_d.nodes[i]);
    out.weights[i] = static_cast<float>(rule_d.weights[i]);
  }
  return out;
}

double FillRootMeanValue(tree::ScalarTreeView const& tree, bst_node_t nidx) {
  if (tree.IsLeaf(nidx)) {
    return tree.LeafValue(nidx);
  }
  auto left = tree.LeftChild(nidx);
  auto right = tree.RightChild(nidx);
  double result = FillRootMeanValue(tree, left) * tree.SumHess(left);
  result += FillRootMeanValue(tree, right) * tree.SumHess(right);
  result /= tree.SumHess(nidx);
  return result;
}

void ValidateQuadratureTreeShapCovers(tree::ScalarTreeView const& tree, bst_node_t nidx) {
  if (tree.IsLeaf(nidx)) {
    return;
  }

  CHECK_GT(tree.SumHess(nidx), 0.0f)
      << "GPU QuadratureTreeSHAP is undefined for trees with non-positive cover at split nodes.";

  auto left = tree.LeftChild(nidx);
  auto right = tree.RightChild(nidx);
  CHECK_GT(tree.SumHess(left), 0.0f)
      << "GPU QuadratureTreeSHAP is undefined for trees with non-positive cover at child nodes.";
  CHECK_GT(tree.SumHess(right), 0.0f)
      << "GPU QuadratureTreeSHAP is undefined for trees with non-positive cover at child nodes.";

  ValidateQuadratureTreeShapCovers(tree, left);
  ValidateQuadratureTreeShapCovers(tree, right);
}

GpuQuadratureTreeShapModelData MakeGpuQuadratureTreeShapModelData(
    Context const* ctx, gbm::GBTreeModel const& model, bst_tree_t tree_end,
    std::vector<float> const* tree_weights) {
  auto const n_groups = model.learner_model_param->num_output_group;
  if (tree_weights != nullptr) {
    CHECK_GE(tree_weights->size(), static_cast<std::size_t>(tree_end));
  }

  GpuQuadratureTreeShapModelData out;
  out.group_root_mean_sums = HostDeviceVector<float>(n_groups, 0.0f, ctx->Device());
  auto& h_root_mean_sums = out.group_root_mean_sums.HostVector();

  for (bst_tree_t i = 0; i < tree_end; ++i) {
    CHECK(!model.trees[i]->IsMultiTarget()) << " SHAP " << MTNotImplemented();
    auto tree = model.trees[i]->HostScView();
    ValidateQuadratureTreeShapCovers(tree, RegTree::kRoot);
    out.max_depth = std::max(out.max_depth, tree.MaxDepth());

    auto gid = model.TreeGroups(DeviceOrd::CPU())[i];
    auto weight = tree_weights == nullptr ? 1.0f : (*tree_weights)[i];
    h_root_mean_sums[gid] += static_cast<float>(FillRootMeanValue(tree, RegTree::kRoot) * weight);
  }

  out.group_root_mean_sums.SetDevice(ctx->Device());
  return out;
}

XGBOOST_DEVICE void AddInPlace(float* lhs, float const* rhs) {
  for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
    lhs[i] += rhs[i];
  }
}

XGBOOST_DEVICE float ExtractQuadratureDelta(QuadratureRule const& rule, float const* h_vals,
                                            float p_enter, float p_exit) {
  float acc = 0.0f;
  if (p_enter != 1.0f) {
    auto const alpha_enter = p_enter - 1.0f;
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      acc += alpha_enter * h_vals[i] / (1.0f + alpha_enter * rule.nodes[i]);
    }
  }
  if (p_exit != 1.0f) {
    auto const alpha_exit = p_exit - 1.0f;
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      acc -= alpha_exit * h_vals[i] / (1.0f + alpha_exit * rule.nodes[i]);
    }
  }
  return acc;
}

XGBOOST_DEVICE float ExtractQuadratureInteractionDelta(QuadratureRule const& rule,
                                                       float const* h_vals, float p_enter,
                                                       float p_exit, float q_partner) {
  if (q_partner == 1.0f) {
    return 0.0f;
  }

  auto const alpha_partner = q_partner - 1.0f;
  auto const alpha_enter = p_enter - 1.0f;

  float acc = 0.0f;
  if (p_exit == 1.0f) {
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      auto const edge_delta = alpha_enter / (1.0f + alpha_enter * rule.nodes[i]);
      acc += alpha_partner * h_vals[i] * edge_delta / (1.0f + alpha_partner * rule.nodes[i]);
    }
  } else {
    auto const alpha_exit = p_exit - 1.0f;
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      auto const edge_delta = alpha_enter / (1.0f + alpha_enter * rule.nodes[i]) -
                              alpha_exit / (1.0f + alpha_exit * rule.nodes[i]);
      acc += alpha_partner * h_vals[i] * edge_delta / (1.0f + alpha_partner * rule.nodes[i]);
    }
  }
  return acc;
}

template <typename Tree>
XGBOOST_DEVICE void WriteWeightedLeafReturn(Tree const& tree, QuadratureRule const& rule,
                                            bst_node_t nidx, float const* c_vals, float w_prod,
                                            float tree_weight, float* out_h) {
  auto const leaf_scale = w_prod * tree.LeafValue(nidx) * tree_weight;
  for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
    out_h[i] = c_vals[i] * leaf_scale * rule.weights[i];
  }
}

template <typename Tree, typename Loader>
XGBOOST_DEVICE bool EvaluateGoesLeft(Tree const& tree, Loader const& loader, bst_idx_t row_idx,
                                     bst_node_t nidx) {
  auto split_index = tree.SplitIndex(nidx);
  auto const& cats = tree.GetCategoriesMatrix();
  auto fvalue = loader.GetElement(row_idx, split_index);
  auto next = predictor::GetNextNode<true, true>(tree, nidx, fvalue, isnan(fvalue), cats);
  return next == tree.LeftChild(nidx);
}

template <typename Tree>
XGBOOST_DEVICE float ChildWeight(Tree const& tree, bst_node_t parent, bst_node_t child) {
  auto parent_cover = tree.SumHess(parent);
  return tree.SumHess(child) / parent_cover;
}

template <typename Loader>
XGBOOST_DEVICE void RunAdditiveNode(tree::ScalarTreeView const& tree, Loader const& loader,
                                    bst_idx_t row_idx, QuadratureRule const& rule, float* path_prob,
                                    bst_node_t nidx, float const* c_vals, float w_prod,
                                    float tree_weight, float* out_h, float* phi);

template <typename Loader>
XGBOOST_DEVICE void VisitAdditiveChild(tree::ScalarTreeView const& tree, Loader const& loader,
                                       bst_idx_t row_idx, QuadratureRule const& rule,
                                       float* path_prob, bst_node_t split_node,
                                       bst_node_t child_node, float child_weight, bool satisfies,
                                       float const* c_vals, float w_prod, float tree_weight,
                                       float* out_h, float* phi) {
  auto split_index = tree.SplitIndex(split_node);
  auto p_old = path_prob[split_index];
  float p_exit = p_old == kQuadratureTreeShapUnseen ? 1.0f : p_old;
  float p_e = satisfies ? p_exit / child_weight : 0.0f;

  float c_child[kQuadratureTreeShapPoints];
  auto alpha_e = p_e - 1.0f;
  for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
    c_child[i] = c_vals[i] * (1.0f + alpha_e * rule.nodes[i]);
  }

  if (p_exit != 1.0f) {
    auto const alpha_exit = p_exit - 1.0f;
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      c_child[i] /= 1.0f + alpha_exit * rule.nodes[i];
    }
  }

  path_prob[split_index] = p_e;
  RunAdditiveNode(tree, loader, row_idx, rule, path_prob, child_node, c_child,
                  w_prod * child_weight, tree_weight, out_h, phi);
  phi[split_index] += ExtractQuadratureDelta(rule, out_h, p_e, p_exit);
  path_prob[split_index] = p_old;
}

template <typename Loader>
XGBOOST_DEVICE void RunAdditiveNode(tree::ScalarTreeView const& tree, Loader const& loader,
                                    bst_idx_t row_idx, QuadratureRule const& rule, float* path_prob,
                                    bst_node_t nidx, float const* c_vals, float w_prod,
                                    float tree_weight, float* out_h, float* phi) {
  if (tree.IsLeaf(nidx)) {
    WriteWeightedLeafReturn(tree, rule, nidx, c_vals, w_prod, tree_weight, out_h);
    return;
  }

  auto left = tree.LeftChild(nidx);
  auto right = tree.RightChild(nidx);
  auto left_weight = ChildWeight(tree, nidx, left);
  auto right_weight = ChildWeight(tree, nidx, right);
  auto goes_left = EvaluateGoesLeft(tree, loader, row_idx, nidx);

  float right_h[kQuadratureTreeShapPoints];
  VisitAdditiveChild(tree, loader, row_idx, rule, path_prob, nidx, left, left_weight, goes_left,
                     c_vals, w_prod, tree_weight, out_h, phi);
  VisitAdditiveChild(tree, loader, row_idx, rule, path_prob, nidx, right, right_weight, !goes_left,
                     c_vals, w_prod, tree_weight, right_h, phi);
  AddInPlace(out_h, right_h);
}

template <typename Loader>
XGBOOST_DEVICE void RunInteractionNode(tree::ScalarTreeView const& tree, Loader const& loader,
                                       bst_idx_t row_idx, QuadratureRule const& rule,
                                       float* path_prob, QuadraturePathElement* path,
                                       std::size_t path_depth, std::size_t ncolumns,
                                       bst_node_t nidx, float const* c_vals, float w_prod,
                                       float tree_weight, float* out_h, float* matrix);

XGBOOST_DEVICE void HandleInteractionReturn(QuadratureRule const& rule, std::size_t ncolumns,
                                            QuadraturePathElement const* path,
                                            std::size_t path_depth, bst_feature_t split_index,
                                            float const* h_vals, float p_enter, float p_exit,
                                            float* matrix) {
  matrix[static_cast<std::size_t>(split_index) * ncolumns + split_index] +=
      ExtractQuadratureDelta(rule, h_vals, p_enter, p_exit);

  auto const current_split = path[path_depth - 1].split_index;
  bool skipped_current = false;
  for (std::size_t i = path_depth; i != 0; --i) {
    auto const idx = i - 1;
    auto const partner_split = path[idx].split_index;
    bool shadowed = false;
    for (std::size_t newer = path_depth; newer > i; --newer) {
      if (path[newer - 1].split_index == partner_split) {
        shadowed = true;
        break;
      }
    }
    if (shadowed) {
      continue;
    }
    if (!skipped_current && partner_split == current_split) {
      skipped_current = true;
      continue;
    }
    auto pair_delta =
        ExtractQuadratureInteractionDelta(rule, h_vals, p_enter, p_exit, path[idx].p_child);
    matrix[static_cast<std::size_t>(split_index) * ncolumns + partner_split] += pair_delta;
  }
}

template <typename Loader>
XGBOOST_DEVICE void VisitInteractionChild(tree::ScalarTreeView const& tree, Loader const& loader,
                                          bst_idx_t row_idx, QuadratureRule const& rule,
                                          float* path_prob, QuadraturePathElement* path,
                                          std::size_t path_depth, std::size_t ncolumns,
                                          bst_node_t split_node, bst_node_t child_node,
                                          float child_weight, bool satisfies, float const* c_vals,
                                          float w_prod, float tree_weight, float* out_h,
                                          float* matrix) {
  auto split_index = tree.SplitIndex(split_node);
  auto p_old = path_prob[split_index];
  float p_exit = p_old == kQuadratureTreeShapUnseen ? 1.0f : p_old;
  float p_e = satisfies ? p_exit / child_weight : 0.0f;

  float c_child[kQuadratureTreeShapPoints];
  auto alpha_e = p_e - 1.0f;
  for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
    c_child[i] = c_vals[i] * (1.0f + alpha_e * rule.nodes[i]);
  }

  if (p_exit != 1.0f) {
    auto const alpha_exit = p_exit - 1.0f;
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      c_child[i] /= 1.0f + alpha_exit * rule.nodes[i];
    }
  }

  path_prob[split_index] = p_e;
  path[path_depth] = QuadraturePathElement{split_index, p_e};
  RunInteractionNode(tree, loader, row_idx, rule, path_prob, path, path_depth + 1, ncolumns,
                     child_node, c_child, w_prod * child_weight, tree_weight, out_h, matrix);
  HandleInteractionReturn(rule, ncolumns, path, path_depth + 1, split_index, out_h, p_e, p_exit,
                          matrix);
  path_prob[split_index] = p_old;
}

template <typename Loader>
XGBOOST_DEVICE void RunInteractionNode(tree::ScalarTreeView const& tree, Loader const& loader,
                                       bst_idx_t row_idx, QuadratureRule const& rule,
                                       float* path_prob, QuadraturePathElement* path,
                                       std::size_t path_depth, std::size_t ncolumns,
                                       bst_node_t nidx, float const* c_vals, float w_prod,
                                       float tree_weight, float* out_h, float* matrix) {
  if (tree.IsLeaf(nidx)) {
    WriteWeightedLeafReturn(tree, rule, nidx, c_vals, w_prod, tree_weight, out_h);
    return;
  }

  auto left = tree.LeftChild(nidx);
  auto right = tree.RightChild(nidx);
  auto left_weight = ChildWeight(tree, nidx, left);
  auto right_weight = ChildWeight(tree, nidx, right);
  auto goes_left = EvaluateGoesLeft(tree, loader, row_idx, nidx);

  float right_h[kQuadratureTreeShapPoints];
  VisitInteractionChild(tree, loader, row_idx, rule, path_prob, path, path_depth, ncolumns, nidx,
                        left, left_weight, goes_left, c_vals, w_prod, tree_weight, out_h, matrix);
  VisitInteractionChild(tree, loader, row_idx, rule, path_prob, path, path_depth, ncolumns, nidx,
                        right, right_weight, !goes_left, c_vals, w_prod, tree_weight, right_h,
                        matrix);
  AddInPlace(out_h, right_h);
}

template <typename Loader>
void LaunchAdditiveKernel(Context const* ctx, Loader const& loader, bst_idx_t base_rowid,
                          DeviceModel const& d_model, QuadratureRule rule,
                          common::OptionalWeights tree_weights,
                          common::Span<float const> group_root_mean_sums,
                          linalg::VectorView<float const> base_score,
                          common::Span<float const> base_margin, float* path_prob, float* phis) {
  auto const n_rows = loader.NumRows();
  auto const n_groups = d_model.n_groups;
  auto const n_features = d_model.n_features;
  auto const ncolumns = static_cast<std::size_t>(n_features) + 1;
  auto const row_stride = static_cast<std::size_t>(n_groups) * ncolumns;
  auto d_trees = d_model.Trees();
  auto d_tree_groups = d_model.tree_groups;

  dh::LaunchN(n_rows * n_groups, ctx->CUDACtx()->Stream(), [=] __device__(std::size_t idx) {
    auto [local_row, gid] = linalg::UnravelIndex(idx, n_rows, n_groups);
    auto global_row = base_rowid + local_row;
    auto* phi = phis + static_cast<std::size_t>(global_row) * row_stride +
                static_cast<std::size_t>(gid) * ncolumns;
    auto* row_path_prob =
        path_prob + (static_cast<std::size_t>(global_row) * n_groups + gid) * n_features;
    for (bst_feature_t i = 0; i < n_features; ++i) {
      row_path_prob[i] = kQuadratureTreeShapUnseen;
    }

    for (bst_tree_t tree_idx = 0; tree_idx < static_cast<bst_tree_t>(d_trees.size()); ++tree_idx) {
      if (d_tree_groups[tree_idx] != gid) {
        continue;
      }
      auto const& tree = cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx]);
      float c_init[kQuadratureTreeShapPoints];
      float h_vals[kQuadratureTreeShapPoints];
      for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
        c_init[i] = 1.0f;
      }
      RunAdditiveNode(tree, loader, local_row, rule, row_path_prob, RegTree::kRoot, c_init, 1.0f,
                      tree_weights[tree_idx], h_vals, phi);
    }

    auto const bias =
        group_root_mean_sums[gid] +
        (base_margin.empty() ? base_score(gid)
                             : base_margin[static_cast<std::size_t>(global_row) * n_groups + gid]);
    phi[n_features] += bias;
  });
}

template <typename Loader>
void LaunchInteractionKernel(Context const* ctx, Loader const& loader, bst_idx_t base_rowid,
                             DeviceModel const& d_model, QuadratureRule rule,
                             common::OptionalWeights tree_weights,
                             common::Span<float const> group_root_mean_sums,
                             linalg::VectorView<float const> base_score,
                             common::Span<float const> base_margin, float* path_prob,
                             QuadraturePathElement* path, bst_node_t max_depth, float* phis) {
  auto const n_rows = loader.NumRows();
  auto const n_groups = d_model.n_groups;
  auto const n_features = d_model.n_features;
  auto const ncolumns = static_cast<std::size_t>(n_features) + 1;
  auto const matrix_size = ncolumns * ncolumns;
  auto d_trees = d_model.Trees();
  auto d_tree_groups = d_model.tree_groups;
  auto const path_stride = std::max<bst_node_t>(max_depth, 1);

  dh::LaunchN(n_rows * n_groups, ctx->CUDACtx()->Stream(), [=] __device__(std::size_t idx) {
    auto [local_row, gid] = linalg::UnravelIndex(idx, n_rows, n_groups);
    auto global_row = base_rowid + local_row;
    auto* matrix = phis + (static_cast<std::size_t>(global_row) * n_groups + gid) * matrix_size;
    auto* row_path_prob =
        path_prob + (static_cast<std::size_t>(global_row) * n_groups + gid) * n_features;
    auto* row_path = path + (static_cast<std::size_t>(global_row) * n_groups + gid) * path_stride;

    for (bst_feature_t i = 0; i < n_features; ++i) {
      row_path_prob[i] = kQuadratureTreeShapUnseen;
    }

    for (bst_tree_t tree_idx = 0; tree_idx < static_cast<bst_tree_t>(d_trees.size()); ++tree_idx) {
      if (d_tree_groups[tree_idx] != gid) {
        continue;
      }
      auto const& tree = cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx]);
      float c_init[kQuadratureTreeShapPoints];
      float h_vals[kQuadratureTreeShapPoints];
      for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
        c_init[i] = 1.0f;
      }
      RunInteractionNode(tree, loader, local_row, rule, row_path_prob, row_path, 0, ncolumns,
                         RegTree::kRoot, c_init, 1.0f, tree_weights[tree_idx], h_vals, matrix);
    }

    matrix[(ncolumns - 1) * ncolumns + (ncolumns - 1)] +=
        group_root_mean_sums[gid] +
        (base_margin.empty() ? base_score(gid)
                             : base_margin[static_cast<std::size_t>(global_row) * n_groups + gid]);

    for (std::size_t r = 0; r < ncolumns; ++r) {
      for (std::size_t c = r + 1; c < ncolumns; ++c) {
        auto const sym = 0.5f * (matrix[r * ncolumns + c] + matrix[c * ncolumns + r]);
        matrix[r * ncolumns + c] = sym;
        matrix[c * ncolumns + r] = sym;
      }
    }
    for (std::size_t r = 0; r < ncolumns; ++r) {
      float value = matrix[r * ncolumns + r];
      for (std::size_t c = 0; c < ncolumns; ++c) {
        if (c != r) {
          value -= matrix[r * ncolumns + c];
        }
      }
      matrix[r * ncolumns + r] = value;
    }
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

common::OptionalWeights MakeOptionalTreeWeights(Context const* ctx,
                                                std::vector<float> const* tree_weights,
                                                bst_tree_t tree_end,
                                                dh::device_vector<float>* d_tree_weights) {
  if (tree_weights == nullptr) {
    return common::OptionalWeights{1.0f};
  }
  d_tree_weights->assign(tree_weights->cbegin(), tree_weights->cbegin() + tree_end);
  return common::OptionalWeights{common::Span<float const>{
      thrust::raw_pointer_cast(d_tree_weights->data()), d_tree_weights->size()}};
}

void ConfigureQuadratureTreeShapStack(bst_node_t max_depth, bool has_categorical) {
  if (max_depth == 0) {
    return;
  }

  auto const per_level = has_categorical ? std::size_t{8 * 1024} : std::size_t{4 * 1024};
  auto const desired = std::size_t{64 * 1024} + static_cast<std::size_t>(max_depth) * per_level;
  std::size_t current{0};
  dh::safe_cuda(cudaDeviceGetLimit(&current, cudaLimitStackSize));
  if (current < desired) {
    dh::safe_cuda(cudaDeviceSetLimit(cudaLimitStackSize, desired));
  }
}
}  // namespace

void ShapValues(Context const* ctx, DMatrix* p_fmat, HostDeviceVector<float>* out_contribs,
                gbm::GBTreeModel const& model, bst_tree_t tree_end,
                std::vector<float> const* tree_weights, int, unsigned) {
  xgboost_NVTX_FN_RANGE();
  CHECK(!p_fmat->Info().IsColumnSplit())
      << "Predict contribution support for column-wise data split is not yet implemented.";
  dh::safe_cuda(cudaSetDevice(ctx->Ordinal()));
  out_contribs->SetDevice(ctx->Device());
  tree_end = predictor::GetTreeLimit(model.trees, tree_end);

  auto const n_groups = model.learner_model_param->num_output_group;
  CHECK_NE(n_groups, 0);
  auto const n_features = model.learner_model_param->num_feature;
  auto const ncolumns = static_cast<std::size_t>(n_features) + 1;
  auto const dim_size = ncolumns * n_groups;
  auto const n_samples = p_fmat->Info().num_row_;
  out_contribs->Resize(n_samples * dim_size);
  out_contribs->Fill(0.0f);
  auto phis = out_contribs->DeviceSpan();

  auto rule = MakeQuadratureRule();
  auto model_data = MakeGpuQuadratureTreeShapModelData(ctx, model, tree_end, tree_weights);
  ConfigureQuadratureTreeShapStack(model_data.max_depth, model.Cats()->HasCategorical());
  auto group_root_mean_sums = model_data.group_root_mean_sums.ConstDeviceSpan();

  DeviceModel d_model{ctx->Device(), model, true, 0, tree_end, CopyViews{ctx}};
  dh::device_vector<float> d_tree_weights;
  auto weights = MakeOptionalTreeWeights(ctx, tree_weights, tree_end, &d_tree_weights);
  dh::device_vector<float> path_prob(n_samples * n_groups * n_features);

  p_fmat->Info().base_margin_.SetDevice(ctx->Device());
  auto margin = p_fmat->Info().base_margin_.Data()->ConstDeviceSpan();
  auto base_score = model.learner_model_param->BaseScore(ctx);

  auto new_enc =
      p_fmat->Cats()->NeedRecode() ? p_fmat->Cats()->DeviceView(ctx) : enc::DeviceColumnsView{};
  LaunchShap(ctx, p_fmat, new_enc, model, [&](auto&& loader, bst_idx_t base_rowid) {
    LaunchAdditiveKernel(ctx, loader, base_rowid, d_model, rule, weights, group_root_mean_sums,
                         base_score, margin, thrust::raw_pointer_cast(path_prob.data()),
                         phis.data());
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
  CHECK(!p_fmat->Info().IsColumnSplit()) << "Predict interaction contribution support for "
                                            "column-wise data split is not yet implemented.";
  dh::safe_cuda(cudaSetDevice(ctx->Ordinal()));
  out_contribs->SetDevice(ctx->Device());
  tree_end = predictor::GetTreeLimit(model.trees, tree_end);

  auto const n_groups = model.learner_model_param->num_output_group;
  CHECK_NE(n_groups, 0);
  auto const n_features = model.learner_model_param->num_feature;
  auto const ncolumns = static_cast<std::size_t>(n_features) + 1;
  auto const matrix_size = ncolumns * ncolumns;
  auto const dim_size = matrix_size * n_groups;
  auto const n_samples = p_fmat->Info().num_row_;
  out_contribs->Resize(n_samples * dim_size);
  out_contribs->Fill(0.0f);
  auto phis = out_contribs->DeviceSpan();

  auto rule = MakeQuadratureRule();
  auto model_data = MakeGpuQuadratureTreeShapModelData(ctx, model, tree_end, tree_weights);
  ConfigureQuadratureTreeShapStack(model_data.max_depth, model.Cats()->HasCategorical());
  auto group_root_mean_sums = model_data.group_root_mean_sums.ConstDeviceSpan();

  DeviceModel d_model{ctx->Device(), model, true, 0, tree_end, CopyViews{ctx}};
  dh::device_vector<float> d_tree_weights;
  auto weights = MakeOptionalTreeWeights(ctx, tree_weights, tree_end, &d_tree_weights);
  dh::device_vector<float> path_prob(n_samples * n_groups * n_features);
  auto const path_stride = std::max<bst_node_t>(model_data.max_depth, 1);
  dh::device_vector<QuadraturePathElement> path(n_samples * n_groups * path_stride);

  p_fmat->Info().base_margin_.SetDevice(ctx->Device());
  auto margin = p_fmat->Info().base_margin_.Data()->ConstDeviceSpan();
  auto base_score = model.learner_model_param->BaseScore(ctx);

  auto new_enc =
      p_fmat->Cats()->NeedRecode() ? p_fmat->Cats()->DeviceView(ctx) : enc::DeviceColumnsView{};
  LaunchShap(ctx, p_fmat, new_enc, model, [&](auto&& loader, bst_idx_t base_rowid) {
    LaunchInteractionKernel(ctx, loader, base_rowid, d_model, rule, weights, group_root_mean_sums,
                            base_score, margin, thrust::raw_pointer_cast(path_prob.data()),
                            thrust::raw_pointer_cast(path.data()), path_stride, phis.data());
  });
}

void ApproxFeatureImportance(Context const*, DMatrix*, HostDeviceVector<float>*,
                             gbm::GBTreeModel const&, bst_tree_t, std::vector<float> const*) {
  StringView not_implemented{
      "contribution is not implemented in the GPU predictor, use CPU instead."};
  LOG(FATAL) << "Approximated " << not_implemented;
}
}  // namespace xgboost::interpretability::cuda_impl
