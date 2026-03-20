/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#include <GPUTreeShap/gpu_treeshap.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cmath>
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

constexpr std::size_t kMaxGpuQuadraturePoints = 16;
constexpr std::size_t kMaxGpuQuadratureDepth = 64;
constexpr double kQuadratureShapQeps = 1e-15;
constexpr double kQuadratureShapUnseen = -999.0;

struct QuadratureRule {
  std::size_t points{0};
  std::array<double, kMaxGpuQuadraturePoints> nodes{};
  std::array<double, kMaxGpuQuadraturePoints> weights{};
};

double LegendrePolynomial(std::size_t n, double x) {
  double p0 = 1.0;
  if (n == 0) {
    return p0;
  }
  double p1 = x;
  if (n == 1) {
    return p1;
  }
  for (std::size_t k = 2; k <= n; ++k) {
    auto kd = static_cast<double>(k);
    double pk = ((2.0 * kd - 1.0) * x * p1 - (kd - 1.0) * p0) / kd;
    p0 = p1;
    p1 = pk;
  }
  return p1;
}

double LegendreDerivative(std::size_t n, double x, double pn) {
  auto n_d = static_cast<double>(n);
  return n_d * (x * pn - LegendrePolynomial(n - 1, x)) / (x * x - 1.0);
}

QuadratureRule MakeEndpointQuadrature(std::size_t n) {
  CHECK_GE(n, 2);
  CHECK_LE(n, kMaxGpuQuadraturePoints) << "GPU QuadratureSHAP currently supports up to "
                                       << kMaxGpuQuadraturePoints << " quadrature points.";

  QuadratureRule rule;
  rule.points = n;
  std::vector<std::pair<double, double>> nodes_weights;
  nodes_weights.reserve(n);

  for (std::size_t i = 0; i < n; ++i) {
    double theta = M_PI * (static_cast<double>(i) + 0.75) / (static_cast<double>(n) + 0.5);
    double x = std::cos(theta);
    for (std::size_t iter = 0; iter < 64; ++iter) {
      auto pn = LegendrePolynomial(n, x);
      auto dpn = LegendreDerivative(n, x, pn);
      auto dx = pn / dpn;
      x -= dx;
      if (std::abs(dx) < kQuadratureShapQeps) {
        break;
      }
    }

    auto pn = LegendrePolynomial(n, x);
    auto dpn = LegendreDerivative(n, x, pn);
    auto w = 2.0 / ((1.0 - x * x) * dpn * dpn);
    double s = 0.5 * (x + 1.0);
    double ws = 0.5 * w;
    nodes_weights.emplace_back(s * s, 2.0 * s * ws);
  }

  std::sort(nodes_weights.begin(), nodes_weights.end(),
            [](auto const& l, auto const& r) { return l.first < r.first; });
  for (std::size_t i = 0; i < n; ++i) {
    rule.nodes[i] = nodes_weights[i].first;
    rule.weights[i] = nodes_weights[i].second;
  }
  return rule;
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

std::vector<float> MakeTreeRootMeanValues(gbm::GBTreeModel const& model, bst_tree_t tree_end,
                                          std::vector<float> const* tree_weights) {
  std::vector<float> mean_values(tree_end);
  for (bst_tree_t tree_idx = 0; tree_idx < tree_end; ++tree_idx) {
    auto weight = tree_weights == nullptr ? 1.0f : (*tree_weights)[tree_idx];
    auto const tree = model.trees.at(tree_idx)->HostScView();
    mean_values[tree_idx] = static_cast<float>(FillRootMeanValue(tree, RegTree::kRoot) * weight);
  }
  return mean_values;
}

template <int MaxPoints>
struct QuadratureFrame {
  bst_node_t node{RegTree::kInvalidNodeId};
  bst_feature_t split_index{0};
  int path_len{0};
  std::uint8_t stage{0};
  double w_prod{1.0};
  double child_p_enter[2]{};
  double child_p_up[2]{};
  bst_node_t child_node[2]{RegTree::kInvalidNodeId, RegTree::kInvalidNodeId};
  double child_weight[2]{};
  double c_vals[MaxPoints]{};
  double h_vals[MaxPoints]{};
};

template <int MaxPoints>
XGBOOST_DEVICE void CopyQuadratureValues(double (&dst)[MaxPoints], double const (&src)[MaxPoints],
                                         std::size_t points) {
  for (std::size_t i = 0; i < points; ++i) {
    dst[i] = src[i];
  }
}

template <int MaxPoints>
XGBOOST_DEVICE void AddQuadratureValues(double (&dst)[MaxPoints], double const (&src)[MaxPoints],
                                        std::size_t points) {
  for (std::size_t i = 0; i < points; ++i) {
    dst[i] += src[i];
  }
}

template <int MaxPoints>
XGBOOST_DEVICE double ExtractQuadratureDelta(std::size_t points, double const* nodes,
                                             double const* weights,
                                             double const (&h_vals)[MaxPoints], double p_enter,
                                             double p_exit) {
  auto alpha_enter = p_enter - 1.0;
  auto alpha_exit = p_exit - 1.0;
  auto has_enter = p_enter != kQuadratureShapUnseen && fabs(alpha_enter) >= kQuadratureShapQeps;
  auto has_exit = p_exit != kQuadratureShapUnseen && fabs(alpha_exit) >= kQuadratureShapQeps;
  if (!has_enter && !has_exit) {
    return 0.0;
  }

  double acc = 0.0;
  for (std::size_t i = 0; i < points; ++i) {
    auto weighted_h = h_vals[i] * weights[i];
    if (has_enter) {
      acc += alpha_enter * weighted_h / (1.0 + alpha_enter * nodes[i]);
    }
    if (has_exit) {
      acc -= alpha_exit * weighted_h / (1.0 + alpha_exit * nodes[i]);
    }
  }
  return acc;
}

template <int MaxPoints>
XGBOOST_DEVICE double FindPathProbability(
    int path_len, bst_feature_t const (&path_features)[kMaxGpuQuadratureDepth],
    double const (&path_p)[kMaxGpuQuadratureDepth], bst_feature_t split_index) {
  for (int i = path_len - 1; i >= 0; --i) {
    if (path_features[i] == split_index) {
      return path_p[i];
    }
  }
  return kQuadratureShapUnseen;
}

template <int MaxPoints, typename Loader>
XGBOOST_DEVICE void QuadratureShapTree(tree::ScalarTreeView const& tree, Loader const& loader,
                                       bst_idx_t ridx, std::size_t points, double const* nodes,
                                       double const* weights, float tree_weight, float* out_row) {
  auto const cats = tree.GetCategoriesMatrix();
  QuadratureFrame<MaxPoints> frames[kMaxGpuQuadratureDepth];
  bst_feature_t path_features[kMaxGpuQuadratureDepth];
  double path_p[kMaxGpuQuadratureDepth];
  double ret_h[MaxPoints]{};
  int stack_size = 1;
  bool have_return = false;

  frames[0].node = RegTree::kRoot;
  frames[0].path_len = 0;
  frames[0].stage = 0;
  frames[0].w_prod = 1.0;
  for (std::size_t i = 0; i < points; ++i) {
    frames[0].c_vals[i] = 1.0;
  }

  while (stack_size > 0 || have_return) {
    if (have_return) {
      if (stack_size == 0) {
        break;
      }
      auto& parent = frames[stack_size - 1];
      auto child_idx = parent.stage - 1;
      out_row[parent.split_index] += static_cast<float>(ExtractQuadratureDelta<MaxPoints>(
          points, nodes, weights, ret_h, parent.child_p_enter[child_idx],
          parent.child_p_up[child_idx]));
      if (child_idx == 0) {
        CopyQuadratureValues<MaxPoints>(parent.h_vals, ret_h, points);
        parent.stage = 2;
      } else {
        AddQuadratureValues<MaxPoints>(parent.h_vals, ret_h, points);
        CopyQuadratureValues<MaxPoints>(ret_h, parent.h_vals, points);
        --stack_size;
      }
      have_return = (child_idx == 1);
      continue;
    }

    auto& frame = frames[stack_size - 1];
    if (tree.IsLeaf(frame.node)) {
      auto leaf_value = static_cast<double>(tree.LeafValue(frame.node) * tree_weight);
      for (std::size_t i = 0; i < points; ++i) {
        ret_h[i] = frame.c_vals[i] * frame.w_prod * leaf_value;
      }
      --stack_size;
      have_return = true;
      continue;
    }

    if (frame.stage == 2) {
      auto child_slot = frame.path_len;
      if (child_slot >= static_cast<int>(kMaxGpuQuadratureDepth)) {
        return;
      }
      auto child = 1;
      auto child_node = frame.child_node[child];
      auto child_weight = frame.child_weight[child];
      auto p_old =
          FindPathProbability<MaxPoints>(frame.path_len, path_features, path_p, frame.split_index);
      double p_e = 0.0;
      double p_up = 0.0;
      auto satisfies = child_node == tree.RightChild(frame.node)
                           ? !(predictor::GetNextNode<true, true>(
                                   tree, frame.node, loader.GetElement(ridx, frame.split_index),
                                   common::CheckNAN(loader.GetElement(ridx, frame.split_index)),
                                   cats) == tree.LeftChild(frame.node))
                           : predictor::GetNextNode<true, true>(
                                 tree, frame.node, loader.GetElement(ridx, frame.split_index),
                                 common::CheckNAN(loader.GetElement(ridx, frame.split_index)),
                                 cats) == tree.LeftChild(frame.node);
      if (p_old == kQuadratureShapUnseen) {
        p_e = satisfies ? 1.0 / child_weight : 0.0;
        p_up = 1.0;
      } else if (fabs(p_old) < kQuadratureShapQeps) {
        p_e = 0.0;
        p_up = 0.0;
      } else {
        p_e = satisfies ? p_old / child_weight : 0.0;
        p_up = p_old;
      }

      path_features[child_slot] = frame.split_index;
      path_p[child_slot] = p_e;
      frame.child_p_enter[child] = p_e;
      frame.child_p_up[child] = p_up;

      auto& child_frame = frames[stack_size++];
      child_frame.node = child_node;
      child_frame.path_len = frame.path_len + 1;
      child_frame.stage = 0;
      child_frame.w_prod = frame.w_prod * child_weight;
      auto alpha_e = p_e - 1.0;
      auto alpha_old = p_old - 1.0;
      auto has_old = p_old != kQuadratureShapUnseen && fabs(alpha_old) >= kQuadratureShapQeps;
      for (std::size_t i = 0; i < points; ++i) {
        auto v = frame.c_vals[i] * (1.0 + alpha_e * nodes[i]);
        if (has_old) {
          v /= 1.0 + alpha_old * nodes[i];
        }
        child_frame.c_vals[i] = v;
      }
      continue;
    }

    auto split_index = tree.SplitIndex(frame.node);
    auto fvalue = loader.GetElement(ridx, split_index);
    auto is_missing = common::CheckNAN(fvalue);
    auto next = predictor::GetNextNode<true, true>(tree, frame.node, fvalue, is_missing, cats);
    auto left = tree.LeftChild(frame.node);
    auto right = tree.RightChild(frame.node);
    auto parent_cover = static_cast<double>(tree.SumHess(frame.node));
    if (!(parent_cover > 0.0)) {
      return;
    }

    frame.split_index = split_index;
    frame.child_node[0] = left;
    frame.child_node[1] = right;
    frame.child_weight[0] = static_cast<double>(tree.SumHess(left)) / parent_cover;
    frame.child_weight[1] = static_cast<double>(tree.SumHess(right)) / parent_cover;
    frame.stage = 1;

    auto child_slot = frame.path_len;
    if (child_slot >= static_cast<int>(kMaxGpuQuadratureDepth)) {
      return;
    }
    auto child = 0;
    auto child_node = frame.child_node[child];
    auto child_weight = frame.child_weight[child];
    auto p_old =
        FindPathProbability<MaxPoints>(frame.path_len, path_features, path_p, frame.split_index);
    double p_e = 0.0;
    double p_up = 0.0;
    auto satisfies = next == child_node;
    if (p_old == kQuadratureShapUnseen) {
      p_e = satisfies ? 1.0 / child_weight : 0.0;
      p_up = 1.0;
    } else if (fabs(p_old) < kQuadratureShapQeps) {
      p_e = 0.0;
      p_up = 0.0;
    } else {
      p_e = satisfies ? p_old / child_weight : 0.0;
      p_up = p_old;
    }

    path_features[child_slot] = frame.split_index;
    path_p[child_slot] = p_e;
    frame.child_p_enter[child] = p_e;
    frame.child_p_up[child] = p_up;

    auto& child_frame = frames[stack_size++];
    child_frame.node = child_node;
    child_frame.path_len = frame.path_len + 1;
    child_frame.stage = 0;
    child_frame.w_prod = frame.w_prod * child_weight;
    auto alpha_e = p_e - 1.0;
    auto alpha_old = p_old - 1.0;
    auto has_old = p_old != kQuadratureShapUnseen && fabs(alpha_old) >= kQuadratureShapQeps;
    for (std::size_t i = 0; i < points; ++i) {
      auto v = frame.c_vals[i] * (1.0 + alpha_e * nodes[i]);
      if (has_old) {
        v /= 1.0 + alpha_old * nodes[i];
      }
      child_frame.c_vals[i] = v;
    }
  }
}

template <int MaxPoints, typename Loader, typename ModelView>
void LaunchQuadratureShap(Context const* ctx, Loader loader, bst_idx_t base_rowid,
                          gbm::GBTreeModel const& model, ModelView const& d_model,
                          common::Span<float const> root_mean_values,
                          common::OptionalWeights tree_weights, std::size_t points,
                          common::Span<double const> nodes, common::Span<double const> weights,
                          HostDeviceVector<float>* out_contribs) {
  auto const ngroup = model.learner_model_param->num_output_group;
  auto const ncolumns = model.learner_model_param->num_feature + 1;
  auto d_trees = d_model.Trees();
  auto d_tree_groups = d_model.tree_groups;
  auto phis = out_contribs->DeviceSpan();

  dh::LaunchN(loader.NumRows(), ctx->CUDACtx()->Stream(), [=] __device__(std::size_t ridx) {
    auto row_idx = base_rowid + static_cast<bst_idx_t>(ridx);
    for (bst_tree_t tree_idx = 0; tree_idx < d_trees.size(); ++tree_idx) {
      auto const& d_tree = cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx]);
      auto gid = d_tree_groups[tree_idx];
      auto out_row = phis.data() + (row_idx * ngroup + gid) * ncolumns;
      out_row[ncolumns - 1] += root_mean_values[tree_idx];
      QuadratureShapTree<MaxPoints>(d_tree, loader, ridx, points, nodes.data(), weights.data(),
                                    tree_weights[tree_idx], out_row);
    }
  });
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
  CHECK_LE(quadrature_points, kMaxGpuQuadraturePoints)
      << "GPU QuadratureSHAP currently supports up to " << kMaxGpuQuadraturePoints
      << " quadrature points.";

  tree_end = predictor::GetTreeLimit(model.trees, tree_end);
  auto const ngroup = model.learner_model_param->num_output_group;
  CHECK_NE(ngroup, 0);
  auto const ncolumns = model.learner_model_param->num_feature + 1;
  auto dim_size = ncolumns * ngroup;
  out_contribs->SetDevice(ctx->Device());
  out_contribs->Resize(p_fmat->Info().num_row_ * dim_size);
  out_contribs->Fill(0.0f);

  bst_node_t max_depth = 0;
  for (bst_tree_t tree_idx = 0; tree_idx < tree_end; ++tree_idx) {
    CHECK(!model.trees[tree_idx]->IsMultiTarget()) << "Predict contribution" << MTNotImplemented();
    max_depth = std::max(max_depth, model.trees[tree_idx]->MaxDepth());
  }
  CHECK_LE(max_depth + 1, static_cast<bst_node_t>(kMaxGpuQuadratureDepth))
      << "GPU QuadratureSHAP currently supports trees of depth up to "
      << (kMaxGpuQuadratureDepth - 1) << ".";

  auto rule = MakeEndpointQuadrature(quadrature_points);
  dh::device_vector<double> d_nodes(rule.nodes.begin(), rule.nodes.begin() + quadrature_points);
  dh::device_vector<double> d_weights(rule.weights.begin(),
                                      rule.weights.begin() + quadrature_points);

  auto h_root_means = MakeTreeRootMeanValues(model, tree_end, tree_weights);
  dh::device_vector<float> d_root_means(h_root_means.cbegin(), h_root_means.cend());

  DeviceModel d_model{ctx->Device(), model, true, 0, tree_end, CopyViews{ctx}};
  dh::device_vector<float> d_tree_weights;
  auto weights_opt = common::OptionalWeights{1.0f};
  if (tree_weights != nullptr) {
    d_tree_weights.assign(tree_weights->cbegin(), tree_weights->cbegin() + tree_end);
    weights_opt = common::OptionalWeights{common::Span<float const>{
        thrust::raw_pointer_cast(d_tree_weights.data()), d_tree_weights.size()}};
  }

  auto new_enc =
      p_fmat->Cats()->NeedRecode() ? p_fmat->Cats()->DeviceView(ctx) : enc::DeviceColumnsView{};
  auto root_means =
      common::Span<float const>{thrust::raw_pointer_cast(d_root_means.data()), d_root_means.size()};
  auto nodes = common::Span<double const>{thrust::raw_pointer_cast(d_nodes.data()), d_nodes.size()};
  auto weights =
      common::Span<double const>{thrust::raw_pointer_cast(d_weights.data()), d_weights.size()};

  LaunchShap(ctx, p_fmat, new_enc, model, [&](auto&& loader, bst_idx_t base_rowid) {
    LaunchQuadratureShap<kMaxGpuQuadraturePoints>(ctx, loader, base_rowid, model, d_model,
                                                  root_means, weights_opt, quadrature_points, nodes,
                                                  weights, out_contribs);
  });

  p_fmat->Info().base_margin_.SetDevice(ctx->Device());
  auto margin = p_fmat->Info().base_margin_.Data()->ConstDeviceSpan();
  auto base_score = model.learner_model_param->BaseScore(ctx);
  auto phis = out_contribs->DeviceSpan();
  auto n_samples = p_fmat->Info().num_row_;
  dh::LaunchN(n_samples * ngroup, ctx->CUDACtx()->Stream(), [=] __device__(std::size_t idx) {
    auto [_, gid] = linalg::UnravelIndex(idx, n_samples, ngroup);
    phis[(idx + 1) * ncolumns - 1] += margin.empty() ? base_score(gid) : margin[idx];
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
