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
#include "../../common/error_msg.h"
#include "../../common/nvtx_utils.h"
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
                  dh::device_vector<uint32_t>* path_categories) {
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

    float v = tree.LeafValue(child_nidx);
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
  if (tree_weights != nullptr) {
    LOG(FATAL) << "Dart booster feature " << not_implemented;
  }
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

  auto new_enc =
      p_fmat->Cats()->NeedRecode() ? p_fmat->Cats()->DeviceView(ctx) : enc::DeviceColumnsView{};

  dh::device_vector<uint32_t> categories;
  ExtractPaths(ctx, &device_paths, model, d_model, &categories);

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

void ShapInteractionValues(Context const* ctx, DMatrix* p_fmat,
                           HostDeviceVector<float>* out_contribs, gbm::GBTreeModel const& model,
                           bst_tree_t tree_end, std::vector<float> const* tree_weights,
                           bool approximate) {
  xgboost_NVTX_FN_RANGE();
  std::string not_implemented{"contribution is not implemented in GPU predictor, use cpu instead."};
  if (approximate) {
    LOG(FATAL) << "Approximated " << not_implemented;
  }
  if (tree_weights != nullptr) {
    LOG(FATAL) << "Dart booster feature " << not_implemented;
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

  dh::device_vector<uint32_t> categories;
  ExtractPaths(ctx, &device_paths, model, d_model, &categories);
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
