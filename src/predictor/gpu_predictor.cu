/**
 * Copyright 2017-2025, XGBoost Contributors
 */
#include <GPUTreeShap/gpu_treeshap.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include <cuda/functional>   // for proclaim_return_type
#include <cuda/std/utility>  // for swap
#include <memory>

#include "../collective/allreduce.h"
#include "../common/bitfield.h"
#include "../common/categorical.h"
#include "../common/common.h"
#include "../common/cuda_context.cuh"  // for CUDAContext
#include "../common/cuda_rt_utils.h"   // for AllVisibleGPUs, SetDevice
#include "../common/device_helpers.cuh"
#include "../common/error_msg.h"      // for InplacePredictProxy
#include "../common/nvtx_utils.h"     // for xgboost_NVTX_FN_RANGE
#include "../data/batch_utils.h"      // for StaticBatch
#include "../data/cat_container.cuh"  // for EncPolicy
#include "../data/device_adapter.cuh"
#include "../data/ellpack_page.cuh"
#include "../data/proxy_dmatrix.cuh"  // for DispatchAny
#include "../data/proxy_dmatrix.h"
#include "../gbm/gbtree_model.h"
#include "../tree/tree_view.h"
#include "gbtree_view.h"  // for GBTreeModelView
#include "predict_fn.h"
#include "utils.h"  // for CheckProxyDMatrix
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/multi_target_tree_model.h"  // for MultiTargetTree, MultiTargetTreeView
#include "xgboost/predictor.h"
#include "xgboost/tree_model.h"
#include "xgboost/tree_updater.h"

namespace xgboost::predictor {
DMLC_REGISTRY_FILE_TAG(gpu_predictor);

using cuda_impl::StaticBatch;

struct SparsePageView {
  common::Span<const Entry> d_data;
  common::Span<const bst_idx_t> d_row_ptr;
  bst_feature_t num_features;

  SparsePageView() = default;
  explicit SparsePageView(Context const* ctx, SparsePage const& page, bst_feature_t n_features)
      : d_data{[&] {
          page.data.SetDevice(ctx->Device());
          return page.data.ConstDeviceSpan();
        }()},
        d_row_ptr{[&] {
          page.offset.SetDevice(ctx->Device());
          return page.offset.ConstDeviceSpan();
        }()},
        num_features{n_features} {}

  [[nodiscard]] __device__ float GetElement(size_t ridx, size_t fidx) const {
    // Binary search
    auto begin_ptr = d_data.begin() + d_row_ptr[ridx];
    auto end_ptr = d_data.begin() + d_row_ptr[ridx + 1];
    if (end_ptr - begin_ptr == this->NumCols()) {
      // Bypass span check for dense data
      return d_data.data()[d_row_ptr[ridx] + fidx].fvalue;
    }
    common::Span<const Entry>::iterator previous_middle;
    while (end_ptr != begin_ptr) {
      auto middle = begin_ptr + (end_ptr - begin_ptr) / 2;
      if (middle == previous_middle) {
        break;
      } else {
        previous_middle = middle;
      }

      if (middle->index == fidx) {
        return middle->fvalue;
      } else if (middle->index < fidx) {
        begin_ptr = middle;
      } else {
        end_ptr = middle;
      }
    }
    // Value is missing
    return std::numeric_limits<float>::quiet_NaN();
  }
  [[nodiscard]] XGBOOST_DEVICE size_t NumRows() const { return d_row_ptr.size() - 1; }
  [[nodiscard]] XGBOOST_DEVICE size_t NumCols() const { return num_features; }
};

template <typename EncAccessor>
struct SparsePageLoader {
 public:
  using SupportShmemLoad = std::true_type;

 private:
  EncAccessor acc_;

 public:
  bool use_shared;
  SparsePageView data;
  float* smem;

  __device__ SparsePageLoader(SparsePageView data, bool use_shared, bst_feature_t num_features,
                              bst_idx_t num_rows, float, EncAccessor&& acc)
      : use_shared(use_shared), data(data), acc_{std::forward<EncAccessor>(acc)} {
    extern __shared__ float _smem[];
    smem = _smem;
    // Copy instances
    if (use_shared) {
      bst_uint global_idx = blockDim.x * blockIdx.x + threadIdx.x;
      int shared_elements = blockDim.x * data.num_features;
      dh::BlockFill(smem, shared_elements, std::numeric_limits<float>::quiet_NaN());
      __syncthreads();
      if (global_idx < num_rows) {
        bst_uint elem_begin = data.d_row_ptr[global_idx];
        bst_uint elem_end = data.d_row_ptr[global_idx + 1];
        for (bst_uint elem_idx = elem_begin; elem_idx < elem_end; elem_idx++) {
          Entry elem = data.d_data[elem_idx];
          smem[threadIdx.x * data.num_features + elem.index] = this->acc_(elem);
        }
      }
      __syncthreads();
    }
  }
  [[nodiscard]] __device__ float GetElement(size_t ridx, size_t fidx) const {
    if (use_shared) {
      return smem[threadIdx.x * data.num_features + fidx];
    } else {
      return this->acc_(data.GetElement(ridx, fidx), fidx);
    }
  }
};

template <typename Accessor, typename EncAccessor>
struct EllpackLoader {
 public:
  using SupportShmemLoad = std::false_type;

  Accessor matrix;
  EncAccessor acc;

  XGBOOST_DEVICE EllpackLoader(Accessor m, bool /*use_shared*/, bst_feature_t /*n_features*/,
                               bst_idx_t /*n_samples*/, float /*missing*/, EncAccessor&& acc)
      : matrix{std::move(m)}, acc{std::forward<EncAccessor>(acc)} {}
  [[nodiscard]] XGBOOST_DEV_INLINE float GetElement(size_t ridx, size_t fidx) const {
    auto gidx = matrix.template GetBinIndex<false>(ridx, fidx);
    if (gidx == -1) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    if (common::IsCat(matrix.feature_types, fidx)) {
      return this->acc(matrix.gidx_fvalue_map[gidx], fidx);
    }
    // The gradient index needs to be shifted by one as min values are not included in the
    // cuts.
    if (gidx == matrix.feature_segments[fidx]) {
      return matrix.min_fvalue[fidx];
    }
    return matrix.gidx_fvalue_map[gidx - 1];
  }
  [[nodiscard]] XGBOOST_DEVICE bst_idx_t NumCols() const { return this->matrix.NumFeatures(); }
  [[nodiscard]] XGBOOST_DEVICE bst_idx_t NumRows() const { return this->matrix.n_rows; }
};

/**
 * @brief Use for in-place predict.
 */
template <typename Batch, typename EncAccessor>
struct DeviceAdapterLoader {
 public:
  using SupportShmemLoad = std::true_type;

 private:
  Batch batch_;
  EncAccessor acc_;

 public:
  bst_feature_t n_features;
  float* smem;
  bool use_shared;
  data::IsValidFunctor is_valid;

  XGBOOST_DEV_INLINE DeviceAdapterLoader(Batch&& batch, bool use_shared, bst_feature_t n_features,
                                         bst_idx_t n_samples, float missing, EncAccessor&& acc)
      : batch_{std::move(batch)},
        acc_{std::forward<EncAccessor>(acc)},
        n_features{n_features},
        use_shared{use_shared},
        is_valid{missing} {
    extern __shared__ float _smem[];
    this->smem = _smem;
    if (this->use_shared) {
      auto global_idx = blockDim.x * blockIdx.x + threadIdx.x;
      size_t shared_elements = blockDim.x * n_features;
      dh::BlockFill(smem, shared_elements, std::numeric_limits<float>::quiet_NaN());
      __syncthreads();
      if (global_idx < n_samples) {
        auto beg = global_idx * n_features;
        auto end = (global_idx + 1) * n_features;
        for (size_t i = beg; i < end; ++i) {
          data::COOTuple const& e = this->batch_.GetElement(i);
          if (is_valid(e)) {
            smem[threadIdx.x * n_features + (i - beg)] = this->acc_(e);
          }
        }
      }
    }
    __syncthreads();
  }

  [[nodiscard]] XGBOOST_DEV_INLINE float GetElement(size_t ridx, size_t fidx) const {
    if (use_shared) {
      return smem[threadIdx.x * n_features + fidx];
    }
    auto value = this->batch_.GetElement(ridx * n_features + fidx).value;
    if (is_valid(value)) {
      return this->acc_(value, fidx);
    } else {
      return std::numeric_limits<float>::quiet_NaN();
    }
  }
};

namespace {
template <bool has_missing, bool has_categorical, typename TreeView, typename Loader>
__device__ bst_node_t GetLeafIndex(bst_idx_t ridx, TreeView const& tree, Loader* loader) {
  bst_node_t nidx = 0;
  while (!tree.IsLeaf(nidx)) {
    float fvalue = loader->GetElement(ridx, tree.SplitIndex(nidx));
    bool is_missing = has_missing && common::CheckNAN(fvalue);
    auto next = GetNextNode<has_missing, has_categorical>(tree, nidx, fvalue, is_missing,
                                                          tree.GetCategoriesMatrix());
    assert(nidx < next);
    nidx = next;
  }
  return nidx;
}

template <bool has_missing, typename TreeView, typename Loader>
__device__ auto GetLeafWeight(bst_idx_t ridx, TreeView const& tree, Loader* loader) {
  bst_node_t nidx = -1;
  if (tree.HasCategoricalSplit()) {
    nidx = GetLeafIndex<has_missing, true>(ridx, tree, loader);
  } else {
    nidx = GetLeafIndex<has_missing, false>(ridx, tree, loader);
  }
  return tree.LeafValue(nidx);
}
}  // namespace

using TreeViewVar = cuda::std::variant<tree::ScalarTreeView, tree::MultiTargetTreeView>;

template <typename Loader, typename Data, bool has_missing, typename EncAccessor>
__global__ void PredictLeafKernel(Data data, common::Span<TreeViewVar const> d_trees,
                                  common::Span<float> d_out_predictions, bst_tree_t tree_begin,
                                  bst_tree_t tree_end, bst_feature_t num_features, bool use_shared,
                                  float missing, EncAccessor acc) {
  bst_idx_t ridx = blockDim.x * blockIdx.x + threadIdx.x;
  if (ridx >= data.NumRows()) {
    return;
  }
  Loader loader{std::move(data), use_shared, num_features, data.NumRows(), missing, std::move(acc)};
  for (bst_tree_t tree_idx = tree_begin; tree_idx < tree_end; ++tree_idx) {
    auto const& d_tree = d_trees[tree_idx - tree_begin];
    cuda::std::visit(
        [&](auto&& tree) {
          bst_node_t leaf = -1;
          if (tree.HasCategoricalSplit()) {
            leaf = GetLeafIndex<has_missing, true>(ridx, tree, &loader);
          } else {
            leaf = GetLeafIndex<has_missing, false>(ridx, tree, &loader);
          }
          d_out_predictions[ridx * (tree_end - tree_begin) + tree_idx] = leaf;
        },
        d_tree);
  }
}

template <typename Loader, typename Data, bool has_missing, typename EncAccessor>
__global__ void PredictKernel(Data data, common::Span<TreeViewVar const> d_trees,
                              common::Span<float> d_out_predictions,
                              common::Span<bst_target_t const> d_tree_groups,
                              bst_feature_t num_features, bool use_shared, bst_target_t n_groups,
                              float missing, EncAccessor acc) {
  bst_idx_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  Loader loader{std::move(data), use_shared, num_features, data.NumRows(), missing, std::move(acc)};
  if (global_idx >= data.NumRows()) {
    return;
  }

  if (n_groups == 1u) {
    float sum = 0;
    for (auto const& d_tree : d_trees) {
      auto const& sc_tree = cuda::std::get<tree::ScalarTreeView>(d_tree);
      float leaf = GetLeafWeight<has_missing>(global_idx, sc_tree, &loader);
      sum += leaf;
    }
    d_out_predictions[global_idx] += sum;
  } else {
    for (bst_tree_t tree_idx = 0, k = d_trees.size(); tree_idx < k; tree_idx++) {
      // Both d_tree_group and d_tress are subset of trees.
      auto tree_group = d_tree_groups[tree_idx];
      auto const& d_tree = d_trees[tree_idx];
      cuda::std::visit(
          enc::Overloaded{[&](tree::ScalarTreeView const& tree) {
                            auto leaf = GetLeafWeight<has_missing>(global_idx, tree, &loader);
                            bst_idx_t out_prediction_idx = global_idx * n_groups + tree_group;
                            d_out_predictions[out_prediction_idx] += leaf;
                          },
                          [&](tree::MultiTargetTreeView const& tree) {
                            // Tree group is 0.
                            auto leaf = GetLeafWeight<has_missing>(global_idx, tree, &loader);
                            for (std::size_t i = 0, n = leaf.Shape(0); i < n; ++i) {
                              bst_idx_t out_prediction_idx = global_idx * n_groups + i;
                              d_out_predictions[out_prediction_idx] += leaf(i);
                            }
                          }},
          d_tree);
    }
  }
}

namespace {
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
}  // namespace

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

  /*! Feature values >= lower and < upper flow down this path. */
  float feature_lower_bound;
  float feature_upper_bound;
  /*! Feature value set to true flow down this path. */
  common::CatBitField categories;
  /*! Do missing values flow down this path? */
  bool is_missing_branch;

  // Does this instance flow down this path?
  [[nodiscard]] XGBOOST_DEVICE bool EvaluateSplit(float x) const {
    // is nan
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

  // the &= op in bitfiled is per cuda thread, this one loops over the entire
  // bitfield.
  XGBOOST_DEVICE static common::CatBitField Intersect(common::CatBitField l,
                                                      common::CatBitField r) {
    if (l.Data() == r.Data()) {
      return l;
    }
    if (l.Capacity() > r.Capacity()) {
      cuda::std::swap(l, r);
    }
    for (size_t i = 0; i < r.Bits().size(); ++i) {
      l.Bits()[i] &= r.Bits()[i];
    }
    return l;
  }

  // Combine two split conditions on the same feature
  XGBOOST_DEVICE void Merge(ShapSplitCondition other) {
    // Combine duplicate features
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
  // Node index in tree.
  // -1 if not a leaf (internal split node)
  bst_node_t nidx;
  bst_tree_t tree_idx;

  [[nodiscard]] XGBOOST_DEVICE bool IsLeaf() const { return nidx != -1; }
};
static_assert(sizeof(PathInfo) == 16);

auto MakeTreeSegments(Context const* ctx, bst_tree_t tree_begin, bst_tree_t tree_end,
                      gbm::GBTreeModel const& model) {
  // Copy decision trees to device
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

// Transform model into path element form for GPUTreeShap
void ExtractPaths(Context const* ctx,
                  dh::device_vector<gpu_treeshap::PathElement<ShapSplitCondition>>* paths,
                  gbm::GBTreeModel const& h_model, DeviceModel const& d_model,
                  dh::device_vector<uint32_t>* path_categories) {
  curt::SetDevice(ctx->Ordinal());

  // Path length and tree index for all leaf nodes
  dh::caching_device_vector<PathInfo> info(d_model.n_nodes);
  auto d_trees = d_model.Trees();  // subset of trees
  auto tree_segments = MakeTreeSegments(ctx, d_model.tree_begin, d_model.tree_end, h_model);
  CHECK_EQ(tree_segments.ConstHostVector().back(), d_model.n_nodes);
  auto d_tree_segments = tree_segments.ConstDeviceSpan();

  auto path_it = dh::MakeIndexTransformIter(
      cuda::proclaim_return_type<PathInfo>([=] __device__(size_t idx) -> PathInfo {
        bst_tree_t const tree_idx = dh::SegmentId(d_tree_segments, idx);
        bst_node_t const nidx = idx - d_tree_segments[tree_idx];
        auto const& tree = cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx]);
        if (!tree.IsLeaf(nidx) || tree.IsDeleted(nidx)) {
          // -1 if it's an internal split node
          return PathInfo{0, -1, 0};
        }
        // Get the path length for leaf
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
  thrust::exclusive_scan(ctx->CUDACtx()->CTP(), length_iterator, length_iterator + info.size() + 1,
                         path_segments.begin());

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
    // Root node has feature -1
    d_paths[output_position] = {idx, -1, group, ShapSplitCondition{-inf, inf, false, {}}, 1.0, v};
  });
}

namespace {
template <std::size_t kBlockThreads>
[[nodiscard]] std::size_t SharedMemoryBytes(std::size_t n_features, std::size_t max_shmem_bytes) {
  CHECK_GT(max_shmem_bytes, 0);
  size_t shared_memory_bytes = static_cast<size_t>(sizeof(float) * n_features * kBlockThreads);
  if (shared_memory_bytes > max_shmem_bytes) {
    shared_memory_bytes = 0;
  }
  return shared_memory_bytes;
}

using BitVector = LBitField64;

__global__ void MaskBitVectorKernel(SparsePageView data, common::Span<TreeViewVar const> d_trees,
                                    BitVector decision_bits, BitVector missing_bits,
                                    bst_tree_t tree_begin, bst_tree_t tree_end,
                                    bst_feature_t num_features, std::size_t num_nodes,
                                    bool use_shared, float missing) {
  // This needs to be always instantiated since the data is loaded cooperatively by all threads.
  SparsePageLoader loader{data, use_shared, num_features, data.NumRows(), missing, NoOpAccessor{}};
  auto const row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_idx >= data.NumRows()) {
    return;
  }

  std::size_t tree_offset = 0;
  for (auto tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
    auto const& d_tree = cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx - tree_begin]);
    auto const tree_nodes = d_tree.Size();
    for (auto nid = 0; nid < tree_nodes; nid++) {
      if (d_tree.IsDeleted(nid) || d_tree.IsLeaf(nid)) {
        continue;
      }
      auto const fvalue = loader.GetElement(row_idx, d_tree.SplitIndex(nid));
      auto const is_missing = common::CheckNAN(fvalue);
      auto const bit_index = row_idx * num_nodes + tree_offset + nid;
      if (is_missing) {
        missing_bits.Set(bit_index);
      } else {
        auto const decision =
            d_tree.HasCategoricalSplit()
                ? GetDecision<true>(d_tree, nid, fvalue, d_tree.GetCategoriesMatrix())
                : GetDecision<false>(d_tree, nid, fvalue, d_tree.GetCategoriesMatrix());
        if (decision) {
          decision_bits.Set(bit_index);
        }
      }
    }
    tree_offset += tree_nodes;
  }
}

template <typename TreeView>
__device__ bst_node_t GetLeafIndexByBitVector(bst_idx_t ridx, TreeView const& tree,
                                              BitVector const& decision_bits,
                                              BitVector const& missing_bits, std::size_t num_nodes,
                                              std::size_t tree_offset) {
  bst_node_t nidx = 0;
  while (!tree.IsLeaf(nidx)) {
    auto const bit_index = ridx * num_nodes + tree_offset + nidx;
    if (missing_bits.Check(bit_index)) {
      nidx = tree.DefaultChild(nidx);
    } else {
      nidx = tree.LeftChild(nidx) + !decision_bits.Check(bit_index);
    }
  }
  return nidx;
}

template <typename TreeView>
__device__ float GetLeafWeightByBitVector(bst_idx_t ridx, TreeView const& tree,
                                          BitVector const& decision_bits,
                                          BitVector const& missing_bits, std::size_t num_nodes,
                                          std::size_t tree_offset) {
  auto const nidx =
      GetLeafIndexByBitVector(ridx, tree, decision_bits, missing_bits, num_nodes, tree_offset);
  return tree.LeafValue(nidx);
}

template <bool predict_leaf>
__global__ void PredictByBitVectorKernel(common::Span<TreeViewVar const> d_trees,
                                         common::Span<float> d_out_predictions,
                                         common::Span<bst_target_t const> d_tree_groups,
                                         BitVector decision_bits, BitVector missing_bits,
                                         bst_tree_t tree_begin, bst_tree_t tree_end,
                                         std::size_t num_rows, std::size_t num_nodes,
                                         std::uint32_t num_group) {
  auto const row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_idx >= num_rows) {
    return;
  }

  std::size_t tree_offset = 0;
  if constexpr (predict_leaf) {
    for (auto tree_idx = tree_begin; tree_idx < tree_end; ++tree_idx) {
      auto const& d_tree = cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx - tree_begin]);
      auto const leaf = GetLeafIndexByBitVector(row_idx, d_tree, decision_bits, missing_bits,
                                                num_nodes, tree_offset);
      d_out_predictions[row_idx * (tree_end - tree_begin) + tree_idx] = static_cast<float>(leaf);
      tree_offset += d_tree.Size();
    }
  } else {
    if (num_group == 1) {
      float sum = 0;
      for (auto tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
        auto const& d_tree = cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx - tree_begin]);
        sum += GetLeafWeightByBitVector(row_idx, d_tree, decision_bits, missing_bits, num_nodes,
                                        tree_offset);
        tree_offset += d_tree.Size();
      }
      d_out_predictions[row_idx] += sum;
    } else {
      for (auto tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
        auto const tree_group = d_tree_groups[tree_idx - tree_begin];
        auto const& d_tree = cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx - tree_begin]);
        bst_uint out_prediction_idx = row_idx * num_group + tree_group;
        d_out_predictions[out_prediction_idx] += GetLeafWeightByBitVector(
            row_idx, d_tree, decision_bits, missing_bits, num_nodes, tree_offset);
        tree_offset += d_tree.Size();
      }
    }
  }
}

class ColumnSplitHelper {
 public:
  explicit ColumnSplitHelper(Context const* ctx) : ctx_{ctx} {}

  void PredictBatch(DMatrix* dmat, HostDeviceVector<float>* out_preds,
                    gbm::GBTreeModel const& model, DeviceModel const& d_model) const {
    CHECK(dmat->PageExists<SparsePage>()) << "Column split for external memory is not support.";
    PredictDMatrix<false>(dmat, out_preds, d_model, model.learner_model_param->num_feature,
                          model.learner_model_param->num_output_group);
  }

  void PredictLeaf(DMatrix* dmat, HostDeviceVector<float>* out_preds, gbm::GBTreeModel const& model,
                   DeviceModel const& d_model) const {
    CHECK(dmat->PageExists<SparsePage>()) << "Column split for external memory is not support.";
    PredictDMatrix<true>(dmat, out_preds, d_model, model.learner_model_param->num_feature,
                         model.learner_model_param->num_output_group);
  }

 private:
  using BitType = BitVector::value_type;

  template <bool predict_leaf>
  void PredictDMatrix(DMatrix* dmat, HostDeviceVector<float>* out_preds, DeviceModel const& d_model,
                      bst_feature_t num_features, std::uint32_t num_group) const {
    dh::safe_cuda(cudaSetDevice(ctx_->Ordinal()));
    dh::caching_device_vector<BitType> decision_storage{};
    dh::caching_device_vector<BitType> missing_storage{};

    auto constexpr kBlockThreads = 128;
    auto const max_shared_memory_bytes = dh::MaxSharedMemory(ctx_->Ordinal());
    auto const shared_memory_bytes =
        SharedMemoryBytes<kBlockThreads>(num_features, max_shared_memory_bytes);
    auto const use_shared = shared_memory_bytes != 0;

    auto const num_nodes = d_model.n_nodes;
    std::size_t batch_offset = 0;
    for (auto const& batch : dmat->GetBatches<SparsePage>()) {
      auto const num_rows = batch.Size();
      ResizeBitVectors(&decision_storage, &missing_storage, num_rows * num_nodes);
      BitVector decision_bits{dh::ToSpan(decision_storage)};
      BitVector missing_bits{dh::ToSpan(missing_storage)};

      SparsePageView data{ctx_, batch, num_features};
      auto const grid = static_cast<uint32_t>(common::DivRoundUp(num_rows, kBlockThreads));
      auto d_tree_groups = d_model.tree_groups;
      dh::LaunchKernel {grid, kBlockThreads, shared_memory_bytes, ctx_->CUDACtx()->Stream()}(
          MaskBitVectorKernel, data, d_model.Trees(), decision_bits, missing_bits,
          d_model.tree_begin, d_model.tree_end, num_features, num_nodes, use_shared,
          std::numeric_limits<float>::quiet_NaN());

      AllReduceBitVectors(&decision_storage, &missing_storage);

      dh::LaunchKernel {grid, kBlockThreads, 0, ctx_->CUDACtx()->Stream()}(
          PredictByBitVectorKernel<predict_leaf>, d_model.Trees(),
          out_preds->DeviceSpan().subspan(batch_offset), d_tree_groups,
          decision_bits, missing_bits, d_model.tree_begin, d_model.tree_end, num_rows, num_nodes,
          num_group);

      batch_offset += batch.Size() * num_group;
    }
  }

  void AllReduceBitVectors(dh::caching_device_vector<BitType>* decision_storage,
                           dh::caching_device_vector<BitType>* missing_storage) const {
    auto rc = collective::Success() << [&] {
      return collective::Allreduce(
          ctx_,
          linalg::MakeVec(decision_storage->data().get(), decision_storage->size(), ctx_->Device()),
          collective::Op::kBitwiseOR);
    } << [&] {
      return collective::Allreduce(
          ctx_,
          linalg::MakeVec(missing_storage->data().get(), missing_storage->size(), ctx_->Device()),
          collective::Op::kBitwiseAND);
    };
    collective::SafeColl(rc);
  }

  void ResizeBitVectors(dh::caching_device_vector<BitType>* decision_storage,
                        dh::caching_device_vector<BitType>* missing_storage,
                        std::size_t total_bits) const {
    auto const size = BitVector::ComputeStorageSize(total_bits);
    if (decision_storage->size() < size) {
      decision_storage->resize(size);
    }
    thrust::fill(ctx_->CUDACtx()->CTP(), decision_storage->begin(), decision_storage->end(), 0);
    if (missing_storage->size() < size) {
      missing_storage->resize(size);
    }
    thrust::fill(ctx_->CUDACtx()->CTP(), missing_storage->begin(), missing_storage->end(), 0);
  }

  Context const* ctx_;
};

using cuda_impl::MakeCatAccessor;

template <typename EncAccessor>
struct ShapSparsePageLoader {
 public:
  using SupportShmemLoad = std::false_type;

  SparsePageView data;
  EncAccessor acc;

  template <typename Fidx>
  [[nodiscard]] __device__ float GetElement(bst_idx_t ridx, Fidx fidx) const {
    auto fvalue = data.GetElement(ridx, fidx);
    return acc(fvalue, fidx);
  }
  [[nodiscard]] XGBOOST_DEVICE bst_idx_t NumRows() const { return data.NumRows(); }
  [[nodiscard]] XGBOOST_DEVICE bst_idx_t NumCols() const { return data.NumCols(); }
};

// Provide configuration for launching the predict kernel.
template <typename IsDense, typename EncAccessor>
class LaunchConfig {
 public:
  static constexpr bool HasMissing() { return !IsDense::value; }
  using EncAccessorT = EncAccessor;

  template <typename T, std::uint32_t block_threads>
  struct LoaderType {
    using Type = T;
    constexpr static std::uint32_t kBlockThreads = block_threads;

    static std::size_t AllocShmem(Context const* ctx, bst_feature_t n_features) {
      if constexpr (typename Type::SupportShmemLoad{}) {
        return SharedMemoryBytes<kBlockThreads>(n_features, ConfigureDevice(ctx->Device()));
      }
      return 0;
    }
  };

 private:
  static auto constexpr NotSet() { return std::numeric_limits<bst_idx_t>::max(); }

  Context const* ctx_;
  bst_feature_t n_features_;
  std::size_t shared_memory_bytes_{0};

 public:
  template <typename Loader, typename K, typename BatchT, typename... Args>
  void Launch(K&& kernel, BatchT&& batch, Args&&... args) const {
    auto grid = static_cast<uint32_t>(common::DivRoundUp(batch.NumRows(), Loader::kBlockThreads));
    dh::LaunchKernel{grid, Loader::kBlockThreads, this->shared_memory_bytes_,  // NOLINT
                     this->ctx_->CUDACtx()->Stream()}(kernel, std::forward<BatchT>(batch),
                                                      std::forward<Args>(args)...);
  }
  template <typename Loader, typename Data>
  void LaunchPredictKernel(Data batch, float missing, bst_feature_t n_features,
                           DeviceModel const& d_model, EncAccessorT acc, bst_idx_t batch_offset,
                           HostDeviceVector<float>* predictions) {
    auto kernel = PredictKernel<typename Loader::Type, common::GetValueT<decltype(batch)>,
                                HasMissing(), EncAccessorT>;
    auto d_tree_groups = d_model.tree_groups;
    this->Launch<Loader>(kernel, std::move(batch), d_model.Trees(),
                         predictions->DeviceSpan().subspan(batch_offset), d_tree_groups, n_features,
                         this->UseShared(), d_model.n_groups, missing, acc);
  }

  [[nodiscard]] bool UseShared() const { return shared_memory_bytes_ != 0; }

  [[nodiscard]] static std::size_t ConfigureDevice(DeviceOrd const& device) {
    thread_local std::unordered_map<std::int32_t, std::size_t> max_shared;
    auto it = max_shared.find(device.ordinal);
    if (it == max_shared.cend()) {
      max_shared[device.ordinal] = dh::MaxSharedMemory(device.ordinal);
      it = max_shared.find(device.ordinal);
    }
    return it->second;
  }

  template <typename Loader>
  void AllocShmem() {
    this->shared_memory_bytes_ = Loader::AllocShmem(this->ctx_, this->n_features_);
  }

 public:
  LaunchConfig(Context const* ctx, bst_feature_t n_features)
      : ctx_{ctx}, n_features_{n_features} {}

  template <typename Fn>
  void ForEachBatch(DMatrix* p_fmat, Fn&& fn) {
    if (p_fmat->PageExists<SparsePage>()) {
      constexpr std::uint32_t kBlockThreads = 128;
      using LoaderImpl = SparsePageLoader<EncAccessor>;
      using Loader = LoaderType<LoaderImpl, kBlockThreads>;
      this->AllocShmem<Loader>();
      for (auto& page : p_fmat->GetBatches<SparsePage>()) {
        SparsePageView batch{ctx_, page, n_features_};
        fn(Loader{}, std::forward<SparsePageView>(batch));
      }
    } else {
      p_fmat->Info().feature_types.SetDevice(ctx_->Device());
      auto feature_types = p_fmat->Info().feature_types.ConstDeviceSpan();

      for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx_, StaticBatch(true))) {
        page.Impl()->Visit(ctx_, feature_types, [&](auto&& batch) {
          using Acc = std::remove_reference_t<decltype(batch)>;
          // No shared memory use for ellpack
          using Loader = EllpackLoader<Acc, EncAccessor>;
          constexpr std::uint32_t kBlockThreads = 256;
          fn(LoaderType<Loader, kBlockThreads>{},
             std::forward<common::GetValueT<decltype(batch)>>(batch));
        });
      }
    }
  }
  // Used by the SHAP methods.
  template <typename Fn>
  void ForEachBatch(DMatrix* p_fmat, EncAccessor&& acc, Fn&& fn) {
    if (p_fmat->PageExists<SparsePage>()) {
      for (auto& page : p_fmat->GetBatches<SparsePage>()) {
        // Shap kernel doesn't use shared memory to stage data.
        SparsePageView batch{ctx_, page, n_features_};
        auto loader = ShapSparsePageLoader<EncAccessor>{batch, acc};
        fn(std::move(loader), page.base_rowid);
      }
    } else {
      p_fmat->Info().feature_types.SetDevice(ctx_->Device());
      auto feature_types = p_fmat->Info().feature_types.ConstDeviceSpan();

      for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx_, StaticBatch(true))) {
        page.Impl()->Visit(ctx_, feature_types, [&](auto&& batch) {
          using Acc = std::remove_reference_t<decltype(batch)>;
          // No shared memory use for ellpack
          auto loader = EllpackLoader{batch,
                                      /*use_shared=*/false,
                                      this->n_features_,
                                      batch.NumRows(),
                                      std::numeric_limits<float>::quiet_NaN(),
                                      std::forward<EncAccessor>(acc)};
          fn(std::move(loader), batch.base_rowid);
        });
      }
    }
  }
};

template <typename Kernel>
void LaunchPredict(Context const* ctx, bool is_dense, enc::DeviceColumnsView const& new_enc,
                   gbm::GBTreeModel const& model, Kernel&& launch) {
  if (is_dense) {
    if (model.Cats() && model.Cats()->HasCategorical() && new_enc.HasCategorical()) {
      auto [acc, mapping] = MakeCatAccessor(ctx, new_enc, model.Cats());
      auto cfg =
          LaunchConfig<std::true_type, decltype(acc)>{ctx, model.learner_model_param->num_feature};
      launch(std::move(cfg), std::move(acc));
    } else {
      auto cfg =
          LaunchConfig<std::true_type, NoOpAccessor>{ctx, model.learner_model_param->num_feature};
      launch(std::move(cfg), NoOpAccessor{});
    }
  } else {
    if (model.Cats() && model.Cats()->HasCategorical() && new_enc.HasCategorical()) {
      auto [acc, mapping] = MakeCatAccessor(ctx, new_enc, model.Cats());
      auto cfg =
          LaunchConfig<std::false_type, decltype(acc)>{ctx, model.learner_model_param->num_feature};
      launch(std::move(cfg), std::move(acc));
    } else {
      auto cfg =
          LaunchConfig<std::false_type, NoOpAccessor>{ctx, model.learner_model_param->num_feature};
      launch(std::move(cfg), NoOpAccessor{});
    }
  }
}

template <typename Kernel>
void LaunchShap(Context const* ctx, enc::DeviceColumnsView const& new_enc,
                gbm::GBTreeModel const& model, Kernel&& launch) {
  if (model.Cats() && model.Cats()->HasCategorical() && new_enc.HasCategorical()) {
    auto [acc, mapping] = MakeCatAccessor(ctx, new_enc, model.Cats());
    auto cfg =
        LaunchConfig<std::true_type, decltype(acc)>{ctx, model.learner_model_param->num_feature};
    launch(std::move(cfg), std::move(acc));
  } else {
    auto cfg =
        LaunchConfig<std::true_type, NoOpAccessor>{ctx, model.learner_model_param->num_feature};
    launch(std::move(cfg), NoOpAccessor{});
  }
}
}  // anonymous namespace

class GPUPredictor : public xgboost::Predictor {
 private:
  void PredictDMatrix(DMatrix* p_fmat, HostDeviceVector<float>* out_preds,
                      gbm::GBTreeModel const& model, bst_tree_t tree_begin,
                      bst_tree_t tree_end) const {
    if (tree_end - tree_begin == 0) {
      return;
    }
    out_preds->SetDevice(ctx_->Device());
    auto const& info = p_fmat->Info();

    DeviceModel d_model{this->ctx_->Device(), model, tree_begin, tree_end, &this->model_mu_,
                        CopyViews{this->ctx_}};

    if (info.IsColumnSplit()) {
      column_split_helper_.PredictBatch(p_fmat, out_preds, model, d_model);
      return;
    }

    CHECK_LE(p_fmat->Info().num_col_, model.learner_model_param->num_feature);
    auto n_features = model.learner_model_param->num_feature;

    auto new_enc =
        p_fmat->Cats()->NeedRecode() ? p_fmat->Cats()->DeviceView(ctx_) : enc::DeviceColumnsView{};
    LaunchPredict(ctx_, p_fmat->IsDense(), new_enc, model, [&](auto&& cfg, auto&& acc) {
      using Config = common::GetValueT<decltype(cfg)>;

      bst_idx_t batch_offset = 0;
      cfg.ForEachBatch(p_fmat, [&](auto&& loader_t, auto&& batch) {
        using Loader = typename common::GetValueT<decltype(loader_t)>;
        cfg.template LaunchPredictKernel<Loader>(std::move(batch),
                                                 std::numeric_limits<float>::quiet_NaN(),
                                                 n_features, d_model, acc, batch_offset, out_preds);
        batch_offset += batch.NumRows() * model.learner_model_param->OutputLength();
      });
    });
  }

 public:
  explicit GPUPredictor(Context const* ctx) : Predictor{ctx}, column_split_helper_{ctx} {}

  ~GPUPredictor() override {
    if (ctx_->IsCUDA() && ctx_->Ordinal() < curt::AllVisibleGPUs()) {
      dh::safe_cuda(cudaSetDevice(ctx_->Ordinal()));
    }
  }

  void PredictBatch(DMatrix* dmat, PredictionCacheEntry* predts, const gbm::GBTreeModel& model,
                    bst_tree_t tree_begin, bst_tree_t tree_end = 0) const override {
    xgboost_NVTX_FN_RANGE();
    CHECK(ctx_->Device().IsCUDA()) << "Set `device' to `cuda` for processing GPU data.";
    auto* out_preds = &predts->predictions;
    if (tree_end == 0) {
      tree_end = model.trees.size();
    }
    this->PredictDMatrix(dmat, out_preds, model, tree_begin, tree_end);
  }

  template <typename Adapter>
  void DispatchedInplacePredict(std::shared_ptr<Adapter> m, std::shared_ptr<DMatrix> p_m,
                                const gbm::GBTreeModel& model, float missing,
                                PredictionCacheEntry* out_preds, bst_tree_t tree_begin,
                                bst_tree_t tree_end) const {
    CHECK_EQ(dh::CurrentDevice(), m->Device().ordinal)
        << "XGBoost is running on device: " << this->ctx_->Device().Name() << ", "
        << "but data is on: " << m->Device().Name();
    this->InitOutPredictions(p_m->Info(), &(out_preds->predictions), model);
    out_preds->predictions.SetDevice(m->Device());
    using BatchT = common::GetValueT<decltype(std::declval<Adapter>().Value())>;

    auto n_samples = m->NumRows();
    auto n_features = model.learner_model_param->num_feature;

    DeviceModel d_model{ctx_->Device(),       model, tree_begin, tree_end, &this->model_mu_,
                        CopyViews{this->ctx_}};

    if constexpr (std::is_same_v<Adapter, data::CudfAdapter>) {
      if (m->HasCategorical()) {
        auto new_enc = m->DCats();
        LaunchPredict(this->ctx_, false, new_enc, model, [&](auto&& cfg, auto&& acc) {
          using EncAccessor = std::remove_reference_t<decltype(acc)>;
          using LoaderImpl = DeviceAdapterLoader<BatchT, EncAccessor>;
          using Loader =
              typename common::GetValueT<decltype(cfg)>::template LoaderType<LoaderImpl, 128>;
          cfg.template AllocShmem<Loader>();
          cfg.template LaunchPredictKernel<Loader>(m->Value(), missing, n_features, d_model, acc, 0,
                                                   &out_preds->predictions);
        });
        return;
      }
    }

    LaunchPredict(this->ctx_, false, enc::DeviceColumnsView{}, model,
                  [&](auto&& cfg, auto&& acc) {
                    using EncAccessor = std::remove_reference_t<decltype(acc)>;
                    CHECK((std::is_same_v<EncAccessor, NoOpAccessor>));
                    using LoaderImpl = DeviceAdapterLoader<BatchT, EncAccessor>;
                    using Loader =
                        typename common::GetValueT<decltype(cfg)>::template LoaderType<LoaderImpl,
                                                                                       128>;
                    cfg.template AllocShmem<Loader>();
                    cfg.template LaunchPredictKernel<Loader>(
                        m->Value(), missing, n_features, d_model, acc, 0, &out_preds->predictions);
                  });
  }

  [[nodiscard]] bool InplacePredict(std::shared_ptr<DMatrix> p_m, gbm::GBTreeModel const& model,
                                    float missing, PredictionCacheEntry* out_preds,
                                    bst_tree_t tree_begin, bst_tree_t tree_end) const override {
    xgboost_NVTX_FN_RANGE();
    auto proxy = dynamic_cast<data::DMatrixProxy*>(p_m.get());
    CHECK(proxy) << error::InplacePredictProxy();
    bool type_error = false;
    data::cuda_impl::DispatchAny<false>(
        proxy,
        [&](auto x) {
          CheckProxyDMatrix(x, proxy, model.learner_model_param);
          this->DispatchedInplacePredict(x, p_m, model, missing, out_preds, tree_begin, tree_end);
        },
        &type_error);
    return !type_error;
  }

  void PredictContribution(DMatrix* p_fmat, HostDeviceVector<float>* out_contribs,
                           const gbm::GBTreeModel& model, bst_tree_t tree_end,
                           std::vector<float> const* tree_weights, bool approximate, int,
                           unsigned) const override {
    xgboost_NVTX_FN_RANGE();
    StringView not_implemented{
        "contribution is not implemented in the GPU predictor, use CPU instead."};
    if (approximate) {
      LOG(FATAL) << "Approximated " << not_implemented;
    }
    if (tree_weights != nullptr) {
      LOG(FATAL) << "Dart booster feature " << not_implemented;
    }
    CHECK(!p_fmat->Info().IsColumnSplit())
        << "Predict contribution support for column-wise data split is not yet implemented.";
    dh::safe_cuda(cudaSetDevice(ctx_->Ordinal()));
    out_contribs->SetDevice(ctx_->Device());
    tree_end = GetTreeLimit(model.trees, tree_end);

    const int ngroup = model.learner_model_param->num_output_group;
    CHECK_NE(ngroup, 0);
    // allocate space for (number of features + bias) times the number of rows
    size_t contributions_columns = model.learner_model_param->num_feature + 1;  // +1 for bias
    auto dim_size = contributions_columns * model.learner_model_param->num_output_group;
    out_contribs->Resize(p_fmat->Info().num_row_ * dim_size);
    out_contribs->Fill(0.0f);
    auto phis = out_contribs->DeviceSpan();

    dh::device_vector<gpu_treeshap::PathElement<ShapSplitCondition>> device_paths;
    DeviceModel d_model{this->ctx_->Device(), model, 0, tree_end, &this->model_mu_,
                        CopyViews{this->ctx_}};

    auto new_enc =
        p_fmat->Cats()->NeedRecode() ? p_fmat->Cats()->DeviceView(ctx_) : enc::DeviceColumnsView{};

    dh::device_vector<uint32_t> categories;
    ExtractPaths(ctx_, &device_paths, model, d_model, &categories);

    LaunchShap(this->ctx_, new_enc, model, [&](auto&& cfg, auto&& acc) {
      using Config = common::GetValueT<decltype(cfg)>;
      using EncAccessor = typename Config::EncAccessorT;

      cfg.ForEachBatch(
          p_fmat, std::forward<EncAccessor>(acc), [&](auto&& loader, bst_idx_t base_rowid) {
            auto begin = dh::tbegin(phis) + base_rowid * dim_size;
            gpu_treeshap::GPUTreeShap<dh::XGBDeviceAllocator<int>>(
                loader, device_paths.begin(), device_paths.end(), ngroup, begin, dh::tend(phis));
          });
    });

    // Add the base margin term to last column
    p_fmat->Info().base_margin_.SetDevice(ctx_->Device());
    const auto margin = p_fmat->Info().base_margin_.Data()->ConstDeviceSpan();

    auto base_score = model.learner_model_param->BaseScore(ctx_);
    dh::LaunchN(p_fmat->Info().num_row_ * model.learner_model_param->num_output_group,
                ctx_->CUDACtx()->Stream(), [=] __device__(size_t idx) {
                  phis[(idx + 1) * contributions_columns - 1] +=
                      margin.empty() ? base_score(0) : margin[idx];
                });
  }

  void PredictInteractionContributions(DMatrix* p_fmat, HostDeviceVector<float>* out_contribs,
                                       gbm::GBTreeModel const& model, bst_tree_t tree_end,
                                       std::vector<float> const* tree_weights,
                                       bool approximate) const override {
    xgboost_NVTX_FN_RANGE();
    std::string not_implemented{
        "contribution is not implemented in GPU predictor, use cpu instead."};
    if (approximate) {
      LOG(FATAL) << "Approximated " << not_implemented;
    }
    if (tree_weights != nullptr) {
      LOG(FATAL) << "Dart booster feature " << not_implemented;
    }
    dh::safe_cuda(cudaSetDevice(ctx_->Ordinal()));
    out_contribs->SetDevice(ctx_->Device());
    tree_end = GetTreeLimit(model.trees, tree_end);

    const int ngroup = model.learner_model_param->num_output_group;
    CHECK_NE(ngroup, 0);
    // allocate space for (number of features + bias) times the number of rows
    size_t contributions_columns = model.learner_model_param->num_feature + 1;  // +1 for bias
    auto dim_size =
        contributions_columns * contributions_columns * model.learner_model_param->num_output_group;
    out_contribs->Resize(p_fmat->Info().num_row_ * dim_size);
    out_contribs->Fill(0.0f);
    auto phis = out_contribs->DeviceSpan();

    dh::device_vector<gpu_treeshap::PathElement<ShapSplitCondition>> device_paths;
    DeviceModel d_model{this->ctx_->Device(), model, 0, tree_end, &this->model_mu_,
                        CopyViews{this->ctx_}};

    dh::device_vector<uint32_t> categories;
    ExtractPaths(ctx_, &device_paths, model, d_model, &categories);
    auto new_enc =
        p_fmat->Cats()->NeedRecode() ? p_fmat->Cats()->DeviceView(ctx_) : enc::DeviceColumnsView{};

    LaunchShap(this->ctx_, new_enc, model, [&](auto&& cfg, auto&& acc) {
      using Config = common::GetValueT<decltype(cfg)>;
      using EncAccessor = typename Config::EncAccessorT;

      cfg.ForEachBatch(
          p_fmat, std::forward<EncAccessor>(acc), [&](auto&& loader, bst_idx_t base_rowid) {
            auto begin = dh::tbegin(phis) + base_rowid * dim_size;
            gpu_treeshap::GPUTreeShapInteractions<dh::XGBDeviceAllocator<int>>(
                loader, device_paths.begin(), device_paths.end(), ngroup, begin, dh::tend(phis));
          });
    });

    // Add the base margin term to last column
    p_fmat->Info().base_margin_.SetDevice(ctx_->Device());
    const auto margin = p_fmat->Info().base_margin_.Data()->ConstDeviceSpan();

    auto base_score = model.learner_model_param->BaseScore(ctx_);
    size_t n_features = model.learner_model_param->num_feature;
    dh::LaunchN(p_fmat->Info().num_row_ * model.learner_model_param->num_output_group,
                ctx_->CUDACtx()->Stream(), [=] __device__(size_t idx) {
                  size_t group = idx % ngroup;
                  size_t row_idx = idx / ngroup;
                  phis[gpu_treeshap::IndexPhiInteractions(row_idx, ngroup, group, n_features,
                                                          n_features, n_features)] +=
                      margin.empty() ? base_score(0) : margin[idx];
                });
  }

  void PredictLeaf(DMatrix* p_fmat, HostDeviceVector<float>* predictions,
                   gbm::GBTreeModel const& model, bst_tree_t tree_end) const override {
    xgboost_NVTX_FN_RANGE();
    dh::safe_cuda(cudaSetDevice(ctx_->Ordinal()));

    const MetaInfo& info = p_fmat->Info();
    bst_idx_t n_samples = info.num_row_;
    tree_end = GetTreeLimit(model.trees, tree_end);
    predictions->SetDevice(ctx_->Device());
    predictions->Resize(n_samples * tree_end);

    DeviceModel d_model{ctx_->Device(),       model, 0, tree_end, &this->model_mu_,
                        CopyViews{this->ctx_}};

    if (info.IsColumnSplit()) {
      column_split_helper_.PredictLeaf(p_fmat, predictions, model, d_model);
      return;
    }

    bst_feature_t n_features = model.learner_model_param->num_feature;
    auto new_enc =
        p_fmat->Cats()->NeedRecode() ? p_fmat->Cats()->DeviceView(ctx_) : enc::DeviceColumnsView{};

    LaunchPredict(ctx_, p_fmat->IsDense(), new_enc, model, [&](auto&& cfg, auto&& acc) {
      bst_idx_t batch_offset = 0;
      cfg.ForEachBatch(p_fmat, [&](auto&& loader_t, auto&& batch) {
        using Loader = typename common::GetValueT<decltype(loader_t)>;
        using Config = common::GetValueT<decltype(cfg)>;
        auto kernel = PredictLeafKernel<typename Loader::Type, common::GetValueT<decltype(batch)>,
                                        Config::HasMissing(), typename Config::EncAccessorT>;
        cfg.template Launch<Loader>(kernel, std::move(batch), d_model.Trees(),
                                    predictions->DeviceSpan().subspan(batch_offset),
                                    d_model.tree_begin, d_model.tree_end, n_features,
                                    cfg.UseShared(), std::numeric_limits<float>::quiet_NaN(),
                                    std::forward<typename Config::EncAccessorT>(acc));

        batch_offset += batch.NumRows();
      });
    });
  }

 private:
  // Prevent multiple threads from pulling the model to device together.
  mutable std::mutex model_mu_;
  ColumnSplitHelper column_split_helper_;
};

XGBOOST_REGISTER_PREDICTOR(GPUPredictor, "gpu_predictor")
    .describe("Make predictions using GPU.")
    .set_body([](Context const* ctx) { return new GPUPredictor(ctx); });

}  // namespace xgboost::predictor
