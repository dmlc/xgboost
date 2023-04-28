/**
 * Copyright 2017-2023 by XGBoost Contributors
 */
#include <GPUTreeShap/gpu_treeshap.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <any>  // for any, any_cast
#include <memory>

#include "../common/bitfield.h"
#include "../common/categorical.h"
#include "../common/common.h"
#include "../common/device_helpers.cuh"
#include "../data/device_adapter.cuh"
#include "../data/ellpack_page.cuh"
#include "../data/proxy_dmatrix.h"
#include "../gbm/gbtree_model.h"
#include "predict_fn.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/predictor.h"
#include "xgboost/tree_model.h"
#include "xgboost/tree_updater.h"

namespace xgboost::predictor {
DMLC_REGISTRY_FILE_TAG(gpu_predictor);

struct TreeView {
  RegTree::CategoricalSplitMatrix cats;
  common::Span<RegTree::Node const> d_tree;

  XGBOOST_DEVICE
  TreeView(size_t tree_begin, size_t tree_idx, common::Span<const RegTree::Node> d_nodes,
           common::Span<size_t const> d_tree_segments,
           common::Span<FeatureType const> d_tree_split_types,
           common::Span<uint32_t const> d_cat_tree_segments,
           common::Span<RegTree::CategoricalSplitMatrix::Segment const> d_cat_node_segments,
           common::Span<uint32_t const> d_categories) {
    auto begin = d_tree_segments[tree_idx - tree_begin];
    auto n_nodes = d_tree_segments[tree_idx - tree_begin + 1] -
                   d_tree_segments[tree_idx - tree_begin];

    d_tree = d_nodes.subspan(begin, n_nodes);

    auto tree_cat_ptrs = d_cat_node_segments.subspan(begin, n_nodes);
    auto tree_split_types = d_tree_split_types.subspan(begin, n_nodes);

    auto tree_categories =
        d_categories.subspan(d_cat_tree_segments[tree_idx - tree_begin],
                             d_cat_tree_segments[tree_idx - tree_begin + 1] -
                                 d_cat_tree_segments[tree_idx - tree_begin]);

    cats.split_type = tree_split_types;
    cats.categories = tree_categories;
    cats.node_ptr = tree_cat_ptrs;
  }

  __device__ bool HasCategoricalSplit() const {
    return !cats.categories.empty();
  }
};

struct SparsePageView {
  common::Span<const Entry> d_data;
  common::Span<const bst_row_t> d_row_ptr;
  bst_feature_t num_features;

  SparsePageView() = default;
  XGBOOST_DEVICE SparsePageView(common::Span<const Entry> data,
                                common::Span<const bst_row_t> row_ptr,
                                bst_feature_t num_features)
      : d_data{data}, d_row_ptr{row_ptr}, num_features(num_features) {}
  __device__ float GetElement(size_t ridx, size_t fidx) const {
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
    return nanf("");
  }
  XGBOOST_DEVICE size_t NumRows() const { return d_row_ptr.size() - 1; }
  XGBOOST_DEVICE size_t NumCols() const { return num_features; }
};

struct SparsePageLoader {
  bool use_shared;
  SparsePageView data;
  float* smem;
  size_t entry_start;

  __device__ SparsePageLoader(SparsePageView data, bool use_shared, bst_feature_t num_features,
                              bst_row_t num_rows, size_t entry_start, float)
      : use_shared(use_shared),
        data(data),
        entry_start(entry_start) {
    extern __shared__ float _smem[];
    smem = _smem;
    // Copy instances
    if (use_shared) {
      bst_uint global_idx = blockDim.x * blockIdx.x + threadIdx.x;
      int shared_elements = blockDim.x * data.num_features;
      dh::BlockFill(smem, shared_elements, nanf(""));
      __syncthreads();
      if (global_idx < num_rows) {
        bst_uint elem_begin = data.d_row_ptr[global_idx];
        bst_uint elem_end = data.d_row_ptr[global_idx + 1];
        for (bst_uint elem_idx = elem_begin; elem_idx < elem_end; elem_idx++) {
          Entry elem = data.d_data[elem_idx - entry_start];
          smem[threadIdx.x * data.num_features + elem.index] = elem.fvalue;
        }
      }
      __syncthreads();
    }
  }
  __device__ float GetElement(size_t  ridx, size_t  fidx) const {
    if (use_shared) {
      return smem[threadIdx.x * data.num_features + fidx];
    } else {
      return data.GetElement(ridx, fidx);
    }
  }
};

struct EllpackLoader {
  EllpackDeviceAccessor const& matrix;
  XGBOOST_DEVICE EllpackLoader(EllpackDeviceAccessor const& m, bool, bst_feature_t, bst_row_t,
                               size_t, float)
      : matrix{m} {}
  __device__ __forceinline__ float GetElement(size_t ridx, size_t fidx) const {
    auto gidx = matrix.GetBinIndex(ridx, fidx);
    if (gidx == -1) {
      return nan("");
    }
    if (common::IsCat(matrix.feature_types, fidx)) {
      return matrix.gidx_fvalue_map[gidx];
    }
    // The gradient index needs to be shifted by one as min values are not included in the
    // cuts.
    if (gidx == matrix.feature_segments[fidx]) {
      return matrix.min_fvalue[fidx];
    }
    return matrix.gidx_fvalue_map[gidx - 1];
  }
};

template <typename Batch>
struct DeviceAdapterLoader {
  Batch batch;
  bst_feature_t columns;
  float* smem;
  bool use_shared;
  data::IsValidFunctor is_valid;

  using BatchT = Batch;

  XGBOOST_DEV_INLINE DeviceAdapterLoader(Batch const batch, bool use_shared,
                                         bst_feature_t num_features, bst_row_t num_rows,
                                         size_t entry_start, float missing) :
    batch{batch},
    columns{num_features},
    use_shared{use_shared},
    is_valid{missing} {
      extern __shared__ float _smem[];
      smem = _smem;
      if (use_shared) {
        uint32_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
        size_t shared_elements = blockDim.x * num_features;
        dh::BlockFill(smem, shared_elements, nanf(""));
        __syncthreads();
        if (global_idx < num_rows) {
          auto beg = global_idx * columns;
          auto end = (global_idx + 1) * columns;
          for (size_t i = beg; i < end; ++i) {
            auto value = batch.GetElement(i).value;
            if (is_valid(value)) {
              smem[threadIdx.x * num_features + (i - beg)] = value;
            }
          }
        }
      }
      __syncthreads();
    }

  XGBOOST_DEV_INLINE  float GetElement(size_t  ridx, size_t  fidx) const {
    if (use_shared) {
      return smem[threadIdx.x * columns + fidx];
    }
    auto value = batch.GetElement(ridx * columns + fidx).value;
    if (is_valid(value)) {
      return value;
    } else {
      return nan("");
    }
  }
};

template <bool has_missing, bool has_categorical, typename Loader>
__device__ bst_node_t GetLeafIndex(bst_row_t ridx, TreeView const &tree,
                                   Loader *loader) {
  bst_node_t nidx = 0;
  RegTree::Node n = tree.d_tree[nidx];
  while (!n.IsLeaf()) {
    float fvalue = loader->GetElement(ridx, n.SplitIndex());
    bool is_missing = common::CheckNAN(fvalue);
    nidx = GetNextNode<has_missing, has_categorical>(n, nidx, fvalue,
                                                     is_missing, tree.cats);
    n = tree.d_tree[nidx];
  }
  return nidx;
}

template <bool has_missing, typename Loader>
__device__ float GetLeafWeight(bst_row_t ridx, TreeView const &tree,
                               Loader *loader) {
  bst_node_t nidx = -1;
  if (tree.HasCategoricalSplit()) {
    nidx = GetLeafIndex<has_missing, true>(ridx, tree, loader);
  } else {
    nidx = GetLeafIndex<has_missing, false>(ridx, tree, loader);
  }
  return tree.d_tree[nidx].LeafValue();
}

template <typename Loader, typename Data>
__global__ void
PredictLeafKernel(Data data, common::Span<const RegTree::Node> d_nodes,
                  common::Span<float> d_out_predictions,
                  common::Span<size_t const> d_tree_segments,

                  common::Span<FeatureType const> d_tree_split_types,
                  common::Span<uint32_t const> d_cat_tree_segments,
                  common::Span<RegTree::CategoricalSplitMatrix::Segment const> d_cat_node_segments,
                  common::Span<uint32_t const> d_categories,

                  size_t tree_begin, size_t tree_end, size_t num_features,
                  size_t num_rows, size_t entry_start, bool use_shared,
                  float missing) {
  bst_row_t ridx = blockDim.x * blockIdx.x + threadIdx.x;
  if (ridx >= num_rows) {
    return;
  }
  Loader loader(data, use_shared, num_features, num_rows, entry_start, missing);
  for (size_t tree_idx = tree_begin; tree_idx < tree_end; ++tree_idx) {
    TreeView d_tree{
        tree_begin,          tree_idx,           d_nodes,
        d_tree_segments,     d_tree_split_types, d_cat_tree_segments,
        d_cat_node_segments, d_categories};

    bst_node_t leaf = -1;
    if (d_tree.HasCategoricalSplit()) {
      leaf = GetLeafIndex<true, true>(ridx, d_tree, &loader);
    } else {
      leaf = GetLeafIndex<true, false>(ridx, d_tree, &loader);
    }
    d_out_predictions[ridx * (tree_end - tree_begin) + tree_idx] = leaf;
  }
}

template <typename Loader, typename Data, bool has_missing = true>
__global__ void
PredictKernel(Data data, common::Span<const RegTree::Node> d_nodes,
              common::Span<float> d_out_predictions,
              common::Span<size_t const> d_tree_segments,
              common::Span<int const> d_tree_group,
              common::Span<FeatureType const> d_tree_split_types,
              common::Span<uint32_t const> d_cat_tree_segments,
              common::Span<RegTree::CategoricalSplitMatrix::Segment const> d_cat_node_segments,
              common::Span<uint32_t const> d_categories, size_t tree_begin,
              size_t tree_end, size_t num_features, size_t num_rows,
              size_t entry_start, bool use_shared, int num_group, float missing) {
  bst_uint global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  Loader loader(data, use_shared, num_features, num_rows, entry_start, missing);
  if (global_idx >= num_rows) return;
  if (num_group == 1) {
    float sum = 0;
    for (size_t tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      TreeView d_tree{
          tree_begin,          tree_idx,           d_nodes,
          d_tree_segments,     d_tree_split_types, d_cat_tree_segments,
          d_cat_node_segments, d_categories};
      float leaf = GetLeafWeight<has_missing>(global_idx, d_tree, &loader);
      sum += leaf;
    }
    d_out_predictions[global_idx] += sum;
  } else {
    for (size_t tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      int tree_group = d_tree_group[tree_idx];
      TreeView d_tree{
          tree_begin,          tree_idx,           d_nodes,
          d_tree_segments,     d_tree_split_types, d_cat_tree_segments,
          d_cat_node_segments, d_categories};
      bst_uint out_prediction_idx = global_idx * num_group + tree_group;
      d_out_predictions[out_prediction_idx] +=
          GetLeafWeight<has_missing>(global_idx, d_tree, &loader);
    }
  }
}

class DeviceModel {
 public:
  // Need to lazily construct the vectors because GPU id is only known at runtime
  HostDeviceVector<RTreeNodeStat> stats;
  HostDeviceVector<size_t> tree_segments;
  HostDeviceVector<RegTree::Node> nodes;
  HostDeviceVector<int> tree_group;
  HostDeviceVector<FeatureType> split_types;

  // Pointer to each tree, segmenting the node array.
  HostDeviceVector<uint32_t> categories_tree_segments;
  // Pointer to each node, segmenting categories array.
  HostDeviceVector<RegTree::CategoricalSplitMatrix::Segment> categories_node_segments;
  HostDeviceVector<uint32_t> categories;

  size_t tree_beg_;  // NOLINT
  size_t tree_end_;  // NOLINT
  int num_group;

  void Init(const gbm::GBTreeModel& model, size_t tree_begin, size_t tree_end, int32_t gpu_id) {
    dh::safe_cuda(cudaSetDevice(gpu_id));

    // Copy decision trees to device
    tree_segments = std::move(HostDeviceVector<size_t>({}, gpu_id));
    auto& h_tree_segments = tree_segments.HostVector();
    h_tree_segments.reserve((tree_end - tree_begin) + 1);
    size_t sum = 0;
    h_tree_segments.push_back(sum);
    for (auto tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      sum += model.trees.at(tree_idx)->GetNodes().size();
      h_tree_segments.push_back(sum);
    }

    nodes = std::move(HostDeviceVector<RegTree::Node>(h_tree_segments.back(), RegTree::Node(),
                                                      gpu_id));
    stats = std::move(HostDeviceVector<RTreeNodeStat>(h_tree_segments.back(),
                                                      RTreeNodeStat(), gpu_id));
    auto d_nodes = nodes.DevicePointer();
    auto d_stats = stats.DevicePointer();
    for (auto tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      auto& src_nodes = model.trees.at(tree_idx)->GetNodes();
      auto& src_stats = model.trees.at(tree_idx)->GetStats();
      dh::safe_cuda(cudaMemcpyAsync(
          d_nodes + h_tree_segments[tree_idx - tree_begin], src_nodes.data(),
          sizeof(RegTree::Node) * src_nodes.size(), cudaMemcpyDefault));
      dh::safe_cuda(cudaMemcpyAsync(
          d_stats + h_tree_segments[tree_idx - tree_begin], src_stats.data(),
          sizeof(RTreeNodeStat) * src_stats.size(), cudaMemcpyDefault));
    }

    tree_group = std::move(HostDeviceVector<int>(model.tree_info.size(), 0, gpu_id));
    auto& h_tree_group = tree_group.HostVector();
    std::memcpy(h_tree_group.data(), model.tree_info.data(), sizeof(int) * model.tree_info.size());

    // Initialize categorical splits.
    split_types.SetDevice(gpu_id);
    std::vector<FeatureType>& h_split_types = split_types.HostVector();
    h_split_types.resize(h_tree_segments.back());
    for (auto tree_idx = tree_begin; tree_idx < tree_end; ++tree_idx) {
      auto const& src_st = model.trees.at(tree_idx)->GetSplitTypes();
      std::copy(src_st.cbegin(), src_st.cend(),
                h_split_types.begin() + h_tree_segments[tree_idx - tree_begin]);
    }

    categories = HostDeviceVector<uint32_t>({}, gpu_id);
    categories_tree_segments = HostDeviceVector<uint32_t>(1, 0, gpu_id);
    std::vector<uint32_t> &h_categories = categories.HostVector();
    std::vector<uint32_t> &h_split_cat_segments = categories_tree_segments.HostVector();
    for (auto tree_idx = tree_begin; tree_idx < tree_end; ++tree_idx) {
      auto const& src_cats = model.trees.at(tree_idx)->GetSplitCategories();
      size_t orig_size = h_categories.size();
      h_categories.resize(orig_size + src_cats.size());
      std::copy(src_cats.cbegin(), src_cats.cend(),
                h_categories.begin() + orig_size);
      h_split_cat_segments.push_back(h_categories.size());
    }

    categories_node_segments = HostDeviceVector<RegTree::CategoricalSplitMatrix::Segment>(
        h_tree_segments.back(), {}, gpu_id);
    std::vector<RegTree::CategoricalSplitMatrix::Segment>& h_categories_node_segments =
        categories_node_segments.HostVector();
    for (auto tree_idx = tree_begin; tree_idx < tree_end; ++tree_idx) {
      auto const &src_cats_ptr = model.trees.at(tree_idx)->GetSplitCategoriesPtr();
      std::copy(src_cats_ptr.cbegin(), src_cats_ptr.cend(),
                h_categories_node_segments.begin() +
                    h_tree_segments[tree_idx - tree_begin]);
    }

    this->tree_beg_ = tree_begin;
    this->tree_end_ = tree_end;
    this->num_group = model.learner_model_param->OutputLength();
  }
};

struct ShapSplitCondition {
  ShapSplitCondition() = default;
  XGBOOST_DEVICE
  ShapSplitCondition(float feature_lower_bound, float feature_upper_bound,
                     bool is_missing_branch, common::CatBitField cats)
      : feature_lower_bound(feature_lower_bound),
        feature_upper_bound(feature_upper_bound),
        is_missing_branch(is_missing_branch), categories{std::move(cats)} {
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
  XGBOOST_DEVICE bool EvaluateSplit(float x) const {
    // is nan
    if (isnan(x)) {
      return is_missing_branch;
    }
    if (categories.Size() != 0) {
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
    if (l.Size() > r.Size()) {
      thrust::swap(l, r);
    }
    for (size_t i = 0; i < r.Bits().size(); ++i) {
      l.Bits()[i] &= r.Bits()[i];
    }
    return l;
  }

  // Combine two split conditions on the same feature
  XGBOOST_DEVICE void Merge(ShapSplitCondition other) {
    // Combine duplicate features
    if (categories.Size() != 0 || other.categories.Size() != 0) {
      categories = Intersect(categories, other.categories);
    } else {
      feature_lower_bound = max(feature_lower_bound, other.feature_lower_bound);
      feature_upper_bound = min(feature_upper_bound, other.feature_upper_bound);
    }
    is_missing_branch = is_missing_branch && other.is_missing_branch;
  }
};

struct PathInfo {
  int64_t leaf_position;  // -1 not a leaf
  size_t length;
  size_t tree_idx;
};

// Transform model into path element form for GPUTreeShap
void ExtractPaths(
    dh::device_vector<gpu_treeshap::PathElement<ShapSplitCondition>> *paths,
    DeviceModel *model, dh::device_vector<uint32_t> *path_categories,
    int gpu_id) {
  dh::safe_cuda(cudaSetDevice(gpu_id));
  auto& device_model = *model;

  dh::caching_device_vector<PathInfo> info(device_model.nodes.Size());
  dh::XGBCachingDeviceAllocator<PathInfo> alloc;
  auto d_nodes = device_model.nodes.ConstDeviceSpan();
  auto d_tree_segments = device_model.tree_segments.ConstDeviceSpan();
  auto nodes_transform = dh::MakeTransformIterator<PathInfo>(
      thrust::make_counting_iterator(0ull), [=] __device__(size_t idx) {
        auto n = d_nodes[idx];
        if (!n.IsLeaf() || n.IsDeleted()) {
          return PathInfo{-1, 0, 0};
        }
        size_t tree_idx =
            dh::SegmentId(d_tree_segments.begin(), d_tree_segments.end(), idx);
        size_t tree_offset = d_tree_segments[tree_idx];
        size_t path_length = 1;
        while (!n.IsRoot()) {
          n = d_nodes[n.Parent() + tree_offset];
          path_length++;
        }
        return PathInfo{static_cast<int64_t>(idx), path_length, tree_idx};
      });
  auto end = thrust::copy_if(
      thrust::cuda::par(alloc), nodes_transform,
      nodes_transform + d_nodes.size(), info.begin(),
      [=] __device__(const PathInfo& e) { return e.leaf_position != -1; });
  info.resize(end - info.begin());
  auto length_iterator = dh::MakeTransformIterator<size_t>(
      info.begin(),
      [=] __device__(const PathInfo& info) { return info.length; });
  dh::caching_device_vector<size_t> path_segments(info.size() + 1);
  thrust::exclusive_scan(thrust::cuda::par(alloc), length_iterator,
                         length_iterator + info.size() + 1,
                         path_segments.begin());

  paths->resize(path_segments.back());

  auto d_paths = dh::ToSpan(*paths);
  auto d_info = info.data().get();
  auto d_stats = device_model.stats.ConstDeviceSpan();
  auto d_tree_group = device_model.tree_group.ConstDeviceSpan();
  auto d_path_segments = path_segments.data().get();

  auto d_split_types = device_model.split_types.ConstDeviceSpan();
  auto d_cat_segments = device_model.categories_tree_segments.ConstDeviceSpan();
  auto d_cat_node_segments = device_model.categories_node_segments.ConstDeviceSpan();

  size_t max_cat = 0;
  if (thrust::any_of(dh::tbegin(d_split_types), dh::tend(d_split_types),
                     common::IsCatOp{})) {
    dh::PinnedMemory pinned;
    auto h_max_cat = pinned.GetSpan<RegTree::CategoricalSplitMatrix::Segment>(1);
    auto max_elem_it = dh::MakeTransformIterator<size_t>(
        dh::tbegin(d_cat_node_segments),
        [] __device__(RegTree::CategoricalSplitMatrix::Segment seg) { return seg.size; });
    size_t max_cat_it =
        thrust::max_element(thrust::device, max_elem_it,
                            max_elem_it + d_cat_node_segments.size()) -
        max_elem_it;
    dh::safe_cuda(cudaMemcpy(h_max_cat.data(),
                             d_cat_node_segments.data() + max_cat_it,
                             h_max_cat.size_bytes(), cudaMemcpyDeviceToHost));
    max_cat = h_max_cat[0].size;
    CHECK_GE(max_cat, 1);
    path_categories->resize(max_cat * paths->size());
  }

  auto d_model_categories = device_model.categories.DeviceSpan();
  common::Span<uint32_t> d_path_categories = dh::ToSpan(*path_categories);

  dh::LaunchN(info.size(), [=] __device__(size_t idx) {
    auto path_info = d_info[idx];
    size_t tree_offset = d_tree_segments[path_info.tree_idx];
    TreeView tree{0,                   path_info.tree_idx, d_nodes,
                  d_tree_segments,     d_split_types,      d_cat_segments,
                  d_cat_node_segments, d_model_categories};
    int group = d_tree_group[path_info.tree_idx];
    size_t child_idx = path_info.leaf_position;
    auto child = d_nodes[child_idx];
    float v = child.LeafValue();
    const float inf = std::numeric_limits<float>::infinity();
    size_t output_position = d_path_segments[idx + 1] - 1;
    while (!child.IsRoot()) {
      size_t parent_idx = tree_offset + child.Parent();
      double child_cover = d_stats[child_idx].sum_hess;
      double parent_cover = d_stats[parent_idx].sum_hess;
      double zero_fraction = child_cover / parent_cover;
      auto parent = tree.d_tree[child.Parent()];

      bool is_left_path = (tree_offset + parent.LeftChild()) == child_idx;
      bool is_missing_path = (!parent.DefaultLeft() && !is_left_path) ||
                             (parent.DefaultLeft() && is_left_path);

      float lower_bound = -inf;
      float upper_bound = inf;
      common::CatBitField bits;
      if (common::IsCat(tree.cats.split_type, child.Parent())) {
        auto path_cats = d_path_categories.subspan(max_cat * output_position, max_cat);
        size_t size = tree.cats.node_ptr[child.Parent()].size;
        auto node_cats = tree.cats.categories.subspan(tree.cats.node_ptr[child.Parent()].beg, size);
        SPAN_CHECK(path_cats.size() >= node_cats.size());
        for (size_t i = 0; i < node_cats.size(); ++i) {
          path_cats[i] = is_left_path ? ~node_cats[i] : node_cats[i];
        }
        bits = common::CatBitField{path_cats};
      } else {
        lower_bound = is_left_path ? -inf : parent.SplitCond();
        upper_bound = is_left_path ? parent.SplitCond() : inf;
      }
      d_paths[output_position--] =
          gpu_treeshap::PathElement<ShapSplitCondition>{
              idx,           parent.SplitIndex(),
              group,         ShapSplitCondition{lower_bound, upper_bound, is_missing_path, bits},
              zero_fraction, v};
      child_idx = parent_idx;
      child = parent;
    }
    // Root node has feature -1
    d_paths[output_position] = {idx, -1, group, ShapSplitCondition{-inf, inf, false, {}}, 1.0, v};
  });
}

namespace {
template <size_t kBlockThreads>
size_t SharedMemoryBytes(size_t cols, size_t max_shared_memory_bytes) {
  // No way max_shared_memory_bytes that is equal to 0.
  CHECK_GT(max_shared_memory_bytes, 0);
  size_t shared_memory_bytes =
      static_cast<size_t>(sizeof(float) * cols * kBlockThreads);
  if (shared_memory_bytes > max_shared_memory_bytes) {
    shared_memory_bytes = 0;
  }
  return shared_memory_bytes;
}
}  // anonymous namespace

class GPUPredictor : public xgboost::Predictor {
 private:
  void PredictInternal(const SparsePage& batch,
                       DeviceModel const& model,
                       size_t num_features,
                       HostDeviceVector<bst_float>* predictions,
                       size_t batch_offset, bool is_dense) const {
    batch.offset.SetDevice(ctx_->gpu_id);
    batch.data.SetDevice(ctx_->gpu_id);
    const uint32_t BLOCK_THREADS = 128;
    size_t num_rows = batch.Size();
    auto GRID_SIZE = static_cast<uint32_t>(common::DivRoundUp(num_rows, BLOCK_THREADS));
    auto max_shared_memory_bytes = ConfigureDevice(ctx_->gpu_id);
    size_t shared_memory_bytes =
        SharedMemoryBytes<BLOCK_THREADS>(num_features, max_shared_memory_bytes);
    bool use_shared = shared_memory_bytes != 0;

    size_t entry_start = 0;
    SparsePageView data(batch.data.DeviceSpan(), batch.offset.DeviceSpan(),
                        num_features);
    auto const kernel = [&](auto predict_fn) {
      dh::LaunchKernel {GRID_SIZE, BLOCK_THREADS, shared_memory_bytes} (
          predict_fn, data, model.nodes.ConstDeviceSpan(),
          predictions->DeviceSpan().subspan(batch_offset),
          model.tree_segments.ConstDeviceSpan(),
          model.tree_group.ConstDeviceSpan(),
          model.split_types.ConstDeviceSpan(),
          model.categories_tree_segments.ConstDeviceSpan(),
          model.categories_node_segments.ConstDeviceSpan(),
          model.categories.ConstDeviceSpan(), model.tree_beg_, model.tree_end_,
          num_features, num_rows, entry_start, use_shared, model.num_group,
          nan(""));
    };
    if (is_dense) {
      kernel(PredictKernel<SparsePageLoader, SparsePageView, false>);
    } else {
      kernel(PredictKernel<SparsePageLoader, SparsePageView, true>);
    }
  }
  void PredictInternal(EllpackDeviceAccessor const& batch,
                       DeviceModel const& model,
                       HostDeviceVector<bst_float>* out_preds,
                       size_t batch_offset) const {
    const uint32_t BLOCK_THREADS = 256;
    size_t num_rows = batch.n_rows;
    auto GRID_SIZE = static_cast<uint32_t>(common::DivRoundUp(num_rows, BLOCK_THREADS));
    DeviceModel d_model;

    bool use_shared = false;
    size_t entry_start = 0;
    dh::LaunchKernel {GRID_SIZE, BLOCK_THREADS} (
        PredictKernel<EllpackLoader, EllpackDeviceAccessor>, batch,
        model.nodes.ConstDeviceSpan(), out_preds->DeviceSpan().subspan(batch_offset),
        model.tree_segments.ConstDeviceSpan(), model.tree_group.ConstDeviceSpan(),
        model.split_types.ConstDeviceSpan(),
        model.categories_tree_segments.ConstDeviceSpan(),
        model.categories_node_segments.ConstDeviceSpan(),
        model.categories.ConstDeviceSpan(), model.tree_beg_, model.tree_end_,
        batch.NumFeatures(), num_rows, entry_start, use_shared,
        model.num_group, nan(""));
  }

  void DevicePredictInternal(DMatrix* dmat, HostDeviceVector<float>* out_preds,
                             const gbm::GBTreeModel& model, size_t tree_begin,
                             size_t tree_end) const {
    if (tree_end - tree_begin == 0) {
      return;
    }
    out_preds->SetDevice(ctx_->gpu_id);
    auto const& info = dmat->Info();
    DeviceModel d_model;
    d_model.Init(model, tree_begin, tree_end, ctx_->gpu_id);

    if (dmat->PageExists<SparsePage>()) {
      size_t batch_offset = 0;
      for (auto &batch : dmat->GetBatches<SparsePage>()) {
        this->PredictInternal(batch, d_model, model.learner_model_param->num_feature,
                              out_preds, batch_offset, dmat->IsDense());
        batch_offset += batch.Size() * model.learner_model_param->num_output_group;
      }
    } else {
      size_t batch_offset = 0;
      for (auto const& page : dmat->GetBatches<EllpackPage>(ctx_, BatchParam{})) {
        dmat->Info().feature_types.SetDevice(ctx_->gpu_id);
        auto feature_types = dmat->Info().feature_types.ConstDeviceSpan();
        this->PredictInternal(
            page.Impl()->GetDeviceAccessor(ctx_->gpu_id, feature_types),
            d_model,
            out_preds,
            batch_offset);
        batch_offset += page.Impl()->n_rows;
      }
    }
  }

 public:
  explicit GPUPredictor(Context const* ctx) : Predictor::Predictor{ctx} {}

  ~GPUPredictor() override {
    if (ctx_->gpu_id >= 0 && ctx_->gpu_id < common::AllVisibleGPUs()) {
      dh::safe_cuda(cudaSetDevice(ctx_->gpu_id));
    }
  }

  void PredictBatch(DMatrix* dmat, PredictionCacheEntry* predts,
                    const gbm::GBTreeModel& model, uint32_t tree_begin,
                    uint32_t tree_end = 0) const override {
    int device = ctx_->gpu_id;
    CHECK_GE(device, 0) << "Set `gpu_id' to positive value for processing GPU data.";
    auto* out_preds = &predts->predictions;
    if (tree_end == 0) {
      tree_end = model.trees.size();
    }
    this->DevicePredictInternal(dmat, out_preds, model, tree_begin, tree_end);
  }

  template <typename Adapter, typename Loader>
  void DispatchedInplacePredict(std::any const& x, std::shared_ptr<DMatrix> p_m,
                                const gbm::GBTreeModel& model, float missing,
                                PredictionCacheEntry* out_preds, uint32_t tree_begin,
                                uint32_t tree_end) const {
    uint32_t const output_groups =  model.learner_model_param->num_output_group;

    auto m = std::any_cast<std::shared_ptr<Adapter>>(x);
    CHECK_EQ(m->NumColumns(), model.learner_model_param->num_feature)
        << "Number of columns in data must equal to trained model.";
    CHECK_EQ(dh::CurrentDevice(), m->DeviceIdx())
        << "XGBoost is running on device: " << this->ctx_->gpu_id << ", "
        << "but data is on: " << m->DeviceIdx();
    if (p_m) {
      p_m->Info().num_row_ = m->NumRows();
      this->InitOutPredictions(p_m->Info(), &(out_preds->predictions), model);
    } else {
      MetaInfo info;
      info.num_row_ = m->NumRows();
      this->InitOutPredictions(info, &(out_preds->predictions), model);
    }
    out_preds->predictions.SetDevice(m->DeviceIdx());

    const uint32_t BLOCK_THREADS = 128;
    auto GRID_SIZE = static_cast<uint32_t>(common::DivRoundUp(m->NumRows(), BLOCK_THREADS));

    auto max_shared_memory_bytes = dh::MaxSharedMemory(m->DeviceIdx());
    size_t shared_memory_bytes =
        SharedMemoryBytes<BLOCK_THREADS>(m->NumColumns(), max_shared_memory_bytes);
    DeviceModel d_model;
    d_model.Init(model, tree_begin, tree_end, m->DeviceIdx());

    bool use_shared = shared_memory_bytes != 0;
    size_t entry_start = 0;

    dh::LaunchKernel {GRID_SIZE, BLOCK_THREADS, shared_memory_bytes} (
        PredictKernel<Loader, typename Loader::BatchT>, m->Value(),
        d_model.nodes.ConstDeviceSpan(), out_preds->predictions.DeviceSpan(),
        d_model.tree_segments.ConstDeviceSpan(), d_model.tree_group.ConstDeviceSpan(),
        d_model.split_types.ConstDeviceSpan(),
        d_model.categories_tree_segments.ConstDeviceSpan(),
        d_model.categories_node_segments.ConstDeviceSpan(),
        d_model.categories.ConstDeviceSpan(), tree_begin, tree_end, m->NumColumns(),
        m->NumRows(), entry_start, use_shared, output_groups, missing);
  }

  bool InplacePredict(std::shared_ptr<DMatrix> p_m, const gbm::GBTreeModel& model, float missing,
                      PredictionCacheEntry* out_preds, uint32_t tree_begin,
                      unsigned tree_end) const override {
    auto proxy = dynamic_cast<data::DMatrixProxy*>(p_m.get());
    CHECK(proxy)<< "Inplace predict accepts only DMatrixProxy as input.";
    auto x = proxy->Adapter();
    if (x.type() == typeid(std::shared_ptr<data::CupyAdapter>)) {
      this->DispatchedInplacePredict<data::CupyAdapter,
                                     DeviceAdapterLoader<data::CupyAdapterBatch>>(
          x, p_m, model, missing, out_preds, tree_begin, tree_end);
    } else if (x.type() == typeid(std::shared_ptr<data::CudfAdapter>)) {
      this->DispatchedInplacePredict<data::CudfAdapter,
                                     DeviceAdapterLoader<data::CudfAdapterBatch>>(
          x, p_m, model, missing, out_preds, tree_begin, tree_end);
    } else {
      return false;
    }
    return true;
  }

  void PredictContribution(DMatrix* p_fmat,
                           HostDeviceVector<bst_float>* out_contribs,
                           const gbm::GBTreeModel& model, unsigned tree_end,
                           std::vector<bst_float> const* tree_weights,
                           bool approximate, int,
                           unsigned) const override {
    std::string not_implemented{"contribution is not implemented in GPU "
                                "predictor, use `cpu_predictor` instead."};
    if (approximate) {
      LOG(FATAL) << "Approximated " << not_implemented;
    }
    if (tree_weights != nullptr) {
      LOG(FATAL) << "Dart booster feature " << not_implemented;
    }
    dh::safe_cuda(cudaSetDevice(ctx_->gpu_id));
    out_contribs->SetDevice(ctx_->gpu_id);
    if (tree_end == 0 || tree_end > model.trees.size()) {
      tree_end = static_cast<uint32_t>(model.trees.size());
    }

    const int ngroup = model.learner_model_param->num_output_group;
    CHECK_NE(ngroup, 0);
    // allocate space for (number of features + bias) times the number of rows
    size_t contributions_columns =
        model.learner_model_param->num_feature + 1;  // +1 for bias
    out_contribs->Resize(p_fmat->Info().num_row_ * contributions_columns *
                    model.learner_model_param->num_output_group);
    out_contribs->Fill(0.0f);
    auto phis = out_contribs->DeviceSpan();

    dh::device_vector<gpu_treeshap::PathElement<ShapSplitCondition>>
        device_paths;
    DeviceModel d_model;
    d_model.Init(model, 0, tree_end, ctx_->gpu_id);
    dh::device_vector<uint32_t> categories;
    ExtractPaths(&device_paths, &d_model, &categories, ctx_->gpu_id);
    for (auto& batch : p_fmat->GetBatches<SparsePage>()) {
      batch.data.SetDevice(ctx_->gpu_id);
      batch.offset.SetDevice(ctx_->gpu_id);
      SparsePageView X(batch.data.DeviceSpan(), batch.offset.DeviceSpan(),
                       model.learner_model_param->num_feature);
      auto begin = dh::tbegin(phis) + batch.base_rowid * contributions_columns;
      gpu_treeshap::GPUTreeShap<dh::XGBDeviceAllocator<int>>(
          X, device_paths.begin(), device_paths.end(), ngroup, begin,
          dh::tend(phis));
    }
    // Add the base margin term to last column
    p_fmat->Info().base_margin_.SetDevice(ctx_->gpu_id);
    const auto margin = p_fmat->Info().base_margin_.Data()->ConstDeviceSpan();

    auto base_score = model.learner_model_param->BaseScore(ctx_);
    dh::LaunchN(p_fmat->Info().num_row_ * model.learner_model_param->num_output_group,
                [=] __device__(size_t idx) {
                  phis[(idx + 1) * contributions_columns - 1] +=
                      margin.empty() ? base_score(0) : margin[idx];
                });
  }

  void PredictInteractionContributions(DMatrix* p_fmat,
                                       HostDeviceVector<bst_float>* out_contribs,
                                       const gbm::GBTreeModel& model,
                                       unsigned tree_end,
                                       std::vector<bst_float> const* tree_weights,
                                       bool approximate) const override {
    std::string not_implemented{"contribution is not implemented in GPU "
                                "predictor, use `cpu_predictor` instead."};
    if (approximate) {
      LOG(FATAL) << "Approximated " << not_implemented;
    }
    if (tree_weights != nullptr) {
      LOG(FATAL) << "Dart booster feature " << not_implemented;
    }
    dh::safe_cuda(cudaSetDevice(ctx_->gpu_id));
    out_contribs->SetDevice(ctx_->gpu_id);
    if (tree_end == 0 || tree_end > model.trees.size()) {
      tree_end = static_cast<uint32_t>(model.trees.size());
    }

    const int ngroup = model.learner_model_param->num_output_group;
    CHECK_NE(ngroup, 0);
    // allocate space for (number of features + bias) times the number of rows
    size_t contributions_columns =
        model.learner_model_param->num_feature + 1;  // +1 for bias
    out_contribs->Resize(p_fmat->Info().num_row_ * contributions_columns *
                         contributions_columns *
                         model.learner_model_param->num_output_group);
    out_contribs->Fill(0.0f);
    auto phis = out_contribs->DeviceSpan();

    dh::device_vector<gpu_treeshap::PathElement<ShapSplitCondition>>
        device_paths;
    DeviceModel d_model;
    d_model.Init(model, 0, tree_end, ctx_->gpu_id);
    dh::device_vector<uint32_t> categories;
    ExtractPaths(&device_paths, &d_model, &categories, ctx_->gpu_id);
    for (auto& batch : p_fmat->GetBatches<SparsePage>()) {
      batch.data.SetDevice(ctx_->gpu_id);
      batch.offset.SetDevice(ctx_->gpu_id);
      SparsePageView X(batch.data.DeviceSpan(), batch.offset.DeviceSpan(),
                       model.learner_model_param->num_feature);
      auto begin = dh::tbegin(phis) + batch.base_rowid * contributions_columns;
      gpu_treeshap::GPUTreeShapInteractions<dh::XGBDeviceAllocator<int>>(
          X, device_paths.begin(), device_paths.end(), ngroup, begin,
          dh::tend(phis));
    }
    // Add the base margin term to last column
    p_fmat->Info().base_margin_.SetDevice(ctx_->gpu_id);
    const auto margin = p_fmat->Info().base_margin_.Data()->ConstDeviceSpan();

    auto base_score = model.learner_model_param->BaseScore(ctx_);
    size_t n_features = model.learner_model_param->num_feature;
    dh::LaunchN(p_fmat->Info().num_row_ * model.learner_model_param->num_output_group,
                [=] __device__(size_t idx) {
                  size_t group = idx % ngroup;
                  size_t row_idx = idx / ngroup;
                  phis[gpu_treeshap::IndexPhiInteractions(row_idx, ngroup, group, n_features,
                                                          n_features, n_features)] +=
                      margin.empty() ? base_score(0) : margin[idx];
                });
  }

  void PredictInstance(const SparsePage::Inst&,
                       std::vector<bst_float>*,
                       const gbm::GBTreeModel&, unsigned) const override {
    LOG(FATAL) << "[Internal error]: " << __func__
               << " is not implemented in GPU Predictor.";
  }

  void PredictLeaf(DMatrix *p_fmat, HostDeviceVector<bst_float> *predictions,
                   const gbm::GBTreeModel &model,
                   unsigned tree_end) const override {
    dh::safe_cuda(cudaSetDevice(ctx_->gpu_id));
    auto max_shared_memory_bytes = ConfigureDevice(ctx_->gpu_id);

    const MetaInfo& info = p_fmat->Info();
    constexpr uint32_t kBlockThreads = 128;
    size_t shared_memory_bytes = SharedMemoryBytes<kBlockThreads>(
        info.num_col_, max_shared_memory_bytes);
    bool use_shared = shared_memory_bytes != 0;
    bst_feature_t num_features = info.num_col_;
    bst_row_t num_rows = info.num_row_;
    size_t entry_start = 0;

    if (tree_end == 0 || tree_end > model.trees.size()) {
      tree_end = static_cast<uint32_t>(model.trees.size());
    }
    predictions->SetDevice(ctx_->gpu_id);
    predictions->Resize(num_rows * tree_end);
    DeviceModel d_model;
    d_model.Init(model, 0, tree_end, this->ctx_->gpu_id);

    if (p_fmat->PageExists<SparsePage>()) {
      for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
        batch.data.SetDevice(ctx_->gpu_id);
        batch.offset.SetDevice(ctx_->gpu_id);
        bst_row_t batch_offset = 0;
        SparsePageView data{batch.data.DeviceSpan(), batch.offset.DeviceSpan(),
                            model.learner_model_param->num_feature};
        size_t num_rows = batch.Size();
        auto grid =
            static_cast<uint32_t>(common::DivRoundUp(num_rows, kBlockThreads));
        dh::LaunchKernel {grid, kBlockThreads, shared_memory_bytes} (
            PredictLeafKernel<SparsePageLoader, SparsePageView>, data,
            d_model.nodes.ConstDeviceSpan(),
            predictions->DeviceSpan().subspan(batch_offset),
            d_model.tree_segments.ConstDeviceSpan(),

            d_model.split_types.ConstDeviceSpan(),
            d_model.categories_tree_segments.ConstDeviceSpan(),
            d_model.categories_node_segments.ConstDeviceSpan(),
            d_model.categories.ConstDeviceSpan(),

            d_model.tree_beg_, d_model.tree_end_, num_features, num_rows,
            entry_start, use_shared, nan(""));
        batch_offset += batch.Size();
      }
    } else {
      for (auto const& batch : p_fmat->GetBatches<EllpackPage>(ctx_, BatchParam{})) {
        bst_row_t batch_offset = 0;
        EllpackDeviceAccessor data{batch.Impl()->GetDeviceAccessor(ctx_->gpu_id)};
        size_t num_rows = batch.Size();
        auto grid =
            static_cast<uint32_t>(common::DivRoundUp(num_rows, kBlockThreads));
        dh::LaunchKernel {grid, kBlockThreads, shared_memory_bytes} (
            PredictLeafKernel<EllpackLoader, EllpackDeviceAccessor>, data,
            d_model.nodes.ConstDeviceSpan(),
            predictions->DeviceSpan().subspan(batch_offset),
            d_model.tree_segments.ConstDeviceSpan(),

            d_model.split_types.ConstDeviceSpan(),
            d_model.categories_tree_segments.ConstDeviceSpan(),
            d_model.categories_node_segments.ConstDeviceSpan(),
            d_model.categories.ConstDeviceSpan(),

            d_model.tree_beg_, d_model.tree_end_, num_features, num_rows,
            entry_start, use_shared, nan(""));
        batch_offset += batch.Size();
      }
    }
  }

  void Configure(const std::vector<std::pair<std::string, std::string>>& cfg) override {
    Predictor::Configure(cfg);
  }

 private:
  /*! \brief Reconfigure the device when GPU is changed. */
  static size_t ConfigureDevice(int device) {
    if (device >= 0) {
      return dh::MaxSharedMemory(device);
    }
    return 0;
  }
};

XGBOOST_REGISTER_PREDICTOR(GPUPredictor, "gpu_predictor")
    .describe("Make predictions using GPU.")
    .set_body([](Context const* ctx) { return new GPUPredictor(ctx); });

}  // namespace xgboost::predictor
