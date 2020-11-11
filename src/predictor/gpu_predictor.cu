/*!
 * Copyright 2017-2020 by Contributors
 */
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <GPUTreeShap/gpu_treeshap.h>
#include <memory>

#include "xgboost/data.h"
#include "xgboost/predictor.h"
#include "xgboost/tree_model.h"
#include "xgboost/tree_updater.h"
#include "xgboost/host_device_vector.h"

#include "../gbm/gbtree_model.h"
#include "../data/ellpack_page.cuh"
#include "../data/device_adapter.cuh"
#include "../common/common.h"
#include "../common/bitfield.h"
#include "../common/categorical.h"
#include "../common/device_helpers.cuh"

namespace xgboost {
namespace predictor {

DMLC_REGISTRY_FILE_TAG(gpu_predictor);

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
                              bst_row_t num_rows, size_t entry_start)
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
  XGBOOST_DEVICE EllpackLoader(EllpackDeviceAccessor const& m, bool,
                               bst_feature_t, bst_row_t, size_t)
      : matrix{m} {}
  __device__ __forceinline__ float GetElement(size_t  ridx, size_t  fidx) const {
    auto gidx = matrix.GetBinIndex(ridx, fidx);
    if (gidx == -1) {
      return nan("");
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

  using BatchT = Batch;

  XGBOOST_DEV_INLINE DeviceAdapterLoader(Batch const batch, bool use_shared,
                                         bst_feature_t num_features, bst_row_t num_rows,
                                         size_t entry_start) :
    batch{batch},
    columns{num_features},
    use_shared{use_shared} {
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
            smem[threadIdx.x * num_features + (i - beg)] = batch.GetElement(i).value;
          }
        }
      }
      __syncthreads();
    }

  XGBOOST_DEV_INLINE  float GetElement(size_t  ridx, size_t  fidx) const {
    if (use_shared) {
      return smem[threadIdx.x * columns + fidx];
    }
    return batch.GetElement(ridx * columns + fidx).value;
  }
};

template <typename Loader>
__device__ float GetLeafWeight(bst_row_t ridx, const RegTree::Node* tree,
                               common::Span<FeatureType const> split_types,
                               common::Span<RegTree::Segment const> d_cat_ptrs,
                               common::Span<uint32_t const> d_categories,
                               Loader* loader) {
  bst_node_t nidx = 0;
  RegTree::Node n = tree[nidx];
  while (!n.IsLeaf()) {
    float fvalue = loader->GetElement(ridx, n.SplitIndex());
    // Missing value
    if (common::CheckNAN(fvalue)) {
      nidx = n.DefaultChild();
    } else {
      bool go_left = true;
      if (common::IsCat(split_types, nidx)) {
        auto categories = d_categories.subspan(d_cat_ptrs[nidx].beg,
                                               d_cat_ptrs[nidx].size);
        go_left = Decision(categories, common::AsCat(fvalue));
      } else {
        go_left = fvalue < n.SplitCond();
      }
      if (go_left) {
        nidx = n.LeftChild();
      } else {
        nidx = n.RightChild();
      }
    }
    n = tree[nidx];
  }
  return tree[nidx].LeafValue();
}

template <typename Loader>
__device__ bst_node_t GetLeafIndex(bst_row_t ridx, const RegTree::Node* tree,
                                   Loader const& loader) {
  bst_node_t nidx = 0;
  RegTree::Node n = tree[nidx];
  while (!n.IsLeaf()) {
    float fvalue = loader.GetElement(ridx, n.SplitIndex());
    // Missing value
    if (isnan(fvalue)) {
      nidx = n.DefaultChild();
      n = tree[nidx];
    } else {
      if (fvalue < n.SplitCond()) {
        nidx = n.LeftChild();
        n = tree[nidx];
      } else {
        nidx = n.RightChild();
        n = tree[nidx];
      }
    }
  }
  return nidx;
}

template <typename Loader, typename Data>
__global__ void PredictLeafKernel(Data data,
                                  common::Span<const RegTree::Node> d_nodes,
                                  common::Span<float> d_out_predictions,
                                  common::Span<size_t const> d_tree_segments,
                                  size_t tree_begin, size_t tree_end, size_t num_features,
                                  size_t num_rows, size_t entry_start, bool use_shared) {
  bst_row_t ridx = blockDim.x * blockIdx.x + threadIdx.x;
  if (ridx >= num_rows) {
    return;
  }
  Loader loader(data, use_shared, num_features, num_rows, entry_start);
  for (int tree_idx = tree_begin; tree_idx < tree_end; ++tree_idx) {
    const RegTree::Node* d_tree = &d_nodes[d_tree_segments[tree_idx - tree_begin]];
    auto leaf = GetLeafIndex(ridx, d_tree, loader);
    d_out_predictions[ridx * (tree_end - tree_begin) + tree_idx] = leaf;
  }
}

template <typename Loader, typename Data>
__global__ void
PredictKernel(Data data, common::Span<const RegTree::Node> d_nodes,
              common::Span<float> d_out_predictions,
              common::Span<size_t const> d_tree_segments,
              common::Span<int const> d_tree_group,
              common::Span<FeatureType const> d_tree_split_types,
              common::Span<uint32_t const> d_cat_tree_segments,
              common::Span<RegTree::Segment const> d_cat_node_segments,
              common::Span<uint32_t const> d_categories, size_t tree_begin,
              size_t tree_end, size_t num_features, size_t num_rows,
              size_t entry_start, bool use_shared, int num_group) {
  bst_uint global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  Loader loader(data, use_shared, num_features, num_rows, entry_start);
  if (global_idx >= num_rows) return;
  if (num_group == 1) {
    float sum = 0;
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      const RegTree::Node* d_tree =
          &d_nodes[d_tree_segments[tree_idx - tree_begin]];
      auto tree_cat_ptrs = d_cat_node_segments.subspan(
          d_tree_segments[tree_idx - tree_begin],
          d_tree_segments[tree_idx - tree_begin + 1] -
              d_tree_segments[tree_idx - tree_begin]);
      auto tree_categories =
          d_categories.subspan(d_cat_tree_segments[tree_idx - tree_begin],
                               d_cat_tree_segments[tree_idx - tree_begin + 1] -
                               d_cat_tree_segments[tree_idx - tree_begin]);
      float leaf = GetLeafWeight(global_idx, d_tree, d_tree_split_types,
                                 tree_cat_ptrs,
                                 tree_categories,
                                 &loader);
      sum += leaf;
    }
    d_out_predictions[global_idx] += sum;
  } else {
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      int tree_group = d_tree_group[tree_idx];
      const RegTree::Node* d_tree =
          &d_nodes[d_tree_segments[tree_idx - tree_begin]];
      bst_uint out_prediction_idx = global_idx * num_group + tree_group;
      auto tree_cat_ptrs = d_cat_node_segments.subspan(
          d_tree_segments[tree_idx - tree_begin],
          d_tree_segments[tree_idx - tree_begin + 1] -
              d_tree_segments[tree_idx - tree_begin]);
      auto tree_categories =
          d_categories.subspan(d_cat_tree_segments[tree_idx - tree_begin],
                               d_cat_tree_segments[tree_idx - tree_begin + 1] -
                               d_cat_tree_segments[tree_idx - tree_begin]);
      d_out_predictions[out_prediction_idx] +=
          GetLeafWeight(global_idx, d_tree, d_tree_split_types,
                        tree_cat_ptrs,
                        tree_categories,
                        &loader);
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
  HostDeviceVector<RegTree::Segment> categories_node_segments;
  HostDeviceVector<uint32_t> categories;

  size_t tree_beg_;  // NOLINT
  size_t tree_end_;  // NOLINT
  int num_group;

  void Init(const gbm::GBTreeModel& model, size_t tree_begin, size_t tree_end, int32_t gpu_id) {
    dh::safe_cuda(cudaSetDevice(gpu_id));

    CHECK_EQ(model.param.size_leaf_vector, 0);
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

    categories_node_segments =
        HostDeviceVector<RegTree::Segment>(h_tree_segments.back(), {}, gpu_id);
    std::vector<RegTree::Segment> &h_categories_node_segments =
        categories_node_segments.HostVector();
    for (auto tree_idx = tree_begin; tree_idx < tree_end; ++tree_idx) {
      auto const &src_cats_ptr = model.trees.at(tree_idx)->GetSplitCategoriesPtr();
      std::copy(src_cats_ptr.cbegin(), src_cats_ptr.cend(),
                h_categories_node_segments.begin() +
                    h_tree_segments[tree_idx - tree_begin]);
    }

    this->tree_beg_ = tree_begin;
    this->tree_end_ = tree_end;
    this->num_group = model.learner_model_param->num_output_group;
  }
};

struct PathInfo {
  int64_t leaf_position;  // -1 not a leaf
  size_t length;
  size_t tree_idx;
};

// Transform model into path element form for GPUTreeShap
void ExtractPaths(dh::device_vector<gpu_treeshap::PathElement>* paths,
                  const gbm::GBTreeModel& model, size_t tree_limit,
                  int gpu_id) {
  DeviceModel device_model;
  device_model.Init(model, 0, tree_limit, gpu_id);
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
        return PathInfo{int64_t(idx), path_length, tree_idx};
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

  auto d_paths = paths->data().get();
  auto d_info = info.data().get();
  auto d_stats = device_model.stats.ConstDeviceSpan();
  auto d_tree_group = device_model.tree_group.ConstDeviceSpan();
  auto d_path_segments = path_segments.data().get();
  dh::LaunchN(gpu_id, info.size(), [=] __device__(size_t idx) {
    auto path_info = d_info[idx];
    size_t tree_offset = d_tree_segments[path_info.tree_idx];
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
      auto parent = d_nodes[parent_idx];
      bool is_left_path = (tree_offset + parent.LeftChild()) == child_idx;
      bool is_missing_path = (!parent.DefaultLeft() && !is_left_path) ||
                             (parent.DefaultLeft() && is_left_path);
      float lower_bound = is_left_path ? -inf : parent.SplitCond();
      float upper_bound = is_left_path ? parent.SplitCond() : inf;
      d_paths[output_position--] = {
          idx,         parent.SplitIndex(), group,         lower_bound,
          upper_bound, is_missing_path,     zero_fraction, v};
      child_idx = parent_idx;
      child = parent;
    }
    // Root node has feature -1
    d_paths[output_position] = {idx, -1, group, -inf, inf, false, 1.0, v};
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
                       size_t num_features,
                       HostDeviceVector<bst_float>* predictions,
                       size_t batch_offset) {
    batch.offset.SetDevice(generic_param_->gpu_id);
    batch.data.SetDevice(generic_param_->gpu_id);
    const uint32_t BLOCK_THREADS = 128;
    size_t num_rows = batch.Size();
    auto GRID_SIZE = static_cast<uint32_t>(common::DivRoundUp(num_rows, BLOCK_THREADS));

    size_t shared_memory_bytes =
        SharedMemoryBytes<BLOCK_THREADS>(num_features, max_shared_memory_bytes_);
    bool use_shared = shared_memory_bytes != 0;

    size_t entry_start = 0;
    SparsePageView data(batch.data.DeviceSpan(), batch.offset.DeviceSpan(),
                        num_features);
    dh::LaunchKernel {GRID_SIZE, BLOCK_THREADS, shared_memory_bytes} (
        PredictKernel<SparsePageLoader, SparsePageView>, data,
        model_.nodes.ConstDeviceSpan(),
        predictions->DeviceSpan().subspan(batch_offset),
        model_.tree_segments.ConstDeviceSpan(), model_.tree_group.ConstDeviceSpan(),
        model_.split_types.ConstDeviceSpan(),
        model_.categories_tree_segments.ConstDeviceSpan(),
        model_.categories_node_segments.ConstDeviceSpan(),
        model_.categories.ConstDeviceSpan(), model_.tree_beg_, model_.tree_end_,
        num_features, num_rows, entry_start, use_shared, model_.num_group);
  }
  void PredictInternal(EllpackDeviceAccessor const& batch,
                       HostDeviceVector<bst_float>* out_preds,
                       size_t batch_offset) {
    const uint32_t BLOCK_THREADS = 256;
    size_t num_rows = batch.n_rows;
    auto GRID_SIZE = static_cast<uint32_t>(common::DivRoundUp(num_rows, BLOCK_THREADS));

    bool use_shared = false;
    size_t entry_start = 0;
    dh::LaunchKernel {GRID_SIZE, BLOCK_THREADS} (
        PredictKernel<EllpackLoader, EllpackDeviceAccessor>, batch,
        model_.nodes.ConstDeviceSpan(), out_preds->DeviceSpan().subspan(batch_offset),
        model_.tree_segments.ConstDeviceSpan(), model_.tree_group.ConstDeviceSpan(),
        model_.split_types.ConstDeviceSpan(),
        model_.categories_tree_segments.ConstDeviceSpan(),
        model_.categories_node_segments.ConstDeviceSpan(),
        model_.categories.ConstDeviceSpan(), model_.tree_beg_, model_.tree_end_,
        batch.NumFeatures(), num_rows, entry_start, use_shared,
        model_.num_group);
  }

  void DevicePredictInternal(DMatrix* dmat, HostDeviceVector<float>* out_preds,
                             const gbm::GBTreeModel& model, size_t tree_begin,
                             size_t tree_end) {
    dh::safe_cuda(cudaSetDevice(generic_param_->gpu_id));
    if (tree_end - tree_begin == 0) {
      return;
    }
    model_.Init(model, tree_begin, tree_end, generic_param_->gpu_id);
    out_preds->SetDevice(generic_param_->gpu_id);
    auto const& info = dmat->Info();

    if (dmat->PageExists<SparsePage>()) {
      size_t batch_offset = 0;
      for (auto &batch : dmat->GetBatches<SparsePage>()) {
        this->PredictInternal(batch, model.learner_model_param->num_feature,
                              out_preds, batch_offset);
        batch_offset += batch.Size() * model.learner_model_param->num_output_group;
      }
    } else {
      size_t batch_offset = 0;
      for (auto const& page : dmat->GetBatches<EllpackPage>()) {
        this->PredictInternal(
            page.Impl()->GetDeviceAccessor(generic_param_->gpu_id),
            out_preds,
            batch_offset);
        batch_offset += page.Impl()->n_rows;
      }
    }
  }

 public:
  explicit GPUPredictor(GenericParameter const* generic_param) :
      Predictor::Predictor{generic_param} {}

  ~GPUPredictor() override {
    if (generic_param_->gpu_id >= 0) {
      dh::safe_cuda(cudaSetDevice(generic_param_->gpu_id));
    }
  }

  void PredictBatch(DMatrix* dmat, PredictionCacheEntry* predts,
                    const gbm::GBTreeModel& model, int tree_begin,
                    unsigned ntree_limit = 0) override {
    // This function is duplicated with CPU predictor PredictBatch, see comments in there.
    // FIXME(trivialfis): Remove the duplication.
    std::lock_guard<std::mutex> const guard(lock_);
    int device = generic_param_->gpu_id;
    CHECK_GE(device, 0) << "Set `gpu_id' to positive value for processing GPU data.";
    ConfigureDevice(device);

    CHECK_EQ(tree_begin, 0);
    auto* out_preds = &predts->predictions;
    CHECK_GE(predts->version, tree_begin);
    if (out_preds->Size() == 0 && dmat->Info().num_row_ != 0) {
      CHECK_EQ(predts->version, 0);
    }
    if (predts->version == 0) {
      this->InitOutPredictions(dmat->Info(), out_preds, model);
    }

    uint32_t const output_groups =  model.learner_model_param->num_output_group;
    CHECK_NE(output_groups, 0);

    uint32_t real_ntree_limit = ntree_limit * output_groups;
    if (real_ntree_limit == 0 || real_ntree_limit > model.trees.size()) {
      real_ntree_limit = static_cast<uint32_t>(model.trees.size());
    }

    uint32_t const end_version = (tree_begin + real_ntree_limit) / output_groups;

    if (predts->version > end_version) {
      CHECK_NE(ntree_limit, 0);
      this->InitOutPredictions(dmat->Info(), out_preds, model);
      predts->version = 0;
    }
    uint32_t const beg_version = predts->version;
    CHECK_LE(beg_version, end_version);

    if (beg_version < end_version) {
      this->DevicePredictInternal(dmat, out_preds, model,
                                  beg_version * output_groups,
                                  end_version * output_groups);
    }

    uint32_t delta = end_version - beg_version;
    CHECK_LE(delta, model.trees.size());
    predts->Update(delta);

    CHECK(out_preds->Size() == output_groups * dmat->Info().num_row_ ||
          out_preds->Size() == dmat->Info().num_row_);
  }

  template <typename Adapter, typename Loader>
  void DispatchedInplacePredict(dmlc::any const &x,
                                const gbm::GBTreeModel &model, float,
                                PredictionCacheEntry *out_preds,
                                uint32_t tree_begin, uint32_t tree_end) const {
    auto max_shared_memory_bytes = dh::MaxSharedMemory(this->generic_param_->gpu_id);
    uint32_t const output_groups =  model.learner_model_param->num_output_group;
    DeviceModel d_model;
    d_model.Init(model, tree_begin, tree_end, this->generic_param_->gpu_id);

    auto m = dmlc::get<std::shared_ptr<Adapter>>(x);
    CHECK_EQ(m->NumColumns(), model.learner_model_param->num_feature)
        << "Number of columns in data must equal to trained model.";
    CHECK_EQ(this->generic_param_->gpu_id, m->DeviceIdx())
        << "XGBoost is running on device: " << this->generic_param_->gpu_id << ", "
        << "but data is on: " << m->DeviceIdx();
    MetaInfo info;
    info.num_col_ = m->NumColumns();
    info.num_row_ = m->NumRows();
    this->InitOutPredictions(info, &(out_preds->predictions), model);

    const uint32_t BLOCK_THREADS = 128;
    auto GRID_SIZE = static_cast<uint32_t>(common::DivRoundUp(info.num_row_, BLOCK_THREADS));

    size_t shared_memory_bytes =
        SharedMemoryBytes<BLOCK_THREADS>(info.num_col_, max_shared_memory_bytes);
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
        info.num_row_, entry_start, use_shared, output_groups);
  }

  void InplacePredict(dmlc::any const &x, const gbm::GBTreeModel &model,
                      float missing, PredictionCacheEntry *out_preds,
                      uint32_t tree_begin, unsigned tree_end) const override {
    if (x.type() == typeid(std::shared_ptr<data::CupyAdapter>)) {
      this->DispatchedInplacePredict<
          data::CupyAdapter, DeviceAdapterLoader<data::CupyAdapterBatch>>(
          x, model, missing, out_preds, tree_begin, tree_end);
    } else if (x.type() == typeid(std::shared_ptr<data::CudfAdapter>)) {
      this->DispatchedInplacePredict<
          data::CudfAdapter, DeviceAdapterLoader<data::CudfAdapterBatch>>(
          x, model, missing, out_preds, tree_begin, tree_end);
    } else {
      LOG(FATAL) << "Only CuPy and CuDF are supported by GPU Predictor.";
    }
  }

  void PredictContribution(DMatrix* p_fmat,
                           HostDeviceVector<bst_float>* out_contribs,
                           const gbm::GBTreeModel& model, unsigned ntree_limit,
                           std::vector<bst_float>*,
                           bool approximate, int,
                           unsigned) override {
    if (approximate) {
      LOG(FATAL) << "Approximated contribution is not implemented in GPU Predictor.";
    }

    dh::safe_cuda(cudaSetDevice(generic_param_->gpu_id));
    out_contribs->SetDevice(generic_param_->gpu_id);
    uint32_t real_ntree_limit =
        ntree_limit * model.learner_model_param->num_output_group;
    if (real_ntree_limit == 0 || real_ntree_limit > model.trees.size()) {
      real_ntree_limit = static_cast<uint32_t>(model.trees.size());
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

    dh::device_vector<gpu_treeshap::PathElement> device_paths;
    ExtractPaths(&device_paths, model, real_ntree_limit,
                 generic_param_->gpu_id);
    for (auto& batch : p_fmat->GetBatches<SparsePage>()) {
      batch.data.SetDevice(generic_param_->gpu_id);
      batch.offset.SetDevice(generic_param_->gpu_id);
      SparsePageView X(batch.data.DeviceSpan(), batch.offset.DeviceSpan(),
                       model.learner_model_param->num_feature);
      gpu_treeshap::GPUTreeShap(
          X, device_paths.begin(), device_paths.end(), ngroup,
          phis.data() + batch.base_rowid * contributions_columns, phis.size());
    }
    // Add the base margin term to last column
    p_fmat->Info().base_margin_.SetDevice(generic_param_->gpu_id);
    const auto margin = p_fmat->Info().base_margin_.ConstDeviceSpan();
    float base_score = model.learner_model_param->base_score;
    dh::LaunchN(
        generic_param_->gpu_id,
        p_fmat->Info().num_row_ * model.learner_model_param->num_output_group,
        [=] __device__(size_t idx) {
          phis[(idx + 1) * contributions_columns - 1] +=
              margin.empty() ? base_score : margin[idx];
        });
  }

  void PredictInteractionContributions(DMatrix* p_fmat,
                                       HostDeviceVector<bst_float>* out_contribs,
                                       const gbm::GBTreeModel& model,
                                       unsigned ntree_limit,
                                       std::vector<bst_float>*,
                                       bool approximate) override {
    if (approximate) {
      LOG(FATAL) << "[Internal error]: " << __func__
                 << " approximate is not implemented in GPU Predictor.";
    }

    dh::safe_cuda(cudaSetDevice(generic_param_->gpu_id));
    out_contribs->SetDevice(generic_param_->gpu_id);
    uint32_t real_ntree_limit =
        ntree_limit * model.learner_model_param->num_output_group;
    if (real_ntree_limit == 0 || real_ntree_limit > model.trees.size()) {
      real_ntree_limit = static_cast<uint32_t>(model.trees.size());
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

    dh::device_vector<gpu_treeshap::PathElement> device_paths;
    ExtractPaths(&device_paths, model, real_ntree_limit,
                 generic_param_->gpu_id);
    for (auto& batch : p_fmat->GetBatches<SparsePage>()) {
      batch.data.SetDevice(generic_param_->gpu_id);
      batch.offset.SetDevice(generic_param_->gpu_id);
      SparsePageView X(batch.data.DeviceSpan(), batch.offset.DeviceSpan(),
                       model.learner_model_param->num_feature);
      gpu_treeshap::GPUTreeShapInteractions(
          X, device_paths.begin(), device_paths.end(), ngroup,
          phis.data() + batch.base_rowid * contributions_columns, phis.size());
    }
    // Add the base margin term to last column
    p_fmat->Info().base_margin_.SetDevice(generic_param_->gpu_id);
    const auto margin = p_fmat->Info().base_margin_.ConstDeviceSpan();
    float base_score = model.learner_model_param->base_score;
    size_t n_features = model.learner_model_param->num_feature;
    dh::LaunchN(
        generic_param_->gpu_id,
        p_fmat->Info().num_row_ * model.learner_model_param->num_output_group,
        [=] __device__(size_t idx) {
          size_t group = idx % ngroup;
          size_t row_idx = idx / ngroup;
          phis[gpu_treeshap::IndexPhiInteractions(
              row_idx, ngroup, group, n_features, n_features, n_features)] +=
              margin.empty() ? base_score : margin[idx];
        });
  }

 protected:
  void InitOutPredictions(const MetaInfo& info,
                          HostDeviceVector<bst_float>* out_preds,
                          const gbm::GBTreeModel& model) const {
    size_t n_classes = model.learner_model_param->num_output_group;
    size_t n = n_classes * info.num_row_;
    const HostDeviceVector<bst_float>& base_margin = info.base_margin_;
    out_preds->SetDevice(generic_param_->gpu_id);
    out_preds->Resize(n);
    if (base_margin.Size() != 0) {
      CHECK_EQ(base_margin.Size(), n);
      out_preds->Copy(base_margin);
    } else {
      out_preds->Fill(model.learner_model_param->base_score);
    }
  }

  void PredictInstance(const SparsePage::Inst&,
                       std::vector<bst_float>*,
                       const gbm::GBTreeModel&, unsigned) override {
    LOG(FATAL) << "[Internal error]: " << __func__
               << " is not implemented in GPU Predictor.";
  }

  void PredictLeaf(DMatrix* p_fmat, HostDeviceVector<bst_float>* predictions,
                   const gbm::GBTreeModel& model,
                   unsigned ntree_limit) override {
    dh::safe_cuda(cudaSetDevice(generic_param_->gpu_id));
    ConfigureDevice(generic_param_->gpu_id);

    const MetaInfo& info = p_fmat->Info();
    constexpr uint32_t kBlockThreads = 128;
    size_t shared_memory_bytes =
        SharedMemoryBytes<kBlockThreads>(info.num_col_, max_shared_memory_bytes_);
    bool use_shared = shared_memory_bytes != 0;
    bst_feature_t num_features = info.num_col_;
    bst_row_t num_rows = info.num_row_;
    size_t entry_start = 0;

    uint32_t real_ntree_limit = ntree_limit * model.learner_model_param->num_output_group;
    if (real_ntree_limit == 0 || real_ntree_limit > model.trees.size()) {
      real_ntree_limit = static_cast<uint32_t>(model.trees.size());
    }
    predictions->SetDevice(generic_param_->gpu_id);
    predictions->Resize(num_rows * real_ntree_limit);
    model_.Init(model, 0, real_ntree_limit, generic_param_->gpu_id);

    if (p_fmat->PageExists<SparsePage>()) {
      for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
        batch.data.SetDevice(generic_param_->gpu_id);
        batch.offset.SetDevice(generic_param_->gpu_id);
        bst_row_t batch_offset = 0;
        SparsePageView data{batch.data.DeviceSpan(), batch.offset.DeviceSpan(),
                            model.learner_model_param->num_feature};
        size_t num_rows = batch.Size();
        auto grid =
            static_cast<uint32_t>(common::DivRoundUp(num_rows, kBlockThreads));
        dh::LaunchKernel {grid, kBlockThreads, shared_memory_bytes} (
            PredictLeafKernel<SparsePageLoader, SparsePageView>, data,
            model_.nodes.ConstDeviceSpan(),
            predictions->DeviceSpan().subspan(batch_offset),
            model_.tree_segments.ConstDeviceSpan(),
            model_.tree_beg_, model_.tree_end_, num_features, num_rows,
            entry_start, use_shared);
        batch_offset += batch.Size();
      }
    } else {
      for (auto const& batch : p_fmat->GetBatches<EllpackPage>()) {
        bst_row_t batch_offset = 0;
        EllpackDeviceAccessor data{batch.Impl()->GetDeviceAccessor(generic_param_->gpu_id)};
        size_t num_rows = batch.Size();
        auto grid =
            static_cast<uint32_t>(common::DivRoundUp(num_rows, kBlockThreads));
        dh::LaunchKernel {grid, kBlockThreads, shared_memory_bytes} (
            PredictLeafKernel<EllpackLoader, EllpackDeviceAccessor>, data,
            model_.nodes.ConstDeviceSpan(),
            predictions->DeviceSpan().subspan(batch_offset),
            model_.tree_segments.ConstDeviceSpan(),
            model_.tree_beg_, model_.tree_end_, num_features, num_rows,
            entry_start, use_shared);
        batch_offset += batch.Size();
      }
    }
  }

  void Configure(const std::vector<std::pair<std::string, std::string>>& cfg) override {
    Predictor::Configure(cfg);
  }

 private:
  /*! \brief Reconfigure the device when GPU is changed. */
  void ConfigureDevice(int device) {
    if (device >= 0) {
      max_shared_memory_bytes_ = dh::MaxSharedMemory(device);
    }
  }

  std::mutex lock_;
  DeviceModel model_;
  size_t max_shared_memory_bytes_ { 0 };
};

XGBOOST_REGISTER_PREDICTOR(GPUPredictor, "gpu_predictor")
.describe("Make predictions using GPU.")
.set_body([](GenericParameter const* generic_param) {
            return new GPUPredictor(generic_param);
          });

}  // namespace predictor
}  // namespace xgboost
