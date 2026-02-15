/**
 * Copyright 2017-2026, XGBoost Contributors
 */
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
#include "gpu_data_accessor.cuh"
#include "interpretability/shap.h"
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
      // clang-format off
      dh::LaunchKernel {grid, kBlockThreads, shared_memory_bytes,
                        ctx_->CUDACtx()->Stream()}(
          // clang-format on
          MaskBitVectorKernel, data, d_model.Trees(), decision_bits, missing_bits,
          d_model.tree_begin, d_model.tree_end, num_features, num_nodes, use_shared,
          std::numeric_limits<float>::quiet_NaN());

      AllReduceBitVectors(&decision_storage, &missing_storage);

      // clang-format off
      dh::LaunchKernel {grid, kBlockThreads, 0,
                        ctx_->CUDACtx()->Stream()}(
          // clang-format on
          PredictByBitVectorKernel<predict_leaf>, d_model.Trees(),
          out_preds->DeviceSpan().subspan(batch_offset), d_tree_groups, decision_bits, missing_bits,
          d_model.tree_begin, d_model.tree_end, num_rows, num_nodes, num_group);

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
  LaunchConfig(Context const* ctx, bst_feature_t n_features) : ctx_{ctx}, n_features_{n_features} {}

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

    DeviceModel d_model{this->ctx_->Device(), model, false, tree_begin, tree_end, &this->model_mu_,
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

    DeviceModel d_model{ctx_->Device(),       model, false, tree_begin, tree_end, &this->model_mu_,
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

    LaunchPredict(this->ctx_, false, enc::DeviceColumnsView{}, model, [&](auto&& cfg, auto&& acc) {
      using EncAccessor = std::remove_reference_t<decltype(acc)>;
      CHECK((std::is_same_v<EncAccessor, NoOpAccessor>));
      using LoaderImpl = DeviceAdapterLoader<BatchT, EncAccessor>;
      using Loader =
          typename common::GetValueT<decltype(cfg)>::template LoaderType<LoaderImpl, 128>;
      cfg.template AllocShmem<Loader>();
      cfg.template LaunchPredictKernel<Loader>(m->Value(), missing, n_features, d_model, acc, 0,
                                               &out_preds->predictions);
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
                           common::Span<float const> tree_weights, bool approximate, int,
                           unsigned) const override {
    xgboost_NVTX_FN_RANGE();
    if (approximate) {
      LOG(FATAL) << "Approximated contribution is not implemented in the GPU predictor, use CPU "
                    "instead.";
    }
    interpretability::ShapValues(ctx_, p_fmat, out_contribs, model, tree_end, tree_weights, 0, 0);
  }

  void PredictInteractionContributions(DMatrix* p_fmat, HostDeviceVector<float>* out_contribs,
                                       gbm::GBTreeModel const& model, bst_tree_t tree_end,
                                       common::Span<float const> tree_weights,
                                       bool approximate) const override {
    xgboost_NVTX_FN_RANGE();
    if (approximate) {
      LOG(FATAL) << "Approximated contribution is not implemented in GPU predictor, use cpu "
                    "instead.";
    }
    interpretability::ShapInteractionValues(ctx_, p_fmat, out_contribs, model, tree_end,
                                            tree_weights, approximate);
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

    DeviceModel d_model{ctx_->Device(),       model, false, 0, tree_end, &this->model_mu_,
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
