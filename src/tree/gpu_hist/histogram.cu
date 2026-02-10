/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <thrust/copy.h>                         // for copy_n
#include <thrust/iterator/transform_iterator.h>  // for make_transform_iterator

#include <algorithm>
#include <cstdint>          // uint32_t, int32_t
#include <cuda/functional>  // for proclaim_copyable_arguments
#include <memory>           // for unique_ptr

#include "../../collective/aggregator.h"
#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../common/cuda_rt_utils.h"   // for GetMpCnt
#include "../../common/deterministic.cuh"
#include "../../common/device_helpers.cuh"
#include "../../common/linalg_op.cuh"  // for tbegin
#include "../../data/ellpack_page.cuh"
#include "histogram.cuh"
#include "quantiser.cuh"
#include "row_partitioner.cuh"
#include "xgboost/base.h"

namespace xgboost::tree {
namespace {
struct Pair {
  GradientPairPrecise first;
  GradientPairPrecise second;
};
__host__ XGBOOST_DEV_INLINE Pair operator+(Pair const& lhs, Pair const& rhs) {
  return {lhs.first + rhs.first, lhs.second + rhs.second};
}

template <typename IterT>
XGBOOST_DEV_INLINE bst_idx_t IterIdx(EllpackAccessorImpl<IterT> const& matrix,
                                     RowPartitioner::RowIndexT ridx, bst_feature_t fidx) {
  // # Row index local to each batch
  // ridx_local = ridx - base_rowid
  // # Starting entry index for this row in the matrix
  // entry_idx = ridx_local * row_stride
  // # Inside a row, first column inside this feature group
  // entry_idx += start_feature
  // # The feature index local to the current feature group
  // idx - ridx * feature_stride == idx % feature_stride
  // # Final index
  // entry_idx += idx % feature_stride
  return (ridx - matrix.base_rowid) * matrix.row_stride + fidx;
}
}  // anonymous namespace

struct Clip {
  static XGBOOST_DEV_INLINE float Pclip(float v) { return v > 0 ? v : 0; }
  static XGBOOST_DEV_INLINE float Nclip(float v) { return v < 0 ? abs(v) : 0; }

  XGBOOST_DEV_INLINE Pair operator()(GradientPair x) const {
    auto pg = Pclip(x.GetGrad());
    auto ph = Pclip(x.GetHess());

    auto ng = Nclip(x.GetGrad());
    auto nh = Nclip(x.GetHess());

    return {GradientPairPrecise{pg, ph}, GradientPairPrecise{ng, nh}};
  }
};

/**
 * In algorithm 5 (see common::CreateRoundingFactor) the bound is calculated as
 * $max(|v_i|) * n$.  Here we use the bound:
 *
 * \begin{equation}
 *   max( fl(\sum^{V}_{v_i>0}{v_i}), fl(\sum^{V}_{v_i<0}|v_i|) )
 * \end{equation}
 *
 * to avoid outliers, as the full reduction is reproducible on GPU with reduction tree.
 */
GradientQuantiser::GradientQuantiser(Context const* ctx,
                                     linalg::VectorView<GradientPair const> gpair,
                                     MetaInfo const& info) {
  using GradientSumT = GradientPairPrecise;
  using T = typename GradientSumT::ValueT;

  auto beg = thrust::make_transform_iterator(linalg::tcbegin(gpair), Clip());
  Pair p =
      dh::Reduce(ctx->CUDACtx()->CTP(), beg, beg + gpair.Size(), Pair{}, cuda::std::plus<Pair>{});
  // Treat pair as array of 4 primitive types to allreduce
  using ReduceT = typename decltype(p.first)::ValueT;
  static_assert(sizeof(Pair) == sizeof(ReduceT) * 4, "Expected to reduce four elements.");
  auto rc = collective::GlobalSum(ctx, info, linalg::MakeVec(reinterpret_cast<ReduceT*>(&p), 4));
  collective::SafeColl(rc);

  GradientSumT positive_sum{p.first}, negative_sum{p.second};

  std::size_t total_rows = gpair.Size();
  rc = collective::GlobalSum(ctx, info, linalg::MakeVec(&total_rows, 1));
  collective::SafeColl(rc);

  auto histogram_rounding =
      GradientSumT{common::CreateRoundingFactor<T>(
                       std::max(positive_sum.GetGrad(), negative_sum.GetGrad()), total_rows),
                   common::CreateRoundingFactor<T>(
                       std::max(positive_sum.GetHess(), negative_sum.GetHess()), total_rows)};

  using IntT = typename GradientPairInt64::ValueT;

  /**
   * Factor for converting gradients from fixed-point to floating-point.
   */
  to_floating_point_ =
      histogram_rounding /
      static_cast<T>(static_cast<IntT>(1)
                     << (sizeof(typename GradientSumT::ValueT) * 8 - 2));  // keep 1 for sign bit
  /**
   * Factor for converting gradients from floating-point to fixed-point. For
   * f64:
   *
   *   Precision = 64 - 1 - log2(rounding)
   *
   * rounding is calcuated as exp(m), see the rounding factor calcuation for
   * details.
   */
  to_fixed_point_ = GradientSumT{static_cast<T>(1) / to_floating_point_.GetGrad(),
                                 static_cast<T>(1) / to_floating_point_.GetHess()};
}

MultiGradientQuantiser::MultiGradientQuantiser(Context const* ctx,
                                               linalg::MatrixView<GradientPair const> gpair,
                                               MetaInfo const& info) {
  std::vector<GradientQuantiser> h_quantizers;
  // TODO(jiamingy): We need to merge this into a single call for improved distributed training.
  for (bst_target_t t = 0, n_targets = gpair.Shape(1); t < n_targets; ++t) {
    h_quantizers.emplace_back(ctx, gpair.Slice(linalg::All(), t), info);
  }
  this->quantizers_ = h_quantizers;
}

void CalcQuantizedGpairs(Context const* ctx, linalg::MatrixView<GradientPair const> gpairs,
                         common::Span<GradientQuantiser const> roundings,
                         linalg::Matrix<GradientPairInt64>* p_out) {
  auto shape = gpairs.Shape();
  if (p_out->Empty()) {
    *p_out = linalg::Matrix<GradientPairInt64>{shape, ctx->Device(), linalg::kF};
  } else {
    p_out->Reshape(shape);
  }

  auto out_gpair = p_out->View(ctx->Device());
  CHECK(out_gpair.FContiguous());
  auto it = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) {
    auto [ridx, target_idx] = linalg::UnravelIndex(i, gpairs.Shape());
    auto g = gpairs(ridx, target_idx);
    return roundings[target_idx].ToFixedPoint(g);
  });
  thrust::copy_n(ctx->CUDACtx()->CTP(), it, gpairs.Size(), linalg::tbegin(out_gpair));
}

XGBOOST_DEV_INLINE void AtomicAddGpairShared(xgboost::GradientPairInt64* dest,
                                             xgboost::GradientPairInt64 const& gpair) {
  auto dst_ptr = reinterpret_cast<int64_t*>(dest);
  auto g = gpair.GetQuantisedGrad();
  auto h = gpair.GetQuantisedHess();

  AtomicAdd64As32(dst_ptr, g);
  AtomicAdd64As32(dst_ptr + 1, h);
}

// Global 64 bit integer atomics at the time of writing do not benefit from being separated into two
// 32 bit atomics
XGBOOST_DEV_INLINE void AtomicAddGpairGlobal(xgboost::GradientPairInt64* dest,
                                             xgboost::GradientPairInt64 const& gpair) {
  auto dst_ptr = reinterpret_cast<uint64_t*>(dest);
  auto g = gpair.GetQuantisedGrad();
  auto h = gpair.GetQuantisedHess();

  atomicAdd(dst_ptr, *reinterpret_cast<uint64_t*>(&g));
  atomicAdd(dst_ptr + 1, *reinterpret_cast<uint64_t*>(&h));
}

template <std::int32_t BlockThreads, std::int32_t MinBlocks>
struct HistTuning {
  static constexpr std::int32_t kBlockThreads = BlockThreads;
  static constexpr std::int32_t kMinBlocks = MinBlocks;
};

namespace {
constexpr std::int32_t kItemsPerThread = 8;

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#feature-set-compiler-targets
// Technical Specifications                  7.5  | 8.0  | 8.6  8.7 | 8.9 | 9.0 10.0 | 11.0 12.0
// Maximum number of resident blocks per SM  16   | 32   | 16       | 24  | 32       | 24
// Maximum number of resident warps per SM   32   | 64   | 48             | 64       | 48
// Maximum number of resident threads per SM 1024 | 2048 | 1536           | 2048     | 1536

using HistSm75 = HistTuning<1024, 1>;

using HistSm80 = HistTuning<1024, 2>;

using HistSm86 = HistTuning<768, 2>;

using HistSm90 = HistTuning<1024, 2>;

using HistSm110 = HistTuning<768, 2>;

// Multi-target launch bounds
#if __CUDA_ARCH__ >= 1100
using MtHistBound = HistSm110;
#elif __CUDA_ARCH__ >= 900
using MtHistBound = HistSm90;
#elif __CUDA_ARCH__ >= 860
using MtHistBound = HistSm86;
#elif __CUDA_ARCH__ >= 800
using MtHistBound = HistSm80;
#else
using MtHistBound = HistSm75;
#endif

// Single-target launch bounds
// Maximize the number of threads instead of tuning for occupancy for single target.
using StHistBound = HistSm75;

template <typename HistArchPolicy, std::int32_t ItemsPerThread, bool Dense, bool Compressed,
          bool SharedMem>
struct HistPolicy : public HistArchPolicy {
  static constexpr std::int32_t kItemsPerThread = ItemsPerThread;
  static constexpr std::int32_t kTileSize = HistArchPolicy::kBlockThreads * ItemsPerThread;
  static constexpr bool kDense = Dense;
  static constexpr bool kCompressed = Compressed;
  static constexpr bool kSharedMem = SharedMem;
};

template <typename Fn>
void DispatchCudaSm(std::int32_t device, Fn&& fn) {
  std::int32_t version = 0;
  dh::safe_cuda(cub::SmVersion(version, device));
  if (version >= 1100) {
    fn(HistSm110{});
  } else if (version >= 900) {
    fn(HistSm90{});
  } else if (version >= 860) {
    fn(HistSm86{});
  } else if (version >= 800) {
    fn(HistSm80{});
  } else {
    fn(HistSm75{});
  }
}

__device__ GradientPairInt64 LoadGpair(GradientPairInt64 const* XGBOOST_RESTRICT gpairs) {
  static_assert(sizeof(int4) == sizeof(GradientPairInt64));
  auto g = *reinterpret_cast<int4 const*>(gpairs);
  return *reinterpret_cast<GradientPairInt64*>(&g);
}

// Build the histogram for a single target in a single node.
template <typename Policy, typename Accessor, typename RidxIterSpan>
__device__ void HistKernelOneNodeTarget(Accessor const& matrix, FeatureGroup const& group,
                                        RidxIterSpan d_ridx_iter, GradientPairInt64 const* gpair,
                                        GradientPairInt64* smem_hist, GradientPairInt64* gmem_hist,
                                        bst_idx_t offset, std::uint32_t stride) {
  bst_feature_t const feature_stride = Policy::kCompressed ? group.num_features : matrix.row_stride;

  using Idx = RowPartitioner::RowIndexT;

  Idx const ridx_size = d_ridx_iter.size();
  auto const d_ridx = d_ridx_iter.data();

  if constexpr (Policy::kSharedMem) {
    dh::BlockFill(smem_hist, group.num_bins, GradientPairInt64{});
    __syncthreads();
  }

  auto atomic_add = [&](auto bin_idx, auto const& adjusted) {
    if constexpr (Policy::kSharedMem) {
      AtomicAddGpairShared(smem_hist + bin_idx, adjusted);
    } else {
      // gmem_hist is a subspan for the current target.
      AtomicAddGpairGlobal(gmem_hist + bin_idx, adjusted);
    }
  };

  auto process_valid_tile = [&](auto idx) {
    // unrolled version unravel to save registers:
    // auto [ridx, fidx] = unravel_index(idx, (n_rows, feature_stride));
    //
    // ridx_in_set: Index into the row batch
    // fidx_in_set: Index into the feature group
    Idx ridx_in_set = idx / feature_stride;
    Idx fidx_in_set = idx - ridx_in_set * feature_stride;

    Idx ridx = d_ridx[ridx_in_set];
    auto fidx = fidx_in_set + group.start_feature;

    bst_bin_t compressed_bin = matrix.gidx_iter[IterIdx(matrix, ridx, fidx)];
    if (Policy::kDense || compressed_bin != matrix.NullValue()) {
      auto g = LoadGpair(gpair + ridx);
      if constexpr (Policy::kCompressed) {
        compressed_bin += matrix.feature_segments[fidx];
      }
      if constexpr (Policy::kSharedMem) {
        compressed_bin -= group.start_bin;
      }
      atomic_add(compressed_bin, g);
    }
  };

  // The number of elements for this grid to process
  bst_idx_t const n_elements = static_cast<std::size_t>(ridx_size) * feature_stride;

  auto process_gpair_tile = [&](auto full_tile, auto offset) {
#pragma unroll 1
    for (std::int32_t j = 0; j < Policy::kItemsPerThread; ++j) {
      bst_idx_t const idx = offset + j * Policy::kBlockThreads + threadIdx.x;
      if (full_tile || idx < n_elements) {
        process_valid_tile(idx);
      }
    }
  };

  while (offset < n_elements) {
    std::int32_t const valid_items =
        cuda::std::min(n_elements - offset, static_cast<bst_idx_t>(Policy::kTileSize));
    if (Policy::kTileSize == valid_items) {
      process_gpair_tile(std::true_type{}, offset);
    } else {
      process_gpair_tile(std::false_type{}, offset);
    }
    offset += stride;
  }

  if constexpr (!Policy::kSharedMem) {
    return;
  }

  // Write shared memory back to global memory
  __syncthreads();

  for (auto bin_idx : dh::BlockStrideRange(0, group.num_bins)) {
    AtomicAddGpairGlobal(gmem_hist + group.start_bin + bin_idx, smem_hist[bin_idx]);
  }
}
}  // namespace

/**
 * @brief Kernel for the single-target histogram.
 */
template <typename Policy, typename Accessor>
__global__ __launch_bounds__(StHistBound::kBlockThreads, StHistBound::kMinBlocks) void StHistKernel(
    Accessor const matrix, FeatureGroupsAccessor const feature_groups,
    common::Span<cuda_impl::RowIndexT const> d_ridx_iter,
    common::Span<GradientPairInt64 const> d_gpair, common::Span<GradientPairInt64> node_hist) {
  extern __align__(cuda::std::alignment_of_v<GradientPairInt64>) __shared__ char shmem[];

  // Privatized histogram
  auto smem_hist = reinterpret_cast<GradientPairInt64*>(shmem);

  // Offset of the first grid
  bst_idx_t offset = blockIdx.x * Policy::kTileSize;
  // Grid-strided loop
  auto const kStride = Policy::kTileSize * gridDim.x;

  FeatureGroup group = feature_groups[blockIdx.y];

  HistKernelOneNodeTarget<Policy>(matrix, group, d_ridx_iter, d_gpair.data(), smem_hist,
                                  node_hist.data(), offset, kStride);
}

/**
 * @brief Kernel for the multi-target histogram.
 *
 * @param matrix         An ellpack accessor.
 * @param feature_groups Grouping for privatized histogram.
 * @param d_ridx_iters   Pointer to row index spans. One span per node.
 * @param blk_ptr        Indptr for mapping blockIdx.x to nidx_in_set.
 */
template <typename Policy, typename Accessor, typename RidxIterSpan>
__global__ __launch_bounds__(MtHistBound::kBlockThreads, MtHistBound::kMinBlocks) void MtHistKernel(
    Accessor const matrix, FeatureGroupsAccessor const feature_groups, RidxIterSpan* d_ridx_iters,
    common::Span<std::uint32_t const> blk_ptr, common::Span<GradientPairInt64>* node_hists,
    GradientPairInt64 const* d_gpair, bst_idx_t n_samples, bst_target_t n_targets) {
  using Idx = RowPartitioner::RowIndexT;

  // Find the node for this block.
  auto const* XGBOOST_RESTRICT p_blk_ptr = blk_ptr.data();
  Idx nidx_in_set = dh::SegmentId(p_blk_ptr, p_blk_ptr + blk_ptr.size(), blockIdx.x);
  Idx starting_blk = p_blk_ptr[nidx_in_set];

  extern __align__(cuda::std::alignment_of_v<GradientPairInt64>) __shared__ char shmem[];

  // Privatized histogram
  auto smem_hist = reinterpret_cast<GradientPairInt64*>(shmem);
  auto d_node_hist = node_hists + nidx_in_set;
  auto const n_bins_per_target = d_node_hist->size() / n_targets;

  // The number of blocks in this sub-grid
  auto n_blks = p_blk_ptr[nidx_in_set + 1] - starting_blk;
  auto blkid_in_set = blockIdx.x - starting_blk;

  // unravel_index(blkdid_in_set, {n_blocks_one_node_target, n_targets})
  auto blkid_for_node = blkid_in_set / n_targets;
  bst_target_t target_idx = blkid_in_set - blkid_for_node * n_targets;

  // Offset of the first grid
  bst_idx_t offset = blkid_for_node * Policy::kTileSize;
  // Grid-strided loop
  auto const kStride = Policy::kTileSize * (n_blks / n_targets);

  FeatureGroup group = feature_groups[blockIdx.y];

  // With a target-major layout, we don't have to pack the histogram for all targets into
  // the shared memory. Since we launch one block for each target, the histogram index can
  // be shared in L2.
  auto gmem_hist = d_node_hist->data() + target_idx * n_bins_per_target;
  d_gpair = d_gpair + n_samples * target_idx;

  HistKernelOneNodeTarget<Policy>(matrix, group, d_ridx_iters[nidx_in_set], d_gpair, smem_hist,
                                  gmem_hist, offset, kStride);
}

// Dispatcher for the histogram kernel.
struct HistKernel {
  /**
   * @brief Partition the grid into sub-grid for nodes.
   *
   * @param sizes_csum          cumulative sum of node sizes (csum of n_samples for each node).
   * @param columns_per_group   Estimated number of columns for each feature group.
   * @param max_blocks_per_node The maximum sub-grid size for a node.
   * @param p_out_blocks        The total number of blocks (grid size).
   */
  template <typename Policy>
  static auto AllocateBlocks(std::vector<std::size_t> const& sizes_csum,
                             std::int32_t columns_per_group, std::size_t max_blocks_per_node,
                             bst_target_t n_targets, std::uint32_t* p_out_blocks) {
    CHECK_GT(max_blocks_per_node, 0);
    std::vector<std::uint32_t> blk_ptr{0};
    bst_idx_t n_total_blocks = 0;
    for (std::size_t j = 1; j < sizes_csum.size(); ++j) {
      auto nidx_in_set = j - 1;
      auto n_samples = sizes_csum[j] - sizes_csum[j - 1];
      std::size_t items_per_group = n_samples * columns_per_group;
      auto n_blocks = common::DivRoundUp(items_per_group, Policy::kTileSize);
      CHECK_GT(n_blocks, 0);  // at least one block for each node.
      n_blocks = std::min(n_blocks, max_blocks_per_node) * n_targets;
      blk_ptr.push_back(blk_ptr[nidx_in_set] + n_blocks);
      n_total_blocks += n_blocks;
    }
    // check overflow
    CHECK_EQ(n_total_blocks, blk_ptr.back());
    *p_out_blocks = blk_ptr.back();
    return dh::device_vector<std::uint32_t>{blk_ptr};
  }

  struct HistKernelConfig {
    std::int32_t n_blocks_per_mp = 0;
    std::size_t shmem_bytes = 0;

    template <typename Policy, typename Kernel>
    void Reset(std::size_t new_shmem_bytes, Kernel* kernel, Policy, std::size_t max_shared_bytes) {
      if (new_shmem_bytes > 0) {
        // This function is the reason for all this trouble to cache the
        // configuration. It blocks the device.
        //
        // Also, it must precede the `cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
        // otherwise the shmem bytes might be invalid.
        dh::safe_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           max_shared_bytes));
      }
      if (new_shmem_bytes > this->shmem_bytes) {
        this->shmem_bytes = new_shmem_bytes;
      }
      // Use this as a limiter, works for root node. Not too bad an option for child nodes.
      dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &this->n_blocks_per_mp, kernel, Policy::kBlockThreads, shmem_bytes));
    }
  };

  // Maps kernel instantiations to their configurations. This is a mutable state, as a
  // result the histogram kernel is not thread safe.
  std::map<void*, HistKernelConfig> cfg;
  // The number of multi-processor for the selected GPU
  std::int32_t const n_mps;
  // Maximum size of the shared memory (optin)
  std::size_t const max_shared_bytes;
  // Use global memory for testing
  bool const force_global;

  template <typename Policy, typename Kernel>
  void SetCfg(Policy policy, std::size_t shmem_bytes, Kernel kernel) {
    auto it = this->cfg.find(reinterpret_cast<void*>(kernel));

    HistKernelConfig v;
    if (it == cfg.cend()) {
      v.Reset(shmem_bytes, kernel, policy, max_shared_bytes);
      this->cfg[reinterpret_cast<void*>(kernel)] = v;
    }
  }

  explicit HistKernel(Context const* ctx, bool force_global)
      : n_mps{curt::GetMpCnt(ctx->Ordinal())},
        max_shared_bytes{dh::MaxSharedMemoryOptin(ctx->Ordinal())},
        force_global{force_global} {}

  // Single target
  template <bool kDense, bool kCompressed, typename Accessor>
  void DispatchHistShmem(Context const* ctx, Accessor const& matrix,
                         FeatureGroupsAccessor const& feature_groups,
                         common::Span<GradientPairInt64 const> gpair,
                         common::Span<cuda_impl::RowIndexT const> ridx,
                         common::Span<GradientPairInt64> hist) {
    std::size_t shmem_bytes = feature_groups.ShmemSize();
    bool use_shared = !this->force_global && shmem_bytes <= this->max_shared_bytes;
    shmem_bytes = use_shared ? shmem_bytes : 0;

    auto launch = [&](auto policy, auto kernel) {
      auto const& v = this->cfg.at(reinterpret_cast<void*>(kernel));
      using Policy = common::GetValueT<decltype(policy)>;
      int columns_per_group = common::DivRoundUp(matrix.row_stride, feature_groups.NumGroups());
      CHECK_GT(v.n_blocks_per_mp, 0);
      std::size_t items_per_group = ridx.size() * columns_per_group;
      std::uint32_t n_blocks =
          std::min(static_cast<cuda_impl::RowIndexT>(v.n_blocks_per_mp * this->n_mps),
                   static_cast<cuda_impl::RowIndexT>(
                       common::DivRoundUp(items_per_group, Policy::kTileSize)));
      dim3 conf(n_blocks, feature_groups.NumGroups());
      dh::LaunchKernel(conf, Policy::kBlockThreads, shmem_bytes, ctx->CUDACtx()->Stream())(
          kernel, matrix, feature_groups, ridx, gpair, hist);
      dh::safe_cuda(cudaPeekAtLastError());
    };
    using Arch = StHistBound;

    if (use_shared) {
      using Policy = HistPolicy<Arch, kItemsPerThread, kDense, kCompressed, true>;
      auto kernel = StHistKernel<Policy, Accessor>;
      this->SetCfg(Policy{}, shmem_bytes, kernel);
      launch(Policy{}, kernel);
    } else {
      using Policy = HistPolicy<Arch, kItemsPerThread, kDense, kCompressed, false>;
      auto kernel = StHistKernel<Policy, Accessor>;
      this->SetCfg(Policy{}, shmem_bytes, kernel);
      launch(Policy{}, kernel);
    }
  }
  // Vector leaf
  template <bool kDense, bool kCompressed, typename Accessor, typename RidxIterSpan>
  void DispatchHistShmem(Context const* ctx, Accessor const& matrix,
                         FeatureGroupsAccessor const& feature_groups,
                         linalg::MatrixView<GradientPairInt64 const> gpair,
                         RidxIterSpan* ridx_iters,
                         common::Span<common::Span<GradientPairInt64>> hists,
                         std::vector<std::size_t> const& h_sizes_csum) {
    CHECK(gpair.FContiguous());
    auto n_samples = gpair.Shape(0);
    auto n_targets = gpair.Shape(1);
    auto d_gpair = gpair.Values().data();

    std::size_t shmem_bytes = feature_groups.ShmemSize();
    bool use_shared = !force_global && shmem_bytes <= this->max_shared_bytes;
    shmem_bytes = use_shared ? shmem_bytes : 0;

    auto launch = [&](auto policy, auto kernel) {
      auto const& v = this->cfg.at(reinterpret_cast<void*>(kernel));
      using Policy = common::GetValueT<decltype(policy)>;
      int columns_per_group = common::DivRoundUp(matrix.row_stride, feature_groups.NumGroups());
      CHECK_GT(v.n_blocks_per_mp, 0);
      std::uint32_t n_blocks = 0;
      auto blk_ptr = AllocateBlocks<Policy>(h_sizes_csum, columns_per_group,
                                            v.n_blocks_per_mp * n_mps, n_targets, &n_blocks);
      CHECK_GE(n_blocks, hists.size());
      dim3 conf(n_blocks, feature_groups.NumGroups());
      dh::LaunchKernel(conf, Policy::kBlockThreads, shmem_bytes, ctx->CUDACtx()->Stream())(
          kernel, matrix, feature_groups, ridx_iters, dh::ToSpan(blk_ptr), hists.data(), d_gpair,
          n_samples, n_targets);
      dh::safe_cuda(cudaPeekAtLastError());
    };

    CHECK(gpair.FContiguous());
    if (use_shared) {
      DispatchCudaSm(ctx->Ordinal(), [&](auto arch) {
        using Arch = common::GetValueT<decltype(arch)>;
        using Policy = HistPolicy<Arch, kItemsPerThread, kDense, kCompressed, true>;
        auto kernel = MtHistKernel<Policy, Accessor, RidxIterSpan>;
        this->SetCfg(Policy{}, shmem_bytes, kernel);
        launch(Policy{}, kernel);
      });
    } else {
      DispatchCudaSm(ctx->Ordinal(), [&](auto arch) {
        using Arch = common::GetValueT<decltype(arch)>;
        using Policy = HistPolicy<Arch, kItemsPerThread, kDense, kCompressed, false>;
        auto kernel = MtHistKernel<Policy, Accessor, RidxIterSpan>;
        this->SetCfg(Policy{}, shmem_bytes, kernel);
        launch(Policy{}, kernel);
      });
    }
  }

  template <typename Accessor, typename... Args>
  void DispatchHistCompress(Context const* ctx, Accessor const& matrix, Args&&... args) {
    if (matrix.IsDense()) {
      DispatchHistShmem<true, true>(ctx, matrix, std::forward<Args>(args)...);
    } else if (matrix.IsDenseCompressed()) {
      DispatchHistShmem<false, true>(ctx, matrix, std::forward<Args>(args)...);
    } else {
      DispatchHistShmem<false, false>(ctx, matrix, std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  void Dispatch(Args&&... args) {
    this->DispatchHistCompress(std::forward<Args>(args)...);
  }
};

template <typename Accessor>
class DeviceHistogramDispatchAccessor {
  std::unique_ptr<HistKernel> kernel_{nullptr};

 public:
  void Reset(Context const* ctx, bool force_global_memory) {
    this->kernel_ = std::make_unique<HistKernel>(ctx, force_global_memory);
  }

  void BuildHistogram(Context const* ctx, Accessor const& matrix,
                      FeatureGroupsAccessor const& feature_groups,
                      common::Span<GradientPairInt64 const> gpair,
                      common::Span<cuda_impl::RowIndexT const> ridx,
                      common::Span<GradientPairInt64> hist) {
    this->kernel_->Dispatch(ctx, matrix, feature_groups, gpair, ridx, hist);
  }

  void BuildHistogram(Context const* ctx, Accessor const& matrix,
                      FeatureGroupsAccessor const& feature_groups,
                      linalg::MatrixView<GradientPairInt64 const> gpair,
                      common::Span<common::Span<cuda_impl::RowIndexT const>> ridxs,
                      common::Span<common::Span<GradientPairInt64>> hists,
                      std::vector<std::size_t> const& h_sizes_csum) {
    std::size_t n_total_samples = h_sizes_csum.back();
    if (ridxs.size() == 1 && n_total_samples == matrix.n_rows) {
      // Special optimization for the root node.
      using RidxIter = thrust::counting_iterator<cuda_impl::RowIndexT>;
      CHECK_LT(matrix.base_rowid, std::numeric_limits<cuda_impl::RowIndexT>::max());
      auto iter = common::IterSpan{
          thrust::make_counting_iterator(static_cast<cuda_impl::RowIndexT>(matrix.base_rowid)),
          matrix.n_rows};
      dh::caching_device_vector<common::IterSpan<RidxIter>> ridx_iters(hists.size(), iter);
      this->kernel_->Dispatch(ctx, matrix, feature_groups, gpair, ridx_iters.data().get(), hists,
                              h_sizes_csum);
    } else {
      using RidxIter = cuda_impl::RowIndexT const;
      this->kernel_->Dispatch(ctx, matrix, feature_groups, gpair, ridxs.data(), hists,
                              h_sizes_csum);
    }
  }
};

// Dispatch between single buffer accessor and double buffer accessor.
struct DeviceHistogramBuilderImpl {
  DeviceHistogramDispatchAccessor<EllpackDeviceAccessor> simpl;
  DeviceHistogramDispatchAccessor<DoubleEllpackAccessor> dimpl;

  template <typename... Args>
  void Reset(Args&&... args) {
    this->simpl.Reset(std::forward<Args>(args)...);
    this->dimpl.Reset(std::forward<Args>(args)...);
  }

  template <typename Accessor, typename... Args>
  void BuildHistogram(Context const* ctx, Accessor const& matrix, Args&&... args) {
    if constexpr (std::is_same_v<Accessor, EllpackDeviceAccessor>) {
      this->simpl.BuildHistogram(ctx, matrix, std::forward<Args>(args)...);
    } else {
      static_assert(std::is_same_v<Accessor, DoubleEllpackAccessor>);
      this->dimpl.BuildHistogram(ctx, matrix, std::forward<Args>(args)...);
    }
  }
};

DeviceHistogramBuilder::DeviceHistogramBuilder()
    : p_impl_{std::make_unique<DeviceHistogramBuilderImpl>()} {
  monitor_.Init(__func__);
}

DeviceHistogramBuilder::~DeviceHistogramBuilder() = default;

void DeviceHistogramBuilder::Reset(Context const* ctx, std::size_t max_cached_hist_nodes,
                                   bst_bin_t n_total_bins, bool force_global_memory) {
  this->monitor_.Start(__func__);
  this->p_impl_->Reset(ctx, force_global_memory);
  this->hist_.Reset(ctx, n_total_bins, max_cached_hist_nodes);
  this->monitor_.Stop(__func__);
}

void DeviceHistogramBuilder::BuildHistogram(Context const* ctx, EllpackAccessor const& matrix,
                                            FeatureGroupsAccessor const& feature_groups,
                                            common::Span<GradientPairInt64 const> gpair,
                                            common::Span<cuda_impl::RowIndexT const> ridx,
                                            common::Span<GradientPairInt64> histogram) {
  this->monitor_.Start(__func__);
  std::visit(
      [&](auto&& matrix) {
        this->p_impl_->BuildHistogram(ctx, matrix, feature_groups, gpair, ridx, histogram);
      },
      matrix);
  this->monitor_.Stop(__func__);
}

void DeviceHistogramBuilder::BuildHistogram(
    Context const* ctx, EllpackAccessor const& matrix, FeatureGroupsAccessor const& feature_groups,
    linalg::MatrixView<GradientPairInt64 const> gpair,
    common::Span<common::Span<cuda_impl::RowIndexT const>> ridxs,
    common::Span<common::Span<GradientPairInt64>> hists,
    std::vector<std::size_t> const& h_sizes_csum) {
  std::visit(
      [&](auto&& matrix) {
        this->p_impl_->BuildHistogram(ctx, matrix, feature_groups, gpair, ridxs, hists,
                                      h_sizes_csum);
      },
      matrix);
}

void DeviceHistogramBuilder::AllReduceHist(Context const* ctx, MetaInfo const& info,
                                           bst_node_t nidx, std::size_t num_histograms) {
  this->monitor_.Start(__func__);
  auto d_node_hist = hist_.GetNodeHistogram(nidx);
  using ReduceT = typename std::remove_pointer<decltype(d_node_hist.data())>::type::ValueT;
  auto rc = collective::GlobalSum(
      ctx, info,
      linalg::MakeVec(reinterpret_cast<ReduceT*>(d_node_hist.data()),
                      d_node_hist.size() * 2 * num_histograms, ctx->Device()));
  SafeColl(rc);
  this->monitor_.Stop(__func__);
}
}  // namespace xgboost::tree
