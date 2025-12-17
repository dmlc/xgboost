/**
 * Copyright 2020-2025, XGBoost Contributors
 */
#include <thrust/iterator/transform_iterator.h>  // for make_transform_iterator

#include <algorithm>
#include <cstdint>  // uint32_t, int32_t

#include "../../collective/aggregator.h"
#include "../../common/cuda_rt_utils.h"  // for GetMpCnt
#include "../../common/deterministic.cuh"
#include "../../common/device_helpers.cuh"
#include "../../common/linalg_op.cuh"  // for tbegin
#include "../../data/array_page_source.h"
#include "../../data/ellpack_page.cuh"
#include "histogram.cuh"
#include "row_partitioner.cuh"
#include "xgboost/base.h"
#include "xgboost/gradient.h"

namespace xgboost::tree {
namespace {
struct Pair {
  GradientPair first;
  GradientPair second;
};
__host__ XGBOOST_DEV_INLINE Pair operator+(Pair const& lhs, Pair const& rhs) {
  return {lhs.first + rhs.first, lhs.second + rhs.second};
}

XGBOOST_DEV_INLINE bst_feature_t FeatIdx(FeatureGroup const& group, bst_idx_t idx,
                                         cuda_impl::RowIndexT ridx, std::int32_t feature_stride) {
  return group.start_feature + idx - ridx * feature_stride;
}

XGBOOST_DEV_INLINE bst_feature_t FeatIdx(FeatureGroup const& group, bst_idx_t idx,
                                         std::int32_t feature_stride) {
  return group.start_feature + idx % feature_stride;
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

    return {GradientPair{pg, ph}, GradientPair{ng, nh}};
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
  Pair p = dh::Reduce(ctx->CUDACtx()->CTP(), beg, beg + gpair.Size(), Pair{}, thrust::plus<Pair>{});
  // Treat pair as array of 4 primitive types to allreduce
  using ReduceT = typename decltype(p.first)::ValueT;
  static_assert(sizeof(Pair) == sizeof(ReduceT) * 4, "Expected to reduce four elements.");
  auto rc = collective::GlobalSum(ctx, info, linalg::MakeVec(reinterpret_cast<ReduceT*>(&p), 4));
  collective::SafeColl(rc);

  GradientPair positive_sum{p.first}, negative_sum{p.second};

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
  to_fixed_point_ = GradientSumT(static_cast<T>(1) / to_floating_point_.GetGrad(),
                                 static_cast<T>(1) / to_floating_point_.GetHess());
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

MultiGradientQuantiser::MultiGradientQuantiser(Context const* ctx, GradientContainer* gpair,
                                               MetaInfo const& info) {
  std::vector<GradientQuantiser> h_quantizers;

  using GradientSumT = GradientPairPrecise;
  using T = typename GradientSumT::ValueT;
  auto n_samples = info.num_row_;

  for (bst_target_t t = 0, n_targets = gpair->NumTargetsN(); t < n_targets; ++t) {
    Pair sum;
    for (auto const& batch : gpair->GetGrad()) {
      std::cout << "sum" << std::endl;
      auto d_gpairs = batch.gpairs.View(ctx->Device()).Slice(linalg::All(), t);
      auto beg = thrust::make_transform_iterator(linalg::tcbegin(d_gpairs), Clip{});
      Pair p = dh::Reduce(ctx->CUDACtx()->CTP(), beg, beg + d_gpairs.Size(), Pair{},
                          thrust::plus<Pair>{});
      sum = sum + p;
    }

    GradientPair positive_sum{sum.first}, negative_sum{sum.second};
    auto histogram_rounding =
        GradientSumT{common::CreateRoundingFactor<T>(
                         std::max(positive_sum.GetGrad(), negative_sum.GetGrad()), n_samples),
                     common::CreateRoundingFactor<T>(
                         std::max(positive_sum.GetHess(), negative_sum.GetHess()), n_samples)};

    using IntT = typename GradientPairInt64::ValueT;

    auto to_floating_point_ =
        histogram_rounding /
        static_cast<T>(static_cast<IntT>(1)
                       << (sizeof(typename GradientSumT::ValueT) * 8 - 2));  // keep 1 for sign bit

    auto to_fixed_point_ = GradientSumT(static_cast<T>(1) / to_floating_point_.GetGrad(),
                                        static_cast<T>(1) / to_floating_point_.GetHess());

    h_quantizers.emplace_back(to_fixed_point_, to_floating_point_);
  }

  this->quantizers_ = h_quantizers;
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

template <typename Accessor, bool kCompressed, bool kDense, int kBlockThreads, int kItemsPerThread>
class HistogramAgent {
  int constexpr static kItemsPerTile = kBlockThreads * kItemsPerThread;

  GradientPairInt64* smem_arr_;
  GradientPairInt64* d_node_hist_;
  using Idx = cuda_impl::RowIndexT;

  dh::LDGIterator<const Idx> d_ridx_;
  const GradientPair* d_gpair_;
  const FeatureGroup group_;
  Accessor const& matrix_;
  const int feature_stride_;
  const bst_idx_t n_elements_;
  const GradientQuantiser& rounding_;

  static_assert(kCompressed >= kDense);

 public:
  __device__ HistogramAgent(GradientPairInt64* smem_arr,
                            GradientPairInt64* __restrict__ d_node_hist, const FeatureGroup& group,
                            Accessor const& matrix, common::Span<const Idx> d_ridx,
                            const GradientQuantiser& rounding, const GradientPair* d_gpair)
      : smem_arr_{smem_arr},
        d_node_hist_{d_node_hist},
        d_ridx_(d_ridx.data()),
        group_{group},
        matrix_(matrix),
        feature_stride_(kCompressed ? group.num_features : matrix.row_stride),
        n_elements_{feature_stride_ * d_ridx.size()},
        rounding_{rounding},
        d_gpair_{d_gpair} {}

  __device__ void ProcessPartialTileShared(std::size_t offset) {
    for (std::size_t idx = offset + threadIdx.x,
                     n = std::min(offset + kBlockThreads * kItemsPerTile, n_elements_);
         idx < n; idx += kBlockThreads) {
      Idx ridx = d_ridx_[idx / feature_stride_];
      auto fidx = FeatIdx(group_, idx, feature_stride_);
      bst_bin_t compressed_bin = matrix_.gidx_iter[IterIdx(matrix_, ridx, fidx)];
      if (kDense || compressed_bin != matrix_.NullValue()) {
        // The matrix is compressed with feature-local bins.
        if (kCompressed) {
          compressed_bin += this->matrix_.feature_segments[fidx];
        }
        // Avoid atomic add if it's a null value.
        auto adjusted = rounding_.ToFixedPoint(d_gpair_[ridx]);
        // Subtract start_bin to write to group-local histogram. If this is not a dense
        // matrix, then start_bin is 0 since featuregrouping doesn't support sparse data.
        AtomicAddGpairShared(smem_arr_ + compressed_bin - group_.start_bin, adjusted);
      }
    }
  }

  // Instruction level parallelism by loop unrolling
  // Allows the kernel to pipeline many operations while waiting for global memory
  // Increases the throughput of this kernel significantly
  __device__ void ProcessFullTileShared(std::size_t offset) {
    std::size_t idx[kItemsPerThread];
    Idx ridx[kItemsPerThread];
    bst_bin_t gidx[kItemsPerThread];
    GradientPair gpair[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      idx[i] = offset + i * kBlockThreads + threadIdx.x;
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      ridx[i] = d_ridx_[idx[i] / feature_stride_];
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      gpair[i] = d_gpair_[ridx[i]];
      auto fidx = FeatIdx(group_, idx[i], feature_stride_);
      gidx[i] = matrix_.gidx_iter[IterIdx(matrix_, ridx[i], fidx)];
      if (kDense || gidx[i] != matrix_.NullValue()) {
        if constexpr (kCompressed) {
          gidx[i] += matrix_.feature_segments[fidx];
        }
      } else {
        // Use -1 to denote missing. Since we need to add the beginning bin to gidx, the
        // result might equal to the `NullValue`.
        gidx[i] = -1;
      }
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      // Avoid atomic add if it's a null value.
      if (kDense || gidx[i] != -1) {
        auto adjusted = rounding_.ToFixedPoint(gpair[i]);
        AtomicAddGpairShared(smem_arr_ + gidx[i] - group_.start_bin, adjusted);
      }
    }
  }
  __device__ void BuildHistogramWithShared() {
    dh::BlockFill(smem_arr_, group_.num_bins, GradientPairInt64{});
    __syncthreads();

    std::size_t offset = blockIdx.x * kItemsPerTile;
    while (offset + kItemsPerTile <= n_elements_) {
      ProcessFullTileShared(offset);
      offset += kItemsPerTile * gridDim.x;
    }
    ProcessPartialTileShared(offset);

    // Write shared memory back to global memory
    __syncthreads();
    for (auto i : dh::BlockStrideRange(0, group_.num_bins)) {
      AtomicAddGpairGlobal(d_node_hist_ + group_.start_bin + i, smem_arr_[i]);
    }
  }

  __device__ void BuildHistogramWithGlobal() {
    for (auto idx : dh::GridStrideRange(static_cast<std::size_t>(0), n_elements_)) {
      Idx ridx = d_ridx_[idx / feature_stride_];
      auto fidx = FeatIdx(group_, idx, feature_stride_);
      bst_bin_t compressed_bin = matrix_.gidx_iter[IterIdx(matrix_, ridx, fidx)];
      if (compressed_bin != matrix_.NullValue()) {
        if (kCompressed) {
          compressed_bin += this->matrix_.feature_segments[fidx];
        }
        auto adjusted = rounding_.ToFixedPoint(d_gpair_[ridx]);
        AtomicAddGpairGlobal(d_node_hist_ + compressed_bin, adjusted);
      }
    }
  }
};

template <typename Accessor, bool kCompressed, bool kDense, bool use_shared_memory_histograms,
          int kBlockThreads, int kItemsPerThread>
__global__ void __launch_bounds__(kBlockThreads)
    SharedMemHistKernel(Accessor const matrix, const FeatureGroupsAccessor feature_groups,
                        common::Span<const RowPartitioner::RowIndexT> d_ridx,
                        GradientPairInt64* __restrict__ d_node_hist,
                        const GradientPair* __restrict__ d_gpair,
                        GradientQuantiser const rounding) {
  extern __shared__ char smem[];
  const FeatureGroup group = feature_groups[blockIdx.y];
  auto smem_arr = reinterpret_cast<GradientPairInt64*>(smem);
  auto agent = HistogramAgent<Accessor, kCompressed, kDense, kBlockThreads, kItemsPerThread>(
      smem_arr, d_node_hist, group, matrix, d_ridx, rounding, d_gpair);
  if (use_shared_memory_histograms) {
    agent.BuildHistogramWithShared();
  } else {
    agent.BuildHistogramWithGlobal();
  }
}

template <std::int32_t kBlockThreadsIn, std::int32_t kItemsPerThreadIn, bool kDenseIn,
          bool kCompressedIn, bool kSharedMemIn>
struct HistPolicy {
  static constexpr std::int32_t kBlockThreads = kBlockThreadsIn;
  static constexpr std::int32_t kItemsPerThread = kItemsPerThreadIn;
  static constexpr std::int32_t kTileSize = kBlockThreadsIn * kItemsPerThreadIn;
  static constexpr bool kDense = kDenseIn;
  static constexpr bool kCompressed = kCompressedIn;
  static constexpr bool kSharedMem = kSharedMemIn;
};

/**
 * @brief Kernel for multi-target histogram.
 *
 * Eventually, we should merge the kernel with the existing single-target version. For
 * now, we define a different version as a playground, without risking regressing the
 * established one.
 *
 * @param matrix         An ellpack accessor.
 * @param feature_groups Grouping for privatized histogram.
 * @param d_ridx_iters   Pointer to row index spans. One span per node.
 * @param blk_ptr        Indptr for mapping blockIdx.x to nidx_in_set.
 */
template <typename Policy, typename Accessor, typename RidxIterSpan>
__global__ __launch_bounds__(Policy::kBlockThreads) void HistKernel(
    Accessor const matrix, FeatureGroupsAccessor const feature_groups, RidxIterSpan* d_ridx_iters,
    common::Span<std::uint32_t const> blk_ptr, common::Span<GradientPairInt64>* node_hists,
    linalg::MatrixView<GradientPair const> d_gpair,
    common::Span<GradientQuantiser const> roundings) {
  auto d_roundings = roundings.data();

  FeatureGroup group = feature_groups[blockIdx.y];
  std::int32_t feature_stride = Policy::kCompressed ? group.num_features : matrix.row_stride;

  // Find the node for this block.
  auto nidx_in_set = dh::SegmentId(blk_ptr.data(), blk_ptr.data() + blk_ptr.size(), blockIdx.x);
  auto starting_blk = blk_ptr[nidx_in_set];
  // Grid-strided loop
  auto const kStride = Policy::kTileSize * (blk_ptr[nidx_in_set + 1] - starting_blk);
  // Offset of the first grid
  std::size_t offset = (blockIdx.x - starting_blk) * Policy::kTileSize;

  auto d_ridx = d_ridx_iters[nidx_in_set];
  auto p_ridx = d_ridx_iters[nidx_in_set].data();

  bst_idx_t n_elements = feature_stride * d_ridx.size();

  using Idx = RowPartitioner::RowIndexT;
  bst_target_t const n_targets = roundings.size();

  extern __align__(cuda::std::alignment_of_v<GradientPairInt64>) __shared__ char shmem[];
  // Privatized histogram
  auto smem_hist = reinterpret_cast<GradientPairInt64*>(shmem);

  if constexpr (Policy::kSharedMem) {
    dh::BlockFill(smem_hist, group.num_bins * n_targets, GradientPairInt64{});
    __syncthreads();
  }

  auto gmem_hist = node_hists[nidx_in_set].data();

  auto atomic_add = [&](auto bin_idx, auto const& adjusted) {
    if constexpr (Policy::kSharedMem) {
      AtomicAddGpairShared(smem_hist + bin_idx, adjusted);
    } else {
      AtomicAddGpairGlobal(gmem_hist + bin_idx, adjusted);
    }
  };

  auto process_valid_tile = [&](auto idx) {
    auto ridx_in_node = idx / feature_stride;
    Idx ridx = p_ridx[ridx_in_node];
    auto fidx = FeatIdx(group, idx, ridx_in_node, feature_stride);
    bst_bin_t compressed_bin = matrix.gidx_iter[IterIdx(matrix, ridx, fidx)];
    if (Policy::kDense || compressed_bin != matrix.NullValue()) {
      if constexpr (Policy::kCompressed) {
        compressed_bin += matrix.feature_segments[fidx];
      }
      if constexpr (Policy::kSharedMem) {
        // Handle privatized histogram indexing.
        compressed_bin = (compressed_bin - group.start_bin) * n_targets;
      } else {
        compressed_bin *= n_targets;
      }
      // TODO(jiamingy): Assign a thread for each target. When there are multiple targets,
      // this can cause significant stall. Enable vector load if possible.
      //
      // TODO(jiamingy): When the number of targets is non-trivial, we need to split up
      // the histograms due to shared memory size.
      for (bst_target_t t = 0; t < n_targets; ++t) {
        auto adjusted = d_roundings[t].ToFixedPoint(d_gpair(ridx - matrix.base_rowid, t));
        atomic_add(compressed_bin + t, adjusted);
      }
    }
  };

  auto process_gpair_tile = [&](auto full_tile, auto offset) {
#pragma unroll 1
    for (std::int32_t j = 0; j < Policy::kItemsPerThread; ++j) {
      std::int32_t const idx = offset + j * Policy::kBlockThreads + threadIdx.x;
      if (full_tile || idx < n_elements) {
        process_valid_tile(idx);
      }
    }
  };

  while (offset < n_elements) {
    std::int32_t const valid_items =
        cuda::std::min(n_elements - offset, static_cast<std::size_t>(Policy::kTileSize));
    if (Policy::kTileSize == valid_items) {
      process_gpair_tile(std::true_type{}, offset);
    } else {
      process_gpair_tile(std::false_type{}, offset);
    }
    offset += kStride;
  }

  if constexpr (!Policy::kSharedMem) {
    return;
  }

  // Write shared memory back to global memory
  __syncthreads();

  auto start_bin = group.start_bin * n_targets;
  for (auto i : dh::BlockStrideRange(0u, group.num_bins * n_targets)) {
    AtomicAddGpairGlobal(gmem_hist + start_bin + i, smem_hist[i]);
  }
}

namespace {
constexpr std::int32_t kBlockThreads = 1024;
constexpr std::int32_t kItemsPerThread = 8;
constexpr std::int32_t ItemsPerTile() { return kBlockThreads * kItemsPerThread; }
template <auto Ker>
using DeduceKernelT = std::decay_t<decltype(Ker)>;
}  // namespace

// Use auto deduction guide to workaround compiler error.
template <typename Accessor>
struct HistogramKernel {
  enum KernelType : std::size_t {
    // single-target
    kGlobalCompr = 0,
    kGlobal = 1,
    kSharedCompr = 2,
    kShared = 3,
    kGlobalDense = 4,
    kSharedDense = 5,
  };
  /**
   * Single-target
   */
  // Kernel for working with compressed sparse Ellpack using the global memory.
  using GlobalCompr = DeduceKernelT<
      SharedMemHistKernel<Accessor, true, false, false, kBlockThreads, kItemsPerThread>>;
  GlobalCompr global_compr_kernel{
      SharedMemHistKernel<Accessor, true, false, false, kBlockThreads, kItemsPerThread>};
  // Kernel for working with sparse Ellpack using the global memory.
  using Global = DeduceKernelT<
      SharedMemHistKernel<Accessor, false, false, false, kBlockThreads, kItemsPerThread>>;
  Global global_kernel{
      SharedMemHistKernel<Accessor, false, false, false, kBlockThreads, kItemsPerThread>};
  // Kernel for working with compressed sparse Ellpack using the shared memory.
  using SharedCompr = DeduceKernelT<
      SharedMemHistKernel<Accessor, true, false, true, kBlockThreads, kItemsPerThread>>;
  SharedCompr shared_compr_kernel{
      SharedMemHistKernel<Accessor, true, false, true, kBlockThreads, kItemsPerThread>};
  // Kernel for working with sparse Ellpack using the shared memory.
  using Shared = DeduceKernelT<
      SharedMemHistKernel<Accessor, false, false, true, kBlockThreads, kItemsPerThread>>;
  Shared shared_kernel{
      SharedMemHistKernel<Accessor, false, false, true, kBlockThreads, kItemsPerThread>};
  // Kernel for working with compressed dense ellpack using the global memory
  using GlobalDense = DeduceKernelT<
      SharedMemHistKernel<Accessor, true, true, false, kBlockThreads, kItemsPerThread>>;
  GlobalDense global_dense_kernel{
      SharedMemHistKernel<Accessor, true, true, false, kBlockThreads, kItemsPerThread>};
  // Kernel for working with compressed dense ellpack using the shared memory
  using SharedDense = DeduceKernelT<
      SharedMemHistKernel<Accessor, true, true, true, kBlockThreads, kItemsPerThread>>;
  SharedDense shared_dense_kernel{
      SharedMemHistKernel<Accessor, true, true, true, kBlockThreads, kItemsPerThread>};

  bool shared{false};
  std::array<std::uint32_t, 12> grid_sizes;
  std::size_t smem_size{0};
  std::size_t const max_shared_memory;
  bool const force_global;

  HistogramKernel(Context const* ctx, FeatureGroupsAccessor const& feature_groups,
                  bool force_global_memory)
      : max_shared_memory{dh::MaxSharedMemoryOptin(ctx->Ordinal())},
        force_global{force_global_memory} {
    std::fill_n(grid_sizes.data(), grid_sizes.size(), 0);
    // Decide whether to use shared memory
    // Opt into maximum shared memory for the kernel if necessary
    this->smem_size = feature_groups.ShmemSize(/*n_targets=*/1);
    this->shared = !force_global_memory && this->smem_size <= this->max_shared_memory;
    this->smem_size = this->shared ? this->smem_size : 0;

    auto init = [&](auto& kernel, KernelType k) {
      if (this->shared) {
        dh::safe_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           this->max_shared_memory));
      }

      // determine the launch configuration
      std::int32_t num_groups = feature_groups.NumGroups();
      std::int32_t n_mps = curt::GetMpCnt(ctx->Ordinal());

      std::int32_t n_blocks_per_mp = 0;
      dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n_blocks_per_mp, kernel,
                                                                  kBlockThreads, this->smem_size));

      // This gives the number of blocks to keep the device occupied Use this as the
      // maximum number of blocks
      this->grid_sizes[static_cast<std::size_t>(k)] = n_blocks_per_mp * n_mps;
    };
    // Initialize all kernel instantiations
    // Single target
    std::array kernel_types{kGlobalCompr, kGlobal,      kSharedCompr,
                            kShared,      kGlobalDense, kSharedDense};
    std::int32_t k = 0;
    for (auto& kernel : {global_compr_kernel, global_kernel, shared_compr_kernel, shared_kernel,
                         global_dense_kernel, shared_dense_kernel}) {
      init(kernel, kernel_types[k]);
      ++k;
    }
  }
};

struct MtHistKernel {
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
                             std::uint32_t* p_out_blocks) {
    CHECK_GT(max_blocks_per_node, 0);
    std::vector<std::uint32_t> blk_ptr{0};
    std::uint32_t n_total_blocks = 0;
    for (std::size_t j = 1; j < sizes_csum.size(); ++j) {
      auto nidx_in_set = j - 1;
      auto n_samples = sizes_csum[j] - sizes_csum[j - 1];
      std::size_t items_per_group = n_samples * columns_per_group;
      auto n_blocks = common::DivRoundUp(items_per_group, Policy::kTileSize);
      CHECK_GT(n_blocks, 0);  // at least one block for each node.
      n_blocks = std::min(n_blocks, max_blocks_per_node);
      blk_ptr.push_back(blk_ptr[nidx_in_set] + n_blocks);
      n_total_blocks += n_blocks;
    }
    CHECK_EQ(n_total_blocks, blk_ptr.back());
    *p_out_blocks = blk_ptr.back();
    return dh::device_vector<std::uint32_t>{blk_ptr};
  }

  struct MtHistKernelConfig {
    std::int32_t n_blocks_per_mp = 0;
    std::size_t shmem_bytes = 0;
  };
  // Maps kernel instantiations to their configurations. This is a mutable state, as a
  // result the histogram kernel is not thread safe.
  std::map<void*, MtHistKernelConfig> cfg;
  // The number of multi-processor for the selected GPU
  std::int32_t const n_mps;
  // Maximum size of the shared memory (optin)
  std::size_t const max_shared_bytes;
  // Use global memory for testing
  bool const force_global;

  explicit MtHistKernel(Context const* ctx, bool force_global)
      : n_mps{curt::GetMpCnt(ctx->Ordinal())},
        max_shared_bytes{dh::MaxSharedMemoryOptin(ctx->Ordinal())},
        force_global{force_global} {}

  template <std::int32_t kBlockThreads, bool kDense, bool kCompressed, typename Accessor,
            typename RidxIterSpan>
  void DispatchHistShmem(Context const* ctx, Accessor const& matrix,
                         FeatureGroupsAccessor const& feature_groups,
                         linalg::MatrixView<GradientPair const> gpair, RidxIterSpan* ridx_iters,
                         common::Span<common::Span<GradientPairInt64>> hists,
                         std::vector<std::size_t> const& h_sizes_csum,
                         common::Span<GradientQuantiser const> roundings) {
    auto n_targets = gpair.Shape(1);

    std::size_t shmem_bytes = feature_groups.ShmemSize(n_targets);
    bool use_shared = !force_global && shmem_bytes <= this->max_shared_bytes;
    shmem_bytes = use_shared ? shmem_bytes : 0;

    auto launch = [&](auto policy, auto kernel) {
      using Policy = common::GetValueT<decltype(policy)>;
      int columns_per_group = common::DivRoundUp(matrix.row_stride, feature_groups.NumGroups());
      auto v = this->cfg.at(reinterpret_cast<void*>(kernel));
      CHECK_GT(v.n_blocks_per_mp, 0);
      std::uint32_t n_blocks = 0;
      auto blk_ptr = AllocateBlocks<Policy>(h_sizes_csum, columns_per_group,
                                            v.n_blocks_per_mp * n_mps, &n_blocks);
      CHECK_GE(n_blocks, hists.size());
      dim3 conf(n_blocks, feature_groups.NumGroups());

      kernel<<<conf, Policy::kBlockThreads, shmem_bytes, ctx->CUDACtx()->Stream()>>>(
          matrix, feature_groups, ridx_iters, dh::ToSpan(blk_ptr), hists.data(), gpair, roundings);
      dh::safe_cuda(cudaPeekAtLastError());
    };

    if (use_shared) {
      using Policy = HistPolicy<kBlockThreads, kItemsPerThread, kDense, kCompressed, true>;
      auto kernel = HistKernel<Policy, Accessor, RidxIterSpan>;
      auto it = cfg.find(reinterpret_cast<void*>(kernel));
      if (it == cfg.cend()) {
        MtHistKernelConfig v;
        // This function is the reason for all this trouble to cache the
        // configuration. It blocks the device.
        //
        // Also, it must precede the `cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
        // otherwise the shmem bytes might be invalid.
        if (shmem_bytes > v.shmem_bytes) {
          dh::safe_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             shmem_bytes));
          v.shmem_bytes = shmem_bytes;
        }
        // Use this as a limiter, works for root node. Not too bad an option for child nodes.
        dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &v.n_blocks_per_mp, kernel, Policy::kBlockThreads, shmem_bytes));
        this->cfg[reinterpret_cast<void*>(kernel)] = v;
      }
      launch(Policy{}, kernel);
    } else {
      using Policy = HistPolicy<kBlockThreads, kItemsPerThread, kDense, kCompressed, false>;
      auto kernel = HistKernel<Policy, Accessor, RidxIterSpan>;
      auto it = cfg.find(reinterpret_cast<void*>(kernel));
      if (it == cfg.cend()) {
        MtHistKernelConfig v;
        dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &v.n_blocks_per_mp, kernel, Policy::kBlockThreads, shmem_bytes));
        this->cfg[reinterpret_cast<void*>(kernel)] = v;
      }
      launch(Policy{}, kernel);
    }
  }

  template <std::int32_t kBlockThreads, typename Accessor, typename... Args>
  void DispatchHistCompress(Context const* ctx, Accessor const& matrix, Args&&... args) {
    if (matrix.IsDense()) {
      DispatchHistShmem<kBlockThreads, true, true>(ctx, matrix, std::forward<Args>(args)...);
    } else if (matrix.IsDenseCompressed()) {
      DispatchHistShmem<kBlockThreads, false, true>(ctx, matrix, std::forward<Args>(args)...);
    } else {
      DispatchHistShmem<kBlockThreads, false, false>(ctx, matrix, std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  void DispatchHistPolicyBlockSize(Args&&... args) {
    // An heuristic to choose the block size based on the number of multi processors.
    //
    // The usual practice is to use the SM version for tuning. But we don't have the
    // resource to followup with benchmarks for every architecture.
    constexpr std::int32_t kMpThreshold = 128;
    if (this->n_mps >= kMpThreshold) {
      constexpr std::int32_t kBlockThreads = 1024;
      DispatchHistCompress<kBlockThreads>(std::forward<Args>(args)...);
    } else {
      constexpr std::int32_t kBlockThreads = 768;
      DispatchHistCompress<kBlockThreads>(std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  void Dispatch(Args&&... args) {
    this->DispatchHistPolicyBlockSize(std::forward<Args>(args)...);
  }
};

template <typename Accessor>
class DeviceHistogramDispatchAccessor {
  std::unique_ptr<HistogramKernel<Accessor>> kernel_{nullptr};
  std::unique_ptr<MtHistKernel> mt_kernel_{nullptr};

 public:
  void Reset(Context const* ctx, FeatureGroupsAccessor const& feature_groups,
             bool force_global_memory) {
    this->kernel_ =
        std::make_unique<HistogramKernel<Accessor>>(ctx, feature_groups, force_global_memory);

    this->mt_kernel_ = std::make_unique<MtHistKernel>(ctx, force_global_memory);

    if (force_global_memory) {
      CHECK(!this->kernel_->shared);
    }
  }

  void BuildHistogram(Context const* ctx, Accessor const& matrix,
                      FeatureGroupsAccessor const& feature_groups,
                      common::Span<GradientPair const> gpair,
                      common::Span<const cuda_impl::RowIndexT> d_ridx,
                      common::Span<GradientPairInt64> histogram, GradientQuantiser rounding) const {
    CHECK(kernel_);
    // Otherwise launch blocks such that each block has a minimum amount of work to do
    // There are fixed costs to launching each block, e.g. zeroing shared memory
    // The below amount of minimum work was found by experimentation
    int columns_per_group = common::DivRoundUp(matrix.row_stride, feature_groups.NumGroups());
    // Average number of matrix elements processed by each group
    std::size_t items_per_group = d_ridx.size() * columns_per_group;

    // Allocate number of blocks such that each block has about kMinItemsPerBlock work
    // Up to a maximum where the device is saturated
    auto constexpr kMinItemsPerBlock = ItemsPerTile();

    auto launcher = [&](auto const& kernel, std::uint32_t grid_size) {
      CHECK_NE(grid_size, 0);
      grid_size = std::min(grid_size, static_cast<std::uint32_t>(
                                          common::DivRoundUp(items_per_group, kMinItemsPerBlock)));
      dh::LaunchKernel{dim3(grid_size, feature_groups.NumGroups()),  // NOLINT
                       static_cast<uint32_t>(kBlockThreads), kernel_->smem_size,
                       ctx->CUDACtx()->Stream()}(kernel, matrix, feature_groups, d_ridx,
                                                 histogram.data(), gpair.data(), rounding);
    };

    using K = HistogramKernel<EllpackDeviceAccessor>::KernelType;
    if (!this->kernel_->shared) {  // Use global memory
      CHECK_EQ(this->kernel_->smem_size, 0);
      if (matrix.IsDense()) {
        CHECK(this->kernel_->force_global ||
              (feature_groups.ShmemSize(1) >= this->kernel_->max_shared_memory));
        launcher(this->kernel_->global_dense_kernel, this->kernel_->grid_sizes[K::kGlobalDense]);
      } else if (matrix.IsDenseCompressed()) {
        CHECK(this->kernel_->force_global ||
              (feature_groups.ShmemSize(1) >= this->kernel_->max_shared_memory));
        launcher(this->kernel_->global_compr_kernel, this->kernel_->grid_sizes[K::kGlobalCompr]);
      } else {
        // Sparse
        launcher(this->kernel_->global_kernel, this->kernel_->grid_sizes[K::kGlobal]);
      }
    } else {  // Use shared memory
      CHECK_NE(this->kernel_->smem_size, 0);
      if (matrix.IsDense()) {
        launcher(this->kernel_->shared_dense_kernel, this->kernel_->grid_sizes[K::kSharedDense]);
      } else if (matrix.IsDenseCompressed()) {
        // Dense
        launcher(this->kernel_->shared_compr_kernel, this->kernel_->grid_sizes[K::kSharedCompr]);
      } else {
        // Sparse
        launcher(this->kernel_->shared_kernel, this->kernel_->grid_sizes[K::kShared]);
      }
    }
  }

  void BuildHistogram(Context const* ctx, Accessor const& matrix,
                      FeatureGroupsAccessor const& feature_groups,
                      linalg::MatrixView<GradientPair const> gpair,
                      common::Span<common::Span<cuda_impl::RowIndexT const>> ridxs,
                      common::Span<common::Span<GradientPairInt64>> hists,
                      std::vector<std::size_t> const& h_sizes_csum,
                      common::Span<GradientQuantiser const> roundings) {
    std::size_t n_total_samples = h_sizes_csum.back();
    if (ridxs.size() == 1 && n_total_samples == matrix.n_rows) {
      // Special optimization for the root node.
      using RidxIter = thrust::counting_iterator<cuda_impl::RowIndexT>;
      CHECK_LT(matrix.base_rowid, std::numeric_limits<cuda_impl::RowIndexT>::max());
      auto iter = common::IterSpan{
          thrust::make_counting_iterator(static_cast<cuda_impl::RowIndexT>(matrix.base_rowid)),
          matrix.n_rows};
      dh::caching_device_vector<common::IterSpan<RidxIter>> ridx_iters(hists.size(), iter);
      this->mt_kernel_->Dispatch(ctx, matrix, feature_groups, gpair, ridx_iters.data().get(), hists,
                                 h_sizes_csum, roundings);
    } else {
      using RidxIter = cuda_impl::RowIndexT const;
      this->mt_kernel_->Dispatch(ctx, matrix, feature_groups, gpair, ridxs.data(), hists,
                                 h_sizes_csum, roundings);
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
                                   FeatureGroupsAccessor const& feature_groups,
                                   bst_bin_t n_total_bins, bool force_global_memory) {
  this->monitor_.Start(__func__);
  this->p_impl_->Reset(ctx, feature_groups, force_global_memory);
  this->hist_.Reset(ctx, n_total_bins, max_cached_hist_nodes);
  this->monitor_.Stop(__func__);
}

void DeviceHistogramBuilder::BuildHistogram(Context const* ctx, EllpackAccessor const& matrix,
                                            FeatureGroupsAccessor const& feature_groups,
                                            common::Span<GradientPair const> gpair,
                                            common::Span<const cuda_impl::RowIndexT> ridx,
                                            common::Span<GradientPairInt64> histogram,
                                            GradientQuantiser rounding) {
  this->monitor_.Start(__func__);
  std::visit(
      [&](auto&& matrix) {
        this->p_impl_->BuildHistogram(ctx, matrix, feature_groups, gpair, ridx, histogram,
                                      rounding);
      },
      matrix);
  this->monitor_.Stop(__func__);
}

void DeviceHistogramBuilder::BuildHistogram(
    Context const* ctx, EllpackAccessor const& matrix, FeatureGroupsAccessor const& feature_groups,
    linalg::MatrixView<GradientPair const> gpair,
    common::Span<common::Span<cuda_impl::RowIndexT const>> ridxs,
    common::Span<common::Span<GradientPairInt64>> hists,
    std::vector<std::size_t> const& h_sizes_csum, common::Span<GradientQuantiser const> roundings) {
  std::visit(
      [&](auto&& matrix) {
        this->p_impl_->BuildHistogram(ctx, matrix, feature_groups, gpair, ridxs, hists,
                                      h_sizes_csum, roundings);
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
