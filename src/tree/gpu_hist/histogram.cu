/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <algorithm>             // for min, max
#include <cstddef>               // for size_t
#include <cstdint>               // uint32_t, int32_t
#include <cuda/std/type_traits>  // for cuda::std::alignment_of_v
#include <memory>                // for unique_ptr
#include <vector>                // for vector

#include "../../collective/aggregator.h"
#include "../../common/cuda_compat.cuh"   // for CUDA compatibility
#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../common/cuda_rt_utils.h"   // for GetMpCnt
#include "../../common/device_helpers.cuh"
#include "../../data/ellpack_page.cuh"
#include "histogram.cuh"
#include "row_partitioner.cuh"
#include "xgboost/base.h"

namespace xgboost::tree {
namespace {
// Convert a global row index and an entry's position within the row to a page-local ELLPACK index.
// For dense layouts, fidx is a feature index; for sparse layouts, it is a padded-row offset.
template <typename IterT>
XGBOOST_DEV_INLINE bst_idx_t IterIdx(EllpackAccessorImpl<IterT> const& matrix,
                                     RowPartitioner::RowIndexT ridx, bst_feature_t fidx) {
  return (ridx - matrix.base_rowid) * matrix.row_stride + fidx;
}
}  // anonymous namespace

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
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications
// Compute capability       7.5  | 8.0  | 8.6 8.7 | 8.9  | 9.0 10.0 10.3 | 11.0 12.0 12.1
// Max resident blocks/SM   16   | 32   | 16      | 24   | 32            | 24
// Max resident warps/SM    32   | 64   | 48      | 48   | 64            | 48
// Max resident threads/SM  1024 | 2048 | 1536    | 1536 | 2048          | 1536

using HistSm75 = HistTuning<1024, 1>;

using HistSm80 = HistTuning<1024, 2>;

using HistSm86 = HistTuning<768, 2>;

using HistSm90 = HistTuning<1024, 2>;

using HistSm110 = HistTuning<768, 2>;

// Resolve launch bounds in the device compilation pass, not through template arguments.
#if __CUDA_ARCH__ >= 1100
using HistLaunchBounds = HistSm110;
#elif __CUDA_ARCH__ >= 900
using HistLaunchBounds = HistSm90;
#elif __CUDA_ARCH__ >= 860
using HistLaunchBounds = HistSm86;
#elif __CUDA_ARCH__ >= 800
using HistLaunchBounds = HistSm80;
#else
using HistLaunchBounds = HistSm75;
#endif

template <typename HistArchPolicy, bool Dense, bool Compressed, bool SharedMem, bool SingleTarget>
struct HistPolicy : public HistArchPolicy {
  static constexpr bool kDense = Dense;
  static constexpr bool kCompressed = Compressed;
  static constexpr bool kSharedMem = SharedMem;
  static constexpr bool kSingleTarget = SingleTarget;
};

template <typename Fn>
decltype(auto) DispatchCudaSm(std::int32_t device, Fn&& fn) {
  std::int32_t version = 0;
  dh::safe_cuda(cub::SmVersion(version, device));
  if (version >= 1100) {
    return fn(HistSm110{});
  } else if (version >= 900) {
    return fn(HistSm90{});
  } else if (version >= 860) {
    return fn(HistSm86{});
  } else if (version >= 800) {
    return fn(HistSm80{});
  }
  return fn(HistSm75{});
}

// Only sm_90 and sm_100 reach the cap, and a larger budget (113KB) is slower on H200.
constexpr std::size_t kMaxShmemBytes = 96 /*kb*/ * 1024;
// The shared memory of a block is allocated in units of this size.
constexpr std::int32_t kShmemAllocGranularity = 128;
}  // anonymous namespace

// Split the SM's shared memory among the same number of blocks targeted by the launch bounds.
std::size_t HistShmemBytes(std::int32_t device) {
  auto min_blocks = DispatchCudaSm(device, [](auto arch) { return decltype(arch)::kMinBlocks; });
  auto optin = dh::MaxSharedMemoryOptin(device);
  std::int32_t smem_per_sm = 0, reserved = 0;
  dh::safe_cuda(
      cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  dh::safe_cuda(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, device));

  // Each block is additionally charged a fixed driver reservation. Round down to the
  // allocation granularity, otherwise the last block does not fit.
  auto n_bytes_per_block =
      (smem_per_sm / min_blocks / kShmemAllocGranularity) * kShmemAllocGranularity - reserved;
  CHECK_GT(n_bytes_per_block, 0);
  return std::min({static_cast<std::size_t>(n_bytes_per_block), optin, kMaxShmemBytes});
}
namespace {
__device__ GradientPairInt64 LoadGpair(GradientPairInt64 const* XGBOOST_RESTRICT gpairs) {
  static_assert(sizeof(int4) == sizeof(GradientPairInt64));
  auto g = *reinterpret_cast<int4 const*>(gpairs);
  return *reinterpret_cast<GradientPairInt64*>(&g);
}

// Accumulate sub-segment entries [begin, end) for one target.
template <typename Policy, typename Accessor, typename RidxIterSpan>
__device__ void HistKernelSegment(Accessor const& matrix, FeatureGroup const& group,
                                  RidxIterSpan d_ridx_iter, GradientPairInt64 const* gpair,
                                  GradientPairInt64* smem_hist, GradientPairInt64* gmem_hist,
                                  bst_idx_t begin, bst_idx_t end) {
  bst_feature_t const feature_stride = Policy::kCompressed ? group.num_features : matrix.row_stride;

  using Idx = RowPartitioner::RowIndexT;

  auto const d_ridx = d_ridx_iter.data();

  auto atomic_add = [&](auto bin_idx, auto const& adjusted) {
    if constexpr (Policy::kSharedMem) {
      AtomicAddGpairShared(smem_hist + bin_idx, adjusted);
    } else {
      // gmem_hist is a subspan for the current target.
      AtomicAddGpairGlobal(gmem_hist + bin_idx, adjusted);
    }
  };

  for (bst_idx_t idx = begin + threadIdx.x; idx < end; idx += Policy::kBlockThreads) {
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
    if (Policy::kDense || compressed_bin != static_cast<bst_bin_t>(matrix.NullValue())) {
      auto g = LoadGpair(gpair + ridx);
      if constexpr (Policy::kCompressed) {
        compressed_bin += matrix.feature_segments[fidx];
      }
      if constexpr (Policy::kSharedMem) {
        compressed_bin -= group.start_bin;
      }
      atomic_add(compressed_bin, g);
    }
  }
}

}  // namespace

/**
 * @brief Each block processes one sub-segment for one target, stopping at the segment's end.
 *
 * Block index:
 *
 *     ((nidx_in_set * n_subsegments_per_segment + subsegment) * n_groups + group) * n_targets + target
 *
 * Targets vary fastest, then feature groups, to encourage cache reuse across blocks.
 * All segments use the largest segment's sub-segment count.
 * Blocks starting beyond their segment exit immediately.
 */
template <typename Policy, typename Accessor, typename RidxIterSpan>
__global__ __launch_bounds__(
    HistLaunchBounds::kBlockThreads,
    HistLaunchBounds::kMinBlocks) void HistogramKernel(Accessor const matrix,
                                                       FeatureGroupsAccessor const feature_groups,
                                                       RidxIterSpan const* d_ridx_iters,
                                                       common::Span<GradientPairInt64> const*
                                                           node_hists,
                                                       GradientPairInt64 const* d_gpair,
                                                       bst_idx_t n_samples, bst_target_t n_targets,
                                                       bst_idx_t n_entries_per_subsegment,
                                                       std::uint32_t n_subsegments_per_segment) {
  if constexpr (Policy::kSingleTarget) {
    // Constant propagation removes the target indexing and saves registers.
    n_targets = 1;
  }
  // The sparse layout scans the whole row, it has a single group.
  std::uint32_t const n_groups = Policy::kCompressed ? feature_groups.NumGroups() : 1;

  // (node, subsegment, feature group, target)
  std::uint32_t blk_idx = blockIdx.x;
  bst_target_t const target_idx = blk_idx % n_targets;
  blk_idx /= n_targets;
  bst_feature_t const gidx = blk_idx % n_groups;
  blk_idx /= n_groups;
  std::uint32_t const subsegment_idx = blk_idx % n_subsegments_per_segment;
  std::size_t const nidx_in_set = blk_idx / n_subsegments_per_segment;

  auto const group = feature_groups[gidx];
  auto const d_ridx = d_ridx_iters[nidx_in_set];
  bst_feature_t const feature_stride = Policy::kCompressed ? group.num_features : matrix.row_stride;
  bst_idx_t const n_segment_entries = d_ridx.size() * feature_stride;
  bst_idx_t const begin = static_cast<bst_idx_t>(subsegment_idx) * n_entries_per_subsegment;
  if (begin >= n_segment_entries) {
    // This block's sub-segment begins beyond the segment's entries.
    return;
  }
  bst_idx_t const end = cuda::std::min(begin + n_entries_per_subsegment, n_segment_entries);

  extern __align__(std::alignment_of_v<GradientPairInt64>) __shared__ char shmem[];
  // Privatized histogram
  auto smem_hist = reinterpret_cast<GradientPairInt64*>(shmem);

  auto d_node_hist = node_hists[nidx_in_set];
  // With a target-major layout.
  auto gmem_hist = d_node_hist.data() + target_idx * (d_node_hist.size() / n_targets);
  // hint for PTX: atom.add.u64 -> atom.global.add.u64
  __builtin_assume(__isGlobal(gmem_hist));

  if constexpr (Policy::kSharedMem) {
    dh::BlockFill(smem_hist, group.num_bins, GradientPairInt64{});
    __syncthreads();
  }
  HistKernelSegment<Policy>(matrix, group, d_ridx, d_gpair + n_samples * target_idx, smem_hist,
                            gmem_hist, begin, end);
  if constexpr (Policy::kSharedMem) {
    __syncthreads();
    // Flush this block's histogram.
    for (auto bin_idx : dh::BlockStrideRange(0, group.num_bins)) {
      AtomicAddGpairGlobal(gmem_hist + group.start_bin + bin_idx, smem_hist[bin_idx]);
    }
  }
}

namespace cuda_impl {
bst_idx_t SliceSegment(bst_idx_t n_total_entries, std::size_t entries_per_tile,
                       std::size_t n_resident_blks_per_target, bst_target_t n_targets,
                       std::size_t hist_bytes_per_block, std::uint32_t symbol_bits) {
  CHECK_GT(n_total_entries, 0);
  CHECK_GT(entries_per_tile, 0);
  CHECK_GT(n_resident_blks_per_target, 0);
  CHECK_GT(n_targets, 0);
  CHECK_GT(symbol_bits, 0);

  std::size_t constexpr kPercent = 100, kBitsPerByte = 8;
  // For the same sub-segment, each target's block flushes a separate histogram. Count
  // ELLPACK entries once, assuming reuse across targets. These are payload estimates.
  auto flush_bytes_all_targets = static_cast<std::size_t>(n_targets) * hist_bytes_per_block;
  auto entry_bits_per_tile = entries_per_tile * static_cast<std::size_t>(symbol_bits);
  // Process enough tiles that flush bytes are at most kMaxFlushPercent of entry bytes:
  //   flush_bytes_all_targets / (tiles_per_subsegment * entry_bits_per_tile / 8) <= 25 / 100.
  auto min_tiles_for_flush = common::DivRoundUp(kPercent * kBitsPerByte * flush_bytes_all_targets,
                                                kMaxFlushPercent * entry_bits_per_tile);

  // Avoid too many small blocks (kTargetWaves), but keep enough blocks for parallelism (kMinWaves).
  // The maximum sub-segment size takes priority when the flush budget cannot be met.
  auto n_total_tiles = common::DivRoundUp(n_total_entries, entries_per_tile);
  auto min_tiles_per_subsegment =
      common::DivRoundUp(n_total_tiles, n_resident_blks_per_target * kTargetWaves);
  auto max_tiles_per_subsegment =
      common::DivRoundUp(n_total_tiles, n_resident_blks_per_target * kMinWaves);
  auto tiles_per_subsegment =
      std::clamp(min_tiles_for_flush, min_tiles_per_subsegment, max_tiles_per_subsegment);
  return static_cast<bst_idx_t>(tiles_per_subsegment) * entries_per_tile;
}
}  // namespace cuda_impl

namespace {
/** @brief Histogram launch dimensions. */
struct Grid {
  /** @brief Entries per full sub-segment. */
  bst_idx_t n_entries_per_subsegment;
  /** @brief Number of sub-segments needed to cover the largest segment. */
  std::uint32_t n_subsegments_per_segment;
  /** @brief The total number of blocks. */
  std::uint32_t n_blks;
};

// Use the largest segment's sub-segment count for every segment and target. Increase
// entries per sub-segment if needed to fit the block-count limit.
[[nodiscard]] Grid MakeGrid(bst_idx_t max_segment_entries, bst_idx_t n_entries_per_subsegment,
                            std::size_t tile_size, std::uint32_t n_groups, bst_target_t n_targets,
                            std::size_t n_nodes) {
  CHECK_GT(max_segment_entries, 0);
  CHECK_GT(n_entries_per_subsegment, 0);
  CHECK_GT(tile_size, 0);
  CHECK_GT(n_groups, 0);
  CHECK_GT(n_targets, 0);
  CHECK_GT(n_nodes, 0);

  constexpr std::uint64_t kMaxGrid = std::numeric_limits<std::int32_t>::max();
  // Each group, target, node requires a different block.
  auto n_unique_blks_per_subsegment = static_cast<std::uint64_t>(n_groups) * n_targets * n_nodes;
  CHECK_LE(n_unique_blks_per_subsegment, kMaxGrid) << "Too many blocks for the histogram kernel.";
  // Keep the policy's sub-segment size unless the grid limit requires more entries,
  // rounded up to whole tiles. This is a safety guard, not an optimization.
  auto const max_subsegments = kMaxGrid / n_unique_blks_per_subsegment;
  auto const min_entries = common::DivRoundUp(max_segment_entries, max_subsegments);
  n_entries_per_subsegment =
      std::max(n_entries_per_subsegment, common::DivRoundUp(min_entries, tile_size) * tile_size);

  auto n_subsegments = common::DivRoundUp(max_segment_entries, n_entries_per_subsegment);
  auto n_blks = n_subsegments * n_unique_blks_per_subsegment;
  // The sub-segments cover the largest segment, and the grid fits.
  CHECK_GE(n_subsegments * n_entries_per_subsegment, max_segment_entries);
  CHECK_LE(n_blks, kMaxGrid);
  return {n_entries_per_subsegment, static_cast<std::uint32_t>(n_subsegments),
          static_cast<std::uint32_t>(n_blks)};
}
}  // anonymous namespace

// Dispatcher for the histogram kernel.
struct HistKernel {
  // Maps kernel instantiations to the number of resident blocks per MP. This is a mutable
  // state, as a result the histogram kernel is not thread safe.
  std::map<void*, std::int32_t> cfg;
  // The number of multi-processor for the selected GPU
  std::int32_t const n_mps;
  // Maximum size of the shared memory (optin)
  std::size_t const max_shared_bytes;
  // Use global memory for testing
  bool const force_global;

  // Obtain the (cached) number of resident blocks per MP for a kernel.
  template <typename Policy, typename Kernel>
  [[nodiscard]] std::int32_t BlocksPerMp(Policy, std::size_t shmem_bytes, Kernel kernel) {
    auto [it, inserted] = this->cfg.try_emplace(reinterpret_cast<void*>(kernel), 0);
    if (inserted) {
      if (shmem_bytes > 0) {
        // This function is the reason for all this trouble to cache the
        // configuration. It blocks the device.
        //
        // Also, it must precede the `cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
        // otherwise the shmem bytes might be invalid.
        dh::safe_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           this->max_shared_bytes));
      }
      dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &it->second, kernel, Policy::kBlockThreads, shmem_bytes));
      CHECK_GT(it->second, 0);
    }
    return it->second;
  }

  explicit HistKernel(Context const* ctx, bool force_global)
      : n_mps{curt::GetMpCnt(ctx->Ordinal())},
        max_shared_bytes{dh::MaxSharedMemoryOptin(ctx->Ordinal())},
        force_global{force_global} {}

  template <bool kDense, bool kCompressed, typename Accessor, typename RidxIterSpan>
  void DispatchHist(Context const* ctx, Accessor const& matrix,
                    FeatureGroups const& h_feature_groups,
                    linalg::MatrixView<GradientPairInt64 const> gpair,
                    std::vector<RidxIterSpan> const& h_ridx_iters,
                    std::vector<common::Span<GradientPairInt64>> const& h_hists) {
    CHECK(gpair.FContiguous());
    CHECK_EQ(h_ridx_iters.size(), h_hists.size());
    auto feature_groups = h_feature_groups.DeviceAccessor(ctx->Device());
    // Sparse data requires one group because the kernel scans every entry in each padded row.
    CHECK(kCompressed || feature_groups.NumGroups() == 1);
    auto n_samples = gpair.Shape(0);
    auto n_targets = gpair.Shape(1);
    auto d_gpair = gpair.Values().data();

    // Count all rows for sub-segment sizing and find the largest node for grid sizing.
    bst_idx_t n_total_rows = 0, max_node_rows = 0;
    for (auto const& ridx : h_ridx_iters) {
      n_total_rows += ridx.size();
      max_node_rows = std::max(max_node_rows, static_cast<bst_idx_t>(ridx.size()));
    }
    // No entry to accumulate, for example all values are missing.
    if (n_total_rows == 0 || matrix.row_stride == 0) {
      return;
    }

    std::size_t shmem_bytes = feature_groups.ShmemSize();
    bool use_shared = !force_global && shmem_bytes <= this->max_shared_bytes;
    shmem_bytes = use_shared ? shmem_bytes : 0;

    // Maximum entries per row in a segment: the widest group, or the full sparse row.
    bst_feature_t max_group_features = matrix.row_stride;
    if (kCompressed) {
      auto const& h_fs = h_feature_groups.feature_segments.ConstHostVector();
      max_group_features = 0;
      for (std::size_t i = 1; i < h_fs.size(); ++i) {
        max_group_features = std::max(max_group_features, h_fs[i] - h_fs[i - 1]);
      }
    }
    auto n_groups = static_cast<std::uint32_t>(kCompressed ? feature_groups.NumGroups() : 1);
    bst_idx_t max_segment_entries = max_node_rows * max_group_features;

    dh::TemporaryArray<RidxIterSpan> ridx_iters(h_ridx_iters.size());
    dh::TemporaryArray<common::Span<GradientPairInt64>> hists(h_hists.size());
    auto stream = ctx->CUDACtx()->Stream();
    dh::CopyTo(h_ridx_iters, &ridx_iters, stream);
    dh::CopyTo(h_hists, &hists, stream);

    bst_idx_t n_total_entries = n_total_rows * matrix.row_stride;
    auto const symbol_bits = matrix.SymbolBits();
    auto launch = [&](auto policy) {
      using Policy = decltype(policy);
      auto kernel = HistogramKernel<Policy, Accessor, RidxIterSpan>;
      auto n_blks_per_mp = this->BlocksPerMp(Policy{}, shmem_bytes, kernel);
      auto n_resident_blks_per_target = std::max<std::size_t>(n_blks_per_mp * n_mps / n_targets, 1);
      auto n_entries_per_subsegment =
          cuda_impl::SliceSegment(n_total_entries, Policy::kBlockThreads,
                                  n_resident_blks_per_target, n_targets, shmem_bytes, symbol_bits);
      auto grid = MakeGrid(max_segment_entries, n_entries_per_subsegment, Policy::kBlockThreads,
                           n_groups, n_targets, h_ridx_iters.size());

      dh::LaunchKernel(grid.n_blks, Policy::kBlockThreads, shmem_bytes, stream)(
          kernel, matrix, feature_groups, ridx_iters.data().get(), hists.data().get(), d_gpair,
          n_samples, n_targets, grid.n_entries_per_subsegment, grid.n_subsegments_per_segment);
    };

    DispatchCudaSm(ctx->Ordinal(), [&](auto arch) {
      using Arch = decltype(arch);
      if (n_targets == 1) {
        if (use_shared) {
          launch(HistPolicy<Arch, kDense, kCompressed, true, /*SingleTarget=*/true>{});
        } else {
          launch(HistPolicy<Arch, kDense, kCompressed, false, /*SingleTarget=*/true>{});
        }
      } else {
        if (use_shared) {
          launch(HistPolicy<Arch, kDense, kCompressed, true, /*SingleTarget=*/false>{});
        } else {
          launch(HistPolicy<Arch, kDense, kCompressed, false, /*SingleTarget=*/false>{});
        }
      }
    });
  }

  // Dispatch the layout of the matrix and the type of the row index.
  template <typename Accessor>
  void BuildHistogram(Context const* ctx, Accessor const& matrix,
                      FeatureGroups const& feature_groups,
                      linalg::MatrixView<GradientPairInt64 const> gpair,
                      std::vector<common::Span<cuda_impl::RowIndexT const>> const& ridxs,
                      std::vector<common::Span<GradientPairInt64>> const& hists) {
    auto dispatch = [&](auto const& ridx_iters) {
      if (matrix.IsDense()) {
        this->DispatchHist<true, true>(ctx, matrix, feature_groups, gpair, ridx_iters, hists);
      } else if (matrix.IsDenseCompressed()) {
        this->DispatchHist<false, true>(ctx, matrix, feature_groups, gpair, ridx_iters, hists);
      } else {
        this->DispatchHist<false, false>(ctx, matrix, feature_groups, gpair, ridx_iters, hists);
      }
    };
    if (ridxs.size() == 1 && ridxs.front().size() == matrix.n_rows) {
      // Special optimization for the root node, the row index is the identity mapping.
      using RidxIter = dh::counting_iterator<cuda_impl::RowIndexT>;
      CHECK_LT(matrix.base_rowid, std::numeric_limits<cuda_impl::RowIndexT>::max());
      dispatch(std::vector<common::IterSpan<RidxIter>>{common::IterSpan{
          dh::make_counting_iterator(static_cast<cuda_impl::RowIndexT>(matrix.base_rowid)),
          matrix.n_rows}});
    } else {
      dispatch(ridxs);
    }
  }
};

DeviceHistogramBuilder::DeviceHistogramBuilder() { monitor_.Init(__func__); }

DeviceHistogramBuilder::~DeviceHistogramBuilder() = default;

void DeviceHistogramBuilder::Reset(Context const* ctx, std::size_t max_cached_hist_nodes,
                                   bst_bin_t n_total_bins, bool force_global_memory) {
  this->monitor_.Start(__func__);
  this->p_impl_ = std::make_unique<HistKernel>(ctx, force_global_memory);
  this->hist_.Reset(ctx, n_total_bins, max_cached_hist_nodes);
  this->monitor_.Stop(__func__);
}

void DeviceHistogramBuilder::BuildHistogram(Context const* ctx, EllpackAccessor const& matrix,
                                            FeatureGroups const& feature_groups,
                                            common::Span<GradientPairInt64 const> gpair,
                                            common::Span<cuda_impl::RowIndexT const> ridx,
                                            common::Span<GradientPairInt64> histogram) {
  this->BuildHistogram(ctx, matrix, feature_groups,
                       linalg::MakeTensorView(ctx, linalg::kF, gpair, gpair.size(), 1), {ridx},
                       {histogram});
}

void DeviceHistogramBuilder::BuildHistogram(
    Context const* ctx, EllpackAccessor const& matrix, FeatureGroups const& feature_groups,
    linalg::MatrixView<GradientPairInt64 const> gpair,
    std::vector<common::Span<cuda_impl::RowIndexT const>> const& ridxs,
    std::vector<common::Span<GradientPairInt64>> const& hists) {
  this->monitor_.Start(__func__);
  std::visit(
      [&](auto&& matrix) {
        this->p_impl_->BuildHistogram(ctx, matrix, feature_groups, gpair, ridxs, hists);
      },
      matrix);
  this->monitor_.Stop(__func__);
}

void DeviceHistogramBuilder::AllReduceHist(Context const* ctx, bst_node_t nidx,
                                           std::size_t num_histograms) {
  this->monitor_.Start(__func__);
  auto d_node_hist = hist_.GetNodeHistogram(nidx);
  using ReduceT = typename std::remove_pointer_t<decltype(d_node_hist.data())>::ValueT;
  auto rc = collective::GlobalSum(
      ctx, linalg::MakeVec(reinterpret_cast<ReduceT*>(d_node_hist.data()),
                           d_node_hist.size() * 2 * num_histograms, ctx->Device()));
  SafeColl(rc);
  this->monitor_.Stop(__func__);
}
}  // namespace xgboost::tree
