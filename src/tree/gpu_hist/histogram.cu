/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <algorithm>             // for min, max
#include <cstdint>               // uint32_t, int32_t
#include <cuda/std/type_traits>  // for cuda::std::alignment_of_v
#include <memory>                // for unique_ptr
#include <utility>               // for pair

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
/**
 * @brief The index of an entry in `matrix.gidx_iter`.
 *
 * Each Ellpack row has `row_stride` entries, and `ridx` is global while the batch starts at
 * `base_rowid`. With the dense layout (`kCompressed`), the entries of a row are its
 * features, and `fidx` is a feature index. With the sparse layout, `fidx` is an entry in the
 * padded row, and the bin stored there identifies the feature.
 */
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

// Single-target launch bounds.
struct StHistBound {
  static constexpr std::int32_t kBlockThreads = 1024;
};

// The number of co-resident single-target blocks: as many as the threads of an SM allow,
// asking for more only tightens the register cap. The multi-target tuning of the arch is for
// full occupancy, so it has the threads of an SM.
template <typename Arch>
constexpr std::int32_t StMinBlocks() {
  return std::max(1, Arch::kBlockThreads * Arch::kMinBlocks / StHistBound::kBlockThreads);
}
using StHistDeviceBound = HistTuning<StHistBound::kBlockThreads, StMinBlocks<MtHistBound>()>;

template <typename HistArchPolicy, std::int32_t ItemsPerThread, bool Dense, bool Compressed,
          bool SharedMem>
struct HistPolicy : public HistArchPolicy {
  static constexpr std::int32_t kItemsPerThread = ItemsPerThread;
  // The smallest unit of work assigned to a block.
  static constexpr std::int32_t kTileSize = HistArchPolicy::kBlockThreads * kItemsPerThread;
  static constexpr bool kDense = Dense;
  static constexpr bool kCompressed = Compressed;
  static constexpr bool kSharedMem = SharedMem;
  // An approximation of the cost (time) of zeroing and flushing the privatized histogram
  // for a segment, modeled as items.
  static constexpr std::int32_t kSegmentCost = SharedMem ? kTileSize : 0;
  static constexpr bool kSingleTarget = std::is_same_v<HistArchPolicy, StHistBound>;
};

// The launch bounds depend on `__CUDA_ARCH__`, they must be resolved in the device
// compilation pass instead of being used as template arguments.
template <typename Policy>
using HistBound = std::conditional_t<Policy::kSingleTarget, StHistDeviceBound, MtHistBound>;

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

/**
 * @brief The shared memory budget for a block when `min_blocks` blocks are co-resident.
 *
 * The launch bounds cap the registers so that `min_blocks` blocks fit an SM, the shared
 * memory of the SM is split between the same number of blocks.
 */
[[nodiscard]] std::size_t HistShmemBytes(std::int32_t device, std::int32_t min_blocks) {
  CHECK_GT(min_blocks, 0);
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
}  // anonymous namespace

std::size_t DftStHistShmemBytes(std::int32_t device) {
  return DispatchCudaSm(device, [&](auto arch) {
    return HistShmemBytes(device, StMinBlocks<common::GetValueT<decltype(arch)>>());
  });
}

std::size_t DftMtHistShmemBytes(std::int32_t device) {
  return DispatchCudaSm(device, [&](auto arch) {
    return HistShmemBytes(device, common::GetValueT<decltype(arch)>::kMinBlocks);
  });
}

namespace {
__device__ GradientPairInt64 LoadGpair(GradientPairInt64 const* XGBOOST_RESTRICT gpairs) {
  static_assert(sizeof(int4) == sizeof(GradientPairInt64));
  auto g = *reinterpret_cast<int4 const*>(gpairs);
  return *reinterpret_cast<GradientPairInt64*>(&g);
}

// Build the histogram for the items [begin, end) of a single node, feature group, and target.
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

  auto process_item = [&](auto idx) {
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
  };

  for (bst_idx_t idx = begin + threadIdx.x; idx < end; idx += Policy::kBlockThreads) {
    process_item(idx);
  }
}

// A range of items inside a (node, feature group) segment.
struct HistSegment {
  std::size_t nidx_in_set;
  bst_feature_t gidx;
  // The range of valid items local to the segment, empty if the position is in the padding.
  bst_idx_t begin;
  bst_idx_t end;
  // The distance to the next segment or to `last`, whichever is closer.
  bst_idx_t step;
};

// The largest index `i` in [0, n) with `begin(i) <= pos`, `begin` must be non-decreasing.
// Taking the largest index skips empty entries.
//
// Same as `std::upper_bound() - 1`. The branchy `upper_bound` of thrust and libcu++ adds
// local memory accesses to the accumulation loop of some kernels on sm_80 and sm_90.
template <typename Fn>
XGBOOST_DEV_INLINE std::size_t UpperBoundIdx(std::size_t n, bst_idx_t pos, Fn&& begin) {
  std::size_t base = 0;
  while (n > 1) {
    auto half = n / 2;
    base = begin(base + half) <= pos ? base + half : base;
    n -= half;
  }
  return base;
}

// Find the segment of the item at `pos`, and the range of items in this segment up to
// `last`. Each segment is followed by `Policy::kSegmentCost` padding items.
template <typename Policy, typename Accessor>
XGBOOST_DEV_INLINE HistSegment FindSegment(Accessor const& matrix,
                                           FeatureGroupsAccessor const& feature_groups,
                                           common::Span<std::size_t const> sizes_csum,
                                           bst_idx_t pos, bst_idx_t last) {
  constexpr bst_idx_t kSegCost = Policy::kSegmentCost;
  // The sparse layout has a single group, replace the variable with a constant here.
  auto const n_groups = Policy::kCompressed ? feature_groups.NumGroups() : 1;
  auto const* XGBOOST_RESTRICT p_sizes = sizes_csum.data();

  // The groups of a node split the entries of its rows.
  HistSegment seg;
  seg.nidx_in_set = UpperBoundIdx(sizes_csum.size() - 1, pos, [&](std::size_t i) {
    return p_sizes[i] * matrix.row_stride + i * n_groups * kSegCost;
  });
  auto nidx = seg.nidx_in_set;
  bst_idx_t const n_rows = p_sizes[nidx + 1] - p_sizes[nidx];
  bst_idx_t offset = pos - (p_sizes[nidx] * matrix.row_stride + nidx * n_groups * kSegCost);
  bst_idx_t group_size;
  if constexpr (Policy::kCompressed) {
    auto const* XGBOOST_RESTRICT p_fs = feature_groups.feature_segments.data();
    auto group_begin = [&](bst_feature_t g) {
      return n_rows * p_fs[g] + g * kSegCost;
    };
    seg.gidx = UpperBoundIdx(n_groups, offset, group_begin);
    offset -= group_begin(seg.gidx);
    group_size = p_fs[seg.gidx + 1] - p_fs[seg.gidx];
  } else {
    seg.gidx = 0;
    group_size = matrix.row_stride;
  }
  bst_idx_t const n_valid = n_rows * group_size;
  seg.begin = offset;
  seg.end = cuda::std::min(n_valid, seg.begin + (last - pos));
  seg.step = cuda::std::min(n_valid + kSegCost - offset, last - pos);
  return seg;
}
}  // namespace

/**
 * @brief Kernel for building histograms of multiple nodes and targets.
 *
 * @param matrix          An ellpack accessor.
 * @param feature_groups  Grouping for privatized histogram.
 * @param d_ridx_iters    Pointer to row index spans. One span per node.
 * @param sizes_csum      Cumulative sum of the number of rows in each node.
 * @param node_hists      Pointer to histograms. One histogram per node.
 * @param n_items_per_blk The number of items processed by each block.
 * @param n_items         The total number of items for each target.
 *
 * The items of all (node, feature group) segments are concatenated in this order, and each
 * block processes `n_items_per_blk` contiguous items for one target. The block flushes its
 * privatized histogram once for each segment in its items. The blocks of all targets for the
 * same items are adjacent, they run concurrently and share the bin indices read in L2.
 *
 * Each segment is padded with the cost of its flush. Otherwise, a block can receive many
 * small segments and flush them sequentially while other blocks are idle.
 */
template <typename Policy, typename Accessor, typename RidxIterSpan>
__global__ __launch_bounds__(
    HistBound<Policy>::kBlockThreads,
    HistBound<Policy>::kMinBlocks) void HistogramKernel(Accessor const matrix,
                                                        FeatureGroupsAccessor const feature_groups,
                                                        RidxIterSpan const* d_ridx_iters,
                                                        common::Span<std::size_t const> sizes_csum,
                                                        common::Span<GradientPairInt64> const*
                                                            node_hists,
                                                        GradientPairInt64 const* d_gpair,
                                                        bst_idx_t n_samples, bst_target_t n_targets,
                                                        bst_idx_t n_items_per_blk,
                                                        bst_idx_t n_items) {
  if constexpr (Policy::kSingleTarget) {
    // Constant propagation removes the target indexing and saves registers.
    n_targets = 1;
  }

  extern __align__(std::alignment_of_v<GradientPairInt64>) __shared__ char shmem[];
  // Privatized histogram
  auto smem_hist = reinterpret_cast<GradientPairInt64*>(shmem);

  auto find_segment = [&](bst_idx_t pos) {
    // The end of the items for this block.
    bst_idx_t last = cuda::std::min(pos - pos % n_items_per_blk + n_items_per_blk, n_items);
    return FindSegment<Policy>(matrix, feature_groups, sizes_csum, pos, last);
  };
  bst_target_t const target_idx = blockIdx.x % n_targets;
  auto target_hist = [&](HistSegment const& seg) {
    auto d_node_hist = node_hists[seg.nidx_in_set];
    // With a target-major layout.
    auto gmem_hist = d_node_hist.data() + target_idx * (d_node_hist.size() / n_targets);
    // hint for PTX: atom.add.u64 -> atom.global.add.u64
    __builtin_assume(__isGlobal(gmem_hist));
    return gmem_hist;
  };

  // The first item of the block. The block ends at the next multiple of `n_items_per_blk`.
  // `volatile` keeps the only long-lived position out of the registers
  // during the accumulation.
  volatile bst_idx_t pos = (blockIdx.x / n_targets) * n_items_per_blk;
  do {
    auto seg = find_segment(pos);
    if (seg.begin < seg.end) {
      auto group = feature_groups[seg.gidx];
      if constexpr (Policy::kSharedMem) {
        // Each thread zeroes the same bins it flushes, no barrier is needed.
        dh::BlockFill(smem_hist, group.num_bins, GradientPairInt64{});
        __syncthreads();
      }
      HistKernelSegment<Policy>(matrix, group, d_ridx_iters[seg.nidx_in_set],
                                d_gpair + n_samples * target_idx, smem_hist, target_hist(seg),
                                seg.begin, seg.end);
      if constexpr (Policy::kSharedMem) {
        __syncthreads();
        // Recompute instead of keeping the segment in registers.
        seg = find_segment(pos);
        group = feature_groups[seg.gidx];
        auto gmem_hist = target_hist(seg);
        // Write shared memory back to global memory
        for (auto bin_idx : dh::BlockStrideRange(0, group.num_bins)) {
          AtomicAddGpairGlobal(gmem_hist + group.start_bin + bin_idx, smem_hist[bin_idx]);
        }
      }
    }
    pos += seg.step;
  } while (pos < n_items && pos % n_items_per_blk != 0);
}

// Dispatcher for the histogram kernel.
struct HistKernel {
  /**
   * @brief Split the items of a target into equal parts of whole tiles, one for each block.
   *
   * A block needs enough tiles to amortize the zeroing and flushing of its histogram, while
   * multiple waves of blocks balance the load between SMs. Small inputs fill the device
   * first.
   *
   * @param n_items                    The number of items for each target, including the
   *                                   padding.
   * @param n_resident_blks_per_target The number of blocks for each target that the device
   *                                   can run concurrently.
   *
   * @return The number of items for each block and the number of blocks for each target.
   */
  template <typename Policy>
  static auto SliceItems(bst_idx_t n_items, std::size_t n_resident_blks_per_target) {
    CHECK_GT(n_items, 0);
    CHECK_GT(n_resident_blks_per_target, 0);
    constexpr std::size_t kMaxWaves = 32;
    constexpr std::size_t kMinTiles = 32;
    auto n_tiles = common::DivRoundUp(n_items, Policy::kTileSize);
    auto min_tiles = std::min(kMinTiles, common::DivRoundUp(n_tiles, n_resident_blks_per_target));
    auto n_tiles_per_blk =
        std::max(common::DivRoundUp(n_tiles, n_resident_blks_per_target * kMaxWaves), min_tiles);
    auto n_blks_per_target = common::DivRoundUp(n_tiles, n_tiles_per_blk);
    return std::make_pair(static_cast<bst_idx_t>(n_tiles_per_blk * Policy::kTileSize),
                          static_cast<std::uint64_t>(n_blks_per_target));
  }

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
  void DispatchHistShmem(Context const* ctx, Accessor const& matrix,
                         FeatureGroupsAccessor const& feature_groups,
                         linalg::MatrixView<GradientPairInt64 const> gpair,
                         std::vector<RidxIterSpan> const& h_ridx_iters,
                         std::vector<common::Span<GradientPairInt64>> const& h_hists) {
    CHECK(gpair.FContiguous());
    CHECK_EQ(h_ridx_iters.size(), h_hists.size());
    // The kernel scans the entire row for sparse data, it doesn't filter out the bins of
    // other groups.
    CHECK(kCompressed || feature_groups.NumGroups() == 1);
    auto n_samples = gpair.Shape(0);
    auto n_targets = gpair.Shape(1);
    auto d_gpair = gpair.Values().data();

    // Cumulative sum of the number of rows in each node.
    std::vector<std::size_t> h_sizes_csum{0};
    h_sizes_csum.reserve(h_ridx_iters.size() + 1);
    for (auto const& ridx : h_ridx_iters) {
      h_sizes_csum.push_back(h_sizes_csum.back() + ridx.size());
    }
    // No entry to accumulate, for example all values are missing.
    if (h_sizes_csum.back() == 0 || matrix.row_stride == 0) {
      return;
    }

    std::size_t shmem_bytes = feature_groups.ShmemSize();
    bool use_shared = !force_global && shmem_bytes <= this->max_shared_bytes;
    shmem_bytes = use_shared ? shmem_bytes : 0;

    dh::TemporaryArray<std::size_t> sizes_csum(h_sizes_csum.size());
    dh::TemporaryArray<RidxIterSpan> ridx_iters(h_ridx_iters.size());
    dh::TemporaryArray<common::Span<GradientPairInt64>> hists(h_hists.size());
    auto stream = ctx->CUDACtx()->Stream();
    dh::CopyTo(h_sizes_csum, &sizes_csum, stream);
    dh::CopyTo(h_ridx_iters, &ridx_iters, stream);
    dh::CopyTo(h_hists, &hists, stream);

    auto launch = [&](auto policy) {
      using Policy = common::GetValueT<decltype(policy)>;
      auto kernel = HistogramKernel<Policy, Accessor, RidxIterSpan>;
      auto n_blks_per_mp = this->BlocksPerMp(Policy{}, shmem_bytes, kernel);
      // Must match the kernel.
      bst_idx_t n_segments = (h_sizes_csum.size() - 1) * feature_groups.NumGroups();
      bst_idx_t n_items =
          h_sizes_csum.back() * matrix.row_stride + n_segments * Policy::kSegmentCost;
      auto n_resident_blks_per_target = std::max<std::size_t>(n_blks_per_mp * n_mps / n_targets, 1);
      auto [n_items_per_blk, n_blks_per_target] =
          SliceItems<Policy>(n_items, n_resident_blks_per_target);
      auto n_blks = static_cast<std::uint64_t>(n_blks_per_target) * n_targets;
      CHECK_LE(n_blks, std::numeric_limits<std::uint32_t>::max());
      dh::LaunchKernel(static_cast<std::uint32_t>(n_blks), Policy::kBlockThreads, shmem_bytes,
                       ctx->CUDACtx()->Stream())(
          kernel, matrix, feature_groups, ridx_iters.data().get(), dh::ToSpan(sizes_csum),
          hists.data().get(), d_gpair, n_samples, n_targets, n_items_per_blk, n_items);
      dh::safe_cuda(cudaPeekAtLastError());
    };

    auto launch_arch = [&](auto arch) {
      using Arch = common::GetValueT<decltype(arch)>;
      if (use_shared) {
        launch(HistPolicy<Arch, kItemsPerThread, kDense, kCompressed, true>{});
      } else {
        launch(HistPolicy<Arch, kItemsPerThread, kDense, kCompressed, false>{});
      }
    };
    // Single target maximizes the number of threads, multi-target tunes for occupancy.
    if (n_targets == 1) {
      launch_arch(StHistBound{});
    } else {
      DispatchCudaSm(ctx->Ordinal(), launch_arch);
    }
  }

  template <typename Accessor, typename... Args>
  void Dispatch(Context const* ctx, Accessor const& matrix, Args&&... args) {
    if (matrix.IsDense()) {
      DispatchHistShmem<true, true>(ctx, matrix, std::forward<Args>(args)...);
    } else if (matrix.IsDenseCompressed()) {
      DispatchHistShmem<false, true>(ctx, matrix, std::forward<Args>(args)...);
    } else {
      DispatchHistShmem<false, false>(ctx, matrix, std::forward<Args>(args)...);
    }
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
                      linalg::MatrixView<GradientPairInt64 const> gpair,
                      std::vector<common::Span<cuda_impl::RowIndexT const>> const& ridxs,
                      std::vector<common::Span<GradientPairInt64>> const& hists) {
    if (ridxs.size() == 1 && ridxs.front().size() == matrix.n_rows) {
      // Special optimization for the root node, the row index is the identity mapping.
      using RidxIter = dh::counting_iterator<cuda_impl::RowIndexT>;
      CHECK_LT(matrix.base_rowid, std::numeric_limits<cuda_impl::RowIndexT>::max());
      std::vector<common::IterSpan<RidxIter>> ridx_iters{common::IterSpan{
          dh::make_counting_iterator(static_cast<cuda_impl::RowIndexT>(matrix.base_rowid)),
          matrix.n_rows}};
      this->kernel_->Dispatch(ctx, matrix, feature_groups, gpair, ridx_iters, hists);
    } else {
      this->kernel_->Dispatch(ctx, matrix, feature_groups, gpair, ridxs, hists);
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
  this->BuildHistogram(ctx, matrix, feature_groups,
                       linalg::MakeTensorView(ctx, linalg::kF, gpair, gpair.size(), 1), {ridx},
                       {histogram});
}

void DeviceHistogramBuilder::BuildHistogram(
    Context const* ctx, EllpackAccessor const& matrix, FeatureGroupsAccessor const& feature_groups,
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
