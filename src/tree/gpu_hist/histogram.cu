/**
 * Copyright 2020-2024, XGBoost Contributors
 */
#include <thrust/iterator/transform_iterator.h>  // for make_transform_iterator

#include <algorithm>
#include <cstdint>  // uint32_t, int32_t

#include "../../collective/aggregator.h"
#include "../../common/deterministic.cuh"
#include "../../common/device_helpers.cuh"
#include "../../data/ellpack_page.cuh"
#include "histogram.cuh"
#include "row_partitioner.cuh"
#include "xgboost/base.h"

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
                                         std::int32_t feature_stride) {
  auto fidx = group.start_feature + idx % feature_stride;
  return fidx;
}

XGBOOST_DEV_INLINE bst_idx_t IterIdx(EllpackDeviceAccessor const& matrix,
                                     RowPartitioner::RowIndexT ridx, bst_feature_t fidx) {
  // ridx_local = ridx - base_rowid  <== Row index local to each batch
  // entry_idx = ridx_local * row_stride <== Starting entry index for this row in the matrix
  // entry_idx += start_feature  <== Inside a row, first column inside this feature group
  // idx % feature_stride <== The feaature index local to the current feature group
  // entry_idx += idx % feature_stride <== Final index.
  return (ridx - matrix.base_rowid) * matrix.row_stride + fidx;
}
}  // anonymous namespace

struct Clip : public thrust::unary_function<GradientPair, Pair> {
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
GradientQuantiser::GradientQuantiser(Context const* ctx, common::Span<GradientPair const> gpair,
                                     MetaInfo const& info) {
  using GradientSumT = GradientPairPrecise;
  using T = typename GradientSumT::ValueT;

  thrust::device_ptr<GradientPair const> gpair_beg{gpair.data()};
  auto beg = thrust::make_transform_iterator(gpair_beg, Clip());
  Pair p = dh::Reduce(ctx->CUDACtx()->CTP(), beg, beg + gpair.size(), Pair{}, thrust::plus<Pair>{});
  // Treat pair as array of 4 primitive types to allreduce
  using ReduceT = typename decltype(p.first)::ValueT;
  static_assert(sizeof(Pair) == sizeof(ReduceT) * 4, "Expected to reduce four elements.");
  auto rc = collective::GlobalSum(ctx, info, linalg::MakeVec(reinterpret_cast<ReduceT*>(&p), 4));
  collective::SafeColl(rc);

  GradientPair positive_sum{p.first}, negative_sum{p.second};

  std::size_t total_rows = gpair.size();
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

XGBOOST_DEV_INLINE void AtomicAddGpairShared(xgboost::GradientPairInt64* dest,
                                             xgboost::GradientPairInt64 const& gpair) {
  auto dst_ptr = reinterpret_cast<int64_t *>(dest);
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

  atomicAdd(dst_ptr,
            *reinterpret_cast<uint64_t*>(&g));
  atomicAdd(dst_ptr + 1,
            *reinterpret_cast<uint64_t*>(&h));
}

template <bool kCompressed, int kBlockThreads, int kItemsPerThread>
class HistogramAgent {
  int constexpr static kItemsPerTile = kBlockThreads * kItemsPerThread;

  GradientPairInt64* smem_arr_;
  GradientPairInt64* d_node_hist_;
  using Idx = cuda_impl::RowIndexT;

  dh::LDGIterator<const Idx> d_ridx_;
  const GradientPair* d_gpair_;
  const FeatureGroup group_;
  const EllpackDeviceAccessor& matrix_;
  const int feature_stride_;
  const bst_idx_t n_elements_;
  const GradientQuantiser& rounding_;

 public:
  __device__ HistogramAgent(GradientPairInt64* smem_arr,
                            GradientPairInt64* __restrict__ d_node_hist, const FeatureGroup& group,
                            const EllpackDeviceAccessor& matrix, common::Span<const Idx> d_ridx,
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
      if (compressed_bin != matrix_.NullValue()) {
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
      if (gidx[i] != matrix_.NullValue()) {
        if (kCompressed) {
          gidx[i] += matrix_.feature_segments[fidx];
        }
      } else {
        gidx[i] = -1;  // missing
      }
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      // Avoid atomic add if it's a null value.
      if (gidx[i] != -1) {
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

template <bool kIsDense, bool use_shared_memory_histograms, int kBlockThreads, int kItemsPerThread>
__global__ void __launch_bounds__(kBlockThreads)
    SharedMemHistKernel(const EllpackDeviceAccessor matrix,
                        const FeatureGroupsAccessor feature_groups,
                        common::Span<const RowPartitioner::RowIndexT> d_ridx,
                        GradientPairInt64* __restrict__ d_node_hist,
                        const GradientPair* __restrict__ d_gpair,
                        GradientQuantiser const rounding) {
  extern __shared__ char smem[];
  const FeatureGroup group = feature_groups[blockIdx.y];
  auto smem_arr = reinterpret_cast<GradientPairInt64*>(smem);
  auto agent = HistogramAgent<kIsDense, kBlockThreads, kItemsPerThread>(
      smem_arr, d_node_hist, group, matrix, d_ridx, rounding, d_gpair);
  if (use_shared_memory_histograms) {
    agent.BuildHistogramWithShared();
  } else {
    agent.BuildHistogramWithGlobal();
  }
}

namespace {
constexpr std::int32_t kBlockThreads = 1024;
constexpr std::int32_t kItemsPerThread = 8;
constexpr std::int32_t ItemsPerTile() { return kBlockThreads * kItemsPerThread; }
}  // namespace

// Use auto deduction guide to workaround compiler error.
template <auto GlobalDense = SharedMemHistKernel<true, false, kBlockThreads, kItemsPerThread>,
          auto Global = SharedMemHistKernel<false, false, kBlockThreads, kItemsPerThread>,
          auto SharedDense = SharedMemHistKernel<true, true, kBlockThreads, kItemsPerThread>,
          auto Shared = SharedMemHistKernel<false, true, kBlockThreads, kItemsPerThread>>
struct HistogramKernel {
  // Kernel for working with dense Ellpack using the global memory.
  decltype(Global) global_dense_kernel{
      SharedMemHistKernel<true, false, kBlockThreads, kItemsPerThread>};
  // Kernel for working with sparse Ellpack using the global memory.
  decltype(Global) global_kernel{SharedMemHistKernel<false, false, kBlockThreads, kItemsPerThread>};
  // Kernel for working with dense Ellpack using the shared memory.
  decltype(Shared) shared_dense_kernel{
      SharedMemHistKernel<true, true, kBlockThreads, kItemsPerThread>};
  // Kernel for working with sparse Ellpack using the shared memory.
  decltype(Shared) shared_kernel{SharedMemHistKernel<false, true, kBlockThreads, kItemsPerThread>};

  bool shared{false};
  std::uint32_t grid_size{0};
  std::size_t smem_size{0};
  bool const force_global;

  HistogramKernel(Context const* ctx, FeatureGroupsAccessor const& feature_groups,
                  bool force_global_memory)
      : force_global{force_global_memory} {
    // Decide whether to use shared memory
    // Opt into maximum shared memory for the kernel if necessary
    std::size_t max_shared_memory = dh::MaxSharedMemoryOptin(ctx->Ordinal());

    this->smem_size = sizeof(GradientPairInt64) * feature_groups.max_group_bins;
    this->shared = !force_global_memory && this->smem_size <= max_shared_memory;
    this->smem_size = this->shared ? this->smem_size : 0;

    auto init = [&](auto& kernel) {
      if (this->shared) {
        dh::safe_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           max_shared_memory));
      }

      // determine the launch configuration
      std::int32_t num_groups = feature_groups.NumGroups();
      std::int32_t n_mps = 0;
      dh::safe_cuda(cudaDeviceGetAttribute(&n_mps, cudaDevAttrMultiProcessorCount, ctx->Ordinal()));

      std::int32_t n_blocks_per_mp = 0;
      dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n_blocks_per_mp, kernel,
                                                                  kBlockThreads, this->smem_size));

      // This gives the number of blocks to keep the device occupied Use this as the
      // maximum number of blocks
      this->grid_size = n_blocks_per_mp * n_mps;
    };
    // Initialize all kernel instantiations
    for (auto& kernel : {global_dense_kernel, global_kernel, shared_dense_kernel, shared_kernel}) {
      init(kernel);
    }
  }
};

class DeviceHistogramBuilderImpl {
  std::unique_ptr<HistogramKernel<>> kernel_{nullptr};

 public:
  void Reset(Context const* ctx, FeatureGroupsAccessor const& feature_groups,
             bool force_global_memory) {
    this->kernel_ = std::make_unique<HistogramKernel<>>(ctx, feature_groups, force_global_memory);
    if (force_global_memory) {
      CHECK(!this->kernel_->shared);
    }
  }

  void BuildHistogram(CUDAContext const* ctx, EllpackDeviceAccessor const& matrix,
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
    auto grid_size = std::min(kernel_->grid_size, static_cast<std::uint32_t>(common::DivRoundUp(
                                                      items_per_group, kMinItemsPerBlock)));
    auto launcher = [&](auto kernel) {
      dh::LaunchKernel{dim3(grid_size, feature_groups.NumGroups()),  // NOLINT
                       static_cast<uint32_t>(kBlockThreads), kernel_->smem_size, ctx->Stream()}(
          kernel, matrix, feature_groups, d_ridx, histogram.data(), gpair.data(), rounding);
    };

    if (!this->kernel_->shared) {  // Use global memory
      CHECK_EQ(this->kernel_->smem_size, 0);
      if (matrix.IsDenseCompressed()) {
        // Dense must use shared memory except for testing.
        CHECK(this->kernel_->force_global);
        launcher(this->kernel_->global_dense_kernel);
      } else {
        launcher(this->kernel_->global_kernel);
      }
    } else {  // Use shared memory
      CHECK_NE(this->kernel_->smem_size, 0);
      if (matrix.IsDenseCompressed()) {
        launcher(this->kernel_->shared_dense_kernel);
      } else {
        launcher(this->kernel_->shared_kernel);
      }
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

void DeviceHistogramBuilder::BuildHistogram(CUDAContext const* ctx,
                                            EllpackDeviceAccessor const& matrix,
                                            FeatureGroupsAccessor const& feature_groups,
                                            common::Span<GradientPair const> gpair,
                                            common::Span<const cuda_impl::RowIndexT> ridx,
                                            common::Span<GradientPairInt64> histogram,
                                            GradientQuantiser rounding) {
  this->monitor_.Start(__func__);
  this->p_impl_->BuildHistogram(ctx, matrix, feature_groups, gpair, ridx, histogram, rounding);
  this->monitor_.Stop(__func__);
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
