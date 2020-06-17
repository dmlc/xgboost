/*!
 * Copyright 2020 by XGBoost Contributors
 */
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <algorithm>
#include <ctgmath>
#include <limits>

#include "xgboost/base.h"
#include "row_partitioner.cuh"

#include "histogram.cuh"

#include "../../data/ellpack_page.cuh"
#include "../../common/device_helpers.cuh"

namespace xgboost {
namespace tree {
// Following 2 functions are slightly modifed version of fbcuda.

/* \brief Constructs a rounding factor used to truncate elements in a sum such that the
   sum of the truncated elements is the same no matter what the order of the sum is.

 * Algorithm 5: Reproducible Sequential Sum in 'Fast Reproducible Floating-Point
 * Summation' by Demmel and Nguyen

 * In algorithm 5 the bound is calculated as $max(|v_i|) * n$.  Here we use the bound
 *
 * \begin{equation}
 *   max( fl(\sum^{V}_{v_i>0}{v_i}), fl(\sum^{V}_{v_i<0}|v_i|) )
 * \end{equation}
 *
 * to avoid outliers, as the full reduction is reproducible on GPU with reduction tree.
 */
template <typename T>
DEV_INLINE __host__ T CreateRoundingFactor(T max_abs, int n) {
  T delta = max_abs / (static_cast<T>(1.0) - 2 * n * std::numeric_limits<T>::epsilon());

  // Calculate ceil(log_2(delta)).
  // frexpf() calculates exp and returns `x` such that
  // delta = x * 2^exp, where `x` in (-1.0, -0.5] U [0.5, 1).
  // Because |x| < 1, exp is exactly ceil(log_2(delta)).
  int exp;
  std::frexp(delta, &exp);

  // return M = 2 ^ ceil(log_2(delta))
  return std::ldexp(static_cast<T>(1.0), exp);
}

namespace {
struct Pair {
  GradientPair first;
  GradientPair second;
};
DEV_INLINE Pair operator+(Pair const& lhs, Pair const& rhs) {
  return {lhs.first + rhs.first, lhs.second + rhs.second};
}
}  // anonymous namespace

struct Clip : public thrust::unary_function<GradientPair, Pair> {
  static DEV_INLINE float Pclip(float v) {
    return v > 0 ? v : 0;
  }
  static DEV_INLINE float Nclip(float v) {
    return v < 0 ? abs(v) : 0;
  }

  DEV_INLINE Pair operator()(GradientPair x) const {
    auto pg = Pclip(x.GetGrad());
    auto ph = Pclip(x.GetHess());

    auto ng = Nclip(x.GetGrad());
    auto nh = Nclip(x.GetHess());

    return { GradientPair{ pg, ph }, GradientPair{ ng, nh } };
  }
};

template <typename GradientSumT>
GradientSumT CreateRoundingFactor(common::Span<GradientPair const> gpair) {
  using T = typename GradientSumT::ValueT;
  dh::XGBCachingDeviceAllocator<char> alloc;

  thrust::device_ptr<GradientPair const> gpair_beg {gpair.data()};
  thrust::device_ptr<GradientPair const> gpair_end {gpair.data() + gpair.size()};
  auto beg = thrust::make_transform_iterator(gpair_beg, Clip());
  auto end = thrust::make_transform_iterator(gpair_end, Clip());
  Pair p = thrust::reduce(thrust::cuda::par(alloc), beg, end, Pair{});
  GradientPair positive_sum {p.first}, negative_sum {p.second};

  auto histogram_rounding = GradientSumT {
    CreateRoundingFactor<T>(std::max(positive_sum.GetGrad(), negative_sum.GetGrad()),
                            gpair.size()),
    CreateRoundingFactor<T>(std::max(positive_sum.GetHess(), negative_sum.GetHess()),
                            gpair.size()) };
  return histogram_rounding;
}

template GradientPairPrecise CreateRoundingFactor(common::Span<GradientPair const> gpair);
template GradientPair CreateRoundingFactor(common::Span<GradientPair const> gpair);

template <typename GradientSumT>
__global__ void SharedMemHistKernel(EllpackDeviceAccessor matrix,
                                    common::Span<const RowPartitioner::RowIndexT> d_ridx,
                                    GradientSumT* __restrict__ d_node_hist,
                                    const GradientPair* __restrict__ d_gpair,
                                    const int* feature_groups, const int* bin_groups,
                                    GradientSumT const rounding,
                                    bool use_shared_memory_histograms) {
  using T = typename GradientSumT::ValueT;
  extern __shared__ char smem[];
  int group = blockIdx.y;
  int start_feature = feature_groups[group];
  int start_bin = bin_groups[group];
  int num_bins = bin_groups[group + 1] - bin_groups[group];
  GradientSumT* smem_arr = reinterpret_cast<GradientSumT*>(smem);  // NOLINT
  if (use_shared_memory_histograms) {
    dh::BlockFill(smem_arr, num_bins, GradientSumT());
    __syncthreads();
  }
  int feature_stride = matrix.is_dense ?
    feature_groups[group + 1] - feature_groups[group] : matrix.row_stride;
  size_t n_elements = feature_stride * d_ridx.size();
  for (auto idx : dh::GridStrideRange(static_cast<size_t>(0), n_elements)) {
    int ridx = d_ridx[idx / feature_stride];
    int gidx =
      matrix.gidx_iter[ridx * matrix.row_stride + start_feature + idx % feature_stride];
    if (gidx != matrix.NumBins()) {
      GradientSumT truncated {
        TruncateWithRoundingFactor<T>(rounding.GetGrad(), d_gpair[ridx].GetGrad()),
        TruncateWithRoundingFactor<T>(rounding.GetHess(), d_gpair[ridx].GetHess()),
      };
      // If we are not using shared memory, accumulate the values directly into
      // global memory
      GradientSumT* atomic_add_ptr =
        use_shared_memory_histograms ? smem_arr : d_node_hist;
      gidx = use_shared_memory_histograms ? gidx - start_bin : gidx;
      dh::AtomicAddGpair(atomic_add_ptr + gidx, truncated);
    }
  }

  if (use_shared_memory_histograms) {
    // Write shared memory back to global memory
    __syncthreads();
    for (auto i : dh::BlockStrideRange(0, num_bins)) {
      GradientSumT truncated {
        TruncateWithRoundingFactor<T>(rounding.GetGrad(), smem_arr[i].GetGrad()),
          TruncateWithRoundingFactor<T>(rounding.GetHess(), smem_arr[i].GetHess()),
          };
      dh::AtomicAddGpair(d_node_hist + start_bin + i, truncated);
    }
  }
}

template <typename GradientSumT>
void BuildGradientHistogram(EllpackDeviceAccessor const& matrix,
                            common::Span<GradientPair const> gpair,
                            common::Span<const uint32_t> d_ridx,
                            common::Span<int const> feature_groups,
                            common::Span<int const> bin_groups,
                            common::Span<GradientSumT> histogram,
                            GradientSumT rounding, int max_group_bins) {
  // decide whether to use shared memory
  int device = 0;
  dh::safe_cuda(cudaGetDevice(&device));
  int max_shared_memory = dh::MaxSharedMemoryOptin(device);
  size_t smem_size = sizeof(GradientSumT) * max_group_bins;
  bool shared = smem_size <= max_shared_memory;
  smem_size = shared ? smem_size : 0;

  // opt into maximum shared memory for the kernel if necessary
  auto kernel = SharedMemHistKernel<GradientSumT>;
  if (shared) {
    dh::safe_cuda(cudaFuncSetAttribute
                  (kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   max_shared_memory));
  }

  // determine the launch configuration
  unsigned block_threads = shared ? 1024 : 256;
  int num_groups = feature_groups.size() - 1;
  int n_mps = 0;
  dh::safe_cuda(cudaDeviceGetAttribute(&n_mps, cudaDevAttrMultiProcessorCount, device));
  int n_blocks_per_mp = 0;
  dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor
                (&n_blocks_per_mp, kernel, block_threads, smem_size));
  // TODO(canonizer): This is really a hack, find a better way to distribute the data
  // among thread blocks
  unsigned grid_size = n_blocks_per_mp * n_mps;
  grid_size = common::DivRoundUp(grid_size, common::DivRoundUp(num_groups, 4));

  kernel<<<dim3(grid_size, num_groups), block_threads, smem_size>>>
    (matrix, d_ridx, histogram.data(), gpair.data(), feature_groups.data(), bin_groups.data(),
     rounding, shared);
  dh::safe_cuda(cudaGetLastError());
}

template void BuildGradientHistogram<GradientPair>(
    EllpackDeviceAccessor const& matrix,
    common::Span<GradientPair const> gpair,
    common::Span<const uint32_t> ridx,
    common::Span<int const> feature_groups,
    common::Span<int const> bin_groups,
    common::Span<GradientPair> histogram,
    GradientPair rounding, int max_group_bins);

template void BuildGradientHistogram<GradientPairPrecise>(
    EllpackDeviceAccessor const& matrix,
    common::Span<GradientPair const> gpair,
    common::Span<const uint32_t> ridx,
    common::Span<int const> feature_groups,
    common::Span<int const> bin_groups,
    common::Span<GradientPairPrecise> histogram,
    GradientPairPrecise rounding, int max_group_bins);

}  // namespace tree
}  // namespace xgboost
