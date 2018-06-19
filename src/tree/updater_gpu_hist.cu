/*!
 * Copyright 2017 XGBoost contributors
 */
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <xgboost/tree_updater.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>
#include <utility>
#include <vector>
#include "../common/compressed_iterator.h"
#include "../common/device_helpers.cuh"
#include "../common/hist_util.h"
#include "../common/host_device_vector.h"
#include "../common/timer.h"
#include "param.h"
#include "updater_gpu_common.cuh"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_gpu_hist);

using GradientPairSumT = GradientPairPrecise;

template <int BLOCK_THREADS, typename ReduceT, typename TempStorageT>
__device__ GradientPairSumT ReduceFeature(const GradientPairSumT* begin,
                                     const GradientPairSumT* end,
                                     TempStorageT* temp_storage) {
  __shared__ cub::Uninitialized<GradientPairSumT> uninitialized_sum;
  GradientPairSumT& shared_sum = uninitialized_sum.Alias();

  GradientPairSumT local_sum = GradientPairSumT();
  for (auto itr = begin; itr < end; itr += BLOCK_THREADS) {
    bool thread_active = itr + threadIdx.x < end;
    // Scan histogram
    GradientPairSumT bin = thread_active ? *(itr + threadIdx.x) : GradientPairSumT();
    local_sum += bin;
  }
  local_sum = ReduceT(temp_storage->sum_reduce).Reduce(local_sum, cub::Sum());

  if (threadIdx.x == 0) {
    shared_sum = local_sum;
  }
  __syncthreads();

  return shared_sum;
}

template <int BLOCK_THREADS, typename ReduceT, typename scan_t,
          typename max_ReduceT, typename TempStorageT>
__device__ void EvaluateFeature(int fidx, const GradientPairSumT* hist,
                                const int* feature_segments, float min_fvalue,
                                const float* gidx_fvalue_map,
                                DeviceSplitCandidate* best_split,
                                const DeviceNodeStats& node,
                                const GPUTrainingParam& param,
                                TempStorageT* temp_storage, int constraint,
                                const ValueConstraint& value_constraint) {
  int gidx_begin = feature_segments[fidx];
  int gidx_end = feature_segments[fidx + 1];

  GradientPairSumT feature_sum = ReduceFeature<BLOCK_THREADS, ReduceT>(
      hist + gidx_begin, hist + gidx_end, temp_storage);

  auto prefix_op = SumCallbackOp<GradientPairSumT>();
  for (int scan_begin = gidx_begin; scan_begin < gidx_end;
       scan_begin += BLOCK_THREADS) {
    bool thread_active = scan_begin + threadIdx.x < gidx_end;

    GradientPairSumT bin =
        thread_active ? hist[scan_begin + threadIdx.x] : GradientPairSumT();
    scan_t(temp_storage->scan).ExclusiveScan(bin, bin, cub::Sum(), prefix_op);

    // Calculate  gain
    GradientPairSumT parent_sum = GradientPairSumT(node.sum_gradients);

    GradientPairSumT missing = parent_sum - feature_sum;

    bool missing_left = true;
    const float null_gain = -FLT_MAX;
    float gain = null_gain;
    if (thread_active) {
      gain = LossChangeMissing(bin, missing, parent_sum, node.root_gain, param,
                              constraint, value_constraint, missing_left);
    }

    __syncthreads();

    // Find thread with best gain
    cub::KeyValuePair<int, float> tuple(threadIdx.x, gain);
    cub::KeyValuePair<int, float> best =
        max_ReduceT(temp_storage->max_reduce).Reduce(tuple, cub::ArgMax());

    __shared__ cub::KeyValuePair<int, float> block_max;
    if (threadIdx.x == 0) {
      block_max = best;
    }

    __syncthreads();

    // Best thread updates split
    if (threadIdx.x == block_max.key) {
      int gidx = scan_begin + threadIdx.x;
      float fvalue =
          gidx == gidx_begin ? min_fvalue : gidx_fvalue_map[gidx - 1];

      GradientPairSumT left = missing_left ? bin + missing : bin;
      GradientPairSumT right = parent_sum - left;

      best_split->Update(gain, missing_left ? kLeftDir : kRightDir, fvalue, fidx,
                         GradientPair(left), GradientPair(right), param);
    }
    __syncthreads();
  }
}

template <int BLOCK_THREADS>
__global__ void evaluate_split_kernel(
    const GradientPairSumT* d_hist, int nidx, uint64_t n_features,
    DeviceNodeStats nodes, const int* d_feature_segments,
    const float* d_fidx_min_map, const float* d_gidx_fvalue_map,
    GPUTrainingParam gpu_param, DeviceSplitCandidate* d_split,
    ValueConstraint value_constraint, int* d_monotonic_constraints) {
  typedef cub::KeyValuePair<int, float> ArgMaxT;
  typedef cub::BlockScan<GradientPairSumT, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>
      BlockScanT;
  typedef cub::BlockReduce<ArgMaxT, BLOCK_THREADS> MaxReduceT;

  typedef cub::BlockReduce<GradientPairSumT, BLOCK_THREADS> SumReduceT;

  union TempStorage {
    typename BlockScanT::TempStorage scan;
    typename MaxReduceT::TempStorage max_reduce;
    typename SumReduceT::TempStorage sum_reduce;
  };

  __shared__ cub::Uninitialized<DeviceSplitCandidate> uninitialized_split;
  DeviceSplitCandidate& best_split = uninitialized_split.Alias();
  __shared__ TempStorage temp_storage;

  if (threadIdx.x == 0) {
    best_split = DeviceSplitCandidate();
  }

  __syncthreads();

  auto fidx = blockIdx.x;
  auto constraint = d_monotonic_constraints[fidx];
  EvaluateFeature<BLOCK_THREADS, SumReduceT, BlockScanT, MaxReduceT>(
      fidx, d_hist, d_feature_segments, d_fidx_min_map[fidx], d_gidx_fvalue_map,
      &best_split, nodes, gpu_param, &temp_storage, constraint,
      value_constraint);

  __syncthreads();

  if (threadIdx.x == 0) {
    // Record best loss
    d_split[fidx] = best_split;
  }
}

// Find a gidx value for a given feature otherwise return -1 if not found
template <typename GidxIterT>
__device__ int BinarySearchRow(bst_uint begin, bst_uint end, GidxIterT data,
                               int fidx_begin, int fidx_end) {
  bst_uint previous_middle = UINT32_MAX;
  while (end != begin) {
    auto middle = begin + (end - begin) / 2;
    if (middle == previous_middle) {
      break;
    }
    previous_middle = middle;

    auto gidx = data[middle];

    if (gidx >= fidx_begin && gidx < fidx_end) {
      return gidx;
    } else if (gidx < fidx_begin) {
      begin = middle;
    } else {
      end = middle;
    }
  }
  // Value is missing
  return -1;
}

struct DeviceHistogram {
  dh::BulkAllocator<dh::MemoryType::kDevice> ba;
  dh::DVec<GradientPairSumT> data;
  int n_bins;
  void Init(int device_idx, int max_nodes, int n_bins, bool silent) {
    this->n_bins = n_bins;
    ba.Allocate(device_idx, silent, &data, size_t(max_nodes) * size_t(n_bins));
  }

  void Reset() { data.Fill(GradientPairSumT()); }
  GradientPairSumT* GetHistPtr(int nidx) { return data.Data() + nidx * n_bins; }

  void PrintNidx(int nidx) const {
    auto h_data = data.AsVector();
    std::cout << "nidx " << nidx << ":\n";
    for (int i = n_bins * nidx; i < n_bins * (nidx + 1); i++) {
      std::cout << h_data[i] << " ";
    }
    std::cout << "\n";
  }
};

struct CalcWeightTrainParam {
  float min_child_weight;
  float reg_alpha;
  float reg_lambda;
  float max_delta_step;
  float learning_rate;
  XGBOOST_DEVICE explicit CalcWeightTrainParam(const TrainParam& p)
      : min_child_weight(p.min_child_weight),
        reg_alpha(p.reg_alpha),
        reg_lambda(p.reg_lambda),
        max_delta_step(p.max_delta_step),
        learning_rate(p.learning_rate) {}
};

// index of the first element in cuts greater than v, or n if none;
// cuts are ordered, and binary search is used
__device__ int upper_bound(const float* __restrict__ cuts, int n, float v) {
  if (n == 0)
    return 0;
  if (cuts[n - 1] <= v)
    return n;
  if (cuts[0] > v)
    return 0;
  int left = 0, right = n - 1;
  while (right - left > 1) {
    int middle = left + (right - left) / 2;
    if (cuts[middle] > v)
      right = middle;
    else
      left = middle;
  }
  return right;
}

__global__ void compress_bin_ellpack_k
(common::CompressedBufferWriter wr, common::CompressedByteT* __restrict__ buffer,
 const size_t* __restrict__ row_ptrs,
 const Entry* __restrict__ entries,
 const float* __restrict__ cuts, const size_t* __restrict__ cut_rows,
 size_t base_row, size_t n_rows, size_t row_ptr_begin, size_t row_stride,
 unsigned int null_gidx_value) {
  size_t irow = threadIdx.x + size_t(blockIdx.x) * blockDim.x;
  int ifeature = threadIdx.y + blockIdx.y * blockDim.y;
  if (irow >= n_rows || ifeature >= row_stride)
    return;
  int row_size = static_cast<int>(row_ptrs[irow + 1] - row_ptrs[irow]);
  unsigned int bin = null_gidx_value;
  if (ifeature < row_size) {
    Entry entry = entries[row_ptrs[irow] - row_ptr_begin + ifeature];
    int feature = entry.index;
    float fvalue = entry.fvalue;
    const float *feature_cuts = &cuts[cut_rows[feature]];
    int ncuts = cut_rows[feature + 1] - cut_rows[feature];
    bin = upper_bound(feature_cuts, ncuts, fvalue);
    if (bin >= ncuts)
      bin = ncuts - 1;
    bin += cut_rows[feature];
  }
  wr.AtomicWriteSymbol(buffer, bin, (irow + base_row) * row_stride + ifeature);
}

__global__ void sharedMemHistKernel(size_t row_stride,
                                    const bst_uint* d_ridx,
                                    common::CompressedIterator<uint32_t> d_gidx,
                                    int null_gidx_value,
                                    GradientPairSumT* d_node_hist,
                                    const GradientPair* d_gpair,
                                    size_t segment_begin,
                                    size_t n_elements) {
  extern __shared__ char smem[];
  GradientPairSumT* smem_arr = reinterpret_cast<GradientPairSumT*>(smem); // NOLINT
  for (auto i : dh::BlockStrideRange(0, null_gidx_value)) {
    smem_arr[i] = GradientPairSumT();
  }
  __syncthreads();
  for (auto idx : dh::GridStrideRange(static_cast<size_t>(0), n_elements)) {
    int ridx = d_ridx[idx / row_stride + segment_begin];
    int gidx = d_gidx[ridx * row_stride + idx % row_stride];
    if (gidx != null_gidx_value) {
      AtomicAddGpair(smem_arr + gidx, d_gpair[ridx]);
    }
  }
  __syncthreads();
  for (auto i : dh::BlockStrideRange(0, null_gidx_value)) {
    AtomicAddGpair(d_node_hist + i, smem_arr[i]);
  }
}

using WXQSketch = common::HistCutMatrix::WXQSketch;

__global__ void find_cuts_k
(WXQSketch::Entry* __restrict__ cuts, const bst_float* __restrict__ data,
 const float* __restrict__ cum_weights, int nsamples, int ncuts) {
  // ncuts < nsamples
  int icut = threadIdx.x + blockIdx.x * blockDim.x;
  if (icut >= ncuts)
    return;
  WXQSketch::Entry v;
  int isample = 0;
  if (icut == 0) {
    isample = 0;
  } else if (icut == ncuts - 1) {
    isample = nsamples - 1;
  } else {
    bst_float rank = cum_weights[nsamples - 1] / static_cast<float>(ncuts - 1)
      * static_cast<float>(icut);
    // -1 is used because cum_weights is an inclusive sum
    isample = upper_bound(cum_weights, nsamples, rank) - 1;
    isample = max(0, min(isample, nsamples - 1));
  }
  // repeated values will be filtered out on the CPU
  bst_float rmin = isample > 0 ? cum_weights[isample - 1] : 0;
  bst_float rmax = cum_weights[isample];
  cuts[icut] = WXQSketch::Entry(rmin, rmax, rmax - rmin, data[isample]);
}

// predictate for thrust filtering that returns true if the element is not a NaN
struct IsNotNaN {
  __device__ bool operator()(float a) const { return !isnan(a); }
};

__global__ void unpack_features_k
(float* __restrict__ fvalues, float* __restrict__ feature_weights,
 const size_t* __restrict__ row_ptrs, const float* __restrict__ weights,
 Entry* entries, size_t nrows_array, int ncols, size_t row_begin_ptr,
 size_t nrows) {
  size_t irow = threadIdx.x + size_t(blockIdx.x) * blockDim.x;
  if (irow >= nrows) {
    return;
  }
  size_t row_length = row_ptrs[irow + 1] - row_ptrs[irow];
  int icol = threadIdx.y + blockIdx.y * blockDim.y;
  if (icol >= row_length) {
    return;
  }
  Entry entry = entries[row_ptrs[irow] - row_begin_ptr + icol];
  size_t ind = entry.index * nrows_array + irow;
  // if weights are present, ensure that a non-NaN value is written to weights
  // if and only if it is also written to features
  if (!isnan(entry.fvalue) && (weights == nullptr || !isnan(weights[irow]))) {
    fvalues[ind] = entry.fvalue;
    if (feature_weights != nullptr) {
      feature_weights[ind] = weights[irow];
    }
  }
}

// Manage memory for a single GPU
struct DeviceShard {
  struct Segment {
    size_t begin;
    size_t end;

    Segment() : begin(0), end(0) {}

    Segment(size_t begin, size_t end) : begin(begin), end(end) {
      CHECK_GE(end, begin);
    }
    size_t Size() const { return end - begin; }
  };

  int device_idx;
  int normalised_device_idx;  // Device index counting from param.gpu_id
  dh::BulkAllocator<dh::MemoryType::kDevice> ba;
  dh::DVec<common::CompressedByteT> gidx_buffer;
  dh::DVec<GradientPair> gpair;
  dh::DVec2<bst_uint> ridx;  // Row index relative to this shard
  dh::DVec2<int> position;
  std::vector<Segment> ridx_segments;
  dh::DVec<int> feature_segments;
  dh::DVec<float> gidx_fvalue_map;
  dh::DVec<float> min_fvalue;
  dh::DVec<int> monotone_constraints;
  dh::DVec<bst_float> prediction_cache;
  std::vector<GradientPair> node_sum_gradients;
  dh::DVec<GradientPair> node_sum_gradients_d;
  std::vector<WXQSketch> sketches;
  thrust::device_vector<size_t> row_ptrs;
  common::CompressedIterator<uint32_t> gidx;
  size_t row_stride;
  bst_uint row_begin_idx;  // The row offset for this shard
  bst_uint row_end_idx;
  bst_uint n_rows;
  int n_bins;
  int null_gidx_value;
  DeviceHistogram hist;
  TrainParam param;
  bool prediction_cache_initialised;
  bool can_use_smem_atomics;

  int64_t* tmp_pinned;  // Small amount of staging memory

  std::vector<cudaStream_t> streams;

  dh::CubMemory temp_memory;

  // TODO(canonizer): do add support multi-batch DMatrix here
  DeviceShard(int device_idx, int normalised_device_idx,
              bst_uint row_begin, bst_uint row_end, TrainParam param)
    : device_idx(device_idx),
      normalised_device_idx(normalised_device_idx),
      row_begin_idx(row_begin),
      row_end_idx(row_end),
      row_stride(0),
      n_rows(row_end - row_begin),
      n_bins(0),
      null_gidx_value(0),
      param(param),
      prediction_cache_initialised(false),
      can_use_smem_atomics(false) {}

  void InitRowPtrs(const SparsePage& row_batch) {
    dh::safe_cuda(cudaSetDevice(device_idx));
    row_ptrs.resize(n_rows + 1);
    thrust::copy(&row_batch.offset[row_begin_idx], &row_batch.offset[row_end_idx + 1],
                 row_ptrs.begin());
    auto row_iter = row_ptrs.begin();
    auto get_size = [=] __device__(size_t row) {
      return row_iter[row + 1] - row_iter[row];
    }; // NOLINT

    auto counting = thrust::make_counting_iterator(size_t(0));
    using TransformT = thrust::transform_iterator<decltype(get_size),
                                                  decltype(counting), size_t>;
    TransformT row_size_iter = TransformT(counting, get_size);
    row_stride = thrust::reduce(row_size_iter, row_size_iter + n_rows, 0,
                                thrust::maximum<size_t>());
  }

  void Sketch(const SparsePage& row_batch, const MetaInfo& info) {
    std::vector<WXQSketch::SummaryContainer> summaries;
    // initialize sketches
    size_t num_cols = info.num_col_;
    sketches.resize(num_cols);
    summaries.resize(num_cols);
    int max_num_bins = param.max_bin;
    constexpr int kFactor = 8;
    double eps = 1.0 / (kFactor * max_num_bins);
    size_t dummy_nlevel;
    size_t ncuts = 0;
    WXQSketch::LimitSizeLevel(row_batch.Size(), eps, &dummy_nlevel, &ncuts);
    // double ncuts to be the same as the number of values
    // in the temporary buffers of the sketches
    ncuts *= 2;
    for (int icol = 0; icol < num_cols; ++icol) {
      sketches[icol].Init(row_batch.Size(), eps);
      summaries[icol].Reserve(ncuts);
    }

    // allocate necessary GPU buffers
    dh::safe_cuda(cudaSetDevice(device_idx));

    size_t gpu_batch_nrows = BatchSize(num_cols);
    thrust::device_vector<Entry> entries(gpu_batch_nrows * num_cols);
    thrust::device_vector<bst_float> fvalues(gpu_batch_nrows * num_cols);
    thrust::device_vector<bst_float> feature_weights;
    thrust::device_vector<bst_float> fvalues_cur(gpu_batch_nrows);
    thrust::device_vector<WXQSketch::Entry> cuts_d(ncuts * num_cols);
    thrust::host_vector<WXQSketch::Entry> cuts_h(ncuts * num_cols);
    thrust::device_vector<bst_float> weights(gpu_batch_nrows);
    thrust::device_vector<bst_float> weights2(gpu_batch_nrows);
    bool has_weights = info.weights_.size() > 0;
    if (has_weights) {
      feature_weights.resize(gpu_batch_nrows * num_cols);
    }
    std::vector<size_t> ncuts_cur(num_cols);

    // temporary storage for number of elements for filtering and reduction using CUB
    thrust::device_vector<size_t> num_elements(1);

    // temporary storage for sorting using CUB
    size_t sort_tmp_size = 0;
    if (has_weights) {
      cub::DeviceRadixSort::SortPairs
        (nullptr, sort_tmp_size, fvalues_cur.data().get(),
         fvalues.data().get(), weights.data().get(), weights2.data().get(),
         gpu_batch_nrows);
    } else {
      cub::DeviceRadixSort::SortKeys
        (nullptr, sort_tmp_size, fvalues_cur.data().get(), fvalues.data().get(),
         gpu_batch_nrows);
    }
    thrust::device_vector<char> sort_tmp_storage(sort_tmp_size);

    // temporary storage for inclusive prefix sum using CUB
    size_t scan_tmp_size = 0;
    if (has_weights) {
      cub::DeviceScan::InclusiveSum
        (nullptr, scan_tmp_size, weights2.begin(), weights.begin(), gpu_batch_nrows);
    }
    thrust::device_vector<char> scan_tmp_storage(scan_tmp_size);

    // temporary storage for reduction by key using CUB
    size_t reduce_tmp_size = 0;
    cub::DeviceReduce::ReduceByKey
      (nullptr, reduce_tmp_size, fvalues.begin(),
       fvalues_cur.begin(), weights.begin(), weights2.begin(), num_elements.begin(),
       thrust::maximum<bst_float>(), gpu_batch_nrows);
    thrust::device_vector<char> reduce_tmp_storage(reduce_tmp_size);

    // temporary storage for filtering using CUB
    size_t if_tmp_size = 0;
    cub::DeviceSelect::If(nullptr, if_tmp_size, fvalues.begin(), fvalues_cur.begin(),
                          num_elements.begin(), gpu_batch_nrows, IsNotNaN());
    thrust::device_vector<char> if_tmp_storage(if_tmp_size);

    size_t gpu_nbatches = dh::DivRoundUp(n_rows, gpu_batch_nrows);

    for (size_t gpu_batch = 0; gpu_batch < gpu_nbatches; ++gpu_batch) {
      // compute start and end indices
      size_t batch_row_begin = gpu_batch * gpu_batch_nrows;
      size_t batch_row_end = (gpu_batch + 1) * gpu_batch_nrows;
      if (batch_row_end > n_rows) {
        batch_row_end = n_rows;
      }
      size_t batch_nrows = batch_row_end - batch_row_begin;
      size_t n_entries =
        row_batch.offset[row_begin_idx + batch_row_end] -
        row_batch.offset[row_begin_idx + batch_row_begin];
      // copy the batch to the GPU
      dh::safe_cuda
        (cudaMemcpy(entries.data().get(),
                    &row_batch.data[row_batch.offset[row_begin_idx + batch_row_begin]],
                    n_entries * sizeof(Entry), cudaMemcpyDefault));
      // copy the weights if necessary
      if (has_weights) {
        dh::safe_cuda
          (cudaMemcpy(weights.data().get(),
                      info.weights_.data() + row_begin_idx + batch_row_begin,
                      batch_nrows * sizeof(bst_float), cudaMemcpyDefault));
      }

      // unpack the features; also unpack weights if present
      thrust::fill(fvalues.begin(), fvalues.end(), NAN);
      thrust::fill(feature_weights.begin(), feature_weights.end(), NAN);

      dim3 block3(64, 4, 1);
      dim3 grid3(dh::DivRoundUp(batch_nrows, block3.x),
                 dh::DivRoundUp(num_cols, block3.y), 1);
      unpack_features_k<<<grid3, block3>>>
        (fvalues.data().get(), has_weights ? feature_weights.data().get() : nullptr,
         row_ptrs.data().get() + batch_row_begin,
         has_weights ? weights.data().get() : nullptr, entries.data().get(),
         gpu_batch_nrows, num_cols,
         row_batch.offset[row_begin_idx + batch_row_begin], batch_nrows);
      dh::safe_cuda(cudaGetLastError());
      dh::safe_cuda(cudaDeviceSynchronize());

      for (int icol = 0; icol < num_cols; ++icol) {
        // filter out NaNs in feature values
        auto fvalues_begin = fvalues.data() + icol * gpu_batch_nrows;
        cub::DeviceSelect::If
          (if_tmp_storage.data().get(), if_tmp_size, fvalues_begin, fvalues_cur.data(),
           num_elements.begin(), batch_nrows, IsNotNaN());
        size_t nfvalues_cur = 0;
        thrust::copy_n(num_elements.begin(), 1, &nfvalues_cur);

        // compute cumulative weights using a prefix scan
        if (has_weights) {
          // filter out NaNs in weights;
          // since cub::DeviceSelect::If performs stable filtering,
          // the weights are stored in the correct positions
          auto feature_weights_begin = feature_weights.data() + icol * gpu_batch_nrows;
          cub::DeviceSelect::If
            (if_tmp_storage.data().get(), if_tmp_size, feature_weights_begin,
             weights.data().get(), num_elements.begin(), batch_nrows, IsNotNaN());

          // sort the values and weights
          cub::DeviceRadixSort::SortPairs
            (sort_tmp_storage.data().get(), sort_tmp_size, fvalues_cur.data().get(),
             fvalues_begin.get(), weights.data().get(), weights2.data().get(), nfvalues_cur);

          // sum the weights to get cumulative weight values
          cub::DeviceScan::InclusiveSum
            (scan_tmp_storage.data().get(), scan_tmp_size, weights2.begin(),
             weights.begin(), nfvalues_cur);
        } else {
          // sort the batch values
          cub::DeviceRadixSort::SortKeys
            (sort_tmp_storage.data().get(), sort_tmp_size,
             fvalues_cur.data().get(), fvalues_begin.get(), nfvalues_cur);

          // fill in cumulative weights with counting iterator
          thrust::copy_n(thrust::make_counting_iterator(1), nfvalues_cur,
                         weights.begin());
        }

        // remove repeated items and sum the weights across them;
        // non-negative weights are assumed
        cub::DeviceReduce::ReduceByKey
          (reduce_tmp_storage.data().get(), reduce_tmp_size, fvalues_begin,
           fvalues_cur.begin(), weights.begin(), weights2.begin(), num_elements.begin(),
           thrust::maximum<bst_float>(), nfvalues_cur);
        size_t n_unique = 0;
        thrust::copy_n(num_elements.begin(), 1, &n_unique);

        // extract cuts
        ncuts_cur[icol] = ncuts < n_unique ? ncuts : n_unique;
        // if less elements than cuts: copy all elements with their weights
        if (ncuts > n_unique) {
          auto weights2_iter = weights2.begin();
          auto fvalues_iter = fvalues_cur.begin();
          auto cuts_iter = cuts_d.begin() + icol * ncuts;
          dh::LaunchN(device_idx, n_unique, [=]__device__(size_t i) {
              bst_float rmax = weights2_iter[i];
              bst_float rmin = i > 0 ? weights2_iter[i - 1] : 0;
              cuts_iter[i] = WXQSketch::Entry(rmin, rmax, rmax - rmin, fvalues_iter[i]);
            });
        } else if (ncuts_cur[icol] > 0) {
          // if more elements than cuts: use binary search on cumulative weights
          int block = 256;
          find_cuts_k<<<dh::DivRoundUp(ncuts_cur[icol], block), block>>>
            (cuts_d.data().get() + icol * ncuts, fvalues_cur.data().get(),
             weights2.data().get(), n_unique, ncuts_cur[icol]);
          dh::safe_cuda(cudaGetLastError());
        }
      }

      dh::safe_cuda(cudaDeviceSynchronize());

      // add cuts into sketches
      thrust::copy(cuts_d.begin(), cuts_d.end(), cuts_h.begin());
      for (int icol = 0; icol < num_cols; ++icol) {
        summaries[icol].MakeFromSorted(&cuts_h[ncuts * icol], ncuts_cur[icol]);
        sketches[icol].PushSummary(summaries[icol]);
      }
    }
  }

  size_t BatchSize(size_t max_num_cols) {
    size_t gpu_batch_nrows = param.gpu_batch_nrows;
    if (param.gpu_batch_nrows == 0) {
      // By default, use no more than 1/16th of GPU memory
      gpu_batch_nrows = dh::TotalMemory(device_idx) /
        (16 * max_num_cols * sizeof(Entry));
    } else if (param.gpu_batch_nrows == -1) {
      gpu_batch_nrows = n_rows;
    }
    if (gpu_batch_nrows > n_rows) {
      gpu_batch_nrows = n_rows;
    }
    return gpu_batch_nrows;
  }

  void InitCompressedData(const common::HistCutMatrix& hmat, const SparsePage& row_batch) {
    n_bins = hmat.row_ptr.back();
    null_gidx_value = hmat.row_ptr.back();

    // copy cuts to the GPU
    dh::safe_cuda(cudaSetDevice(device_idx));
    thrust::device_vector<float> cuts_d(hmat.cut);
    thrust::device_vector<size_t> cut_row_ptrs_d(hmat.row_ptr);

    // allocate compressed bin data
    int num_symbols = n_bins + 1;
    size_t compressed_size_bytes =
        common::CompressedBufferWriter::CalculateBufferSize(row_stride * n_rows,
                                                            num_symbols);

    CHECK(!(param.max_leaves == 0 && param.max_depth == 0))
        << "Max leaves and max depth cannot both be unconstrained for "
           "gpu_hist.";
    ba.Allocate(device_idx, param.silent, &gidx_buffer, compressed_size_bytes);
    gidx_buffer.Fill(0);

    int nbits = common::detail::SymbolBits(num_symbols);

    // bin and compress entries in batches of rows
    size_t gpu_batch_nrows = BatchSize(row_stride);
    thrust::device_vector<Entry> entries_d(gpu_batch_nrows * row_stride);

    size_t gpu_nbatches = dh::DivRoundUp(n_rows, gpu_batch_nrows);
    for (size_t gpu_batch = 0; gpu_batch < gpu_nbatches; ++gpu_batch) {
      size_t batch_row_begin = gpu_batch * gpu_batch_nrows;
      size_t batch_row_end = (gpu_batch + 1) * gpu_batch_nrows;
      if (batch_row_end > n_rows) {
        batch_row_end = n_rows;
      }
      size_t batch_nrows = batch_row_end - batch_row_begin;
      size_t n_entries =
        row_batch.offset[row_begin_idx + batch_row_end] -
        row_batch.offset[row_begin_idx + batch_row_begin];
      dh::safe_cuda
        (cudaMemcpy
         (entries_d.data().get(),
          &row_batch.data[row_batch.offset[row_begin_idx + batch_row_begin]],
          n_entries * sizeof(Entry), cudaMemcpyDefault));
      dim3 block3(32, 8, 1);
      dim3 grid3(dh::DivRoundUp(n_rows, block3.x),
                 dh::DivRoundUp(row_stride, block3.y), 1);
      compress_bin_ellpack_k<<<grid3, block3>>>
        (common::CompressedBufferWriter(num_symbols), gidx_buffer.Data(),
         row_ptrs.data().get() + batch_row_begin,
         entries_d.data().get(), cuts_d.data().get(), cut_row_ptrs_d.data().get(),
         batch_row_begin, batch_nrows,
         row_batch.offset[row_begin_idx + batch_row_begin],
         row_stride, null_gidx_value);

      dh::safe_cuda(cudaGetLastError());
      dh::safe_cuda(cudaDeviceSynchronize());
    }

    // free the memory that is no longer needed
    row_ptrs.resize(0);
    row_ptrs.shrink_to_fit();
    entries_d.resize(0);
    entries_d.shrink_to_fit();

    gidx = common::CompressedIterator<uint32_t>(gidx_buffer.Data(), num_symbols);

    // allocate the rest
    int max_nodes =
        param.max_leaves > 0 ? param.max_leaves * 2 : MaxNodesDepth(param.max_depth);
    ba.Allocate(device_idx, param.silent,
                &gpair, n_rows, &ridx, n_rows, &position, n_rows,
                &prediction_cache, n_rows, &node_sum_gradients_d, max_nodes,
                &feature_segments, hmat.row_ptr.size(), &gidx_fvalue_map,
                hmat.cut.size(), &min_fvalue, hmat.min_val.size(),
                &monotone_constraints, param.monotone_constraints.size());
    gidx_fvalue_map = hmat.cut;
    min_fvalue = hmat.min_val;
    feature_segments = hmat.row_ptr;
    monotone_constraints = param.monotone_constraints;

    node_sum_gradients.resize(max_nodes);
    ridx_segments.resize(max_nodes);

    // check if we can use shared memory for building histograms
    // (assuming atleast we need 2 CTAs per SM to maintain decent latency hiding)
    auto histogram_size = sizeof(GradientPairSumT) * null_gidx_value;
    auto max_smem = dh::MaxSharedMemory(device_idx);
    can_use_smem_atomics = histogram_size <= max_smem;

    // Init histogram
    hist.Init(device_idx, max_nodes, hmat.row_ptr.back(), param.silent);

    dh::safe_cuda(cudaMallocHost(&tmp_pinned, sizeof(int64_t)));
  }

  ~DeviceShard() {
    for (auto& stream : streams) {
      dh::safe_cuda(cudaStreamDestroy(stream));
    }
    dh::safe_cuda(cudaFreeHost(tmp_pinned));
  }

  // Get vector of at least n initialised streams
  std::vector<cudaStream_t>& GetStreams(int n) {
    if (n > streams.size()) {
      for (auto& stream : streams) {
        dh::safe_cuda(cudaStreamDestroy(stream));
      }

      streams.clear();
      streams.resize(n);

      for (auto& stream : streams) {
        dh::safe_cuda(cudaStreamCreate(&stream));
      }
    }

    return streams;
  }

  // Reset values for each update iteration
  void Reset(HostDeviceVector<GradientPair>* dh_gpair) {
    dh::safe_cuda(cudaSetDevice(device_idx));
    position.CurrentDVec().Fill(0);
    std::fill(node_sum_gradients.begin(), node_sum_gradients.end(),
              GradientPair());

    thrust::sequence(ridx.CurrentDVec().tbegin(), ridx.CurrentDVec().tend());

    std::fill(ridx_segments.begin(), ridx_segments.end(), Segment(0, 0));
    ridx_segments.front() = Segment(0, ridx.Size());
    this->gpair.copy(dh_gpair->tbegin(device_idx), dh_gpair->tend(device_idx));
    SubsampleGradientPair(&gpair, param.subsample, row_begin_idx);
    hist.Reset();
  }

  void BuildHistUsingGlobalMem(int nidx) {
    auto segment = ridx_segments[nidx];
    auto d_node_hist = hist.GetHistPtr(nidx);
    auto d_gidx = gidx;
    auto d_ridx = ridx.Current();
    auto d_gpair = gpair.Data();
    auto row_stride = this->row_stride;
    auto null_gidx_value = this->null_gidx_value;
    auto n_elements = segment.Size() * row_stride;

    dh::LaunchN(device_idx, n_elements, [=] __device__(size_t idx) {
      int ridx = d_ridx[(idx / row_stride) + segment.begin];
      int gidx = d_gidx[ridx * row_stride + idx % row_stride];

      if (gidx != null_gidx_value) {
        AtomicAddGpair(d_node_hist + gidx, d_gpair[ridx]);
      }
    });
  }

  void BuildHistUsingSharedMem(int nidx) {
    auto segment = ridx_segments[nidx];
    auto segment_begin = segment.begin;
    auto d_node_hist = hist.GetHistPtr(nidx);
    auto d_gidx = gidx;
    auto d_ridx = ridx.Current();
    auto d_gpair = gpair.Data();
    auto row_stride = this->row_stride;
    auto null_gidx_value = this->null_gidx_value;
    auto n_elements = segment.Size() * row_stride;

    const size_t smem_size = sizeof(GradientPairSumT) * null_gidx_value;
    const int items_per_thread = 8;
    const int block_threads = 256;
    const int grid_size =
        static_cast<int>(dh::DivRoundUp(n_elements,
                                        items_per_thread * block_threads));
    if (grid_size <= 0) {
      return;
    }
    dh::safe_cuda(cudaSetDevice(device_idx));
    sharedMemHistKernel<<<grid_size, block_threads, smem_size>>>
        (row_stride, d_ridx, d_gidx, null_gidx_value, d_node_hist, d_gpair,
         segment_begin, n_elements);
  }

  void BuildHist(int nidx) {
    if (can_use_smem_atomics) {
      BuildHistUsingSharedMem(nidx);
    } else {
      BuildHistUsingGlobalMem(nidx);
    }
  }

  void SubtractionTrick(int nidx_parent, int nidx_histogram,
                        int nidx_subtraction) {
    auto d_node_hist_parent = hist.GetHistPtr(nidx_parent);
    auto d_node_hist_histogram = hist.GetHistPtr(nidx_histogram);
    auto d_node_hist_subtraction = hist.GetHistPtr(nidx_subtraction);

    dh::LaunchN(device_idx, hist.n_bins, [=] __device__(size_t idx) {
      d_node_hist_subtraction[idx] =
          d_node_hist_parent[idx] - d_node_hist_histogram[idx];
    });
  }

  __device__ void CountLeft(int64_t* d_count, int val, int left_nidx) {
    unsigned ballot = __ballot(val == left_nidx);
    if (threadIdx.x % 32 == 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(d_count),    // NOLINT
                static_cast<unsigned long long>(__popc(ballot)));  // NOLINT
    }
  }

  void UpdatePosition(int nidx, int left_nidx, int right_nidx, int fidx,
                      int split_gidx, bool default_dir_left, bool is_dense,
                      int fidx_begin, int fidx_end) {
    dh::safe_cuda(cudaSetDevice(device_idx));
    temp_memory.LazyAllocate(sizeof(int64_t));
    auto d_left_count = temp_memory.Pointer<int64_t>();
    dh::safe_cuda(cudaMemset(d_left_count, 0, sizeof(int64_t)));
    auto segment = ridx_segments[nidx];
    auto d_ridx = ridx.Current();
    auto d_position = position.Current();
    auto d_gidx = gidx;
    auto row_stride = this->row_stride;
    dh::LaunchN<1, 512>(
        device_idx, segment.Size(), [=] __device__(bst_uint idx) {
          idx += segment.begin;
          auto ridx = d_ridx[idx];
          auto row_begin = row_stride * ridx;
          auto row_end = row_begin + row_stride;
          auto gidx = -1;
          if (is_dense) {
            gidx = d_gidx[row_begin + fidx];
          } else {
            gidx = BinarySearchRow(row_begin, row_end, d_gidx, fidx_begin,
                                   fidx_end);
          }

          int position;
          if (gidx >= 0) {
            // Feature is found
            position = gidx <= split_gidx ? left_nidx : right_nidx;
          } else {
            // Feature is missing
            position = default_dir_left ? left_nidx : right_nidx;
          }

          CountLeft(d_left_count, position, left_nidx);
          d_position[idx] = position;
        });

    dh::safe_cuda(cudaMemcpy(tmp_pinned, d_left_count, sizeof(int64_t),
                             cudaMemcpyDeviceToHost));
    auto left_count = *tmp_pinned;

    SortPosition(segment, left_nidx, right_nidx);
    // dh::safe_cuda(cudaStreamSynchronize(stream));
    ridx_segments[left_nidx] =
        Segment(segment.begin, segment.begin + left_count);
    ridx_segments[right_nidx] =
        Segment(segment.begin + left_count, segment.end);
  }

  void SortPosition(const Segment& segment, int left_nidx, int right_nidx) {
    int min_bits = 0;
    int max_bits = static_cast<int>(
        std::ceil(std::log2((std::max)(left_nidx, right_nidx) + 1)));

    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes, position.Current() + segment.begin,
        position.other() + segment.begin, ridx.Current() + segment.begin,
        ridx.other() + segment.begin, segment.Size(), min_bits, max_bits);

    temp_memory.LazyAllocate(temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(
        temp_memory.d_temp_storage, temp_memory.temp_storage_bytes,
        position.Current() + segment.begin, position.other() + segment.begin,
        ridx.Current() + segment.begin, ridx.other() + segment.begin,
        segment.Size(), min_bits, max_bits);
    dh::safe_cuda(cudaMemcpy(
        position.Current() + segment.begin, position.other() + segment.begin,
        segment.Size() * sizeof(int), cudaMemcpyDeviceToDevice));
    dh::safe_cuda(cudaMemcpy(
        ridx.Current() + segment.begin, ridx.other() + segment.begin,
        segment.Size() * sizeof(bst_uint), cudaMemcpyDeviceToDevice));
  }

  void UpdatePredictionCache(bst_float* out_preds_d) {
    dh::safe_cuda(cudaSetDevice(device_idx));
    if (!prediction_cache_initialised) {
      dh::safe_cuda(cudaMemcpy(
          prediction_cache.Data(), out_preds_d,
          prediction_cache.Size() * sizeof(bst_float), cudaMemcpyDefault));
    }
    prediction_cache_initialised = true;

    CalcWeightTrainParam param_d(param);

    thrust::copy(node_sum_gradients.begin(), node_sum_gradients.end(),
                 node_sum_gradients_d.tbegin());
    auto d_position = position.Current();
    auto d_ridx = ridx.Current();
    auto d_node_sum_gradients = node_sum_gradients_d.Data();
    auto d_prediction_cache = prediction_cache.Data();

    dh::LaunchN(
        device_idx, prediction_cache.Size(), [=] __device__(int local_idx) {
          int pos = d_position[local_idx];
          bst_float weight = CalcWeight(param_d, d_node_sum_gradients[pos]);
          d_prediction_cache[d_ridx[local_idx]] +=
              weight * param_d.learning_rate;
        });

    dh::safe_cuda(cudaMemcpy(
        out_preds_d, prediction_cache.Data(),
        prediction_cache.Size() * sizeof(bst_float), cudaMemcpyDefault));
  }
};

class GPUHistMaker : public TreeUpdater {
 public:
  struct ExpandEntry;

  GPUHistMaker() : initialised_(false), p_last_fmat_(nullptr) {}
  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) override {
    param_.InitAllowUnknown(args);
    CHECK(param_.n_gpus != 0) << "Must have at least one device";
    n_devices_ = param_.n_gpus;
    devices_ = GPUSet::Range(param_.gpu_id, dh::NDevicesAll(param_.n_gpus));

    dh::CheckComputeCapability();

    if (param_.grow_policy == TrainParam::kLossGuide) {
      qexpand_.reset(new ExpandQueue(LossGuide));
    } else {
      qexpand_.reset(new ExpandQueue(DepthWise));
    }

    monitor_.Init("updater_gpu_hist", param_.debug_verbose);
  }

  void Update(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    monitor_.Start("Update", device_list_);
    GradStats::CheckInfo(dmat->Info());
    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    ValueConstraint::Init(&param_, dmat->Info().num_col_);
    // build tree
    try {
      for (size_t i = 0; i < trees.size(); ++i) {
        this->UpdateTree(gpair, dmat, trees[i]);
      }
    } catch (const std::exception& e) {
      LOG(FATAL) << "GPU plugin exception: " << e.what() << std::endl;
    }
    param_.learning_rate = lr;
    monitor_.Stop("Update", device_list_);
  }

  void InitDataOnce(DMatrix* dmat) {
    info_ = &dmat->Info();

    int n_devices = dh::NDevices(param_.n_gpus, info_->num_row_);

    bst_uint row_begin = 0;
    bst_uint shard_size =
        std::ceil(static_cast<double>(info_->num_row_) / n_devices);

    device_list_.resize(n_devices);
    for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
      int device_idx = (param_.gpu_id + d_idx) % dh::NVisibleDevices();
      device_list_[d_idx] = device_idx;
    }

    reducer_.Init(device_list_);

    // Partition input matrix into row segments
    std::vector<size_t> row_segments;
    shards_.resize(n_devices);
    row_segments.push_back(0);
    for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
      bst_uint row_end =
          std::min(static_cast<size_t>(row_begin + shard_size), info_->num_row_);
      row_segments.push_back(row_end);
      row_begin = row_end;
    }

    dmlc::DataIter<SparsePage>* iter = dmat->RowIterator();
    iter->BeforeFirst();
    CHECK(iter->Next()) << "Empty batches are not supported";
    const SparsePage& batch = iter->Value();
    // Create device shards
    dh::ExecuteIndexShards(&shards_, [&](int i, std::unique_ptr<DeviceShard>& shard) {
        shard = std::unique_ptr<DeviceShard>
          (new DeviceShard(device_list_[i], i,
                           row_segments[i], row_segments[i + 1], param_));
        shard->InitRowPtrs(batch);
      });

    monitor_.Start("Quantiles", device_list_);

    // compute sketches for each shard
    dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {
        shard->Sketch(batch, *info_);
      });

    // merge the sketches from all shards
    // TODO(canonizer): do it in a tree-like reduction
    int num_cols = dmat->Info().num_col_;
    std::vector<WXQSketch> sketches(num_cols);
    WXQSketch::SummaryContainer summary;
    for (int icol = 0; icol < num_cols; ++icol) {
      sketches[icol].Init(batch.Size(), 1.0 / (8 * param_.max_bin));
      for (int shard = 0; shard < shards_.size(); ++shard) {
        shards_[shard]->sketches[icol].GetSummary(&summary);
        sketches[icol].PushSummary(summary);
      }
    }

    hmat_.Init(&sketches, param_.max_bin);
    n_bins_ = hmat_.row_ptr.back();
    monitor_.Stop("Quantiles", device_list_);

    monitor_.Start("BinningCompression", device_list_);
    dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {
        shard->InitCompressedData(hmat_, batch);
      });
    monitor_.Stop("BinningCompression", device_list_);

    CHECK(!iter->Next()) << "External memory not supported";

    p_last_fmat_ = dmat;
    initialised_ = true;
  }

  void InitData(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
                const RegTree& tree) {
    monitor_.Start("InitDataOnce", device_list_);
    if (!initialised_) {
      this->InitDataOnce(dmat);
    }
    monitor_.Stop("InitDataOnce", device_list_);

    column_sampler_.Init(info_->num_col_, param_);

    // Copy gpair & reset memory
    monitor_.Start("InitDataReset", device_list_);

    gpair->Reshard(devices_);
    dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {shard->Reset(gpair); });
    monitor_.Stop("InitDataReset", device_list_);
  }

  void AllReduceHist(int nidx) {
    for (auto& shard : shards_) {
      auto d_node_hist = shard->hist.GetHistPtr(nidx);
      reducer_.AllReduceSum(
          shard->normalised_device_idx,
          reinterpret_cast<GradientPairSumT::ValueT*>(d_node_hist),
          reinterpret_cast<GradientPairSumT::ValueT*>(d_node_hist),
          n_bins_ * (sizeof(GradientPairSumT) / sizeof(GradientPairSumT::ValueT)));
    }

    reducer_.Synchronize();
  }

  void BuildHistLeftRight(int nidx_parent, int nidx_left, int nidx_right) {
    size_t left_node_max_elements = 0;
    size_t right_node_max_elements = 0;
    for (auto& shard : shards_) {
      left_node_max_elements = (std::max)(
          left_node_max_elements, shard->ridx_segments[nidx_left].Size());
      right_node_max_elements = (std::max)(
          right_node_max_elements, shard->ridx_segments[nidx_right].Size());
    }

    auto build_hist_nidx = nidx_left;
    auto subtraction_trick_nidx = nidx_right;

    if (right_node_max_elements < left_node_max_elements) {
      build_hist_nidx = nidx_right;
      subtraction_trick_nidx = nidx_left;
    }

    dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {
        shard->BuildHist(build_hist_nidx);
      });

    this->AllReduceHist(build_hist_nidx);

    dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {
        shard->SubtractionTrick(nidx_parent, build_hist_nidx,
                               subtraction_trick_nidx);
      });
  }

  // Returns best loss
  std::vector<DeviceSplitCandidate> EvaluateSplits(
      const std::vector<int>& nidx_set, RegTree* p_tree) {
    auto columns = info_->num_col_;
    std::vector<DeviceSplitCandidate> best_splits(nidx_set.size());
    std::vector<DeviceSplitCandidate> candidate_splits(nidx_set.size() *
                                                       columns);
    // Use first device
    auto& shard = shards_.front();
    dh::safe_cuda(cudaSetDevice(shard->device_idx));
    shard->temp_memory.LazyAllocate(sizeof(DeviceSplitCandidate) * columns *
                                    nidx_set.size());
    auto d_split = shard->temp_memory.Pointer<DeviceSplitCandidate>();

    auto& streams = shard->GetStreams(static_cast<int>(nidx_set.size()));

    // Use streams to process nodes concurrently
    for (auto i = 0; i < nidx_set.size(); i++) {
      auto nidx = nidx_set[i];
      DeviceNodeStats node(shard->node_sum_gradients[nidx], nidx, param_);

      const int BLOCK_THREADS = 256;
      evaluate_split_kernel<BLOCK_THREADS>
          <<<uint32_t(columns), BLOCK_THREADS, 0, streams[i]>>>(
              shard->hist.GetHistPtr(nidx), nidx, info_->num_col_, node,
              shard->feature_segments.Data(), shard->min_fvalue.Data(),
              shard->gidx_fvalue_map.Data(), GPUTrainingParam(param_),
              d_split + i * columns, node_value_constraints_[nidx],
              shard->monotone_constraints.Data());
    }

    dh::safe_cuda(
        cudaMemcpy(candidate_splits.data(), shard->temp_memory.d_temp_storage,
                   sizeof(DeviceSplitCandidate) * columns * nidx_set.size(),
                   cudaMemcpyDeviceToHost));

    for (auto i = 0; i < nidx_set.size(); i++) {
      auto nidx = nidx_set[i];
      DeviceSplitCandidate nidx_best;
      for (auto fidx = 0; fidx < columns; fidx++) {
        auto& candidate = candidate_splits[i * columns + fidx];
        if (column_sampler_.ColumnUsed(candidate.findex,
                                      p_tree->GetDepth(nidx))) {
          nidx_best.Update(candidate_splits[i * columns + fidx], param_);
        }
      }
      best_splits[i] = nidx_best;
    }
    return std::move(best_splits);
  }

  void InitRoot(RegTree* p_tree) {
    auto root_nidx = 0;
    // Sum gradients
    std::vector<GradientPair> tmp_sums(shards_.size());

    dh::ExecuteIndexShards(&shards_, [&](int i, std::unique_ptr<DeviceShard>& shard) {
        dh::safe_cuda(cudaSetDevice(shard->device_idx));
      tmp_sums[i] =
        dh::SumReduction(shard->temp_memory, shard->gpair.Data(),
                         shard->gpair.Size());
      });
    auto sum_gradient =
        std::accumulate(tmp_sums.begin(), tmp_sums.end(), GradientPair());

    // Generate root histogram
    dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {
        shard->BuildHist(root_nidx);
      });

    this->AllReduceHist(root_nidx);

    // Remember root stats
    p_tree->Stat(root_nidx).sum_hess = sum_gradient.GetHess();
    auto weight = CalcWeight(param_, sum_gradient);
    p_tree->Stat(root_nidx).base_weight = weight;
    (*p_tree)[root_nidx].SetLeaf(param_.learning_rate * weight);

    // Store sum gradients
    for (auto& shard : shards_) {
      shard->node_sum_gradients[root_nidx] = sum_gradient;
    }

    // Initialise root constraint
    node_value_constraints_.resize(p_tree->GetNodes().size());

    // Generate first split
    auto splits = this->EvaluateSplits({root_nidx}, p_tree);
    qexpand_->push(
        ExpandEntry(root_nidx, p_tree->GetDepth(root_nidx), splits.front(), 0));
  }

  void UpdatePosition(const ExpandEntry& candidate, RegTree* p_tree) {
    auto nidx = candidate.nid;
    auto left_nidx = (*p_tree)[nidx].LeftChild();
    auto right_nidx = (*p_tree)[nidx].RightChild();

    // convert floating-point split_pt into corresponding bin_id
    // split_cond = -1 indicates that split_pt is less than all known cut points
    auto split_gidx = -1;
    auto fidx = candidate.split.findex;
    auto default_dir_left = candidate.split.dir == kLeftDir;
    auto fidx_begin = hmat_.row_ptr[fidx];
    auto fidx_end = hmat_.row_ptr[fidx + 1];
    for (auto i = fidx_begin; i < fidx_end; ++i) {
      if (candidate.split.fvalue == hmat_.cut[i]) {
        split_gidx = static_cast<int32_t>(i);
      }
    }

    auto is_dense = info_->num_nonzero_ == info_->num_row_ * info_->num_col_;

    dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {
      shard->UpdatePosition(nidx, left_nidx, right_nidx, fidx,
                           split_gidx, default_dir_left,
                           is_dense, fidx_begin, fidx_end);
      });
  }

  void ApplySplit(const ExpandEntry& candidate, RegTree* p_tree) {
    // Add new leaves
    RegTree& tree = *p_tree;
    tree.AddChilds(candidate.nid);
    auto& parent = tree[candidate.nid];
    parent.SetSplit(candidate.split.findex, candidate.split.fvalue,
                     candidate.split.dir == kLeftDir);
    tree.Stat(candidate.nid).loss_chg = candidate.split.loss_chg;

    // Set up child constraints
    node_value_constraints_.resize(tree.GetNodes().size());
    GradStats left_stats(param_);
    left_stats.Add(candidate.split.left_sum);
    GradStats right_stats(param_);
    right_stats.Add(candidate.split.right_sum);
    node_value_constraints_[candidate.nid].SetChild(
        param_, parent.SplitIndex(), left_stats, right_stats,
        &node_value_constraints_[parent.LeftChild()],
        &node_value_constraints_[parent.RightChild()]);

    // Configure left child
    auto left_weight =
        node_value_constraints_[parent.LeftChild()].CalcWeight(param_, left_stats);
    tree[parent.LeftChild()].SetLeaf(left_weight * param_.learning_rate, 0);
    tree.Stat(parent.LeftChild()).base_weight = left_weight;
    tree.Stat(parent.LeftChild()).sum_hess = candidate.split.left_sum.GetHess();

    // Configure right child
    auto right_weight =
        node_value_constraints_[parent.RightChild()].CalcWeight(param_, right_stats);
    tree[parent.RightChild()].SetLeaf(right_weight * param_.learning_rate, 0);
    tree.Stat(parent.RightChild()).base_weight = right_weight;
    tree.Stat(parent.RightChild()).sum_hess = candidate.split.right_sum.GetHess();
    // Store sum gradients
    for (auto& shard : shards_) {
      shard->node_sum_gradients[parent.LeftChild()] = candidate.split.left_sum;
      shard->node_sum_gradients[parent.RightChild()] = candidate.split.right_sum;
    }
    this->UpdatePosition(candidate, p_tree);
  }

  void UpdateTree(HostDeviceVector<GradientPair>* gpair, DMatrix* p_fmat,
                  RegTree* p_tree) {
    auto& tree = *p_tree;

    monitor_.Start("InitData", device_list_);
    this->InitData(gpair, p_fmat, *p_tree);
    monitor_.Stop("InitData", device_list_);
    monitor_.Start("InitRoot", device_list_);
    this->InitRoot(p_tree);
    monitor_.Stop("InitRoot", device_list_);

    auto timestamp = qexpand_->size();
    auto num_leaves = 1;

    while (!qexpand_->empty()) {
      auto candidate = qexpand_->top();
      qexpand_->pop();
      if (!candidate.IsValid(param_, num_leaves)) continue;
      // std::cout << candidate;
      monitor_.Start("ApplySplit", device_list_);
      this->ApplySplit(candidate, p_tree);
      monitor_.Stop("ApplySplit", device_list_);
      num_leaves++;

      auto left_child_nidx = tree[candidate.nid].LeftChild();
      auto right_child_nidx = tree[candidate.nid].RightChild();

      // Only create child entries if needed
      if (ExpandEntry::ChildIsValid(param_, tree.GetDepth(left_child_nidx),
                                    num_leaves)) {
        monitor_.Start("BuildHist", device_list_);
        this->BuildHistLeftRight(candidate.nid, left_child_nidx,
                                 right_child_nidx);
        monitor_.Stop("BuildHist", device_list_);

        monitor_.Start("EvaluateSplits", device_list_);
        auto splits =
            this->EvaluateSplits({left_child_nidx, right_child_nidx}, p_tree);
        qexpand_->push(ExpandEntry(left_child_nidx,
                                   tree.GetDepth(left_child_nidx), splits[0],
                                   timestamp++));
        qexpand_->push(ExpandEntry(right_child_nidx,
                                   tree.GetDepth(right_child_nidx), splits[1],
                                   timestamp++));
        monitor_.Stop("EvaluateSplits", device_list_);
      }
    }
  }

  bool UpdatePredictionCache(
      const DMatrix* data, HostDeviceVector<bst_float>* p_out_preds) override {
    monitor_.Start("UpdatePredictionCache", device_list_);
    if (shards_.empty() || p_last_fmat_ == nullptr || p_last_fmat_ != data)
      return false;
    p_out_preds->Reshard(devices_);
    dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {
        shard->UpdatePredictionCache(p_out_preds->DevicePointer(shard->device_idx));
      });
    monitor_.Stop("UpdatePredictionCache", device_list_);
    return true;
  }

  struct ExpandEntry {
    int nid;
    int depth;
    DeviceSplitCandidate split;
    uint64_t timestamp;
    ExpandEntry(int nid, int depth, const DeviceSplitCandidate& split,
                uint64_t timestamp)
        : nid(nid), depth(depth), split(split), timestamp(timestamp) {}
    bool IsValid(const TrainParam& param, int num_leaves) const {
      if (split.loss_chg <= kRtEps) return false;
      if (split.left_sum.GetHess() == 0 || split.right_sum.GetHess() == 0)
        return false;
      if (param.max_depth > 0 && depth == param.max_depth) return false;
      if (param.max_leaves > 0 && num_leaves == param.max_leaves) return false;
      return true;
    }

    static bool ChildIsValid(const TrainParam& param, int depth,
                             int num_leaves) {
      if (param.max_depth > 0 && depth == param.max_depth) return false;
      if (param.max_leaves > 0 && num_leaves == param.max_leaves) return false;
      return true;
    }

    friend std::ostream& operator<<(std::ostream& os, const ExpandEntry& e) {
      os << "ExpandEntry: \n";
      os << "nidx: " << e.nid << "\n";
      os << "depth: " << e.depth << "\n";
      os << "loss: " << e.split.loss_chg << "\n";
      os << "left_sum: " << e.split.left_sum << "\n";
      os << "right_sum: " << e.split.right_sum << "\n";
      return os;
    }
  };

  inline static bool DepthWise(ExpandEntry lhs, ExpandEntry rhs) {
    if (lhs.depth == rhs.depth) {
      return lhs.timestamp > rhs.timestamp;  // favor small timestamp
    } else {
      return lhs.depth > rhs.depth;  // favor small depth
    }
  }
  inline static bool LossGuide(ExpandEntry lhs, ExpandEntry rhs) {
    if (lhs.split.loss_chg == rhs.split.loss_chg) {
      return lhs.timestamp > rhs.timestamp;  // favor small timestamp
    } else {
      return lhs.split.loss_chg < rhs.split.loss_chg;  // favor large loss_chg
    }
  }
  TrainParam param_;
  common::HistCutMatrix hmat_;
  common::GHistIndexMatrix gmat_;
  MetaInfo* info_;
  bool initialised_;
  int n_devices_;
  int n_bins_;

  std::vector<std::unique_ptr<DeviceShard>> shards_;
  ColumnSampler column_sampler_;
  typedef std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
                              std::function<bool(ExpandEntry, ExpandEntry)>>
      ExpandQueue;
  std::unique_ptr<ExpandQueue> qexpand_;
  common::Monitor monitor_;
  dh::AllReducer reducer_;
  std::vector<ValueConstraint> node_value_constraints_;
  std::vector<int> device_list_;

  DMatrix* p_last_fmat_;
  GPUSet devices_;
};

XGBOOST_REGISTER_TREE_UPDATER(GPUHistMaker, "grow_gpu_hist")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUHistMaker(); });
}  // namespace tree
}  // namespace xgboost
