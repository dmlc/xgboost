/*!
 * Copyright 2017-2018 XGBoost contributors
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
#include <limits>
#include <queue>
#include <utility>
#include <vector>
#include "../common/common.h"
#include "../common/compressed_iterator.h"
#include "../common/device_helpers.cuh"
#include "../common/hist_util.h"
#include "../common/host_device_vector.h"
#include "../common/timer.h"
#include "../common/span.h"
#include "param.h"
#include "updater_gpu_common.cuh"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_gpu_hist);

using GradientPairSumT = GradientPairPrecise;

/*!
 * \brief
 *
 * \tparam ReduceT     BlockReduce Type.
 * \tparam TempStorage Cub Shared memory
 *
 * \param begin
 * \param end
 * \param temp_storage Shared memory for intermediate result.
 */
template <int BLOCK_THREADS, typename ReduceT, typename TempStorageT>
__device__ GradientPairSumT ReduceFeature(const GradientPairSumT* begin,
                                          const GradientPairSumT* end,
                                          TempStorageT* temp_storage) {
  __shared__ cub::Uninitialized<GradientPairSumT> uninitialized_sum;
  GradientPairSumT& shared_sum = uninitialized_sum.Alias();

  GradientPairSumT local_sum = GradientPairSumT();
  // For loop sums features into one block size
  for (auto itr = begin; itr < end; itr += BLOCK_THREADS) {
    bool thread_active = itr + threadIdx.x < end;
    // Scan histogram
    GradientPairSumT bin = thread_active ? *(itr + threadIdx.x) : GradientPairSumT();
    local_sum += bin;
  }
  local_sum = ReduceT(temp_storage->sum_reduce).Reduce(local_sum, cub::Sum());
  // Reduction result is stored in thread 0.
  if (threadIdx.x == 0) {
    shared_sum = local_sum;
  }
  __syncthreads();
  return shared_sum;
}

/*! \brief Find the thread with best gain. */
template <int BLOCK_THREADS, typename ReduceT, typename scan_t,
          typename max_ReduceT, typename TempStorageT>
__device__ void EvaluateFeature(
    int fidx,
    const GradientPairSumT* hist,

    const uint32_t* feature_segments,  // cut.row_ptr
    float min_fvalue,                  // cut.min_value
    const float* gidx_fvalue_map,      // cut.cut

    DeviceSplitCandidate* best_split,  // shared memory storing best split
    const DeviceNodeStats& node,
    const GPUTrainingParam& param,
    TempStorageT* temp_storage,  // temp memory for cub operations
    int constraint,              // monotonic_constraints
    const ValueConstraint& value_constraint) {
  // Use pointer from cut to indicate begin and end of bins for each feature.
  uint32_t gidx_begin = feature_segments[fidx];    // begining bin
  uint32_t gidx_end = feature_segments[fidx + 1];  // end bin for i^th feature

  // Sum histogram bins for current feature
  GradientPairSumT const feature_sum = ReduceFeature<BLOCK_THREADS, ReduceT>(
      hist + gidx_begin, hist + gidx_end, temp_storage);

  GradientPairSumT const parent_sum = GradientPairSumT(node.sum_gradients);
  GradientPairSumT const missing = parent_sum - feature_sum;
  float const null_gain = -std::numeric_limits<bst_float>::infinity();

  SumCallbackOp<GradientPairSumT> prefix_op =
      SumCallbackOp<GradientPairSumT>();
  for (int scan_begin = gidx_begin; scan_begin < gidx_end;
       scan_begin += BLOCK_THREADS) {
    bool thread_active = (scan_begin + threadIdx.x) < gidx_end;

    // Gradient value for current bin.
    GradientPairSumT bin =
        thread_active ? hist[scan_begin + threadIdx.x] : GradientPairSumT();
    scan_t(temp_storage->scan).ExclusiveScan(bin, bin, cub::Sum(), prefix_op);

    // Whether the gradient of missing values is put to the left side.
    bool missing_left = true;
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
      best_split->Update(gain, missing_left ? kLeftDir : kRightDir,
                         fvalue, fidx,
                         GradientPair(left),
                         GradientPair(right),
                         param);
    }
    __syncthreads();
  }
}

template <int BLOCK_THREADS>
__global__ void EvaluateSplitKernel(
    const GradientPairSumT* d_hist,  // histogram for gradients
    uint64_t n_features,
    int* feature_set,  // Selected features
    DeviceNodeStats node,

    const uint32_t* d_feature_segments,  // row_ptr form HistCutMatrix
    const float* d_fidx_min_map,         // min_value
    const float* d_gidx_fvalue_map,      // cut

    GPUTrainingParam gpu_param,
    DeviceSplitCandidate* d_split,  // resulting split
    ValueConstraint value_constraint,
    int* d_monotonic_constraints) {
  // KeyValuePair here used as threadIdx.x -> gain_value
  typedef cub::KeyValuePair<int, float> ArgMaxT;
  typedef cub::BlockScan<
    GradientPairSumT, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS> BlockScanT;
  typedef cub::BlockReduce<ArgMaxT, BLOCK_THREADS> MaxReduceT;

  typedef cub::BlockReduce<GradientPairSumT, BLOCK_THREADS> SumReduceT;

  union TempStorage {
    typename BlockScanT::TempStorage scan;
    typename MaxReduceT::TempStorage max_reduce;
    typename SumReduceT::TempStorage sum_reduce;
  };

  // Aligned && shared storage for best_split
  __shared__ cub::Uninitialized<DeviceSplitCandidate> uninitialized_split;
  DeviceSplitCandidate& best_split = uninitialized_split.Alias();
  __shared__ TempStorage temp_storage;

  if (threadIdx.x == 0) {
    best_split = DeviceSplitCandidate();
  }

  __syncthreads();

  // One block for each feature. Features are sampled, so fidx != blockIdx.x
  int fidx = feature_set[blockIdx.x];
  int constraint = d_monotonic_constraints[fidx];
  EvaluateFeature<BLOCK_THREADS, SumReduceT, BlockScanT, MaxReduceT>(
      fidx,
      d_hist,

      d_feature_segments,
      d_fidx_min_map[fidx],
      d_gidx_fvalue_map,

      &best_split,
      node,
      gpu_param,
      &temp_storage,
      constraint,
      value_constraint);

  __syncthreads();

  if (threadIdx.x == 0) {
    // Record best loss for each feature
    d_split[fidx] = best_split;
  }
}

// Find a gidx value for a given feature otherwise return -1 if not found
template <typename GidxIterT>
__device__ int BinarySearchRow(bst_uint begin, bst_uint end, GidxIterT data,
                               int const fidx_begin, int const fidx_end) {
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

/**
 * \struct  DeviceHistogram
 *
 * \summary Data storage for node histograms on device. Automatically expands.
 *
 * \author  Rory
 * \date    28/07/2018
 */
struct DeviceHistogram {
  /*! \brief Map nidx to starting index of its histogram. */
  std::map<int, size_t> nidx_map;
  thrust::device_vector<GradientPairSumT::ValueT> data;
  const size_t kStopGrowingSize = 1 << 26;  // Do not grow beyond this size
  int n_bins;
  int device_idx;

  void Init(int device_idx, int n_bins) {
    this->n_bins = n_bins;
    this->device_idx = device_idx;
  }

  void Reset() {
    dh::safe_cuda(cudaSetDevice(device_idx));
    data.resize(0);
    nidx_map.clear();
  }

  bool HistogramExists(int nidx) {
    return nidx_map.find(nidx) != nidx_map.end();
  }

  void AllocateHistogram(int nidx) {
    if (HistogramExists(nidx)) return;

    if (data.size() > kStopGrowingSize) {
      // Recycle histogram memory
      std::pair<int, size_t> old_entry = *nidx_map.begin();
      nidx_map.erase(old_entry.first);
      dh::safe_cuda(cudaMemset(data.data().get() + old_entry.second, 0,
                               n_bins * sizeof(GradientPairSumT)));
      nidx_map[nidx] = old_entry.second;
    } else {
      // Append new node histogram
      nidx_map[nidx] = data.size();
      dh::safe_cuda(cudaSetDevice(device_idx));
      // x 2: Hess and Grad.
      data.resize(data.size() + (n_bins * 2));
    }
  }

  /**
   * \summary   Return pointer to histogram memory for a given node.
   * \param nidx    Tree node index.
   * \return    hist pointer.
   */
  GradientPairSumT* GetHistPtr(int nidx) {
    CHECK(this->HistogramExists(nidx));
    auto ptr = data.data().get() + nidx_map[nidx];
    return reinterpret_cast<GradientPairSumT*>(ptr);
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

// Bin each input data entry, store the bin indices in compressed form.
__global__ void compress_bin_ellpack_k(
    common::CompressedBufferWriter wr,
    common::CompressedByteT* __restrict__ buffer,  // gidx_buffer
    const size_t* __restrict__ row_ptrs,           // row offset of input data
    const Entry* __restrict__ entries,      // One batch of input data
    const float* __restrict__ cuts,         // HistCutMatrix::cut
    const uint32_t* __restrict__ cut_rows,  // HistCutMatrix::row_ptrs
    size_t base_row,                        // batch_row_begin
    size_t n_rows,
    // row_ptr_begin: row_offset[base_row], the start position of base_row
    size_t row_ptr_begin,
    size_t row_stride,
    unsigned int null_gidx_value) {
  size_t irow = threadIdx.x + blockIdx.x * blockDim.x;
  int ifeature = threadIdx.y + blockIdx.y * blockDim.y;
  if (irow >= n_rows || ifeature >= row_stride)
    return;
  int row_length = static_cast<int>(row_ptrs[irow + 1] - row_ptrs[irow]);
  unsigned int bin = null_gidx_value;
  if (ifeature < row_length) {
    Entry entry = entries[row_ptrs[irow] - row_ptr_begin + ifeature];
    int feature = entry.index;
    float fvalue = entry.fvalue;
    // {feature_cuts, ncuts} forms the array of cuts of `feature'.
    const float *feature_cuts = &cuts[cut_rows[feature]];
    int ncuts = cut_rows[feature + 1] - cut_rows[feature];
    // Assigning the bin in current entry.
    // S.t.: fvalue < feature_cuts[bin]
    bin = dh::UpperBound(feature_cuts, ncuts, fvalue);
    if (bin >= ncuts)
      bin = ncuts - 1;
    // Add the number of bins in previous features.
    bin += cut_rows[feature];
  }
  // Write to gidx buffer.
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

struct Segment {
  size_t begin;
  size_t end;

  Segment() : begin(0), end(0) {}

  Segment(size_t begin, size_t end) : begin(begin), end(end) {
    CHECK_GE(end, begin);
  }
  size_t Size() const { return end - begin; }
};

struct DeviceShard;

struct GPUHistBuilderBase {
 public:
  virtual void Build(DeviceShard* shard, int idx) = 0;
};

// Manage memory for a single GPU
struct DeviceShard {
  int device_idx;
  /*! \brief Device index counting from param.gpu_id */
  int normalised_device_idx;
  dh::BulkAllocator<dh::MemoryType::kDevice> ba;

  /*! \brief HistCutMatrix stored in device. */
  struct DeviceHistCutMatrix {
    /*! \brief row_ptr form HistCutMatrix. */
    dh::DVec<uint32_t> feature_segments;
    /*! \brief minimum value for each feature. */
    dh::DVec<bst_float> min_fvalue;
    /*! \brief Cut. */
    dh::DVec<bst_float> gidx_fvalue_map;
  } cut_;

  /*! \brief Range of rows for each node. */
  std::vector<Segment> ridx_segments;
  DeviceHistogram hist;

  /*! \brief global index of histogram, which is stored in ELLPack format. */
  dh::DVec<common::CompressedByteT> gidx_buffer;
  /*! \brief row length for ELLPack. */
  size_t row_stride;
  common::CompressedIterator<uint32_t> gidx;

  /*! \brief  Row indices relative to this shard, necessary for sorting rows. */
  dh::DVec2<bst_uint> ridx;
  /*! \brief Gradient pair for each row. */
  dh::DVec<GradientPair> gpair;

  /*! \brief The last histogram index. */
  int null_gidx_value;

  dh::DVec2<int> position;

  dh::DVec<int> monotone_constraints;
  dh::DVec<bst_float> prediction_cache;

  /*! \brief Sum gradient for each node. */
  std::vector<GradientPair> node_sum_gradients;
  dh::DVec<GradientPair> node_sum_gradients_d;
  /*! \brief row offset in SparsePage (the input data). */
  thrust::device_vector<size_t> row_ptrs;
  /*! The row offset for this shard. */
  bst_uint row_begin_idx;
  bst_uint row_end_idx;
  bst_uint n_rows;
  int n_bins;

  TrainParam param;
  bool prediction_cache_initialised;

  // FIXME: Remove this
  int64_t* tmp_pinned;  // Small amount of staging memory

  // Used to process nodes concurrently
  std::vector<cudaStream_t> streams;

  dh::CubMemory temp_memory;

  std::unique_ptr<GPUHistBuilderBase> hist_builder;

  // TODO(canonizer): do add support multi-batch DMatrix here
  DeviceShard(int device_idx, int normalised_device_idx,
              bst_uint row_begin, bst_uint row_end, TrainParam _param) :
    device_idx(device_idx),
    normalised_device_idx(normalised_device_idx),
    row_begin_idx(row_begin),
    row_end_idx(row_end),
    row_stride(0),
    n_rows(row_end - row_begin),
    n_bins(0),
    null_gidx_value(0),
    param(_param),
    prediction_cache_initialised(false),
    tmp_pinned(nullptr) {}

  /* Init row_ptrs and row_stride */
  void InitRowPtrs(const SparsePage& row_batch) {
    dh::safe_cuda(cudaSetDevice(device_idx));
    const auto& offset_vec = row_batch.offset.HostVector();
    row_ptrs.resize(n_rows + 1);
    thrust::copy(offset_vec.data() + row_begin_idx,
                 offset_vec.data() + row_end_idx + 1,
                 row_ptrs.begin());
    auto row_iter = row_ptrs.begin();
    // find the maximum row size for converting to ELLPack
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

  /*
     Init:
     n_bins, null_gidx_value, gidx_buffer, row_ptrs, gidx, gidx_fvalue_map,
     min_fvalue, feature_segments, node_sum_gradients, ridx_segments,
     hist
  */
  void InitCompressedData(
      const common::HistCutMatrix& hmat, const SparsePage& row_batch);

  void CreateHistIndices(const SparsePage& row_batch);

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
    this->gpair.copy(dh_gpair->tcbegin(device_idx), dh_gpair->tcend(device_idx));
    SubsampleGradientPair(&gpair, param.subsample, row_begin_idx);
    hist.Reset();
  }

  void BuildHist(int nidx) {
    hist.AllocateHistogram(nidx);
    hist_builder->Build(this, nidx);
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

  bool CanDoSubtractionTrick(int nidx_parent, int nidx_histogram,
                             int nidx_subtraction) {
    // Make sure histograms are already allocated
    hist.AllocateHistogram(nidx_subtraction);
    return hist.HistogramExists(nidx_histogram) &&
           hist.HistogramExists(nidx_parent);
  }

  /*! \brief Count how many rows are assigned to left node. */
  __device__ void CountLeft(int64_t* d_count, int val, int left_nidx) {
    unsigned ballot = __ballot(val == left_nidx);
    if (threadIdx.x % 32 == 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(d_count),    // NOLINT
                static_cast<unsigned long long>(__popc(ballot)));  // NOLINT
    }
  }

  void UpdatePosition(int nidx, int left_nidx, int right_nidx, int fidx,
                      int64_t split_gidx, bool default_dir_left, bool is_dense,
                      int fidx_begin,  // cut.row_ptr[fidx]
                      int fidx_end) {  // cut.row_ptr[fidx + 1]
    dh::safe_cuda(cudaSetDevice(device_idx));
    temp_memory.LazyAllocate(sizeof(int64_t));
    int64_t* d_left_count = temp_memory.Pointer<int64_t>();
    dh::safe_cuda(cudaMemset(d_left_count, 0, sizeof(int64_t)));
    Segment segment = ridx_segments[nidx];
    bst_uint* d_ridx = ridx.Current();
    int* d_position = position.Current();
    common::CompressedIterator<uint32_t> d_gidx = gidx;
    size_t row_stride = this->row_stride;
    // Launch 1 thread for each row
    dh::LaunchN<1, 512>(
        device_idx, segment.Size(), [=] __device__(bst_uint idx) {
          idx += segment.begin;
          bst_uint ridx = d_ridx[idx];
          auto row_begin = row_stride * ridx;
          auto row_end = row_begin + row_stride;
          auto gidx = -1;
          if (is_dense) {
            // FIXME: Maybe just search the cuts again.
            gidx = d_gidx[row_begin + fidx];
          } else {
            gidx = BinarySearchRow(row_begin, row_end, d_gidx, fidx_begin,
                                   fidx_end);
          }

          // belong to left node or right node.
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

    ridx_segments[left_nidx] =
        Segment(segment.begin, segment.begin + left_count);
    ridx_segments[right_nidx] =
        Segment(segment.begin + left_count, segment.end);
  }

  /*! \brief Sort row indices according to position. */
  void SortPosition(const Segment& segment, int left_nidx, int right_nidx) {
    int min_bits = 0;
    int max_bits = static_cast<int>(
        std::ceil(std::log2((std::max)(left_nidx, right_nidx) + 1)));

    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        position.Current() + segment.begin, position.other() + segment.begin,
        ridx.Current() + segment.begin, ridx.other() + segment.begin,
        segment.Size(), min_bits, max_bits);

    temp_memory.LazyAllocate(temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(
        temp_memory.d_temp_storage, temp_memory.temp_storage_bytes,
        position.Current() + segment.begin, position.other() + segment.begin,
        ridx.Current() + segment.begin, ridx.other() + segment.begin,
        segment.Size(), min_bits, max_bits);
    // Copy back key
    dh::safe_cuda(cudaMemcpy(
        position.Current() + segment.begin, position.other() + segment.begin,
        segment.Size() * sizeof(int), cudaMemcpyDeviceToDevice));
    // Copy back value
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

    dh::safe_cuda(cudaMemcpy(node_sum_gradients_d.Data(),
                             node_sum_gradients.data(),
                             sizeof(GradientPair) * node_sum_gradients.size(),
                             cudaMemcpyHostToDevice));
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

struct SharedMemHistBuilder : public GPUHistBuilderBase {
  void Build(DeviceShard* shard, int nidx) override {
    auto segment = shard->ridx_segments[nidx];
    auto segment_begin = segment.begin;
    auto d_node_hist = shard->hist.GetHistPtr(nidx);
    auto d_gidx = shard->gidx;
    auto d_ridx = shard->ridx.Current();
    auto d_gpair = shard->gpair.Data();

    int null_gidx_value = shard->null_gidx_value;
    auto n_elements = segment.Size() * shard->row_stride;

    const size_t smem_size = sizeof(GradientPairSumT) * shard->null_gidx_value;
    const int items_per_thread = 8;
    const int block_threads = 256;
    const int grid_size =
        static_cast<int>(dh::DivRoundUp(n_elements,
                                        items_per_thread * block_threads));
    if (grid_size <= 0) {
      return;
    }
    dh::safe_cuda(cudaSetDevice(shard->device_idx));
    sharedMemHistKernel<<<grid_size, block_threads, smem_size>>>
        (shard->row_stride, d_ridx, d_gidx, null_gidx_value, d_node_hist, d_gpair,
         segment_begin, n_elements);
  }
};

struct GlobalMemHistBuilder : public GPUHistBuilderBase {
  void Build(DeviceShard* shard, int nidx) override {
    Segment segment = shard->ridx_segments[nidx];
    GradientPairSumT* d_node_hist = shard->hist.GetHistPtr(nidx);
    common::CompressedIterator<uint32_t> d_gidx = shard->gidx;
    bst_uint* d_ridx = shard->ridx.Current();
    GradientPair* d_gpair = shard->gpair.Data();

    size_t const n_elements = segment.Size() * shard->row_stride;
    size_t const row_stride = shard->row_stride;
    int const null_gidx_value = shard->null_gidx_value;

    dh::LaunchN(shard->device_idx, n_elements, [=] __device__(size_t idx) {
        int ridx = d_ridx[(idx / row_stride) + segment.begin];
        // lookup the index (bin) of histogram.
        int gidx = d_gidx[ridx * row_stride + idx % row_stride];

        if (gidx != null_gidx_value) {
          AtomicAddGpair(d_node_hist + gidx, d_gpair[ridx]);
        }
      });
  }
};

inline void DeviceShard::InitCompressedData(
    const common::HistCutMatrix& hmat, const SparsePage& row_batch) {
  n_bins = hmat.row_ptr.back();
  null_gidx_value = hmat.row_ptr.back();

  int max_nodes =
      param.max_leaves > 0 ? param.max_leaves * 2 : MaxNodesDepth(param.max_depth);

  ba.Allocate(device_idx, param.silent,
              &gpair, n_rows,
              &ridx, n_rows,
              &position, n_rows,
              &prediction_cache, n_rows,
              &node_sum_gradients_d, max_nodes,
              &cut_.feature_segments, hmat.row_ptr.size(),
              &cut_.gidx_fvalue_map, hmat.cut.size(),
              &cut_.min_fvalue, hmat.min_val.size(),
              &monotone_constraints, param.monotone_constraints.size());
  cut_.gidx_fvalue_map = hmat.cut;
  cut_.min_fvalue = hmat.min_val;
  cut_.feature_segments = hmat.row_ptr;
  monotone_constraints = param.monotone_constraints;

  node_sum_gradients.resize(max_nodes);
  ridx_segments.resize(max_nodes);

  dh::safe_cuda(cudaSetDevice(device_idx));

  // allocate compressed bin data
  int num_symbols = n_bins + 1;
  // Required buffer size for storing data matrix in ELLPack format.
  size_t compressed_size_bytes =
      common::CompressedBufferWriter::CalculateBufferSize(row_stride * n_rows,
                                                          num_symbols);

  CHECK(!(param.max_leaves == 0 && param.max_depth == 0))
      << "Max leaves and max depth cannot both be unconstrained for "
      "gpu_hist.";
  ba.Allocate(device_idx, param.silent, &gidx_buffer, compressed_size_bytes);
  gidx_buffer.Fill(0);

  int nbits = common::detail::SymbolBits(num_symbols);

  CreateHistIndices(row_batch);

  gidx = common::CompressedIterator<uint32_t>(gidx_buffer.Data(), num_symbols);

  // check if we can use shared memory for building histograms
  // (assuming atleast we need 2 CTAs per SM to maintain decent latency hiding)
  auto histogram_size = sizeof(GradientPairSumT) * null_gidx_value;
  auto max_smem = dh::MaxSharedMemory(device_idx);
  if (histogram_size <= max_smem) {
    hist_builder.reset(new SharedMemHistBuilder);
  } else {
    hist_builder.reset(new GlobalMemHistBuilder);
  }

  // Init histogram
  hist.Init(device_idx, hmat.row_ptr.back());

  dh::safe_cuda(cudaMallocHost(&tmp_pinned, sizeof(int64_t)));
}

inline void DeviceShard::CreateHistIndices(const SparsePage& row_batch) {
  int num_symbols = n_bins + 1;
  // bin and compress entries in batches of rows
  size_t gpu_batch_nrows = std::min
                           (dh::TotalMemory(device_idx) / (16 * row_stride * sizeof(Entry)),
                            static_cast<size_t>(n_rows));
  const std::vector<Entry>& data_vec = row_batch.data.HostVector();

  thrust::device_vector<Entry> entries_d(gpu_batch_nrows * row_stride);
  size_t gpu_nbatches = dh::DivRoundUp(n_rows, gpu_batch_nrows);

  for (size_t gpu_batch = 0; gpu_batch < gpu_nbatches; ++gpu_batch) {
    size_t batch_row_begin = gpu_batch * gpu_batch_nrows;
    size_t batch_row_end = (gpu_batch + 1) * gpu_batch_nrows;
    if (batch_row_end > n_rows) {
      batch_row_end = n_rows;
    }
    size_t batch_nrows = batch_row_end - batch_row_begin;
    // number of entries in this batch.
    size_t n_entries = row_ptrs[batch_row_end] - row_ptrs[batch_row_begin];
    // copy data entries to device.
    dh::safe_cuda
        (cudaMemcpy
         (entries_d.data().get(), data_vec.data() + row_ptrs[batch_row_begin],
          n_entries * sizeof(Entry), cudaMemcpyDefault));
    const dim3 block3(32, 8, 1);  // 256 threads
    const dim3 grid3(dh::DivRoundUp(n_rows, block3.x),
                     dh::DivRoundUp(row_stride, block3.y), 1);
    compress_bin_ellpack_k<<<grid3, block3>>>
        (common::CompressedBufferWriter(num_symbols),
         gidx_buffer.Data(),
         row_ptrs.data().get() + batch_row_begin,
         entries_d.data().get(),
         cut_.gidx_fvalue_map.Data(), cut_.feature_segments.Data(),
         batch_row_begin, batch_nrows,
         row_ptrs[batch_row_begin],
         row_stride, null_gidx_value);

    dh::safe_cuda(cudaGetLastError());
    dh::safe_cuda(cudaDeviceSynchronize());
  }

  // free the memory that is no longer needed
  row_ptrs.resize(0);
  row_ptrs.shrink_to_fit();
  entries_d.resize(0);
  entries_d.shrink_to_fit();
}

class GPUHistMaker : public TreeUpdater {
 public:
  struct ExpandEntry;

  GPUHistMaker() : initialised_(false), p_last_fmat_(nullptr) {}
  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) override {
    param_.InitAllowUnknown(args);
    CHECK(param_.n_gpus != 0) << "Must have at least one device";
    n_devices_ = param_.n_gpus;
    dist_ = GPUDistribution::Block(GPUSet::All(param_.n_gpus)
                                   .Normalised(param_.gpu_id));

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
    monitor_.Start("Update", dist_.Devices());
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
      dh::safe_cuda(cudaGetLastError());
    } catch (const std::exception& e) {
      LOG(FATAL) << "Exception in gpu_hist: " << e.what() << std::endl;
    }
    param_.learning_rate = lr;
    monitor_.Stop("Update", dist_.Devices());
  }

  void InitDataOnce(DMatrix* dmat) {
    info_ = &dmat->Info();

    int n_devices = GPUSet::All(param_.n_gpus, info_->num_row_).Size();

    device_list_.resize(n_devices);
    for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
      int device_idx = GPUSet::GetDeviceIdx(param_.gpu_id + d_idx);
      device_list_[d_idx] = device_idx;
    }

    reducer_.Init(device_list_);

    auto batch_iter = dmat->GetRowBatches().begin();
    const SparsePage& batch = *batch_iter;
    // Create device shards
    shards_.resize(n_devices);
    dh::ExecuteIndexShards(&shards_, [&](int i, std::unique_ptr<DeviceShard>& shard) {
        size_t start = dist_.ShardStart(info_->num_row_, i);
        size_t size = dist_.ShardSize(info_->num_row_, i);
        shard = std::unique_ptr<DeviceShard>
          (new DeviceShard(device_list_.at(i), i,
                           start, start + size, param_));
        shard->InitRowPtrs(batch);
      });

    // Find the cuts.
    monitor_.Start("Quantiles", dist_.Devices());
    common::DeviceSketch(batch, *info_, param_, &hmat_);
    n_bins_ = hmat_.row_ptr.back();
    monitor_.Stop("Quantiles", dist_.Devices());

    monitor_.Start("BinningCompression", dist_.Devices());
    dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {
        shard->InitCompressedData(hmat_, batch);
      });
    monitor_.Stop("BinningCompression", dist_.Devices());
    ++batch_iter;
    CHECK(batch_iter.AtEnd()) << "External memory not supported";

    p_last_fmat_ = dmat;
    initialised_ = true;
  }

  void InitData(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat) {
    monitor_.Start("InitDataOnce", dist_.Devices());
    if (!initialised_) {
      this->InitDataOnce(dmat);
    }
    monitor_.Stop("InitDataOnce", dist_.Devices());

    column_sampler_.Init(info_->num_col_, param_.colsample_bylevel, param_.colsample_bytree);

    // Copy gpair & reset memory
    monitor_.Start("InitDataReset", dist_.Devices());

    gpair->Reshard(dist_);
    dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {
        shard->Reset(gpair);
      });
    monitor_.Stop("InitDataReset", dist_.Devices());
  }

  void AllReduceHist(int nidx) {
    reducer_.GroupStart();
    for (auto& shard : shards_) {
      auto d_node_hist = shard->hist.GetHistPtr(nidx);
      reducer_.AllReduceSum(
          shard->normalised_device_idx,
          reinterpret_cast<GradientPairSumT::ValueT*>(d_node_hist),
          reinterpret_cast<GradientPairSumT::ValueT*>(d_node_hist),
          n_bins_ * (sizeof(GradientPairSumT) / sizeof(GradientPairSumT::ValueT)));
    }
    reducer_.GroupEnd();

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

    // Build histogram for node with the smallest number of training examples
    dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {
        shard->BuildHist(build_hist_nidx);
      });

    this->AllReduceHist(build_hist_nidx);

    // Check whether we can use the subtraction trick to calculate the other
    bool do_subtraction_trick = true;
    for (auto& shard : shards_) {
      do_subtraction_trick &= shard->CanDoSubtractionTrick(
          nidx_parent, build_hist_nidx, subtraction_trick_nidx);
    }

    if (do_subtraction_trick) {
      // Calculate other histogram using subtraction trick
      dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {
        shard->SubtractionTrick(nidx_parent, build_hist_nidx,
                                subtraction_trick_nidx);
      });
    } else {
      // Calculate other histogram manually
      dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {
        shard->BuildHist(subtraction_trick_nidx);
      });

      this->AllReduceHist(subtraction_trick_nidx);
    }
  }

  // Returns best loss
  std::vector<DeviceSplitCandidate> EvaluateSplits(
      const std::vector<int>& nidx_set, RegTree* p_tree) {
    size_t const columns = info_->num_col_;
    std::vector<DeviceSplitCandidate> best_splits(nidx_set.size());
    // Every feature is a candidate
    size_t const candidates_size_bytes =
        nidx_set.size() * columns * sizeof(DeviceSplitCandidate);
    // Storage for all candidates from all nodes.
    std::vector<DeviceSplitCandidate> candidate_splits(nidx_set.size() * columns);
    // FIXME: Multi-gpu support?
    // Use first device
    auto& shard = shards_.front();
    dh::safe_cuda(cudaSetDevice(shard->device_idx));
    shard->temp_memory.LazyAllocate(candidates_size_bytes);
    auto d_split = shard->temp_memory.Pointer<DeviceSplitCandidate>();

    auto& streams = shard->GetStreams(static_cast<int>(nidx_set.size()));

    // Use streams to process nodes concurrently
    for (auto i = 0; i < nidx_set.size(); i++) {
      auto nidx = nidx_set[i];
      DeviceNodeStats node(shard->node_sum_gradients[nidx], nidx, param_);
      int depth = p_tree->GetDepth(nidx);

      HostDeviceVector<int>& feature_set = column_sampler_.GetFeatureSet(depth);
      feature_set.Reshard(GPUSet::Range(shard->device_idx, 1));
      auto& h_feature_set = feature_set.HostVector();
      // One block for each feature
      int constexpr BLOCK_THREADS = 256;
      EvaluateSplitKernel<BLOCK_THREADS>
          <<<uint32_t(feature_set.Size()), BLOCK_THREADS, 0, streams[i]>>>(
              shard->hist.GetHistPtr(nidx),
              info_->num_col_,
              feature_set.DevicePointer(shard->device_idx),
              node,
              shard->cut_.feature_segments.Data(),
              shard->cut_.min_fvalue.Data(),
              shard->cut_.gidx_fvalue_map.Data(),
              GPUTrainingParam(param_),
              d_split + i * columns,  // split candidate for i^th node.
              node_value_constraints_[nidx],
              shard->monotone_constraints.Data());
    }

    dh::safe_cuda(cudaDeviceSynchronize());
    dh::safe_cuda(
        cudaMemcpy(candidate_splits.data(), shard->temp_memory.d_temp_storage,
                   candidates_size_bytes, cudaMemcpyDeviceToHost));
    for (auto i = 0; i < nidx_set.size(); i++) {
      auto depth = p_tree->GetDepth(nidx_set[i]);
      DeviceSplitCandidate nidx_best;
      for (auto fidx : column_sampler_.GetFeatureSet(depth).HostVector()) {
        DeviceSplitCandidate& candidate =
            candidate_splits[i * columns + fidx];
        nidx_best.Update(candidate, param_);
      }
      best_splits[i] = nidx_best;
    }
    return std::move(best_splits);
  }

  void InitRoot(RegTree* p_tree) {
    constexpr int root_nidx = 0;
    // Sum gradients
    std::vector<GradientPair> tmp_sums(shards_.size());

    dh::ExecuteIndexShards(&shards_, [&](int i, std::unique_ptr<DeviceShard>& shard) {
        dh::safe_cuda(cudaSetDevice(shard->device_idx));
      tmp_sums[i] =
        dh::SumReduction(shard->temp_memory, shard->gpair.Data(),
                         shard->gpair.Size());
      });
    GradientPair sum_gradient =
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
    int nidx = candidate.nid;
    int left_nidx = (*p_tree)[nidx].LeftChild();
    int right_nidx = (*p_tree)[nidx].RightChild();

    // convert floating-point split_pt into corresponding bin_id
    // split_cond = -1 indicates that split_pt is less than all known cut points
    int64_t split_gidx = -1;
    int64_t fidx = candidate.split.findex;
    bool default_dir_left = candidate.split.dir == kLeftDir;
    uint32_t fidx_begin = hmat_.row_ptr[fidx];
    uint32_t fidx_end = hmat_.row_ptr[fidx + 1];
    // split_gidx = i where i is the i^th bin containing split value.
    for (auto i = fidx_begin; i < fidx_end; ++i) {
      if (candidate.split.fvalue == hmat_.cut[i]) {
        split_gidx = static_cast<int64_t>(i);
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

    monitor_.Start("InitData", dist_.Devices());
    this->InitData(gpair, p_fmat);
    monitor_.Stop("InitData", dist_.Devices());
    monitor_.Start("InitRoot", dist_.Devices());
    this->InitRoot(p_tree);
    monitor_.Stop("InitRoot", dist_.Devices());

    auto timestamp = qexpand_->size();
    auto num_leaves = 1;

    while (!qexpand_->empty()) {
      ExpandEntry candidate = qexpand_->top();
      qexpand_->pop();
      if (!candidate.IsValid(param_, num_leaves)) continue;

      monitor_.Start("ApplySplit", dist_.Devices());
      this->ApplySplit(candidate, p_tree);
      monitor_.Stop("ApplySplit", dist_.Devices());
      num_leaves++;

      int left_child_nidx = tree[candidate.nid].LeftChild();
      int right_child_nidx = tree[candidate.nid].RightChild();

      // Only create child entries if needed
      if (ExpandEntry::ChildIsValid(param_, tree.GetDepth(left_child_nidx),
                                    num_leaves)) {
        monitor_.Start("BuildHist", dist_.Devices());
        this->BuildHistLeftRight(candidate.nid, left_child_nidx,
                                 right_child_nidx);
        monitor_.Stop("BuildHist", dist_.Devices());

        monitor_.Start("EvaluateSplits", dist_.Devices());
        auto splits =
            this->EvaluateSplits({left_child_nidx, right_child_nidx}, p_tree);
        qexpand_->push(ExpandEntry(left_child_nidx,
                                   tree.GetDepth(left_child_nidx), splits[0],
                                   timestamp++));
        qexpand_->push(ExpandEntry(right_child_nidx,
                                   tree.GetDepth(right_child_nidx), splits[1],
                                   timestamp++));
        monitor_.Stop("EvaluateSplits", dist_.Devices());
      }
    }
  }

  bool UpdatePredictionCache(
      const DMatrix* data, HostDeviceVector<bst_float>* p_out_preds) override {
    monitor_.Start("UpdatePredictionCache", dist_.Devices());
    if (shards_.empty() || p_last_fmat_ == nullptr || p_last_fmat_ != data)
      return false;
    p_out_preds->Reshard(dist_.Devices());
    dh::ExecuteShards(&shards_, [&](std::unique_ptr<DeviceShard>& shard) {
        shard->UpdatePredictionCache(p_out_preds->DevicePointer(shard->device_idx));
      });
    monitor_.Stop("UpdatePredictionCache", dist_.Devices());
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
      if (param.max_depth > 0 && depth >= param.max_depth) return false;
      if (param.max_leaves > 0 && num_leaves >= param.max_leaves) return false;
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
  common::ColumnSampler column_sampler_;
  using ExpandQueue = std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
    std::function<bool(ExpandEntry, ExpandEntry)>>;
  std::unique_ptr<ExpandQueue> qexpand_;
  common::Monitor monitor_;
  dh::AllReducer reducer_;
  std::vector<ValueConstraint> node_value_constraints_;
  std::vector<int> device_list_;

  DMatrix* p_last_fmat_;
  GPUDistribution dist_;
};

XGBOOST_REGISTER_TREE_UPDATER(GPUHistMaker, "grow_gpu_hist")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUHistMaker(); });
}  // namespace tree
}  // namespace xgboost
