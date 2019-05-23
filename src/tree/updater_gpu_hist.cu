/*!
 * Copyright 2017-2019 XGBoost contributors
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
#include "split_evaluator.h"

namespace xgboost {
namespace tree {

#if !defined(GTEST_TEST)
DMLC_REGISTRY_FILE_TAG(updater_gpu_hist);
#endif  // !defined(GTEST_TEST)

// training parameters specific to this algorithm
struct GPUHistMakerTrainParam
    : public dmlc::Parameter<GPUHistMakerTrainParam> {
  bool single_precision_histogram;
  // number of rows in a single GPU batch
  int gpu_batch_nrows;
  bool debug_synchronize;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GPUHistMakerTrainParam) {
    DMLC_DECLARE_FIELD(single_precision_histogram).set_default(false).describe(
        "Use single precision to build histograms.");
    DMLC_DECLARE_FIELD(gpu_batch_nrows)
        .set_lower_bound(-1)
        .set_default(0)
        .describe("Number of rows in a GPU batch, used for finding quantiles on GPU; "
                  "-1 to use all rows assignted to a GPU, and 0 to auto-deduce");
    DMLC_DECLARE_FIELD(debug_synchronize).set_default(false).describe(
        "Check if all distributed tree are identical after tree construction.");
  }
};
#if !defined(GTEST_TEST)
DMLC_REGISTER_PARAMETER(GPUHistMakerTrainParam);
#endif  // !defined(GTEST_TEST)

struct ExpandEntry {
  int nid;
  int depth;
  DeviceSplitCandidate split;
  uint64_t timestamp;
  ExpandEntry() = default;
  ExpandEntry(int nid, int depth, DeviceSplitCandidate split,
              uint64_t timestamp)
      : nid(nid), depth(depth), split(std::move(split)), timestamp(timestamp) {}
  bool IsValid(const TrainParam& param, int num_leaves) const {
    if (split.loss_chg <= kRtEps) return false;
    if (split.left_sum.GetHess() == 0 || split.right_sum.GetHess() == 0) {
      return false;
    }
    if (param.max_depth > 0 && depth == param.max_depth) return false;
    if (param.max_leaves > 0 && num_leaves == param.max_leaves) return false;
    return true;
  }

  static bool ChildIsValid(const TrainParam& param, int depth, int num_leaves) {
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

// Find a gidx value for a given feature otherwise return -1 if not found
__forceinline__ __device__ int BinarySearchRow(
    bst_uint begin, bst_uint end,
    common::CompressedIterator<uint32_t> data,
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

/** \brief Struct for accessing and manipulating an ellpack matrix on the
 * device. Does not own underlying memory and may be trivially copied into
 * kernels.*/
struct ELLPackMatrix {
  common::Span<uint32_t> feature_segments;
  /*! \brief minimum value for each feature. */
  common::Span<bst_float> min_fvalue;
  /*! \brief Cut. */
  common::Span<bst_float> gidx_fvalue_map;
  /*! \brief row length for ELLPack. */
  size_t row_stride{0};
  common::CompressedIterator<uint32_t> gidx_iter;
  bool is_dense;
  int null_gidx_value;

  XGBOOST_DEVICE size_t BinCount() const { return gidx_fvalue_map.size(); }

  // Get a matrix element, uses binary search for look up
  // Return NaN if missing
  __device__ bst_float GetElement(size_t ridx, size_t fidx) const {
    auto row_begin = row_stride * ridx;
    auto row_end = row_begin + row_stride;
    auto gidx = -1;
    if (is_dense) {
      gidx = gidx_iter[row_begin + fidx];
    } else {
      gidx =
          BinarySearchRow(row_begin, row_end, gidx_iter, feature_segments[fidx],
                          feature_segments[fidx + 1]);
    }
    if (gidx == -1) {
      return nan("");
    }
    return gidx_fvalue_map[gidx];
  }
  void Init(common::Span<uint32_t> feature_segments,
    common::Span<bst_float> min_fvalue,
    common::Span<bst_float> gidx_fvalue_map, size_t row_stride,
    common::CompressedIterator<uint32_t> gidx_iter, bool is_dense,
    int null_gidx_value) {
    this->feature_segments = feature_segments;
    this->min_fvalue = min_fvalue;
    this->gidx_fvalue_map = gidx_fvalue_map;
    this->row_stride = row_stride;
    this->gidx_iter = gidx_iter;
    this->is_dense = is_dense;
    this->null_gidx_value = null_gidx_value;
  }
};

// With constraints
template <typename GradientPairT>
XGBOOST_DEVICE float inline LossChangeMissing(
    const GradientPairT& scan, const GradientPairT& missing, const GradientPairT& parent_sum,
    const float& parent_gain, const GPUTrainingParam& param, int constraint,
    const ValueConstraint& value_constraint,
    bool& missing_left_out) {  // NOLINT
  float missing_left_gain = value_constraint.CalcSplitGain(
      param, constraint, GradStats(scan + missing),
      GradStats(parent_sum - (scan + missing)));
  float missing_right_gain = value_constraint.CalcSplitGain(
      param, constraint, GradStats(scan), GradStats(parent_sum - scan));

  if (missing_left_gain >= missing_right_gain) {
    missing_left_out = true;
    return missing_left_gain - parent_gain;
  } else {
    missing_left_out = false;
    return missing_right_gain - parent_gain;
  }
}

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
template <int BLOCK_THREADS, typename ReduceT, typename TempStorageT, typename GradientSumT>
__device__ GradientSumT ReduceFeature(common::Span<const GradientSumT> feature_histogram,
                                      TempStorageT* temp_storage) {
  __shared__ cub::Uninitialized<GradientSumT> uninitialized_sum;
  GradientSumT& shared_sum = uninitialized_sum.Alias();

  GradientSumT local_sum = GradientSumT();
  // For loop sums features into one block size
  auto begin = feature_histogram.data();
  auto end = begin + feature_histogram.size();
  for (auto itr = begin; itr < end; itr += BLOCK_THREADS) {
    bool thread_active = itr + threadIdx.x < end;
    // Scan histogram
    GradientSumT bin = thread_active ? *(itr + threadIdx.x) : GradientSumT();
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
template <int BLOCK_THREADS, typename ReduceT, typename ScanT,
          typename MaxReduceT, typename TempStorageT, typename GradientSumT>
__device__ void EvaluateFeature(
    int fidx, common::Span<const GradientSumT> node_histogram,
    const ELLPackMatrix& matrix,
    DeviceSplitCandidate* best_split,  // shared memory storing best split
    const DeviceNodeStats& node, const GPUTrainingParam& param,
    TempStorageT* temp_storage,  // temp memory for cub operations
    int monotonic_constraint,
    const ValueConstraint& value_constraint) {
  // Use pointer from cut to indicate begin and end of bins for each feature.
  uint32_t gidx_begin = matrix.feature_segments[fidx];  // begining bin
  uint32_t gidx_end =
      matrix.feature_segments[fidx + 1];  // end bin for i^th feature

  // Sum histogram bins for current feature
  GradientSumT const feature_sum = ReduceFeature<BLOCK_THREADS, ReduceT>(
      node_histogram.subspan(gidx_begin, gidx_end - gidx_begin), temp_storage);

  GradientSumT const parent_sum = GradientSumT(node.sum_gradients);
  GradientSumT const missing = parent_sum - feature_sum;
  float const null_gain = -std::numeric_limits<bst_float>::infinity();

  SumCallbackOp<GradientSumT> prefix_op =
      SumCallbackOp<GradientSumT>();
  for (int scan_begin = gidx_begin; scan_begin < gidx_end;
       scan_begin += BLOCK_THREADS) {
    bool thread_active = (scan_begin + threadIdx.x) < gidx_end;

    // Gradient value for current bin.
    GradientSumT bin =
        thread_active ? node_histogram[scan_begin + threadIdx.x] : GradientSumT();
    ScanT(temp_storage->scan).ExclusiveScan(bin, bin, cub::Sum(), prefix_op);

    // Whether the gradient of missing values is put to the left side.
    bool missing_left = true;
    float gain = null_gain;
    if (thread_active) {
      gain = LossChangeMissing(bin, missing, parent_sum, node.root_gain, param,
                               monotonic_constraint, value_constraint, missing_left);
    }

    __syncthreads();

    // Find thread with best gain
    cub::KeyValuePair<int, float> tuple(threadIdx.x, gain);
    cub::KeyValuePair<int, float> best =
        MaxReduceT(temp_storage->max_reduce).Reduce(tuple, cub::ArgMax());

    __shared__ cub::KeyValuePair<int, float> block_max;
    if (threadIdx.x == 0) {
      block_max = best;
    }

    __syncthreads();

    // Best thread updates split
    if (threadIdx.x == block_max.key) {
      int split_gidx = (scan_begin + threadIdx.x) - 1;
      float fvalue;
      if (split_gidx < static_cast<int>(gidx_begin)) {
        fvalue =  matrix.min_fvalue[fidx];
      } else {
        fvalue = matrix.gidx_fvalue_map[split_gidx];
      }
      GradientSumT left = missing_left ? bin + missing : bin;
      GradientSumT right = parent_sum - left;
      best_split->Update(gain, missing_left ? kLeftDir : kRightDir, fvalue,
                         fidx, GradientPair(left), GradientPair(right), param);
    }
    __syncthreads();
  }
}

template <int BLOCK_THREADS, typename GradientSumT>
__global__ void EvaluateSplitKernel(
    common::Span<const GradientSumT> node_histogram,  // histogram for gradients
    common::Span<const int> feature_set,  // Selected features
    DeviceNodeStats node,
  ELLPackMatrix matrix,
    GPUTrainingParam gpu_param,
    common::Span<DeviceSplitCandidate> split_candidates,  // resulting split
    ValueConstraint value_constraint,
    common::Span<int> d_monotonic_constraints) {
  // KeyValuePair here used as threadIdx.x -> gain_value
  using ArgMaxT = cub::KeyValuePair<int, float>;
  using BlockScanT =
      cub::BlockScan<GradientSumT, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>;
  using MaxReduceT = cub::BlockReduce<ArgMaxT, BLOCK_THREADS>;

  using SumReduceT = cub::BlockReduce<GradientSumT, BLOCK_THREADS>;

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
      fidx, node_histogram, matrix, &best_split, node, gpu_param, &temp_storage,
      constraint, value_constraint);

  __syncthreads();

  if (threadIdx.x == 0) {
    // Record best loss for each feature
    split_candidates[blockIdx.x] = best_split;
  }
}

/**
 * \struct  DeviceHistogram
 *
 * \summary Data storage for node histograms on device. Automatically expands.
 *
 * \tparam GradientSumT      histogram entry type.
 * \tparam kStopGrowingSize  Do not grow beyond this size
 *
 * \author  Rory
 * \date    28/07/2018
 */
template <typename GradientSumT, size_t kStopGrowingSize = 1 << 26>
class DeviceHistogram {
 private:
  /*! \brief Map nidx to starting index of its histogram. */
  std::map<int, size_t> nidx_map_;
  thrust::device_vector<typename GradientSumT::ValueT> data_;
  int n_bins_;
  int device_id_;
  static constexpr size_t kNumItemsInGradientSum =
      sizeof(GradientSumT) / sizeof(typename GradientSumT::ValueT);
  static_assert(kNumItemsInGradientSum == 2,
                "Number of items in gradient type should be 2.");

 public:
  void Init(int device_id, int n_bins) {
    this->n_bins_ = n_bins;
    this->device_id_ = device_id;
  }

  void Reset() {
    dh::safe_cuda(cudaMemsetAsync(
        data_.data().get(), 0,
        data_.size() * sizeof(typename decltype(data_)::value_type)));
    nidx_map_.clear();
  }
  bool HistogramExists(int nidx) const {
    return nidx_map_.find(nidx) != nidx_map_.cend();
  }
  size_t HistogramSize() const {
    return n_bins_ * kNumItemsInGradientSum;
  }

  thrust::device_vector<typename GradientSumT::ValueT>& Data() {
    return data_;
  }

  void AllocateHistogram(int nidx) {
    if (HistogramExists(nidx)) return;
    // Number of items currently used in data
    const size_t used_size = nidx_map_.size() * HistogramSize();
    const size_t new_used_size = used_size + HistogramSize();
    dh::safe_cuda(cudaSetDevice(device_id_));
    if (data_.size() >= kStopGrowingSize) {
      // Recycle histogram memory
      if (new_used_size <= data_.size()) {
        // no need to remove old node, just insert the new one.
        nidx_map_[nidx] = used_size;
        // memset histogram size in bytes
        dh::safe_cuda(cudaMemsetAsync(data_.data().get() + used_size, 0,
                                      n_bins_ * sizeof(GradientSumT)));
      } else {
        std::pair<int, size_t> old_entry = *nidx_map_.begin();
        nidx_map_.erase(old_entry.first);
        dh::safe_cuda(cudaMemsetAsync(data_.data().get() + old_entry.second, 0,
                                      n_bins_ * sizeof(GradientSumT)));
        nidx_map_[nidx] = old_entry.second;
      }
    } else {
      // Append new node histogram
      nidx_map_[nidx] = used_size;
      size_t new_required_memory = std::max(data_.size() * 2, HistogramSize());
      if (data_.size() < new_required_memory) {
        data_.resize(new_required_memory);
      }
    }
  }

  /**
   * \summary   Return pointer to histogram memory for a given node.
   * \param nidx    Tree node index.
   * \return    hist pointer.
   */
  common::Span<GradientSumT> GetNodeHistogram(int nidx) {
    CHECK(this->HistogramExists(nidx));
    auto ptr = data_.data().get() + nidx_map_[nidx];
    return common::Span<GradientSumT>(
        reinterpret_cast<GradientSumT*>(ptr), n_bins_);
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
template<typename std::enable_if<true,  int>::type = 0>
__global__ void CompressBinEllpackKernel(
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
  if (irow >= n_rows || ifeature >= row_stride) {
    return;
  }
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
    if (bin >= ncuts) {
      bin = ncuts - 1;
    }
    // Add the number of bins in previous features.
    bin += cut_rows[feature];
  }
  // Write to gidx buffer.
  wr.AtomicWriteSymbol(buffer, bin, (irow + base_row) * row_stride + ifeature);
}

template <typename GradientSumT>
__global__ void SharedMemHistKernel(ELLPackMatrix matrix, const bst_uint* d_ridx,
                                    GradientSumT* d_node_hist,
                                    const GradientPair* d_gpair,
                                    size_t segment_begin, size_t n_elements) {
  extern __shared__ char smem[];
  GradientSumT* smem_arr = reinterpret_cast<GradientSumT*>(smem); // NOLINT
  for (auto i :
       dh::BlockStrideRange(static_cast<size_t>(0), matrix.BinCount())) {
    smem_arr[i] = GradientSumT();
  }
  __syncthreads();
  for (auto idx : dh::GridStrideRange(static_cast<size_t>(0), n_elements)) {
    int ridx = d_ridx[idx / matrix.row_stride + segment_begin];
    int gidx = matrix.gidx_iter[ridx * matrix.row_stride + idx % matrix.row_stride];
    if (gidx != matrix.null_gidx_value) {
      AtomicAddGpair(smem_arr + gidx, d_gpair[ridx]);
    }
  }
  __syncthreads();
  for (auto i :
       dh::BlockStrideRange(static_cast<size_t>(0), matrix.BinCount())) {
    AtomicAddGpair(d_node_hist + i, smem_arr[i]);
  }
}

struct Segment {
  size_t begin;
  size_t end;

  Segment() : begin{0}, end{0} {}

  Segment(size_t begin, size_t end) : begin(begin), end(end) {
    CHECK_GE(end, begin);
  }
  size_t Size() const { return end - begin; }
};

/** \brief Returns a one if the left node index is encountered, otherwise return
 * zero. */
struct IndicateLeftTransform {
  int left_nidx;
  explicit IndicateLeftTransform(int left_nidx) : left_nidx(left_nidx) {}
  __host__ __device__ __forceinline__ int operator()(const int& x) const {
    return x == left_nidx ? 1 : 0;
  }
};

/**
 * \brief Optimised routine for sorting key value pairs into left and right
 * segments. Based on a single pass of exclusive scan, uses iterators to
 * redirect inputs and outputs.
 */
inline void SortPosition(dh::CubMemory* temp_memory, common::Span<int> position,
                         common::Span<int> position_out, common::Span<bst_uint> ridx,
                         common::Span<bst_uint> ridx_out, int left_nidx,
                         int right_nidx, int64_t* d_left_count,
                         cudaStream_t stream = nullptr) {
  auto d_position_out = position_out.data();
  auto d_position_in = position.data();
  auto d_ridx_out = ridx_out.data();
  auto d_ridx_in = ridx.data();
  auto write_results = [=] __device__(size_t idx, int ex_scan_result) {
    int scatter_address;
    if (d_position_in[idx] == left_nidx) {
      scatter_address = ex_scan_result;
    } else {
      scatter_address = (idx - ex_scan_result) + *d_left_count;
    }
    d_position_out[scatter_address] = d_position_in[idx];
    d_ridx_out[scatter_address] = d_ridx_in[idx];
  };  // NOLINT

  IndicateLeftTransform conversion_op(left_nidx);
  cub::TransformInputIterator<int, IndicateLeftTransform, int*> in_itr(
      d_position_in, conversion_op);
  dh::DiscardLambdaItr<decltype(write_results)> out_itr(write_results);
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, in_itr, out_itr,
                                position.size(), stream);
  temp_memory->LazyAllocate(temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(temp_memory->d_temp_storage,
                                temp_memory->temp_storage_bytes, in_itr,
                                out_itr, position.size(), stream);
}

/*! \brief Count how many rows are assigned to left node. */
__forceinline__ __device__ void CountLeft(int64_t* d_count, int val,
                                          int left_nidx) {
#if __CUDACC_VER_MAJOR__ > 8
  int mask = __activemask();
  unsigned ballot = __ballot_sync(mask, val == left_nidx);
  int leader = __ffs(mask) - 1;
  if (threadIdx.x % 32 == leader) {
    atomicAdd(reinterpret_cast<unsigned long long*>(d_count),    // NOLINT
              static_cast<unsigned long long>(__popc(ballot)));  // NOLINT
  }
#else
  unsigned ballot = __ballot(val == left_nidx);
  if (threadIdx.x % 32 == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(d_count),    // NOLINT
              static_cast<unsigned long long>(__popc(ballot)));  // NOLINT
  }
#endif
}

template <typename GradientSumT>
struct DeviceShard;

template <typename GradientSumT>
struct GPUHistBuilderBase {
 public:
  virtual void Build(DeviceShard<GradientSumT>* shard, int idx) = 0;
  virtual ~GPUHistBuilderBase() = default;
};

// Manage memory for a single GPU
template <typename GradientSumT>
struct DeviceShard {
  int n_bins;
  int device_id;
  int shard_idx;  // Position in the local array of shards

  dh::BulkAllocator ba;

  ELLPackMatrix ellpack_matrix;

  /*! \brief Range of rows for each node. */
  std::vector<Segment> ridx_segments;
  DeviceHistogram<GradientSumT> hist;

  /*! \brief row_ptr form HistCutMatrix. */
  common::Span<uint32_t> feature_segments;
  /*! \brief minimum value for each feature. */
  common::Span<bst_float> min_fvalue;
  /*! \brief Cut. */
  common::Span<bst_float> gidx_fvalue_map;
  /*! \brief global index of histogram, which is stored in ELLPack format. */
  common::Span<common::CompressedByteT> gidx_buffer;

  /*! \brief  Row indices relative to this shard, necessary for sorting rows. */
  dh::DoubleBuffer<bst_uint> ridx;
  dh::DoubleBuffer<int> position;
  /*! \brief Gradient pair for each row. */
  common::Span<GradientPair> gpair;

  common::Span<int> monotone_constraints;
  common::Span<bst_float> prediction_cache;

  /*! \brief Sum gradient for each node. */
  std::vector<GradientPair> node_sum_gradients;
  common::Span<GradientPair> node_sum_gradients_d;
  /*! \brief row offset in SparsePage (the input data). */
  thrust::device_vector<size_t> row_ptrs;
  /*! \brief On-device feature set, only actually used on one of the devices */
  thrust::device_vector<int> feature_set_d;
  thrust::device_vector<int64_t>
      left_counts;  // Useful to keep a bunch of zeroed memory for sort position
  /*! The row offset for this shard. */
  bst_uint row_begin_idx;
  bst_uint row_end_idx;
  bst_uint n_rows;

  TrainParam param;
  bool prediction_cache_initialised;

  dh::CubMemory temp_memory;
  dh::PinnedMemory pinned_memory;

  std::vector<cudaStream_t> streams;

  common::Monitor monitor;
  std::vector<ValueConstraint> node_value_constraints;
  common::ColumnSampler column_sampler;

  std::unique_ptr<GPUHistBuilderBase<GradientSumT>> hist_builder;

  using ExpandQueue =
      std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
                          std::function<bool(ExpandEntry, ExpandEntry)>>;
  std::unique_ptr<ExpandQueue> qexpand;

  bool has_interaction_constraint;
  /* \brief Used for interaction constraints. */
  std::unique_ptr<SplitEvaluator> split_evaluator;

  // TODO(canonizer): do add support multi-batch DMatrix here
  DeviceShard(int _device_id, int shard_idx,
              bst_uint row_begin, bst_uint row_end,
              TrainParam _param,
              bool has_interaction_constraint_,
              std::unique_ptr<SplitEvaluator> spliteval,
              uint32_t column_sampler_seed)
      : device_id(_device_id),
        shard_idx(shard_idx),
        row_begin_idx(row_begin),
        row_end_idx(row_end),
        n_rows(row_end - row_begin),
        n_bins(0),
        param(std::move(_param)),
        prediction_cache_initialised(false),
        has_interaction_constraint{has_interaction_constraint_},
        split_evaluator{std::move(spliteval)},
        column_sampler(column_sampler_seed) {
    monitor.Init(std::string("DeviceShard") + std::to_string(device_id));
  }

  /* Init row_ptrs and row_stride */
  size_t InitRowPtrs(const SparsePage& row_batch) {
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
    size_t row_stride = thrust::reduce(row_size_iter, row_size_iter + n_rows, 0,
                                       thrust::maximum<size_t>());
    return row_stride;
  }

  void InitCompressedData(
      const common::HistCutMatrix& hmat, const SparsePage& row_batch, bool is_dense);

  void CreateHistIndices(const SparsePage& row_batch, size_t row_stride, int null_gidx_value);

  ~DeviceShard() {
    dh::safe_cuda(cudaSetDevice(device_id));
    for (auto& stream : streams) {
      dh::safe_cuda(cudaStreamDestroy(stream));
    }
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
  // Note that the column sampler must be passed by value because it is not
  // thread safe
  void Reset(HostDeviceVector<GradientPair>* dh_gpair, int64_t num_columns) {
    if (param.grow_policy == TrainParam::kLossGuide) {
      qexpand.reset(new ExpandQueue(LossGuide));
    } else {
      qexpand.reset(new ExpandQueue(DepthWise));
    }
    this->column_sampler.Init(num_columns, param.colsample_bynode,
      param.colsample_bylevel, param.colsample_bytree);
    dh::safe_cuda(cudaSetDevice(device_id));
    thrust::fill(
        thrust::device_pointer_cast(position.Current()),
        thrust::device_pointer_cast(position.Current() + position.Size()), 0);
    std::fill(node_sum_gradients.begin(), node_sum_gradients.end(),
              GradientPair());
    if (left_counts.size() < 256) {
      left_counts.resize(256);
    } else {
      dh::safe_cuda(cudaMemsetAsync(left_counts.data().get(), 0,
                                    sizeof(int64_t) * left_counts.size()));
    }
    thrust::sequence(
        thrust::device_pointer_cast(ridx.CurrentSpan().data()),
        thrust::device_pointer_cast(ridx.CurrentSpan().data() + ridx.Size()));

    std::fill(ridx_segments.begin(), ridx_segments.end(), Segment(0, 0));
    ridx_segments.front() = Segment(0, ridx.Size());
    dh::safe_cuda(cudaMemcpyAsync(
        gpair.data(), dh_gpair->ConstDevicePointer(device_id),
        gpair.size() * sizeof(GradientPair), cudaMemcpyHostToHost));
    SubsampleGradientPair(device_id, gpair, param.subsample, row_begin_idx);
    hist.Reset();

    split_evaluator->Reset();
  }

  // Run feature sampling and use interaction constraint.
  common::Span<int32_t const> GetFeaturesSet(
      int32_t nidx,
      std::shared_ptr<HostDeviceVector<int32_t> const> p_sampled_features,
      HostDeviceVector<int32_t>* buffer) const {
    common::Span<int32_t const> d_feature_set;
    if (has_interaction_constraint) {
      auto const& h_sampled_features = p_sampled_features->ConstHostVector();
      auto& h_buffer = buffer->HostVector();
      for (auto f : h_sampled_features) {
        if (split_evaluator->CheckFeatureConstraint(nidx, f)) {
          h_buffer.emplace_back(f);
        }
      }
      if (h_buffer.size() == 0) {
        LOG(INFO) << "No sampled feature satifies constraints.";
      }
      buffer->Shard(GPUDistribution(GPUSet(device_id, 1)));
      d_feature_set = buffer->DeviceSpan(device_id);
    } else {
      d_feature_set = p_sampled_features->DeviceSpan(device_id);
    }

    return d_feature_set;
  }

  std::vector<DeviceSplitCandidate> EvaluateSplits(
      std::vector<int> nidxs, const RegTree& tree,
      size_t num_columns) {
    dh::safe_cuda(cudaSetDevice(device_id));
    auto result = pinned_memory.GetSpan<DeviceSplitCandidate>(nidxs.size());

    // Work out cub temporary memory requirement
    GPUTrainingParam gpu_param(param);
    DeviceSplitCandidateReduceOp op(gpu_param);
    size_t temp_storage_bytes;
    DeviceSplitCandidate*dummy = nullptr;
    cub::DeviceReduce::Reduce(
        nullptr, temp_storage_bytes, dummy,
        dummy, num_columns, op,
        DeviceSplitCandidate());
    // size in terms of DeviceSplitCandidate
    size_t cub_memory_size =
      std::ceil(static_cast<double>(temp_storage_bytes) /
        sizeof(DeviceSplitCandidate));

    // Allocate enough temporary memory
    // Result for each nidx
    // + intermediate result for each column
    // + cub reduce memory
    auto temp_span = temp_memory.GetSpan<DeviceSplitCandidate>(
        nidxs.size() + nidxs.size() * num_columns +cub_memory_size*nidxs.size());
    auto d_result_all = temp_span.subspan(0, nidxs.size());
    auto d_split_candidates_all =
        temp_span.subspan(d_result_all.size(), nidxs.size() * num_columns);
    auto d_cub_memory_all =
        temp_span.subspan(d_result_all.size() + d_split_candidates_all.size(),
                          cub_memory_size * nidxs.size());

    auto& streams = this->GetStreams(nidxs.size());
    for (auto i = 0ull; i < nidxs.size(); i++) {
      auto nidx = nidxs[i];
      std::shared_ptr<HostDeviceVector<int32_t>> p_sampled_features =
          column_sampler.GetFeatureSet(tree.GetDepth(nidx));
      HostDeviceVector<int32_t> constrainted_feature_set;
      p_sampled_features->Shard(GPUSet(device_id, 1));
      common::Span<int32_t const> d_feature_set =
          GetFeaturesSet(nidx, p_sampled_features, &constrainted_feature_set);

      auto d_split_candidates =
          d_split_candidates_all.subspan(i * num_columns, d_feature_set.size());
      DeviceNodeStats node(node_sum_gradients[nidx], nidx, param);

      // One block for each feature
      int constexpr kBlockThreads = 256;
      EvaluateSplitKernel<kBlockThreads, GradientSumT>
          <<<uint32_t(d_feature_set.size()), kBlockThreads, 0, streams[i]>>>(
              hist.GetNodeHistogram(nidx), d_feature_set, node, ellpack_matrix,
              gpu_param, d_split_candidates, node_value_constraints[nidx],
              monotone_constraints);

      // Reduce over features to find best feature
      auto d_result = d_result_all.subspan(i, 1);
      auto d_cub_memory =
          d_cub_memory_all.subspan(i * cub_memory_size, cub_memory_size);
      size_t cub_bytes = d_cub_memory.size() * sizeof(DeviceSplitCandidate);
      cub::DeviceReduce::Reduce(reinterpret_cast<void*>(d_cub_memory.data()),
                                cub_bytes, d_split_candidates.data(),
                                d_result.data(), d_split_candidates.size(), op,
                                DeviceSplitCandidate(), streams[i]);
    }

    dh::safe_cuda(cudaMemcpy(result.data(), d_result_all.data(),
                             sizeof(DeviceSplitCandidate) * d_result_all.size(),
                             cudaMemcpyDeviceToHost));

    return std::vector<DeviceSplitCandidate>(result.begin(), result.end());
  }

  void BuildHist(int nidx) {
    hist.AllocateHistogram(nidx);
    hist_builder->Build(this, nidx);
  }

  void SubtractionTrick(int nidx_parent, int nidx_histogram,
                        int nidx_subtraction) {
    auto d_node_hist_parent = hist.GetNodeHistogram(nidx_parent);
    auto d_node_hist_histogram = hist.GetNodeHistogram(nidx_histogram);
    auto d_node_hist_subtraction = hist.GetNodeHistogram(nidx_subtraction);

    dh::LaunchN(device_id, n_bins, [=] __device__(size_t idx) {
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

  void UpdatePosition(int nidx, RegTree::Node split_node) {
    CHECK(!split_node.IsLeaf()) <<"Node must not be leaf";
    Segment segment = ridx_segments[nidx];
    bst_uint* d_ridx = ridx.Current();
    int* d_position = position.Current();
    if (left_counts.size() <= nidx) {
      left_counts.resize((nidx * 2) + 1);
    }
    int64_t* d_left_count = left_counts.data().get() + nidx;
    auto d_matrix = this->ellpack_matrix;
    // Launch 1 thread for each row
    dh::LaunchN<1, 128>(
        device_id, segment.Size(), [=] __device__(bst_uint idx) {
          idx += segment.begin;
          bst_uint ridx = d_ridx[idx];
          bst_float element = d_matrix.GetElement(ridx, split_node.SplitIndex());
          // Missing value
          int new_position = 0;
          if (isnan(element)) {
            new_position = split_node.DefaultChild();
          } else {
            if (element <= split_node.SplitCond()) {
              new_position = split_node.LeftChild();
            } else {
              new_position = split_node.RightChild();
            }
          }
          CountLeft(d_left_count, new_position, split_node.LeftChild());
          d_position[idx] = new_position;
        });

    // Overlap device to host memory copy (left_count) with sort
    auto& streams = this->GetStreams(2);
    auto tmp_pinned = pinned_memory.GetSpan<int64_t>(1);
    dh::safe_cuda(cudaMemcpyAsync(tmp_pinned.data(), d_left_count, sizeof(int64_t),
                                  cudaMemcpyDeviceToHost, streams[0]));

    SortPositionAndCopy(segment, split_node.LeftChild(), split_node.RightChild(), d_left_count,
                        streams[1]);

    dh::safe_cuda(cudaStreamSynchronize(streams[0]));
    int64_t left_count = tmp_pinned[0];
    CHECK_LE(left_count, segment.Size());
    CHECK_GE(left_count, 0);
    ridx_segments[split_node.LeftChild()] =
        Segment(segment.begin, segment.begin + left_count);
    ridx_segments[split_node.RightChild()] =
        Segment(segment.begin + left_count, segment.end);
  }

  /*! \brief Sort row indices according to position. */
  void SortPositionAndCopy(const Segment& segment, int left_nidx,
                           int right_nidx, int64_t* d_left_count,
                           cudaStream_t stream) {
    SortPosition(
        &temp_memory,
        common::Span<int>(position.Current() + segment.begin, segment.Size()),
        common::Span<int>(position.other() + segment.begin, segment.Size()),
        common::Span<bst_uint>(ridx.Current() + segment.begin, segment.Size()),
        common::Span<bst_uint>(ridx.other() + segment.begin, segment.Size()),
        left_nidx, right_nidx, d_left_count, stream);
    // Copy back key/value
    const auto d_position_current = position.Current() + segment.begin;
    const auto d_position_other = position.other() + segment.begin;
    const auto d_ridx_current = ridx.Current() + segment.begin;
    const auto d_ridx_other = ridx.other() + segment.begin;
    dh::LaunchN(device_id, segment.Size(), stream, [=] __device__(size_t idx) {
      d_position_current[idx] = d_position_other[idx];
      d_ridx_current[idx] = d_ridx_other[idx];
    });
  }

  // After tree update is finished, update the position of all training
  // instances to their final leaf This information is used later to update the
  // prediction cache
  void FinalisePosition(RegTree* p_tree) {
    const auto d_nodes =
        temp_memory.GetSpan<RegTree::Node>(p_tree->GetNodes().size());
    dh::safe_cuda(cudaMemcpy(d_nodes.data(), p_tree->GetNodes().data(),
                             d_nodes.size() * sizeof(RegTree::Node),
                             cudaMemcpyHostToDevice));
    auto d_position = position.Current();
    const auto d_ridx = ridx.Current();
    auto d_matrix = this->ellpack_matrix;
    dh::LaunchN(device_id, position.Size(), [=] __device__(size_t idx) {
      auto position = d_position[idx];
      auto node = d_nodes[position];
      bst_uint ridx = d_ridx[idx];

      while (!node.IsLeaf()) {
        bst_float element = d_matrix.GetElement(ridx, node.SplitIndex());
        // Missing value
        if (isnan(element)) {
          position = node.DefaultChild();
        } else {
          if (element <= node.SplitCond()) {
            position = node.LeftChild();
          } else {
            position = node.RightChild();
          }
        }
        node = d_nodes[position];
      }
      d_position[idx] = position;
    });
  }

  void UpdatePredictionCache(bst_float* out_preds_d) {
    dh::safe_cuda(cudaSetDevice(device_id));
    if (!prediction_cache_initialised) {
      dh::safe_cuda(cudaMemcpyAsync(prediction_cache.data(), out_preds_d,
                                    prediction_cache.size() * sizeof(bst_float),
                                    cudaMemcpyDefault));
    }
    prediction_cache_initialised = true;

    CalcWeightTrainParam param_d(param);

    dh::safe_cuda(
        cudaMemcpyAsync(node_sum_gradients_d.data(), node_sum_gradients.data(),
                        sizeof(GradientPair) * node_sum_gradients.size(),
                        cudaMemcpyHostToDevice));
    auto d_position = position.Current();
    auto d_ridx = ridx.Current();
    auto d_node_sum_gradients = node_sum_gradients_d.data();
    auto d_prediction_cache = prediction_cache.data();

    dh::LaunchN(
        device_id, prediction_cache.size(), [=] __device__(int local_idx) {
          int pos = d_position[local_idx];
          bst_float weight = CalcWeight(param_d, d_node_sum_gradients[pos]);
          d_prediction_cache[d_ridx[local_idx]] +=
              weight * param_d.learning_rate;
        });

    dh::safe_cuda(cudaMemcpy(
        out_preds_d, prediction_cache.data(),
        prediction_cache.size() * sizeof(bst_float), cudaMemcpyDefault));
  }

  void AllReduceHist(int nidx, dh::AllReducer* reducer) {
    monitor.StartCuda("AllReduce");
    auto d_node_hist = hist.GetNodeHistogram(nidx).data();
    reducer->AllReduceSum(
        shard_idx,
        reinterpret_cast<typename GradientSumT::ValueT*>(d_node_hist),
        reinterpret_cast<typename GradientSumT::ValueT*>(d_node_hist),
        ellpack_matrix.BinCount() *
            (sizeof(GradientSumT) / sizeof(typename GradientSumT::ValueT)));
    reducer->Synchronize(device_id);

    monitor.StopCuda("AllReduce");
  }

  /**
   * \brief Build GPU local histograms for the left and right child of some parent node
   */
  void BuildHistLeftRight(int nidx_parent, int nidx_left, int nidx_right, dh::AllReducer* reducer) {
    auto build_hist_nidx = nidx_left;
    auto subtraction_trick_nidx = nidx_right;

    // If we are using a single GPU, build the histogram for the node with the
    // fewest training instances
    // If we are distributed, don't bother
    if (reducer->IsSingleGPU()) {
      bool fewer_right =
          ridx_segments[nidx_right].Size() < ridx_segments[nidx_left].Size();
      if (fewer_right) {
        std::swap(build_hist_nidx, subtraction_trick_nidx);
      }
    }

    this->BuildHist(build_hist_nidx);
    this->AllReduceHist(build_hist_nidx, reducer);

    // Check whether we can use the subtraction trick to calculate the other
    bool do_subtraction_trick = this->CanDoSubtractionTrick(
        nidx_parent, build_hist_nidx, subtraction_trick_nidx);

    if (do_subtraction_trick) {
      // Calculate other histogram using subtraction trick
      this->SubtractionTrick(nidx_parent, build_hist_nidx,
                             subtraction_trick_nidx);
    } else {
      // Calculate other histogram manually
      this->BuildHist(subtraction_trick_nidx);
      this->AllReduceHist(subtraction_trick_nidx, reducer);
    }
  }

  void ApplySplit(const ExpandEntry& candidate, RegTree* p_tree) {
    RegTree& tree = *p_tree;

    GradStats left_stats;
    left_stats.Add(candidate.split.left_sum);
    GradStats right_stats;
    right_stats.Add(candidate.split.right_sum);
    GradStats parent_sum;
    parent_sum.Add(left_stats);
    parent_sum.Add(right_stats);
    node_value_constraints.resize(tree.GetNodes().size());
    auto base_weight = node_value_constraints[candidate.nid].CalcWeight(param, parent_sum);
    auto left_weight =
        node_value_constraints[candidate.nid].CalcWeight(param, left_stats)*param.learning_rate;
    auto right_weight =
        node_value_constraints[candidate.nid].CalcWeight(param, right_stats)*param.learning_rate;
    tree.ExpandNode(candidate.nid, candidate.split.findex,
                    candidate.split.fvalue, candidate.split.dir == kLeftDir,
                    base_weight, left_weight, right_weight,
                    candidate.split.loss_chg, parent_sum.sum_hess);
    // Set up child constraints
    node_value_constraints.resize(tree.GetNodes().size());
    node_value_constraints[candidate.nid].SetChild(
        param, tree[candidate.nid].SplitIndex(), left_stats, right_stats,
        &node_value_constraints[tree[candidate.nid].LeftChild()],
        &node_value_constraints[tree[candidate.nid].RightChild()]);
    node_sum_gradients[tree[candidate.nid].LeftChild()] =
        candidate.split.left_sum;
    node_sum_gradients[tree[candidate.nid].RightChild()] =
        candidate.split.right_sum;

    split_evaluator->AddSplit(candidate.nid,
                              tree[candidate.nid].LeftChild(), tree[candidate.nid].RightChild(),
                              tree[candidate.nid].SplitIndex(),
                              left_weight, right_weight);
  }

  void InitRoot(RegTree* p_tree, HostDeviceVector<GradientPair>* gpair_all,
                dh::AllReducer* reducer, int64_t num_columns) {
    constexpr int kRootNIdx = 0;

    const auto &gpair = gpair_all->DeviceSpan(device_id);

    dh::SumReduction(temp_memory, gpair, node_sum_gradients_d,
                     gpair.size());
    reducer->AllReduceSum(
        shard_idx, reinterpret_cast<float*>(node_sum_gradients_d.data()),
        reinterpret_cast<float*>(node_sum_gradients_d.data()), 2);
    reducer->Synchronize(device_id);
    dh::safe_cuda(cudaMemcpy(node_sum_gradients.data(),
                             node_sum_gradients_d.data(), sizeof(GradientPair),
                             cudaMemcpyDeviceToHost));

    this->BuildHist(kRootNIdx);
    this->AllReduceHist(kRootNIdx, reducer);

    // Remember root stats
    p_tree->Stat(kRootNIdx).sum_hess = node_sum_gradients[kRootNIdx].GetHess();
    auto weight = CalcWeight(param, node_sum_gradients[kRootNIdx]);
    p_tree->Stat(kRootNIdx).base_weight = weight;
    (*p_tree)[kRootNIdx].SetLeaf(param.learning_rate * weight);

    // Initialise root constraint
    node_value_constraints.resize(p_tree->GetNodes().size());

    // Generate first split
    auto split = this->EvaluateSplits({kRootNIdx}, *p_tree, num_columns);
    qexpand->push(
        ExpandEntry(kRootNIdx, p_tree->GetDepth(kRootNIdx), split.at(0), 0));
  }

  void UpdateTree(HostDeviceVector<GradientPair>* gpair_all, DMatrix* p_fmat,
                  RegTree* p_tree, dh::AllReducer* reducer) {
    auto& tree = *p_tree;
    monitor.StartCuda("Reset");
    this->Reset(gpair_all, p_fmat->Info().num_col_);
    monitor.StopCuda("Reset");

    monitor.StartCuda("InitRoot");
    this->InitRoot(p_tree, gpair_all, reducer, p_fmat->Info().num_col_);
    monitor.StopCuda("InitRoot");

    auto timestamp = qexpand->size();
    auto num_leaves = 1;

    while (!qexpand->empty()) {
      ExpandEntry candidate = qexpand->top();
      qexpand->pop();
      if (!candidate.IsValid(param, num_leaves)) {
        continue;
      }

      this->ApplySplit(candidate, p_tree);

      num_leaves++;

      int left_child_nidx = tree[candidate.nid].LeftChild();
      int right_child_nidx = tree[candidate.nid].RightChild();
      // Only create child entries if needed
      if (ExpandEntry::ChildIsValid(param, tree.GetDepth(left_child_nidx),
        num_leaves)) {
        monitor.StartCuda("UpdatePosition");
        this->UpdatePosition(candidate.nid, (*p_tree)[candidate.nid]);
        monitor.StopCuda("UpdatePosition");

        monitor.StartCuda("BuildHist");
        this->BuildHistLeftRight(candidate.nid, left_child_nidx, right_child_nidx, reducer);
        monitor.StopCuda("BuildHist");

        monitor.StartCuda("EvaluateSplits");
        auto splits = this->EvaluateSplits({left_child_nidx, right_child_nidx},
                                           *p_tree, p_fmat->Info().num_col_);
        monitor.StopCuda("EvaluateSplits");

        qexpand->push(ExpandEntry(left_child_nidx,
                                   tree.GetDepth(left_child_nidx), splits.at(0),
                                   timestamp++));
        qexpand->push(ExpandEntry(right_child_nidx,
                                   tree.GetDepth(right_child_nidx),
                                   splits.at(1), timestamp++));
      }
    }

    monitor.StartCuda("FinalisePosition");
    this->FinalisePosition(p_tree);
    monitor.StopCuda("FinalisePosition");
  }
};  // end DeviceShard

template <typename GradientSumT>
struct SharedMemHistBuilder : public GPUHistBuilderBase<GradientSumT> {
  void Build(DeviceShard<GradientSumT>* shard, int nidx) override {
    auto segment = shard->ridx_segments[nidx];
    auto segment_begin = segment.begin;
    auto d_node_hist = shard->hist.GetNodeHistogram(nidx);
    auto d_ridx = shard->ridx.Current();
    auto d_gpair = shard->gpair.data();

    auto n_elements = segment.Size() * shard->ellpack_matrix.row_stride;

    const size_t smem_size = sizeof(GradientSumT) * shard->ellpack_matrix.BinCount();
    const int items_per_thread = 8;
    const int block_threads = 256;
    const int grid_size =
        static_cast<int>(dh::DivRoundUp(n_elements,
                                        items_per_thread * block_threads));
    if (grid_size <= 0) {
      return;
    }
    SharedMemHistKernel<<<grid_size, block_threads, smem_size>>>(
        shard->ellpack_matrix, d_ridx, d_node_hist.data(), d_gpair,
        segment_begin, n_elements);
  }
};

template <typename GradientSumT>
struct GlobalMemHistBuilder : public GPUHistBuilderBase<GradientSumT> {
  void Build(DeviceShard<GradientSumT>* shard, int nidx) override {
    Segment segment = shard->ridx_segments[nidx];
    auto d_node_hist = shard->hist.GetNodeHistogram(nidx).data();
    bst_uint* d_ridx = shard->ridx.Current();
    GradientPair* d_gpair = shard->gpair.data();

    size_t const n_elements = segment.Size() * shard->ellpack_matrix.row_stride;
    auto d_matrix = shard->ellpack_matrix;

    dh::LaunchN(shard->device_id, n_elements, [=] __device__(size_t idx) {
        int ridx = d_ridx[(idx / d_matrix.row_stride) + segment.begin];
        // lookup the index (bin) of histogram.
        int gidx = d_matrix.gidx_iter[ridx * d_matrix.row_stride + idx % d_matrix.row_stride];

        if (gidx != d_matrix.null_gidx_value) {
          AtomicAddGpair(d_node_hist + gidx, d_gpair[ridx]);
        }
      });
  }
};

template <typename GradientSumT>
inline void DeviceShard<GradientSumT>::InitCompressedData(
    const common::HistCutMatrix& hmat, const SparsePage& row_batch, bool is_dense) {
  size_t row_stride = this->InitRowPtrs(row_batch);
  n_bins = hmat.row_ptr.back();
  int null_gidx_value = hmat.row_ptr.back();

  int max_nodes =
      param.max_leaves > 0 ? param.max_leaves * 2 : MaxNodesDepth(param.max_depth);

  ba.Allocate(device_id,
              &gpair, n_rows,
              &ridx, n_rows,
              &position, n_rows,
              &prediction_cache, n_rows,
              &node_sum_gradients_d, max_nodes,
              &feature_segments, hmat.row_ptr.size(),
              &gidx_fvalue_map, hmat.cut.size(),
              &min_fvalue, hmat.min_val.size(),
              &monotone_constraints, param.monotone_constraints.size());

  dh::CopyVectorToDeviceSpan(gidx_fvalue_map, hmat.cut);
  dh::CopyVectorToDeviceSpan(min_fvalue, hmat.min_val);
  dh::CopyVectorToDeviceSpan(feature_segments, hmat.row_ptr);
  dh::CopyVectorToDeviceSpan(monotone_constraints, param.monotone_constraints);

  node_sum_gradients.resize(max_nodes);
  ridx_segments.resize(max_nodes);


  // allocate compressed bin data
  int num_symbols = n_bins + 1;
  // Required buffer size for storing data matrix in ELLPack format.
  size_t compressed_size_bytes =
      common::CompressedBufferWriter::CalculateBufferSize(row_stride * n_rows,
                                                          num_symbols);

  CHECK(!(param.max_leaves == 0 && param.max_depth == 0))
      << "Max leaves and max depth cannot both be unconstrained for "
      "gpu_hist.";
  ba.Allocate(device_id, &gidx_buffer, compressed_size_bytes);
  thrust::fill(
      thrust::device_pointer_cast(gidx_buffer.data()),
      thrust::device_pointer_cast(gidx_buffer.data() + gidx_buffer.size()), 0);

  this->CreateHistIndices(row_batch, row_stride, null_gidx_value);

  ellpack_matrix.Init(
      feature_segments, min_fvalue,
      gidx_fvalue_map, row_stride,
      common::CompressedIterator<uint32_t>(gidx_buffer.data(), num_symbols),
      is_dense, null_gidx_value);

  // check if we can use shared memory for building histograms
  // (assuming atleast we need 2 CTAs per SM to maintain decent latency
  // hiding)
  auto histogram_size = sizeof(GradientSumT) * hmat.row_ptr.back();
  auto max_smem = dh::MaxSharedMemory(device_id);
  if (histogram_size <= max_smem) {
    hist_builder.reset(new SharedMemHistBuilder<GradientSumT>);
  } else {
    hist_builder.reset(new GlobalMemHistBuilder<GradientSumT>);
  }

  // Init histogram
  hist.Init(device_id, hmat.NumBins());
}

template <typename GradientSumT>
inline void DeviceShard<GradientSumT>::CreateHistIndices(
    const SparsePage& row_batch, size_t row_stride, int null_gidx_value) {
  int num_symbols = n_bins + 1;
  // bin and compress entries in batches of rows
  size_t gpu_batch_nrows =
      std::min
      (dh::TotalMemory(device_id) / (16 * row_stride * sizeof(Entry)),
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
    CompressBinEllpackKernel<<<grid3, block3>>>
        (common::CompressedBufferWriter(num_symbols),
         gidx_buffer.data(),
         row_ptrs.data().get() + batch_row_begin,
         entries_d.data().get(),
         gidx_fvalue_map.data(), feature_segments.data(),
         batch_row_begin, batch_nrows,
         row_ptrs[batch_row_begin],
         row_stride, null_gidx_value);
  }

  // free the memory that is no longer needed
  row_ptrs.resize(0);
  row_ptrs.shrink_to_fit();
  entries_d.resize(0);
  entries_d.shrink_to_fit();
}

template <typename GradientSumT>
class GPUHistMakerSpecialised {
 public:
  GPUHistMakerSpecialised() :
      initialised_{false}, has_interaction_constraint_{false}, p_last_fmat_{nullptr} {}
  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) {
    param_.InitAllowUnknown(args);
    hist_maker_param_.InitAllowUnknown(args);
    CHECK(param_.n_gpus != 0) << "Must have at least one device";
    n_devices_ = param_.n_gpus;
    dist_ = GPUDistribution::Block(GPUSet::All(param_.gpu_id, param_.n_gpus));

    for (auto const& kv : args) {
      if (kv.first == "interaction_constraints") {
        has_interaction_constraint_ = true;
        break;
      }
    }
    // elastic_net is only used as an internal base evaluator for
    // interaction, doens't contribute to GPU Hist.
    split_evaluator_.reset(SplitEvaluator::Create("elastic_net,interaction"));
    split_evaluator_->Init(args);

    dh::CheckComputeCapability();

    monitor_.Init("updater_gpu_hist");
  }

  void Update(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) {
    monitor_.StartCuda("Update");
    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    ValueConstraint::Init(&param_, dmat->Info().num_col_);
    // build tree
    try {
      for (xgboost::RegTree* tree : trees) {
        this->UpdateTree(gpair, dmat, tree);
      }
      dh::safe_cuda(cudaGetLastError());
    } catch (const std::exception& e) {
      LOG(FATAL) << "Exception in gpu_hist: " << e.what() << std::endl;
    }
    param_.learning_rate = lr;
    monitor_.StopCuda("Update");
  }

  void InitDataOnce(DMatrix* dmat) {
    info_ = &dmat->Info();

    int n_devices = dist_.Devices().Size();

    device_list_.resize(n_devices);
    for (int index = 0; index < n_devices; ++index) {
      int device_id = dist_.Devices().DeviceId(index);
      device_list_[index] = device_id;
    }

    reducer_.Init(device_list_);

    auto batch_iter = dmat->GetRowBatches().begin();
    const SparsePage& batch = *batch_iter;

    // Synchronise the column sampling seed
    uint32_t column_sampling_seed = common::GlobalRandom()();
    rabit::Broadcast(&column_sampling_seed, sizeof(column_sampling_seed), 0);

    // Create device shards
    shards_.resize(n_devices);
    dh::ExecuteIndexShards(
        &shards_,
        [&](int idx, std::unique_ptr<DeviceShard<GradientSumT>>& shard) {
          dh::safe_cuda(cudaSetDevice(dist_.Devices().DeviceId(idx)));
          size_t start = dist_.ShardStart(info_->num_row_, idx);
          size_t size = dist_.ShardSize(info_->num_row_, idx);
          shard = std::unique_ptr<DeviceShard<GradientSumT>>(
              new DeviceShard<GradientSumT>(
                  dist_.Devices().DeviceId(idx), idx,
                  start, start + size, param_,
                  has_interaction_constraint_,
                  std::unique_ptr<SplitEvaluator>(split_evaluator_->GetHostClone()),
                  column_sampling_seed));
        });

    // Find the cuts.
    monitor_.StartCuda("Quantiles");
    common::DeviceSketch(batch, *info_, param_, &hmat_, hist_maker_param_.gpu_batch_nrows);
    n_bins_ = hmat_.row_ptr.back();
    monitor_.StopCuda("Quantiles");
    auto is_dense = info_->num_nonzero_ == info_->num_row_ * info_->num_col_;

    monitor_.StartCuda("BinningCompression");
    dh::ExecuteIndexShards(
        &shards_,
        [&](int idx, std::unique_ptr<DeviceShard<GradientSumT>>& shard) {
          dh::safe_cuda(cudaSetDevice(shard->device_id));
          shard->InitCompressedData(hmat_, batch, is_dense);
        });
    monitor_.StopCuda("BinningCompression");
    ++batch_iter;
    CHECK(batch_iter.AtEnd()) << "External memory not supported";

    p_last_fmat_ = dmat;
    initialised_ = true;
  }

  void InitData(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat) {
    if (!initialised_) {
      monitor_.StartCuda("InitDataOnce");
      this->InitDataOnce(dmat);
      monitor_.StopCuda("InitDataOnce");
    }
  }

  // Only call this method for testing
  void CheckTreesSynchronized(const std::vector<RegTree>& local_trees) const {
    std::string s_model;
    common::MemoryBufferStream fs(&s_model);
    int rank = rabit::GetRank();
    if (rank == 0) {
      local_trees.front().Save(&fs);
    }
    fs.Seek(0);
    rabit::Broadcast(&s_model, 0);
    RegTree reference_tree;
    reference_tree.Load(&fs);
    for (const auto& tree : local_trees) {
      CHECK(tree == reference_tree);
    }
  }

  void UpdateTree(HostDeviceVector<GradientPair>* gpair, DMatrix* p_fmat,
                  RegTree* p_tree) {
    monitor_.StartCuda("InitData");
    this->InitData(gpair, p_fmat);
    monitor_.StopCuda("InitData");

    std::vector<RegTree> trees(shards_.size());
    for (auto& tree : trees) {
      tree = *p_tree;
    }
    gpair->Reshard(dist_);

    // Launch one thread for each device "shard" containing a subset of rows.
    // Threads will cooperatively build the tree, synchronising over histograms.
    // Each thread will redundantly build its own copy of the tree
    dh::ExecuteIndexShards(
        &shards_,
        [&](int idx, std::unique_ptr<DeviceShard<GradientSumT>>& shard) {
          shard->UpdateTree(gpair, p_fmat, &trees.at(idx), &reducer_);
        });

    // All trees are expected to be identical
    if (hist_maker_param_.debug_synchronize) {
      this->CheckTreesSynchronized(trees);
    }

    // Write the output tree
    *p_tree = trees.front();
  }

  bool UpdatePredictionCache(
      const DMatrix* data, HostDeviceVector<bst_float>* p_out_preds) {
    if (shards_.empty() || p_last_fmat_ == nullptr || p_last_fmat_ != data) {
      return false;
    }
    monitor_.StartCuda("UpdatePredictionCache");
    p_out_preds->Shard(dist_.Devices());
    dh::ExecuteIndexShards(
        &shards_,
        [&](int idx, std::unique_ptr<DeviceShard<GradientSumT>>& shard) {
          dh::safe_cuda(cudaSetDevice(shard->device_id));
          shard->UpdatePredictionCache(
              p_out_preds->DevicePointer(shard->device_id));
        });
    monitor_.StopCuda("UpdatePredictionCache");
    return true;
  }

  TrainParam param_;            // NOLINT
  common::HistCutMatrix hmat_;  // NOLINT
  MetaInfo* info_;              // NOLINT

  std::vector<std::unique_ptr<DeviceShard<GradientSumT>>> shards_;  // NOLINT

 private:
  bool initialised_;
  bool has_interaction_constraint_;

  int n_devices_;
  int n_bins_;

  GPUHistMakerTrainParam hist_maker_param_;
  common::GHistIndexMatrix gmat_;
  // Only used for interaction constraint.
  std::unique_ptr<SplitEvaluator> split_evaluator_;

  dh::AllReducer reducer_;

  DMatrix* p_last_fmat_;
  GPUDistribution dist_;

  common::Monitor monitor_;
  /*! List storing device id. */
  std::vector<int> device_list_;
};

class GPUHistMaker : public TreeUpdater {
 public:
  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) override {
    hist_maker_param_.InitAllowUnknown(args);
    float_maker_.reset();
    double_maker_.reset();
    if (hist_maker_param_.single_precision_histogram) {
      float_maker_.reset(new GPUHistMakerSpecialised<GradientPair>());
      float_maker_->Init(args);
    } else {
      double_maker_.reset(new GPUHistMakerSpecialised<GradientPairPrecise>());
      double_maker_->Init(args);
    }
  }

  void Update(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    if (hist_maker_param_.single_precision_histogram) {
      float_maker_->Update(gpair, dmat, trees);
    } else {
      double_maker_->Update(gpair, dmat, trees);
    }
  }

  bool UpdatePredictionCache(
      const DMatrix* data, HostDeviceVector<bst_float>* p_out_preds) override {
    if (hist_maker_param_.single_precision_histogram) {
      return float_maker_->UpdatePredictionCache(data, p_out_preds);
    } else {
      return double_maker_->UpdatePredictionCache(data, p_out_preds);
    }
  }

 private:
  GPUHistMakerTrainParam hist_maker_param_;
  std::unique_ptr<GPUHistMakerSpecialised<GradientPair>> float_maker_;
  std::unique_ptr<GPUHistMakerSpecialised<GradientPairPrecise>> double_maker_;
};

#if !defined(GTEST_TEST)
XGBOOST_REGISTER_TREE_UPDATER(GPUHistMaker, "grow_gpu_hist")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUHistMaker(); });
#endif  // !defined(GTEST_TEST)

}  // namespace tree
}  // namespace xgboost
