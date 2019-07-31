/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
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
#include "constraints.cuh"
#include "gpu_hist/row_partitioner.cuh"

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

// A context that is created for every row that is processed during binning. This is then
// handed off to the different matrices to write to the underlying stream it manages
struct CompressRowContext {
  int bin_;  // NOLINT: Feature bin
  size_t irow_;  // NOLINT: Row to process
  size_t base_row_;  // NOLINT: Total number of rows processed thus far
  size_t row_offset_in_batch_;  // NOLINT: Offset to current row in the batch
  int ifeature_;  // NOLINT: Feature to process
  size_t base_item_offset_;  // NOLINT: Offset to the item in the current batch
  size_t total_items_processed_;  // NOLINT: Total number of items processed thus far

  __device__ explicit CompressRowContext(
    int bin, size_t irow, size_t base_row, size_t row_offset_in_batch,
    int ifeature, size_t base_item_offset, size_t total_items_processed)
    : bin_(bin), irow_(irow), base_row_(base_row), row_offset_in_batch_(row_offset_in_batch),
      ifeature_(ifeature), base_item_offset_(base_item_offset),
      total_items_processed_(total_items_processed) {}
};

/*! \brief How is the compressed data laid out? */
enum class CompressedDataLayout {
  kRowStride,  // Every row is evenly sized with row stride number of items
  kCSR  // Every row is sized based on the actual number of items in that row
};

// Base type of all matrices containing the histograms for all the features that are needed
// for binning. It also abstracts some of the common matrix properties
struct MatrixBase {
  common::Span<uint32_t> feature_segments;
  /*! \brief minimum value for each feature. */
  common::Span<bst_float> min_fvalue;
  /*! \brief Cut. */
  common::Span<bst_float> gidx_fvalue_map;
  int null_gidx_value;

  /*! \brief row length for ELLPack. */
  size_t row_stride{0};

  common::CompressedBufferWriter gidx_buffer_writer;
  common::CompressedIterator<uint32_t> gidx_buffer_iter;
  common::Span<common::CompressedByteT> gidx_buffer;

  __device__  explicit MatrixBase(
    common::Span<uint32_t> fsegs, common::Span<bst_float> min_fvals,
    common::Span<bst_float> fval_map, common::CompressedBufferWriter buf_wr,
    common::CompressedIterator<uint32_t> buf_itr, common::Span<common::CompressedByteT> buf,
    int ngidx, size_t rstride)
      : feature_segments(fsegs), min_fvalue(min_fvals), gidx_fvalue_map(fval_map),
        null_gidx_value(ngidx), gidx_buffer_writer(buf_wr), gidx_buffer_iter(buf_itr),
        gidx_buffer(buf), row_stride(rstride) {}
  __device__  virtual ~MatrixBase() {}  // NOLINT

  __forceinline__ __device__ virtual bst_float GetElement(size_t ridx, size_t fidx) const = 0;
  __forceinline__ __device__ virtual int GetGidx(size_t ridx, size_t gidx_pos) const {
    return gidx_buffer_iter[ridx * row_stride + gidx_pos % row_stride];
  }
  __forceinline__ __device__ virtual void Write(const CompressRowContext &com_ctx) {
    gidx_buffer_writer.AtomicWriteSymbol(
      gidx_buffer.data(), com_ctx.bin_,
      (com_ctx.irow_ + com_ctx.base_row_) * row_stride + com_ctx.ifeature_);
  }

  __forceinline__ __device__ uint32_t GetFeatureBin(int fidx) const {
    return feature_segments[fidx];
  }
  __forceinline__ __device__ bst_float GetMinFeatureValue(int fidx) const {
    return min_fvalue[fidx];
  }
  __forceinline__ __device__ const bst_float *GetFeatureValue(int fbin) const {
    return &gidx_fvalue_map[fbin];
  }
  __forceinline__ __device__ size_t BinCount() const { return gidx_fvalue_map.size(); }
  __forceinline__ __device__ size_t RowStride() const { return row_stride; }
  __forceinline__ __device__ int NullGidxValue() const { return null_gidx_value; }
};

// A dense matrix representation, where every row contains every feature
struct DenseMatrix : MatrixBase {
  __forceinline__ __device__ bst_float GetElement(size_t ridx, size_t fidx) const override {
      auto row_begin = row_stride * ridx;
      auto gidx = gidx_buffer_iter[row_begin + fidx];
      return gidx_fvalue_map[gidx];
  }

  __device__  explicit DenseMatrix(
    common::Span<uint32_t> fsegs, common::Span<bst_float> min_fvals,
    common::Span<bst_float> fval_map, common::CompressedBufferWriter buf_wr,
    common::CompressedIterator<uint32_t> buf_itr, common::Span<common::CompressedByteT> buf,
    int ngidx, size_t rstride)
      : MatrixBase(fsegs, min_fvals, fval_map, buf_wr, buf_itr, buf, ngidx, rstride) {}
};

// A sparse matrix representation, where each row contains a constant number of features
struct RowStrideMatrix : MatrixBase {
  __forceinline__ __device__ bst_float GetElement(size_t ridx, size_t fidx) const override {
      auto row_begin = row_stride * ridx;
      auto row_end = row_begin + row_stride;
      auto gidx = BinarySearchRow(row_begin, row_end, gidx_buffer_iter, feature_segments[fidx],
                                  feature_segments[fidx + 1]);
      return (gidx == -1) ? nan("") : gidx_fvalue_map[gidx];
  }

  __device__  explicit RowStrideMatrix(
    common::Span<uint32_t> fsegs, common::Span<bst_float> min_fvals,
    common::Span<bst_float> fval_map, common::CompressedBufferWriter buf_wr,
    common::CompressedIterator<uint32_t> buf_itr, common::Span<common::CompressedByteT> buf,
    int ngidx, size_t rstride)
      : MatrixBase(fsegs, min_fvals, fval_map, buf_wr, buf_itr, buf, ngidx, rstride) {}
};

// A sparse matrix representation in the CSR format, where it contains the exact number of items
// present in the matrix. A sparse matrix can either be a RowStrideMatrix/CSRMatrix based on
// which representation consumes less GPU memory
struct CSRMatrix : MatrixBase {
  common::CompressedBufferWriter gidx_row_writer;
  common::CompressedIterator<uint32_t> gidx_row_iter;
  common::Span<common::CompressedByteT> gidx_row_buffer;

  size_t n_rows;  // Number of rows in this matrix
  size_t n_items;  // Number of items in this matrix

  __forceinline__ __device__ bst_float GetElement(size_t ridx, size_t fidx) const override {
      auto row_begin = gidx_row_iter[ridx];
      auto row_end = gidx_row_iter[ridx + 1];
      auto gidx = BinarySearchRow(row_begin, row_end, gidx_buffer_iter, feature_segments[fidx],
                                  feature_segments[fidx + 1]);
      return (gidx == -1) ? nan("") : gidx_fvalue_map[gidx];
  }

  __forceinline__ __device__ int GetGidx(size_t ridx, size_t gidx_pos) const override {
    uint32_t n_elems = gidx_row_iter[ridx + 1] - gidx_row_iter[ridx];
    if (gidx_pos % row_stride < n_elems) {
      return gidx_buffer_iter[gidx_row_iter[ridx] + gidx_pos % row_stride];
    }
    return null_gidx_value;
  }

  __forceinline__ __device__ void Write(const CompressRowContext &com_ctx) override {
    if (com_ctx.bin_ != null_gidx_value) {
      gidx_buffer_writer.AtomicWriteSymbol(gidx_buffer.data(), com_ctx.bin_,
        com_ctx.row_offset_in_batch_ - com_ctx.base_item_offset_ +
        com_ctx.total_items_processed_ + com_ctx.ifeature_);

      // TODO(sriramch): There may be multiple writes to the row_buffer at irow + base_row
      // It should be harmless, as the writes are atomic. Explore if there is a way to avoid it,
      // as the atomic ops are needless after the first write
      gidx_row_writer.AtomicWriteSymbol(gidx_row_buffer.data(),
        com_ctx.row_offset_in_batch_ - com_ctx.base_item_offset_ + com_ctx.total_items_processed_,
        (com_ctx.irow_ + com_ctx.base_row_));

      // Write to the last element of the row index containing total number of items
      if (com_ctx.irow_ + com_ctx.base_row_ + 1 == n_rows) {
        gidx_row_writer.AtomicWriteSymbol(gidx_row_buffer.data(), n_items, n_rows);
      }
    }
  }

  __device__  explicit CSRMatrix(
    common::Span<uint32_t> fsegs, common::Span<bst_float> min_fvals,
    common::Span<bst_float> fval_map, common::CompressedBufferWriter buf_wr,
    common::CompressedIterator<uint32_t> buf_itr, common::Span<common::CompressedByteT> buf,
    int ngidx, size_t rstride, common::CompressedBufferWriter row_wr,
    common::CompressedIterator<uint32_t> row_itr, common::Span<common::CompressedByteT> row_buf,
    size_t nrows, size_t nitems)
      : MatrixBase(fsegs, min_fvals, fval_map, buf_wr, buf_itr, buf, ngidx, rstride),
        gidx_row_writer(row_wr), gidx_row_iter(row_itr), gidx_row_buffer(row_buf),
        n_rows(nrows), n_items(nitems) {}
};

template <typename BaseType, typename DerivedType, typename... Args>
__global__ void DeviceMatrixTypeCreatorKernel(BaseType **obj, Args... args) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *obj = new DerivedType(args...);
  }
}

template<typename std::enable_if<true,  int>::type = 0>
__global__ void DeviceMatrixTypeDestroyerKernel(MatrixBase **ptr) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    delete *ptr;
  }
}

/** \brief Struct for accessing and manipulating an ellpack matrix on the
 * device. Does not own underlying memory and may be trivially copied into
 * kernels.*/
struct ELLPackMatrix {
  __forceinline__ __device__ size_t BinCount() const { return (*matrix)->BinCount(); }
  __forceinline__ __device__ size_t RowStride() const { return (*matrix)->RowStride(); }
  __forceinline__ __device__ uint32_t GetFeatureBin(int fidx) const {
    return (*matrix)->GetFeatureBin(fidx);
  }
  __forceinline__ __device__ uint32_t GetMinFeatureValue(int fidx) const {
    return (*matrix)->GetMinFeatureValue(fidx);
  }
  __forceinline__ __device__ const bst_float *GetFeatureValue(int fbin) const {
    return (*matrix)->GetFeatureValue(fbin);
  }
  __forceinline__ __device__ int NullGidxValue() const { return (*matrix)->NullGidxValue(); }

  bool is_dense;  // Is the matrix dense? Kept here for tests
  CompressedDataLayout data_layout;  // Kept here for tests
  MatrixBase **matrix;  // Base matrix reference that can be handled polymorphically

  // Get a matrix element, uses binary search for look up
  // Return NaN if missing
  __forceinline__ __device__ bst_float GetElement(size_t ridx, size_t fidx) const {
    return (*matrix)->GetElement(ridx, fidx);
  }

  // Get the gidx value for row ridx and the feature at the gidx_pos in the gidx_buffer
  __forceinline__ __device__ int GetGidx(size_t ridx, size_t gidx_pos) const {
    return (*matrix)->GetGidx(ridx, gidx_pos);
  }

  __forceinline__ __device__ void Write(const CompressRowContext &com_ctx) {
    return (*matrix)->Write(com_ctx);
  }

  ELLPackMatrix(
    common::Span<uint32_t> feature_segments,
    common::Span<bst_float> min_fvalue,
    common::Span<bst_float> gidx_fvalue_map,
    common::CompressedBufferWriter buf_wr,
    common::CompressedIterator<uint32_t> buf_iter,
    common::Span<common::CompressedByteT> buf,
    common::CompressedBufferWriter row_wr,
    common::CompressedIterator<uint32_t> row_iter,
    common::Span<common::CompressedByteT> row_buf,
    size_t row_stride,
    bool is_dense,
    int null_gidx_value,
    size_t n_rows,
    size_t n_items,
    CompressedDataLayout data_layout) {
      // Allocate memory for the base type pointer on device
      dh::safe_cuda(cudaMalloc(&matrix, sizeof(MatrixBase **)));

      if (is_dense) {
        DeviceMatrixTypeCreatorKernel<MatrixBase, DenseMatrix><<<1, 1>>>(
          matrix, feature_segments, min_fvalue, gidx_fvalue_map,
          buf_wr, buf_iter, buf, null_gidx_value, row_stride);
      } else if (data_layout == CompressedDataLayout::kRowStride) {
        DeviceMatrixTypeCreatorKernel<MatrixBase, RowStrideMatrix><<<1, 1>>>(
          matrix, feature_segments, min_fvalue, gidx_fvalue_map,
          buf_wr, buf_iter, buf, null_gidx_value, row_stride);
      } else if (data_layout == CompressedDataLayout::kCSR) {
        DeviceMatrixTypeCreatorKernel<MatrixBase, CSRMatrix><<<1, 1>>>(
          matrix, feature_segments, min_fvalue, gidx_fvalue_map,
          buf_wr, buf_iter, buf, null_gidx_value, row_stride, row_wr, row_iter, row_buf,
          n_rows, n_items);
      }

      this->is_dense = is_dense;
      this->data_layout = data_layout;
  }
};

struct DeviceMatrixTypeDestroyer {
  void operator()(ELLPackMatrix *ellpack) {
    DeviceMatrixTypeDestroyerKernel<<<1, 1>>>(ellpack->matrix);
    cudaFree(ellpack->matrix);
    delete ellpack;
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
    int constraint,              // monotonic_constraints
    const ValueConstraint& value_constraint) {
  // Use pointer from cut to indicate begin and end of bins for each feature.
  uint32_t gidx_begin = matrix.GetFeatureBin(fidx);  // begining bin
  uint32_t gidx_end = matrix.GetFeatureBin(fidx + 1);  // end bin for i^th feature

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
                               constraint, value_constraint, missing_left);
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
        fvalue =  matrix.GetMinFeatureValue(fidx);
      } else {
        fvalue = *(matrix.GetFeatureValue(split_gidx));
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
    common::Span<const int> feature_set,              // Selected features
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
  dh::device_vector<typename GradientSumT::ValueT> data_;
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

  dh::device_vector<typename GradientSumT::ValueT>& Data() {
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
    ELLPackMatrix matrix,
    const size_t* __restrict__ row_ptrs,    // row offset of input data
    const Entry* __restrict__ entries,      // One batch of input data
    size_t base_row,                        // batch_row_begin
    size_t batch_nrows,                     // number of rows in the batch
    size_t base_item_offset,                // item offset from the beginning of the batch
    size_t total_items_processed            // Number of row items processed in the previous batch
    ) {
  size_t irow = threadIdx.x + blockIdx.x * blockDim.x;
  int ifeature = threadIdx.y + blockIdx.y * blockDim.y;
  if (irow >= batch_nrows || ifeature >= matrix.RowStride()) {
    return;
  }
  int row_length = static_cast<int>(row_ptrs[irow + 1] - row_ptrs[irow]);
  unsigned int bin = matrix.NullGidxValue();
  if (ifeature < row_length) {
    Entry entry = entries[row_ptrs[irow] - row_ptrs[0] + ifeature];
    int feature = entry.index;
    float fvalue = entry.fvalue;
    // {feature_cuts, ncuts} forms the array of cuts of `feature'.
    const float *feature_cuts = matrix.GetFeatureValue(matrix.GetFeatureBin(feature));
    int ncuts = matrix.GetFeatureBin(feature + 1) - matrix.GetFeatureBin(feature);
    // Assigning the bin in current entry.
    // S.t.: fvalue < feature_cuts[bin]
    bin = dh::UpperBound(feature_cuts, ncuts, fvalue);
    if (bin >= ncuts) {
      bin = ncuts - 1;
    }
    // Add the number of bins in previous features.
    bin += matrix.GetFeatureBin(feature);
  }

  // Write to gidx buffer.
  CompressRowContext comp_row_ctx(
    bin, irow, base_row, row_ptrs[irow], ifeature, base_item_offset, total_items_processed);
  matrix.Write(comp_row_ctx);
}

template <typename GradientSumT>
__global__ void SharedMemHistKernel(ELLPackMatrix matrix,
                                    common::Span<const RowPartitioner::RowIndexT> d_ridx,
                                    GradientSumT* d_node_hist,
                                    const GradientPair* d_gpair, size_t n_elements,
                                    bool use_shared_memory_histograms) {
  extern __shared__ char smem[];
  GradientSumT* smem_arr = reinterpret_cast<GradientSumT*>(smem);  // NOLINT
  if (use_shared_memory_histograms) {
    dh::BlockFill(smem_arr, matrix.BinCount(), GradientSumT());
    __syncthreads();
  }
  for (auto idx : dh::GridStrideRange(static_cast<size_t>(0), n_elements)) {
    int ridx = d_ridx[idx / matrix.RowStride() ];
    int gidx = matrix.GetGidx(ridx, idx);
    if (gidx != matrix.NullGidxValue()) {
      // If we are not using shared memory, accumulate the values directly into
      // global memory
      GradientSumT* atomic_add_ptr =
          use_shared_memory_histograms ? smem_arr : d_node_hist;
      AtomicAddGpair(atomic_add_ptr + gidx, d_gpair[ridx]);
    }
  }

  if (use_shared_memory_histograms) {
    // Write shared memory back to global memory
    __syncthreads();
    for (auto i :
         dh::BlockStrideRange(static_cast<size_t>(0), matrix.BinCount())) {
      AtomicAddGpair(d_node_hist + i, smem_arr[i]);
    }
  }
}

// Instances of this type are created while creating the histogram bins for the
// entire dataset across multiple sparse page batches. This keeps track of the number
// of rows to process from a batch and the position from which to process on each device.
struct RowStateOnDevice {
  // Number of rows assigned to this device
  const size_t total_rows_assigned_to_device;
  // Number of rows processed thus far
  size_t total_rows_processed;
  // Number of rows to process from the current sparse page batch
  size_t rows_to_process_from_batch;
  // Offset from the current sparse page batch to begin processing
  size_t row_offset_in_current_batch;
  // Total number of items processed thus far
  size_t total_items_processed;

  explicit RowStateOnDevice(size_t total_rows)
    : total_rows_assigned_to_device(total_rows), total_rows_processed(0),
      rows_to_process_from_batch(0), row_offset_in_current_batch(0),
      total_items_processed(0) {
  }

  explicit RowStateOnDevice(size_t total_rows, size_t batch_rows)
    : total_rows_assigned_to_device(total_rows), total_rows_processed(0),
      rows_to_process_from_batch(batch_rows), row_offset_in_current_batch(0),
      total_items_processed(0) {
  }

  // Advance the row state by the number of rows processed
  void Advance(const SparsePage &batch) {
    if (rows_to_process_from_batch) {
      const auto &offset_vec = batch.offset.ConstHostVector();
      total_items_processed += offset_vec[row_offset_in_current_batch + rows_to_process_from_batch]
                               - offset_vec[row_offset_in_current_batch];
    }
    total_rows_processed += rows_to_process_from_batch;
    CHECK_LE(total_rows_processed, total_rows_assigned_to_device);
    rows_to_process_from_batch = row_offset_in_current_batch = 0;
  }
};

// Manage memory for a single GPU
template <typename GradientSumT>
struct DeviceShard {
  int device_id;
  int shard_idx;  // Position in the local array of shards

  dh::BulkAllocator ba;

  std::unique_ptr<ELLPackMatrix, DeviceMatrixTypeDestroyer> ellpack_matrix;

  std::unique_ptr<RowPartitioner> row_partitioner;
  DeviceHistogram<GradientSumT> hist;

  /*! \brief row_ptr form HistogramCuts. */
  common::Span<uint32_t> feature_segments;
  /*! \brief minimum value for each feature. */
  common::Span<bst_float> min_fvalue;
  /*! \brief Cut. */
  common::Span<bst_float> gidx_fvalue_map;
  /*! \brief global index of histogram, which is stored in ELLPack format. */
  common::Span<common::CompressedByteT> gidx_buffer;
  /*! \brief for sparse matrices, where an alternate representation can save memory, this
      contains the row indices for the different entries present in gidx_buffer */
  common::Span<common::CompressedByteT> gidx_row_buffer;

  /*! \brief Gradient pair for each row. */
  common::Span<GradientPair> gpair;

  common::Span<int> monotone_constraints;
  common::Span<bst_float> prediction_cache;

  /*! \brief Sum gradient for each node. */
  std::vector<GradientPair> node_sum_gradients;
  common::Span<GradientPair> node_sum_gradients_d;
  /*! The row offset for this shard. */
  bst_uint row_begin_idx;
  bst_uint row_end_idx;
  bst_uint n_rows;
  bst_uint n_items;  // Number of items assigned to this shard
  size_t row_stride;
  int n_bins;

  TrainParam param;
  bool prediction_cache_initialised;
  bool use_shared_memory_histograms {false};

  dh::CubMemory temp_memory;
  dh::PinnedMemory pinned_memory;

  std::vector<cudaStream_t> streams;

  common::Monitor monitor;
  std::vector<ValueConstraint> node_value_constraints;
  common::ColumnSampler column_sampler;
  FeatureInteractionConstraint interaction_constraints;

  using ExpandQueue =
      std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
                          std::function<bool(ExpandEntry, ExpandEntry)>>;
  std::unique_ptr<ExpandQueue> qexpand;

  DeviceShard(int _device_id, int shard_idx, bst_uint row_begin,
              bst_uint row_end, TrainParam _param,
              uint32_t column_sampler_seed,
              uint32_t n_features)
      : device_id(_device_id),
        shard_idx(shard_idx),
        row_begin_idx(row_begin),
        row_end_idx(row_end),
        n_rows(row_end - row_begin),
        n_items(0),
        row_stride(0),
        n_bins(0),
        param(std::move(_param)),
        prediction_cache_initialised(false),
        column_sampler(column_sampler_seed),
        interaction_constraints(param, n_features) {
    monitor.Init(std::string("DeviceShard") + std::to_string(device_id));
  }

  void ComputeItemsInShard(const SparsePage &row_batch, const RowStateOnDevice &device_row_state);

  void InitCompressedData(
      const common::HistogramCuts& hmat, size_t row_stride, bool is_dense);

  void CreateHistIndices(
      const SparsePage &row_batch, const common::HistogramCuts &hmat,
      const RowStateOnDevice &device_row_state, int rows_per_batch);

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
    this->interaction_constraints.Reset();
    std::fill(node_sum_gradients.begin(), node_sum_gradients.end(),
              GradientPair());
    row_partitioner.reset();  // Release the device memory first before reallocating
    row_partitioner.reset(new RowPartitioner(device_id, n_rows));

    gpair = dh_gpair->DeviceSpan(device_id);

    SubsampleGradientPair(device_id, gpair, param.subsample, row_begin_idx);
    hist.Reset();
  }

  std::vector<DeviceSplitCandidate> EvaluateSplits(
      std::vector<int> nidxs, const RegTree& tree,
      size_t num_columns) {
    dh::safe_cuda(cudaSetDevice(device_id));
    auto result_all = pinned_memory.GetSpan<DeviceSplitCandidate>(nidxs.size());

    // Work out cub temporary memory requirement
    GPUTrainingParam gpu_param(param);
    DeviceSplitCandidateReduceOp op(gpu_param);
    size_t temp_storage_bytes = 0;
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
      auto p_feature_set = column_sampler.GetFeatureSet(tree.GetDepth(nidx));
      p_feature_set->Shard(GPUSet(device_id, 1));
      auto d_sampled_features = p_feature_set->DeviceSpan(device_id);
      common::Span<int32_t> d_feature_set =
          interaction_constraints.Query(d_sampled_features, nidx);
      auto d_split_candidates =
          d_split_candidates_all.subspan(i * num_columns, d_feature_set.size());

      DeviceNodeStats node(node_sum_gradients[nidx], nidx, param);

      auto d_result = d_result_all.subspan(i, 1);
      if (d_feature_set.size() == 0) {
        // Acting as a device side constructor for DeviceSplitCandidate.
        // DeviceSplitCandidate::IsValid is false so that ApplySplit can reject this
        // candidate.
        auto worst_candidate = DeviceSplitCandidate();
        dh::safe_cuda(cudaMemcpyAsync(d_result.data(), &worst_candidate,
                                      sizeof(DeviceSplitCandidate),
                                      cudaMemcpyHostToDevice));
        continue;
      }

      // One block for each feature
      int constexpr kBlockThreads = 256;
      EvaluateSplitKernel<kBlockThreads, GradientSumT>
          <<<uint32_t(d_feature_set.size()), kBlockThreads, 0, streams[i]>>>(
              hist.GetNodeHistogram(nidx), d_feature_set, node, *ellpack_matrix,
              gpu_param, d_split_candidates, node_value_constraints[nidx],
              monotone_constraints);

      // Reduce over features to find best feature
      auto d_cub_memory =
          d_cub_memory_all.subspan(i * cub_memory_size, cub_memory_size);
      size_t cub_bytes = d_cub_memory.size() * sizeof(DeviceSplitCandidate);
      cub::DeviceReduce::Reduce(reinterpret_cast<void*>(d_cub_memory.data()),
                                cub_bytes, d_split_candidates.data(),
                                d_result.data(), d_split_candidates.size(), op,
                                DeviceSplitCandidate(), streams[i]);
    }

    dh::safe_cuda(cudaMemcpy(result_all.data(), d_result_all.data(),
                             sizeof(DeviceSplitCandidate) * d_result_all.size(),
                             cudaMemcpyDeviceToHost));
    return std::vector<DeviceSplitCandidate>(result_all.begin(), result_all.end());
  }

  void BuildHist(int nidx) {
    hist.AllocateHistogram(nidx);
    auto d_node_hist = hist.GetNodeHistogram(nidx);

    auto d_ridx = row_partitioner->GetRows(nidx);
    if (!d_ridx.size()) return;

    auto d_gpair = gpair.data();

    auto n_elements = d_ridx.size() * row_stride;

    const size_t smem_size =
        use_shared_memory_histograms
            ? sizeof(GradientSumT) * gidx_fvalue_map.size()
            : 0;
    const int items_per_thread = 8;
    const int block_threads = 256;
    const int grid_size = static_cast<int>(
        common::DivRoundUp(n_elements, items_per_thread * block_threads));
    SharedMemHistKernel<<<grid_size, block_threads, smem_size>>>(
        *ellpack_matrix, d_ridx, d_node_hist.data(), d_gpair, n_elements,
        use_shared_memory_histograms);
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
    auto d_matrix = *ellpack_matrix;

    row_partitioner->UpdatePosition(
        nidx, split_node.LeftChild(), split_node.RightChild(),
        [=] __device__(bst_uint ridx) {
          bst_float element =
              d_matrix.GetElement(ridx, split_node.SplitIndex());
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
          return new_position;
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
    auto d_matrix = *ellpack_matrix;
    row_partitioner->FinalisePosition(
        [=] __device__(bst_uint ridx, int position) {
          auto node = d_nodes[position];

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
          return position;
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
    auto d_position = row_partitioner->GetPosition();
    auto d_ridx = row_partitioner->GetRows();
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
    row_partitioner.reset();
  }

  void AllReduceHist(int nidx, dh::AllReducer* reducer) {
    monitor.StartCuda("AllReduce");
    auto d_node_hist = hist.GetNodeHistogram(nidx).data();
    reducer->AllReduceSum(
        shard_idx,
        reinterpret_cast<typename GradientSumT::ValueT*>(d_node_hist),
        reinterpret_cast<typename GradientSumT::ValueT*>(d_node_hist),
        gidx_fvalue_map.size() *
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

    auto left_node_rows = row_partitioner->GetRows(nidx_left).size();
    auto right_node_rows = row_partitioner->GetRows(nidx_right).size();
    // Decide whether to build the left histogram or right histogram
    // Find the largest number of training instances on any given Shard
    // Assume this will be the bottleneck and avoid building this node if
    // possible
    std::vector<size_t> max_reduce;
    max_reduce.push_back(left_node_rows);
    max_reduce.push_back(right_node_rows);
    reducer->HostMaxAllReduce(&max_reduce);
    bool fewer_right = max_reduce[1] < max_reduce[0];
    if (fewer_right) {
      std::swap(build_hist_nidx, subtraction_trick_nidx);
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

    interaction_constraints.Split(candidate.nid, tree[candidate.nid].SplitIndex(),
                                  tree[candidate.nid].LeftChild(),
                                  tree[candidate.nid].RightChild());
  }

  void InitRoot(RegTree* p_tree, dh::AllReducer* reducer, int64_t num_columns) {
    constexpr int kRootNIdx = 0;

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
    this->InitRoot(p_tree, reducer, p_fmat->Info().num_col_);
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
};

template <typename GradientSumT>
inline void DeviceShard<GradientSumT>::ComputeItemsInShard(
  const SparsePage &row_batch,
  const RowStateOnDevice &device_row_state) {
  // Has any been allocated for me in this batch?
  size_t rows_to_process = device_row_state.rows_to_process_from_batch;
  if (!rows_to_process) return;
  const auto &offset_vec = row_batch.offset.ConstHostVector();
  n_items += offset_vec[device_row_state.row_offset_in_current_batch + rows_to_process] -
             + offset_vec[device_row_state.row_offset_in_current_batch];
}

template <typename GradientSumT>
inline void DeviceShard<GradientSumT>::InitCompressedData(
    const common::HistogramCuts &hmat, size_t row_stride, bool is_dense) {
  this->row_stride = row_stride;
  n_bins = hmat.Ptrs().back();
  int null_gidx_value = hmat.Ptrs().back();

  CHECK(!(param.max_leaves == 0 && param.max_depth == 0))
      << "Max leaves and max depth cannot both be unconstrained for "
      "gpu_hist.";

  int max_nodes =
      param.max_leaves > 0 ? param.max_leaves * 2 : MaxNodesDepth(param.max_depth);

  ba.Allocate(device_id,
              &prediction_cache, n_rows,
              &node_sum_gradients_d, max_nodes,
              &feature_segments, hmat.Ptrs().size(),
              &gidx_fvalue_map, hmat.Values().size(),
              &min_fvalue, hmat.MinValues().size(),
              &monotone_constraints, param.monotone_constraints.size());

  dh::CopyVectorToDeviceSpan(gidx_fvalue_map, hmat.Values());
  dh::CopyVectorToDeviceSpan(min_fvalue, hmat.MinValues());
  dh::CopyVectorToDeviceSpan(feature_segments, hmat.Ptrs());
  dh::CopyVectorToDeviceSpan(monotone_constraints, param.monotone_constraints);

  node_sum_gradients.resize(max_nodes);

  // allocate compressed bin data
  int num_symbols = n_bins + 1;
  int num_row_symbols = n_items + 1;

  CompressedDataLayout data_layout = CompressedDataLayout::kRowStride;
  // Required buffer size for storing data matrix in ELLPack format.
  size_t compressed_size_bytes =
    common::CompressedBufferWriter::CalculateBufferSize(row_stride * n_rows,
                                                        num_symbols);
  if (!is_dense) {
    size_t item_compressed_size_bytes =
      common::CompressedBufferWriter::CalculateBufferSize(n_items, num_symbols);

    // +1 for the first element in the row index that contains a value of 0
    size_t row_compressed_size_bytes =
      common::CompressedBufferWriter::CalculateBufferSize(n_rows + 1, num_row_symbols);

    if (item_compressed_size_bytes + row_compressed_size_bytes < compressed_size_bytes) {
      compressed_size_bytes = item_compressed_size_bytes;

      ba.Allocate(device_id, &gidx_row_buffer, row_compressed_size_bytes);
        thrust::fill(
          thrust::device_pointer_cast(gidx_row_buffer.data()),
          thrust::device_pointer_cast(gidx_row_buffer.data() + gidx_row_buffer.size()), 0);

      data_layout = CompressedDataLayout::kCSR;
    }
  }

  ba.Allocate(device_id, &gidx_buffer, compressed_size_bytes);
  thrust::fill(
      thrust::device_pointer_cast(gidx_buffer.data()),
      thrust::device_pointer_cast(gidx_buffer.data() + gidx_buffer.size()), 0);

  ellpack_matrix.reset(
    new ELLPackMatrix(
      feature_segments, min_fvalue, gidx_fvalue_map,
      common::CompressedBufferWriter(num_symbols),
      common::CompressedIterator<uint32_t>(gidx_buffer.data(), num_symbols), gidx_buffer,
      common::CompressedBufferWriter(num_row_symbols),
      common::CompressedIterator<uint32_t>(gidx_row_buffer.data(), num_row_symbols),
      gidx_row_buffer,
      row_stride, is_dense, null_gidx_value, n_rows, n_items, data_layout));

  // check if we can use shared memory for building histograms
  // (assuming atleast we need 2 CTAs per SM to maintain decent latency
  // hiding)
  auto histogram_size = sizeof(GradientSumT) * hmat.Ptrs().back();
  auto max_smem = dh::MaxSharedMemory(device_id);
  if (histogram_size <= max_smem) {
    use_shared_memory_histograms = true;
  }

  // Init histogram
  hist.Init(device_id, hmat.Ptrs().back());
}

template <typename GradientSumT>
inline void DeviceShard<GradientSumT>::CreateHistIndices(
    const SparsePage &row_batch,
    const common::HistogramCuts &hmat,
    const RowStateOnDevice &device_row_state,
    int rows_per_batch) {
  // Has any been allocated for me in this batch?
  if (!device_row_state.rows_to_process_from_batch) return;

  unsigned int null_gidx_value = hmat.Ptrs().back();

  const auto &offset_vec = row_batch.offset.ConstHostVector();
  size_t base_offset = offset_vec[device_row_state.row_offset_in_current_batch];

  // bin and compress entries in batches of rows
  size_t gpu_batch_nrows = std::min(
    dh::TotalMemory(device_id) / (16 * row_stride * sizeof(Entry)),
    static_cast<size_t>(device_row_state.rows_to_process_from_batch));
  const std::vector<Entry>& data_vec = row_batch.data.ConstHostVector();

  size_t gpu_nbatches = common::DivRoundUp(device_row_state.rows_to_process_from_batch,
                                           gpu_batch_nrows);

  for (size_t gpu_batch = 0; gpu_batch < gpu_nbatches; ++gpu_batch) {
    size_t batch_row_begin = gpu_batch * gpu_batch_nrows;
    size_t batch_row_end = (gpu_batch + 1) * gpu_batch_nrows;
    if (batch_row_end > device_row_state.rows_to_process_from_batch) {
      batch_row_end = device_row_state.rows_to_process_from_batch;
    }
    size_t batch_nrows = batch_row_end - batch_row_begin;

    const auto ent_cnt_begin =
      offset_vec[device_row_state.row_offset_in_current_batch + batch_row_begin];
    const auto ent_cnt_end =
      offset_vec[device_row_state.row_offset_in_current_batch + batch_row_end];

    /*! \brief row offset in SparsePage (the input data). */
    dh::device_vector<size_t> row_ptrs(batch_nrows+1);
    thrust::copy(
      offset_vec.data() + device_row_state.row_offset_in_current_batch + batch_row_begin,
      offset_vec.data() + device_row_state.row_offset_in_current_batch + batch_row_end + 1,
      row_ptrs.begin());

    // number of entries in this batch.
    size_t n_entries = ent_cnt_end - ent_cnt_begin;
    dh::device_vector<Entry> entries_d(n_entries);
    // copy data entries to device.
    dh::safe_cuda
        (cudaMemcpy
         (entries_d.data().get(), data_vec.data() + ent_cnt_begin,
          n_entries * sizeof(Entry), cudaMemcpyDefault));
    const dim3 block3(32, 8, 1);  // 256 threads
    const dim3 grid3(common::DivRoundUp(batch_nrows, block3.x),
                     common::DivRoundUp(row_stride, block3.y), 1);
    CompressBinEllpackKernel<<<grid3, block3>>>
        (*this->ellpack_matrix,
         row_ptrs.data().get(),
         entries_d.data().get(),
         device_row_state.total_rows_processed + batch_row_begin,
         batch_nrows,
         base_offset,
         device_row_state.total_items_processed);
  }
}

// An instance of this type is created which keeps track of total number of rows to process,
// rows processed thus far, rows to process and the offset from the current sparse page batch
// to begin processing on each device
class DeviceHistogramBuilderState {
 public:
  template <typename GradientSumT>
  explicit DeviceHistogramBuilderState(
    const std::vector<std::unique_ptr<DeviceShard<GradientSumT>>> &shards) {
    device_row_states_.reserve(shards.size());
    for (const auto &shard : shards) {
      device_row_states_.push_back(RowStateOnDevice(shard->n_rows));
    }
  }

  const RowStateOnDevice &GetRowStateOnDevice(int idx) const {
    return device_row_states_[idx];
  }

  // This method is invoked at the beginning of each sparse page batch. This distributes
  // the rows in the sparse page to the different devices.
  // TODO(sriramch): Think of a way to utilize *all* the GPUs to build the compressed bins.
  void BeginBatch(const SparsePage &batch) {
    size_t rem_rows = batch.Size();
    size_t row_offset_in_current_batch = 0;
    for (auto &device_row_state : device_row_states_) {
      // Do we have anymore left to process from this batch on this device?
      if (device_row_state.total_rows_assigned_to_device > device_row_state.total_rows_processed) {
        // There are still some rows that needs to be assigned to this device
        device_row_state.rows_to_process_from_batch =
          std::min(
            device_row_state.total_rows_assigned_to_device - device_row_state.total_rows_processed,
            rem_rows);
      } else {
        // All rows have been assigned to this device
        device_row_state.rows_to_process_from_batch = 0;
      }

      device_row_state.row_offset_in_current_batch = row_offset_in_current_batch;
      row_offset_in_current_batch += device_row_state.rows_to_process_from_batch;
      rem_rows -= device_row_state.rows_to_process_from_batch;
    }
  }

  // This method is invoked after completion of each sparse page batch
  void EndBatch(const SparsePage &batch) {
    for (auto &rs : device_row_states_) {
      rs.Advance(batch);
    }
  }

 private:
  std::vector<RowStateOnDevice> device_row_states_;
};

template <typename GradientSumT>
class GPUHistMakerSpecialised {
 public:
  GPUHistMakerSpecialised() : initialised_{false}, p_last_fmat_{nullptr} {}
  void Configure(const Args& args, GenericParameter const* generic_param) {
    param_.InitAllowUnknown(args);
    generic_param_ = generic_param;
    hist_maker_param_.InitAllowUnknown(args);
    auto devices = GPUSet::All(generic_param_->gpu_id,
                               generic_param_->n_gpus);
    n_devices_ = devices.Size();
    CHECK(n_devices_ != 0) << "Must have at least one device";
    dist_ = GPUDistribution::Block(devices);

    dh::CheckComputeCapability();

    monitor_.Init("updater_gpu_hist");
  }

  ~GPUHistMakerSpecialised() { dh::GlobalMemoryLogger().Log(); }

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
            new DeviceShard<GradientSumT>(dist_.Devices().DeviceId(idx), idx,
                                          start, start + size, param_,
                                          column_sampling_seed,
                                          info_->num_col_));
        });

    monitor_.StartCuda("Quantiles");
    // Create the quantile sketches for the dmatrix and initialize HistogramCuts
    size_t row_stride = common::DeviceSketch(param_, *generic_param_,
                                             hist_maker_param_.gpu_batch_nrows,
                                             dmat, &hmat_);
    monitor_.StopCuda("Quantiles");

    n_bins_ = hmat_.Ptrs().back();

    auto is_dense = info_->num_nonzero_ == info_->num_row_ * info_->num_col_;

    // Init global data for each shard
    monitor_.StartCuda("InitCompressedData");
    {
      DeviceHistogramBuilderState hist_builder_row_state(shards_);
      for (const auto &batch : dmat->GetRowBatches()) {
        hist_builder_row_state.BeginBatch(batch);

        dh::ExecuteIndexShards(
          &shards_,
          [&](int idx, std::unique_ptr<DeviceShard<GradientSumT>>& shard) {
            dh::safe_cuda(cudaSetDevice(shard->device_id));
            shard->ComputeItemsInShard(batch, hist_builder_row_state.GetRowStateOnDevice(idx));
          });

        hist_builder_row_state.EndBatch(batch);
      }
    }

    dh::ExecuteIndexShards(
        &shards_,
        [&](int idx, std::unique_ptr<DeviceShard<GradientSumT>>& shard) {
          dh::safe_cuda(cudaSetDevice(shard->device_id));
          shard->InitCompressedData(hmat_, row_stride, is_dense);
        });
    monitor_.StopCuda("InitCompressedData");

    monitor_.StartCuda("BinningCompression");
    DeviceHistogramBuilderState hist_builder_row_state(shards_);
    for (const auto &batch : dmat->GetRowBatches()) {
      hist_builder_row_state.BeginBatch(batch);

      dh::ExecuteIndexShards(
        &shards_,
        [&](int idx, std::unique_ptr<DeviceShard<GradientSumT>>& shard) {
          dh::safe_cuda(cudaSetDevice(shard->device_id));
          shard->CreateHistIndices(batch, hmat_, hist_builder_row_state.GetRowStateOnDevice(idx),
                                   hist_maker_param_.gpu_batch_nrows);
        });

      hist_builder_row_state.EndBatch(batch);
    }
    monitor_.StopCuda("BinningCompression");

    p_last_fmat_ = dmat;
    initialised_ = true;
  }

  void InitData(DMatrix* dmat) {
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
    this->InitData(p_fmat);
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
          dh::safe_cuda(cudaSetDevice(shard->device_id));
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
    bool cpu_predictor = p_out_preds->Devices().IsEmpty();
    if (!cpu_predictor) p_out_preds->Shard(dist_.Devices());
    dh::ExecuteIndexShards(
        &shards_,
        [&](int idx, std::unique_ptr<DeviceShard<GradientSumT>>& shard) {
          dh::safe_cuda(cudaSetDevice(shard->device_id));
          if (cpu_predictor) {
            size_t start = dist_.ShardStart(p_out_preds->Size(), idx);
            shard->UpdatePredictionCache(&(p_out_preds->HostVector())[start]);
          } else {
            shard->UpdatePredictionCache(
              p_out_preds->DevicePointer(shard->device_id));
          }
        });
    monitor_.StopCuda("UpdatePredictionCache");
    return true;
  }

  TrainParam param_;           // NOLINT
  common::HistogramCuts hmat_; // NOLINT
  MetaInfo* info_;             // NOLINT

  std::vector<std::unique_ptr<DeviceShard<GradientSumT>>> shards_;  // NOLINT

 private:
  bool initialised_;

  int n_devices_;
  int n_bins_;

  GPUHistMakerTrainParam hist_maker_param_;
  GenericParameter const* generic_param_;

  dh::AllReducer reducer_;

  DMatrix* p_last_fmat_;
  GPUDistribution dist_;

  common::Monitor monitor_;
  /*! List storing device id. */
  std::vector<int> device_list_;
};

class GPUHistMaker : public TreeUpdater {
 public:
  void Configure(const Args& args) override {
    hist_maker_param_.InitAllowUnknown(args);
    float_maker_.reset();
    double_maker_.reset();
    if (hist_maker_param_.single_precision_histogram) {
      float_maker_.reset(new GPUHistMakerSpecialised<GradientPair>());
      float_maker_->Configure(args, tparam_);
    } else {
      double_maker_.reset(new GPUHistMakerSpecialised<GradientPairPrecise>());
      double_maker_->Configure(args, tparam_);
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

  char const* Name() const override {
    return "gpu_hist";
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
