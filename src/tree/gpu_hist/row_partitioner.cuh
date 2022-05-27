/*!
 * Copyright 2017-2022 XGBoost contributors
 */
#pragma once
#include <limits>
#include <vector>
#include "xgboost/base.h"
#include "../../common/device_helpers.cuh"
#include "xgboost/generic_parameters.h"
#include "xgboost/task.h"
#include "xgboost/tree_model.h"
#include <thrust/execution_policy.h>

namespace xgboost {
namespace tree {

  /** \brief Used to demarcate a contiguous set of row indices associated with
 * some tree node. */
struct Segment {
  size_t begin{0};
  size_t end{0};

  Segment() = default;

  Segment(size_t begin, size_t end) : begin(begin), end(end) { CHECK_GE(end, begin); }
  __host__ __device__ size_t Size() const { return end - begin; }
};

template <typename OpDataT>
struct KernelBatchArgs {
  static const int kMaxBatch = 32;
  Segment segments[kMaxBatch];
  OpDataT data[kMaxBatch];

  KernelBatchArgs() = default;
  // Given a global thread idx, assign it to an item from one of the segments
  __device__ void AssignBatch(std::size_t idx, int16_t& batch_idx, std::size_t& item_idx) const {
    std::size_t sum = 0;
    for (int16_t i = 0; i < kMaxBatch; i++) {
      if (sum + segments[i].Size() > idx) {
        batch_idx = i;
        item_idx = (idx - sum) + segments[i].begin;
        break;
      }
      sum += segments[i].Size();
    }
  }
  std::size_t TotalRows() const {
    std::size_t total_rows = 0;
    for (auto segment : segments) {
      total_rows += segment.Size();
    }
    return total_rows;
  }
};

// We can scan over this tuple, where the scan gives us information on how to partition inputs
// according to the flag
struct IndexFlagTuple {
  bst_uint idx;            // The location of the item we are working on in ridx_
  bst_uint flag_scan;      // This gets populated after scanning
  bst_uint segment_start;  // Start offset of this node segment
  int16_t batch_idx;       // Which node in the batch does this item belong to
  bool flag;               // Result of op (is this item going left?)
};

struct IndexFlagOp {
  __device__ IndexFlagTuple operator()(const IndexFlagTuple& a, const IndexFlagTuple& b) const {
    // Segmented scan - resets if we cross batch boundaries
    if (a.batch_idx == b.batch_idx) {
      // Accumulate the flags, everything else stays the same
      return {b.idx, a.flag_scan + b.flag_scan, b.segment_start, b.batch_idx, b.flag};
    } else {
      return b;
    }
  }
};


/*! \brief Count how many rows are assigned to left node. */
__forceinline__ __device__ void AtomicIncrement(unsigned long long* d_counts, bool increment,
                                                int batch_idx) {
  int mask = __activemask();
  int leader = __ffs(mask) - 1;
  bool group_is_contiguous = __all_sync(mask, batch_idx == __shfl_sync(mask, batch_idx, leader));
  // If all threads here are working on the same node
  // we can do a more efficient reduction with warp intrinsics
  if (group_is_contiguous) {
    unsigned ballot = __ballot_sync(mask, increment);
    if (threadIdx.x % 32 == leader) {
      atomicAdd(d_counts + batch_idx,  // NOLINT
                __popc(ballot));   // NOLINT
    }
  } else {
    atomicAdd(d_counts + batch_idx, increment);
  }
}

template <typename RowIndexT,typename OpT, typename OpDataT>
__device__ __forceinline__ IndexFlagTuple GetTuple(std::size_t idx, common::Span<RowIndexT> ridx, common::Span<unsigned long long int> d_left_counts,const KernelBatchArgs<OpDataT> &args, OpT op){
    int16_t batch_idx;
    std::size_t item_idx;
    args.AssignBatch(idx, batch_idx, item_idx);
    auto op_res = op(ridx[item_idx], args.data[batch_idx]);
    AtomicIncrement(d_left_counts.data(), op_res, batch_idx);
    return IndexFlagTuple{bst_uint(item_idx), op_res,
                                  bst_uint(args.segments[batch_idx].begin), batch_idx, op_res};
}

template <int kBlockSize, typename RowIndexT, typename OpT, typename OpDataT>
__global__ __launch_bounds__(kBlockSize) void GetLeftCountsKernel(
    const KernelBatchArgs<OpDataT> args, common::Span<RowIndexT> ridx,
    common::Span<IndexFlagTuple> scan_inputs, common::Span<unsigned long long int> d_left_counts,
    OpT op, std::size_t n) {
  // Load this large struct in shared memory
  // if left to its own devices the compiler loads this very slowly
  //__shared__ KernelBatchArgs<OpDataT> s_args;
  /*
  __shared__ cub::Uninitialized<KernelBatchArgs<OpDataT>> s_temp;
  KernelBatchArgs<OpDataT>& s_args = s_temp.Alias();
  for (int i = threadIdx.x; i < sizeof(KernelBatchArgs<OpDataT>) / 4; i += kBlockSize) {
    reinterpret_cast<int*>(&s_args)[i] = reinterpret_cast<const int*>(&args)[i];
  }
  
  __syncthreads();
  */

  // Global writes of IndexFlagTuple are inefficient due to its 16b size
  // we can use cub to optimise this
  static_assert(sizeof(IndexFlagTuple) == 16, "Expected IndexFlagTuple to be 16 bytes.");
  constexpr int kTupleWords = sizeof(IndexFlagTuple)/sizeof(int);
  typedef cub::BlockStore<int, kBlockSize, kTupleWords, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;
  __shared__ typename BlockStoreT::TempStorage temp_storage;

  // Use the raw pointer because the performance of global writes matters here
  // We don't really need the bounds checking
  IndexFlagTuple* out_ptr = scan_inputs.data();

  /*
  auto get_tuple = [&]__device__ (auto idx){
    int16_t batch_idx;
    std::size_t item_idx;
    s_args.AssignBatch(idx, batch_idx, item_idx);
    auto op_res = op(ridx[item_idx], s_args.data[batch_idx]);
    AtomicIncrement(d_left_counts.data(), op_res, batch_idx);
    return IndexFlagTuple{bst_uint(item_idx), op_res,
                                  bst_uint(s_args.segments[batch_idx].begin), batch_idx, op_res};
  };
  */

  // Process full tiles
  std::size_t tile_offset = blockIdx.x * kBlockSize;
  while (tile_offset + kBlockSize <= n) {
    std::size_t idx = tile_offset + threadIdx.x;
    //auto tuple = get_tuple(idx);
    auto tuple = GetTuple(idx,ridx,d_left_counts,args,op);
    auto block_write_ptr = reinterpret_cast<int*>(out_ptr + tile_offset);
    BlockStoreT(temp_storage).Store(block_write_ptr, *static_cast<int(*)[kTupleWords]>(static_cast<void*>(&tuple)));
    tile_offset += kBlockSize * gridDim.x;
  }

  // Process partial tile
  if (tile_offset < n) {
    // Make sure we don't compute a negative number with unsigned integers
    int valid_items = int(int64_t(n) - int64_t(tile_offset));
    std::size_t idx = tile_offset + threadIdx.x;
    IndexFlagTuple tuple;
    if (idx < n) {
      tuple = GetTuple(idx,ridx,d_left_counts,args,op);
      //tuple = get_tuple(idx);
    }

    auto block_write_ptr = reinterpret_cast<int*>(out_ptr + tile_offset);
    BlockStoreT(temp_storage)
        .Store(block_write_ptr, *static_cast<int(*)[kTupleWords]>(static_cast<void*>(&tuple)),
               valid_items * kTupleWords);
  }
}

template <typename RowIndexT, typename OpT, typename OpDataT>
void GetLeftCounts(const KernelBatchArgs<OpDataT>& args, common::Span<RowIndexT> ridx,
                   common::Span<IndexFlagTuple> scan_inputs,
                   common::Span<unsigned long long int> d_left_counts, OpT op) {
  // Launch 1 thread for each row
  constexpr int kBlockSize = 256;
  const int grid_size = 
      std::max(256,static_cast<int>(xgboost::common::DivRoundUp(args.TotalRows(), kBlockSize)));


  GetLeftCountsKernel<kBlockSize><<<grid_size,kBlockSize>>>(args, ridx, scan_inputs,d_left_counts,op, args.TotalRows());

/*
  dh::LaunchN<1, kBlockSize>(args.TotalRows(), [=] __device__(std::size_t idx) {
    __shared__ KernelBatchArgs<OpDataT> s_args;

    for (int i = threadIdx.x; i < sizeof(KernelBatchArgs<OpDataT>); i += kBlockSize) {
      reinterpret_cast<char*>(&s_args)[i] = reinterpret_cast<const char*>(&args)[i];
    }
    __syncthreads();

    // Assign this thread to a row
    int16_t batch_idx;
    std::size_t item_idx;
    s_args.AssignBatch(idx, batch_idx, item_idx);
    auto op_res = op(ridx[item_idx], s_args.data[batch_idx]);
    scan_inputs[idx] = IndexFlagTuple{bst_uint(item_idx), op_res,
                                      bst_uint(args.segments[batch_idx].begin), batch_idx, op_res};

    AtomicIncrement(d_left_counts.data(), op(ridx[item_idx], s_args.data[batch_idx]), batch_idx);
  });
  */
}

// This is a transformer output iterator
// It takes the result of the scan and performs the partition
// To understand how a scan is used to partition elements see:
// Harris, Mark, Shubhabrata Sengupta, and John D. Owens. "Parallel prefix sum (scan) with CUDA."
// GPU gems 3.39 (2007): 851-876.
struct WriteResultsFunctor {
  bst_uint* ridx_in;
  bst_uint* ridx_out;
  unsigned long long int* left_counts;

  __device__ IndexFlagTuple operator()(const IndexFlagTuple& x) {
    // the ex_scan_result represents how many rows have been assigned to left
    // node so far during scan.
    std::size_t scatter_address;
    if (x.flag) {
      scatter_address = x.segment_start + x.flag_scan - 1;  // -1 because inclusive scan
    } else {
      // current number of rows belong to right node + total number of rows
      // belong to left node
      scatter_address  = (x.idx - x.flag_scan) + left_counts[x.batch_idx];
    }
    ridx_out[scatter_address] = ridx_in[x.idx];
    // Discard
    return {};
  }
};

template <typename RowIndexT, typename OpDataT>
void SortPositionBatch(const KernelBatchArgs<OpDataT>& args, common::Span<RowIndexT> ridx,
                       common::Span<RowIndexT> ridx_tmp, common::Span<IndexFlagTuple> scan_inputs,
                       common::Span<unsigned long long int> left_counts,
                       cudaStream_t stream) {
  static_assert(sizeof(IndexFlagTuple) == 16, "Struct should be 16 bytes aligned.");
  WriteResultsFunctor write_results{ridx.data(), ridx_tmp.data(), left_counts.data()};

  auto discard_write_iterator =
      thrust::make_transform_output_iterator(dh::TypedDiscard<IndexFlagTuple>(), write_results);
  auto counting = thrust::make_counting_iterator(0llu);
  size_t temp_bytes = 0;
  cub::DeviceScan::InclusiveScan(nullptr, temp_bytes, scan_inputs.data(),
                                 discard_write_iterator, IndexFlagOp(),
                                 args.TotalRows(), stream);
  dh::TemporaryArray<int8_t> temp(temp_bytes);
  cub::DeviceScan::InclusiveScan(temp.data().get(), temp_bytes,  scan_inputs.data(),
                                 discard_write_iterator, IndexFlagOp(), args.TotalRows(), stream);

  // copy active segments back to original buffer
  dh::LaunchN(args.TotalRows(), stream, [=] __device__(std::size_t idx) {
    auto item_idx = scan_inputs[idx].idx;
    ridx[item_idx] = ridx_tmp[item_idx];
  });
}

/** \brief Class responsible for tracking subsets of rows as we add splits and
 * partition training rows into different leaf nodes. */
class RowPartitioner {
 public:
  using RowIndexT = bst_uint;

 private:
  int device_idx_;
  /*! \brief In here if you want to find the rows belong to a node nid, first you need to
   * get the indices segment from ridx_segments[nid], then get the row index that
   * represents position of row in input data X.  `RowPartitioner::GetRows` would be a
   * good starting place to get a sense what are these vector storing.
   *
   * node id -> segment -> indices of rows belonging to node
   */
  /*! \brief Range of row index for each node, pointers into ridx below. */
  std::vector<Segment> ridx_segments_;
  /*! \brief mapping for node id -> rows.
   * This looks like:
   * node id  |    1    |    2   |
   * rows idx | 3, 5, 1 | 13, 31 |
   */
  dh::TemporaryArray<RowIndexT> ridx_;
  // Staging area for sorting ridx
  dh::TemporaryArray<RowIndexT> ridx_tmp_;
  dh::TemporaryArray<IndexFlagTuple> scan_inputs_;
  dh::PinnedMemory pinned_;
  std::vector<cudaStream_t> streams_;

 public:
  RowPartitioner(int device_idx, size_t num_rows);
  ~RowPartitioner();
  RowPartitioner(const RowPartitioner&) = delete;
  RowPartitioner& operator=(const RowPartitioner&) = delete;

  /**
   * \brief Gets the row indices of training instances in a given node.
   */
  common::Span<const RowIndexT> GetRows(bst_node_t nidx);

  /**
   * \brief Gets all training rows in the set.
   */
  common::Span<const RowIndexT> GetRows();

  /**
   * \brief Convenience method for testing
   */
  std::vector<RowIndexT> GetRowsHost(bst_node_t nidx);

  template <typename UpdatePositionOpT, typename OpDataT>
  void UpdatePositionBatch(const std::vector<bst_node_t>& nidx,
                           const std::vector<bst_node_t>& left_nidx,
                           const std::vector<bst_node_t>& right_nidx,
                           const std::vector<OpDataT>& op_data, UpdatePositionOpT op) {
    if (nidx.empty()) return;
    CHECK_EQ(nidx.size(), left_nidx.size());
    CHECK_EQ(nidx.size(), right_nidx.size());
    CHECK_EQ(nidx.size(), op_data.size());

    // Process nodes in batches to amortise the fixed latency costs of launching kernels and copying
    // memory from device to host
    for (std::size_t batch_start = 0; batch_start < nidx.size();
         batch_start += KernelBatchArgs<OpDataT>::kMaxBatch) {
      // Temporary arrays
      auto h_left_counts = pinned_.GetSpan<int64_t>(KernelBatchArgs<OpDataT>::kMaxBatch, 0);
      dh::TemporaryArray<unsigned long long int> d_left_counts(KernelBatchArgs<OpDataT>::kMaxBatch, 0);

      std::size_t batch_end = std::min(batch_start + KernelBatchArgs<OpDataT>::kMaxBatch, nidx.size());
      // Prepare kernel arguments
      KernelBatchArgs<OpDataT> args;
      std::copy(op_data.begin() + batch_start, op_data.begin() + batch_end, args.data);
      for (int i = 0; i < (batch_end - batch_start); i++) {
        args.segments[i] = ridx_segments_.at(nidx[batch_start + i]);
      }

      // Evaluate the operator for each row, where true means 'go left'
      // Store the result of the operator for the next step
      // Count the number of rows going left, store in d_left_counts
      GetLeftCounts(args, dh::ToSpan(ridx_), dh::ToSpan(scan_inputs_), dh::ToSpan(d_left_counts), op);

      // Start copying the counts to the host
      // We overlap this transfer with the sort step using streams
      // We only need the result after sorting to update the segment boundaries
      dh::safe_cuda(
          cudaMemcpyAsync(h_left_counts.data(), d_left_counts.data().get(),
                          sizeof(decltype(d_left_counts)::value_type) * d_left_counts.size(),
                          cudaMemcpyDefault, streams_[0]));

      // Partition the rows according to the operator
      SortPositionBatch(args, dh::ToSpan(ridx_), dh::ToSpan(ridx_tmp_), dh::ToSpan(scan_inputs_),
                        dh::ToSpan(d_left_counts), streams_[1]);

      dh::safe_cuda(cudaStreamSynchronize(streams_[0]));

      // Update segments
      for (int i = 0; i < (batch_end - batch_start); i++) {
        auto segment = ridx_segments_.at(nidx[batch_start + i]);
        auto left_count = h_left_counts[i];
        CHECK_LE(left_count, segment.Size());
        CHECK_GE(left_count, 0);
        ridx_segments_.resize(
            std::max(static_cast<bst_node_t>(ridx_segments_.size()),
                     std::max(left_nidx[batch_start + i], right_nidx[batch_start + i]) + 1));
        ridx_segments_[left_nidx[batch_start + i]] =
            Segment(segment.begin, segment.begin + left_count);
        ridx_segments_[right_nidx[batch_start + i]] =
            Segment(segment.begin + left_count, segment.end);
      }
    }
  }
};
};  // namespace tree
};  // namespace xgboost
