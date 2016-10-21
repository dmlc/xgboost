/*!
 * Copyright 2016 Rory mitchell
*/
#pragma once
#include <cub/cub.cuh>
#include <xgboost/base.h>
#include "cuda_helpers.cuh"
#include "types_functions.cuh"

namespace xgboost {
namespace tree {

struct ScanTuple {
  gpu_gpair gpair;
  NodeIdT node_id;

  __device__ ScanTuple() {}

  __device__ ScanTuple(gpu_gpair gpair, NodeIdT node_id)
      : gpair(gpair), node_id(node_id) {}

  __device__ ScanTuple operator+=(const ScanTuple &rhs) {
    if (node_id != rhs.node_id) {
      *this = rhs;
      return *this;
    } else {
      gpair += rhs.gpair;
      return *this;
    }
  }
  __device__ ScanTuple operator+(const ScanTuple &rhs) const {
    ScanTuple t = *this;
    return t += rhs;
  }
};

struct GpairTupleCallbackOp {
  // Running prefix
  ScanTuple running_total;
  // Constructor
  __device__ GpairTupleCallbackOp()
      : running_total(ScanTuple(gpu_gpair(), -1)) {}
  __device__ ScanTuple operator()(ScanTuple block_aggregate) {
    ScanTuple old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

struct GpairCallbackOp {
  // Running prefix
  gpu_gpair running_total;
  // Constructor
  __device__ GpairCallbackOp() : running_total(gpu_gpair()) {}
  __device__ gpu_gpair operator()(gpu_gpair block_aggregate) {
    gpu_gpair old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

template <int _BLOCK_THREADS, bool _DEBUG_VALIDATE>
struct FindSplitParamsSorting {
  enum {
    BLOCK_THREADS = _BLOCK_THREADS,
    TILE_ITEMS = BLOCK_THREADS,
    N_WARPS = _BLOCK_THREADS / 32,
    DEBUG_VALIDATE = _DEBUG_VALIDATE,
    ITEMS_PER_THREAD = 1
  };
};

template <int _BLOCK_THREADS, bool _DEBUG_VALIDATE> struct ReduceParamsSorting {
  enum {
    BLOCK_THREADS = _BLOCK_THREADS,
    ITEMS_PER_THREAD = 1,
    TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
    N_WARPS = _BLOCK_THREADS / 32,
    DEBUG_VALIDATE = _DEBUG_VALIDATE
  };
};

template <typename ParamsT> struct ReduceEnactorSorting {
  typedef cub::BlockScan<ScanTuple, ParamsT::BLOCK_THREADS> GpairScanT;

  struct _TempStorage {
    typename GpairScanT::TempStorage gpair_scan;
  };

  struct TempStorage : cub::Uninitialized<_TempStorage> {};

  // Thread local member variables
  gpu_gpair *d_block_node_sums;
  int *d_block_node_offsets;
  const NodeIdT *d_node_id;
  const Item *d_items;
  _TempStorage &temp_storage;
  Item item;
  NodeIdT node_id;
  NodeIdT right_node_id;
  // Contains node_id relative to the current level only
  NodeIdT node_id_adjusted;
  GpairTupleCallbackOp callback_op;
  const int level;

  __device__ __forceinline__ ReduceEnactorSorting(
      TempStorage &temp_storage, // NOLINT
      gpu_gpair *d_block_node_sums, int *d_block_node_offsets,
      const Item *d_items, const NodeIdT *d_node_id, const int level)
      : temp_storage(temp_storage.Alias()),
        d_block_node_sums(d_block_node_sums),
        d_block_node_offsets(d_block_node_offsets), d_items(d_items),
        d_node_id(d_node_id), callback_op(), level(level) {}

  __device__ __forceinline__ void ResetSumsOffsets() {
    const int max_nodes = 1 << level;

    for (auto i : block_stride_range(0, max_nodes)) {
      d_block_node_sums[i] = gpu_gpair();
      d_block_node_offsets[i] = -1;
    }
  }

  __device__ __forceinline__ void LoadTile(const bst_uint &offset,
                                           const bst_uint &num_remaining) {
    if (threadIdx.x < num_remaining) {
      item = d_items[offset + threadIdx.x];
      node_id = d_node_id[offset + threadIdx.x];
      right_node_id = threadIdx.x == num_remaining - 1
                          ? -1
                          : d_node_id[offset + threadIdx.x + 1];
      // Prevent overflow
      const int level_begin = (1 << level) - 1;
      node_id_adjusted =
          max(static_cast<int>(node_id) - level_begin, -1); // NOLINT
    }
  }

  __device__ __forceinline__ void ProcessTile(const bst_uint &offset,
                                              const bst_uint &num_remaining) {
    LoadTile(offset, num_remaining);

    ScanTuple t(item.gpair, node_id);
    GpairScanT(temp_storage.gpair_scan).InclusiveSum(t, t, callback_op);
    __syncthreads();

    // If tail of segment
    if (node_id != right_node_id && node_id_adjusted >= 0 &&
        threadIdx.x < num_remaining) {
      // Write sum
      d_block_node_sums[node_id_adjusted] = t.gpair;
      // Write offset
      d_block_node_offsets[node_id_adjusted] = offset + threadIdx.x + 1;
    }
  }

  __device__ __forceinline__ void ProcessRegion(const bst_uint &segment_begin,
                                                const bst_uint &segment_end) {
    // Current position
    bst_uint offset = segment_begin;

    ResetSumsOffsets();

    __syncthreads();

    // Process full tiles
    while (offset < segment_end) {
      ProcessTile(offset, segment_end - offset);
      offset += ParamsT::TILE_ITEMS;
    }
  }
};

template <typename ParamsT> struct FindSplitEnactorSorting {
  typedef cub::BlockScan<gpu_gpair, ParamsT::BLOCK_THREADS> GpairScanT;
  typedef cub::BlockReduce<Split, ParamsT::BLOCK_THREADS> SplitReduceT;
  typedef cub::WarpReduce<float> WarpLossReduceT;

  struct _TempStorage {
    union {
      typename GpairScanT::TempStorage gpair_scan;
      typename SplitReduceT::TempStorage split_reduce;
      typename WarpLossReduceT::TempStorage loss_reduce[ParamsT::N_WARPS];
    };
    Split warp_best_splits[ParamsT::N_WARPS];
  };

  struct TempStorage : cub::Uninitialized<_TempStorage> {};

  // Thread local member variables
  _TempStorage &temp_storage;
  gpu_gpair *d_block_node_sums;
  int *d_block_node_offsets;
  const Item *d_items;
  const NodeIdT *d_node_id;
  const Node *d_nodes;
  Item item;
  NodeIdT node_id;
  float left_fvalue;
  const GPUTrainingParam &param;
  Split *d_split_candidates_out;
  const int level;

  __device__ __forceinline__ FindSplitEnactorSorting(
      TempStorage &temp_storage, gpu_gpair *d_block_node_sums, // NOLINT
      int *d_block_node_offsets, const Item *d_items, const NodeIdT *d_node_id,
      const Node *d_nodes, const GPUTrainingParam &param,
      Split *d_split_candidates_out, const int level)
      : temp_storage(temp_storage.Alias()),
        d_block_node_sums(d_block_node_sums),
        d_block_node_offsets(d_block_node_offsets), d_items(d_items),
        d_node_id(d_node_id), d_nodes(d_nodes),
        d_split_candidates_out(d_split_candidates_out), level(level),
        param(param) {}

  __device__ __forceinline__ void LoadTile(NodeIdT node_id_adjusted,
                                           const bst_uint &node_begin,
                                           const bst_uint &offset,
                                           const bst_uint &num_remaining) {
    if (threadIdx.x < num_remaining) {
      node_id = d_node_id[offset + threadIdx.x];

      item = d_items[offset + threadIdx.x];
      bool first_item = offset + threadIdx.x == node_begin;
      left_fvalue = first_item ? item.fvalue - FVALUE_EPS
                               : d_items[offset + threadIdx.x - 1].fvalue;
    }
  }

  __device__ void PrintTileScan(int block_id, bool thread_active,
                                float loss_chg, gpu_gpair missing) {
    if (blockIdx.x != block_id) {
      return;
    }

    for (int warp = 0; warp < ParamsT::N_WARPS; warp++) {
      if (threadIdx.x / 32 == warp) {
        for (int lane = 0; lane < 32; lane++) {
          gpu_gpair g = cub::ShuffleIndex(item.gpair, lane);
          gpu_gpair missing_broadcast = cub::ShuffleIndex(missing, lane);
          float fvalue_broadcast = __shfl(item.fvalue, lane);
          bool thread_active_broadcast = __shfl(thread_active, lane);
          float loss_chg_broadcast = __shfl(loss_chg, lane);
          if (threadIdx.x == 32 * warp) {
            printf("tid %d, fvalue %1.2f, active %c, loss %1.2f, scan ",
                   threadIdx.x + lane, fvalue_broadcast,
                   thread_active_broadcast ? 'y' : 'n',
                   loss_chg_broadcast < 0.0f ? 0 : loss_chg_broadcast);
            g.print();
          }
        }
      }

      __syncthreads();
    }
  }

  __device__ __forceinline__ bool QueryUpdateWarpSplit(float loss_chg,
                                                       float warp_best_loss,
                                                       bool thread_active) {
    int warp_id = threadIdx.x / 32;
    int ballot = __ballot(loss_chg > warp_best_loss && thread_active);
    if (ballot == 0) {
      return false;
    } else {
      // Warp reduce best loss
      float best = WarpLossReduceT(temp_storage.loss_reduce[warp_id])
                       .Reduce(loss_chg, cub::Max());
      // Broadcast
      best = cub::ShuffleIndex(best, 0);

      if (loss_chg == best) {
        return true;
      }
    }

    return false;
  }

  __device__ __forceinline__ bool LeftmostFvalue() {
    return item.fvalue != left_fvalue;
  }

  __device__ __forceinline__ void
  EvaluateSplits(const NodeIdT &node_id_adjusted, const bst_uint &node_begin,
                 const bst_uint &offset, const bst_uint &num_remaining) {
    bool thread_active = LeftmostFvalue() && threadIdx.x < num_remaining &&
                         node_id_adjusted >= 0 && node_id >= 0;

    Node n = thread_active ? d_nodes[node_id] : Node();
    gpu_gpair missing =
        thread_active ? n.sum_gradients - d_block_node_sums[node_id_adjusted]
                      : gpu_gpair();

    bool missing_left;
    float loss_chg =
        thread_active ? loss_chg_missing(item.gpair, missing, n.sum_gradients,
                                         n.root_gain, param, missing_left)
                      : -FLT_MAX;

    int warp_id = threadIdx.x / 32;
    volatile float warp_best_loss =
        temp_storage.warp_best_splits[warp_id].loss_chg;

    if (QueryUpdateWarpSplit(loss_chg, warp_best_loss, thread_active)) {
      float fvalue_split = (item.fvalue + left_fvalue) / 2.0f;

      gpu_gpair left_sum = item.gpair;
      if (missing_left) {
        left_sum += missing;
      }
      gpu_gpair right_sum = n.sum_gradients - left_sum;
      temp_storage.warp_best_splits[warp_id].Update(loss_chg, missing_left,
                                                    fvalue_split, blockIdx.x,
                                                    left_sum, right_sum, param);
    }
  }

  __device__ __forceinline__ void
  ProcessTile(const NodeIdT &node_id_adjusted, const bst_uint &node_begin,
              const bst_uint &offset, const bst_uint &num_remaining,
              GpairCallbackOp &callback_op) { // NOLINT
    LoadTile(node_id_adjusted, node_begin, offset, num_remaining);

    // Scan gpair
    const bool thread_active = threadIdx.x < num_remaining && node_id >= 0;
    GpairScanT(temp_storage.gpair_scan)
        .ExclusiveSum(thread_active ? item.gpair : gpu_gpair(), item.gpair,
                      callback_op);
    __syncthreads();
    // Evaluate split
    EvaluateSplits(node_id_adjusted, node_begin, offset, num_remaining);
  }

  __device__ __forceinline__ void ResetWarpSplits() {
    if (threadIdx.x < ParamsT::N_WARPS) {
      temp_storage.warp_best_splits[threadIdx.x] = Split();
    }
  }

  __device__ __forceinline__ void
  WriteBestSplit(const NodeIdT &node_id_adjusted) {
    if (threadIdx.x < 32) {
      bool active = threadIdx.x < ParamsT::N_WARPS;
      float warp_loss =
          active ? temp_storage.warp_best_splits[threadIdx.x].loss_chg
                 : -FLT_MAX;
      if (QueryUpdateWarpSplit(warp_loss, 0, active)) {
        const int max_nodes = 1 << level;
        d_split_candidates_out[blockIdx.x * max_nodes + node_id_adjusted] =
            temp_storage.warp_best_splits[threadIdx.x];
      }
    }
  }

  __device__ __forceinline__ void ProcessNode(const NodeIdT &node_id_adjusted,
                                              const bst_uint &node_begin,
                                              const bst_uint &node_end) {
    ResetWarpSplits();

    GpairCallbackOp callback_op = GpairCallbackOp();

    bst_uint offset = node_begin;

    while (offset < node_end) {
      ProcessTile(node_id_adjusted, node_begin, offset, node_end - offset,
                  callback_op);
      offset += ParamsT::TILE_ITEMS;
      __syncthreads();
    }

    WriteBestSplit(node_id_adjusted);
  }

  __device__ __forceinline__ void ResetSplitCandidates() {
    const int max_nodes = 1 << level;
    const int begin = blockIdx.x * max_nodes;
    const int end = begin + max_nodes;

    for (auto i : block_stride_range(begin, end)) {
      d_split_candidates_out[i] = Split();
    }
  }

  __device__ __forceinline__ void ProcessFeature(const bst_uint &segment_begin,
                                                 const bst_uint &segment_end) {
    ResetSplitCandidates();

    int node_begin = segment_begin;

    const int max_nodes = 1 << level;

    // Iterate through nodes
    int active_nodes = 0;
    for (int i = 0; i < max_nodes; i++) {
      int node_end = d_block_node_offsets[i];

      if (node_end == -1) {
        continue;
      }

      active_nodes++;

      ProcessNode(i, node_begin, node_end);

      __syncthreads();

      node_begin = node_end;
    }
  }
};

template <typename ReduceParamsT, typename FindSplitParamsT>
__global__ __launch_bounds__(1024, 1) void find_split_candidates_sorted_kernel(
    const Item *d_items, Split *d_split_candidates_out,
    const NodeIdT *d_node_id, const Node *d_nodes, bst_uint num_items,
    const int num_features, const int *d_feature_offsets,
    gpu_gpair *d_node_sums, int *d_node_offsets, const GPUTrainingParam param,
    const int level) {

  if (num_items <= 0) {
    return;
  }

  bst_uint segment_begin = d_feature_offsets[blockIdx.x];
  bst_uint segment_end = d_feature_offsets[blockIdx.x + 1];

  typedef ReduceEnactorSorting<ReduceParamsT> ReduceT;
  typedef FindSplitEnactorSorting<FindSplitParamsT> FindSplitT;

  __shared__ union {
    typename ReduceT::TempStorage reduce;
    typename FindSplitT::TempStorage find_split;
  } temp_storage;


  const int max_modes_level = 1 << level;
  gpu_gpair *d_block_node_sums = d_node_sums + blockIdx.x * max_modes_level;
  int *d_block_node_offsets = d_node_offsets + blockIdx.x * max_modes_level;

  ReduceT(temp_storage.reduce, d_block_node_sums, d_block_node_offsets, d_items,
          d_node_id, level)
      .ProcessRegion(segment_begin, segment_end);
  __syncthreads();

  FindSplitT(temp_storage.find_split, d_block_node_sums, d_block_node_offsets,
             d_items, d_node_id, d_nodes, param, d_split_candidates_out, level)
      .ProcessFeature(segment_begin, segment_end);
}

void find_split_candidates_sorted(
    const Item *d_items, Split *d_split_candidates, const NodeIdT *d_node_id,
    Node *d_nodes, bst_uint num_items, int num_features,
    const int *d_feature_offsets, gpu_gpair *d_node_sums, int *d_node_offsets,
    const GPUTrainingParam param, const int level) {

  const int BLOCK_THREADS = 512;

  CHECK(BLOCK_THREADS / 32 < 32) << "Too many active warps.";

  typedef FindSplitParamsSorting<BLOCK_THREADS, false> find_split_params;
  typedef ReduceParamsSorting<BLOCK_THREADS, false> reduce_params;
  int grid_size = num_features;

  find_split_candidates_sorted_kernel<
      reduce_params, find_split_params><<<grid_size, BLOCK_THREADS>>>(
      d_items, d_split_candidates, d_node_id, d_nodes, num_items, num_features,
      d_feature_offsets, d_node_sums, d_node_offsets, param, level);

  safe_cuda(cudaGetLastError());
  safe_cuda(cudaDeviceSynchronize());
}
}  // namespace tree
}  // namespace xgboost
