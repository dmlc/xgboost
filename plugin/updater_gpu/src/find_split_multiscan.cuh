/*!
 * Copyright 2016 Rory mitchell
*/
#pragma once
#include <cub/cub.cuh>
#include <xgboost/base.h>
#include "device_helpers.cuh"
#include "gpu_data.cuh"
#include "types.cuh"
#include "common.cuh"

namespace xgboost {
namespace tree {

typedef uint64_t BitFlagSet;

__device__ __inline__ void set_bit(BitFlagSet &bf, int index) { // NOLINT
  bf |= 1 << index;
}

__device__ __inline__ bool check_bit(BitFlagSet bf, int index) {
  return (bf >> index) & 1;
}

// Carryover prefix for scanning multiple tiles of bit flags
struct FlagPrefixCallbackOp {
  BitFlagSet tile_carry;

  __device__ FlagPrefixCallbackOp() : tile_carry(0) {}

  __device__ BitFlagSet operator()(BitFlagSet block_aggregate) {
    BitFlagSet old_prefix = tile_carry;
    tile_carry |= block_aggregate;
    return old_prefix;
  }
};

// Scan op for bit flags that resets if the final bit is set
struct FlagScanOp {
  __device__ __forceinline__ BitFlagSet operator()(const BitFlagSet &a,
                                                   const BitFlagSet &b) {
    if (check_bit(b, 63)) {
      return b;
    } else {
      return a | b;
    }
  }
};

template <int _BLOCK_THREADS, int _N_NODES, bool _DEBUG_VALIDATE>
struct FindSplitParamsMultiscan {
  enum {
    BLOCK_THREADS = _BLOCK_THREADS,
    TILE_ITEMS = BLOCK_THREADS,
    N_NODES = _N_NODES,
    N_WARPS = _BLOCK_THREADS / 32,
    DEBUG_VALIDATE = _DEBUG_VALIDATE,
    ITEMS_PER_THREAD = 1
  };
};

template <int _BLOCK_THREADS, int _N_NODES, bool _DEBUG_VALIDATE>
struct ReduceParamsMultiscan {
  enum {
    BLOCK_THREADS = _BLOCK_THREADS,
    ITEMS_PER_THREAD = 1,
    TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
    N_NODES = _N_NODES,
    N_WARPS = _BLOCK_THREADS / 32,
    DEBUG_VALIDATE = _DEBUG_VALIDATE
  };
};

template <typename ParamsT> struct ReduceEnactorMultiscan {
  typedef cub::WarpReduce<gpu_gpair> WarpReduceT;

  struct _TempStorage {
    typename WarpReduceT::TempStorage warp_reduce[ParamsT::N_WARPS];
    gpu_gpair partial_sums[ParamsT::N_NODES][ParamsT::N_WARPS];
  };

  struct TempStorage : cub::Uninitialized<_TempStorage> {};

  struct _Reduction {
    gpu_gpair node_sums[ParamsT::N_NODES];
  };

  struct Reduction : cub::Uninitialized<_Reduction> {};

  // Thread local member variables
  const ItemIter item_iter;
  _TempStorage &temp_storage;
  _Reduction &reduction;
  gpu_gpair gpair;
  NodeIdT node_id;
  NodeIdT node_id_adjusted;
  const int node_begin;

  __device__ __forceinline__
  ReduceEnactorMultiscan(TempStorage &temp_storage, // NOLINT
                         Reduction &reduction,      // NOLINT
                         const ItemIter item_iter, const int node_begin)
      : temp_storage(temp_storage.Alias()), reduction(reduction.Alias()),
        item_iter(item_iter), node_begin(node_begin) {}

  __device__ __forceinline__ void ResetPartials() {
    if (threadIdx.x < ParamsT::N_WARPS) {
      for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++) {
        temp_storage.partial_sums[NODE][threadIdx.x] = gpu_gpair();
      }
    }
  }

  __device__ __forceinline__ void ResetReduction() {
    if (threadIdx.x < ParamsT::N_NODES) {
      reduction.node_sums[threadIdx.x] = gpu_gpair();
    }
  }

  __device__ __forceinline__ void LoadTile(const bst_uint &offset,
                                           const bst_uint &num_remaining) {
    if (threadIdx.x < num_remaining) {
      bst_uint i = offset + threadIdx.x;
      gpair = thrust::get<0>(item_iter[i]);
      // gpair = d_items[offset + threadIdx.x].gpair;
      // node_id = d_node_id[offset + threadIdx.x];
      node_id = thrust::get<2>(item_iter[i]);
      node_id_adjusted = node_id - node_begin;
    } else {
      gpair = gpu_gpair();
      node_id = -1;
      node_id_adjusted = -1;
    }
  }

  __device__ __forceinline__ void ProcessTile(const bst_uint &offset,
                                              const bst_uint &num_remaining) {
    LoadTile(offset, num_remaining);

    // Warp synchronous reduction
    for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++) {
      bool active = node_id_adjusted == NODE;

      unsigned int ballot = __ballot(active);

      int warp_id = threadIdx.x / 32;
      int lane_id = threadIdx.x % 32;

      if (ballot == 0) {
        continue;
      } else if (__popc(ballot) == 1) {
        if (active) {
          temp_storage.partial_sums[NODE][warp_id] += gpair;
        }
      } else {
        gpu_gpair sum = WarpReduceT(temp_storage.warp_reduce[warp_id])
                            .Sum(active ? gpair : gpu_gpair());
        if (lane_id == 0) {
          temp_storage.partial_sums[NODE][warp_id] += sum;
        }
      }
    }
  }

  __device__ __forceinline__ void ReducePartials() {
    // Use single warp to reduce partials
    if (threadIdx.x < 32) {
      for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++) {
        gpu_gpair sum =
            WarpReduceT(temp_storage.warp_reduce[0])
                .Sum(threadIdx.x < ParamsT::N_WARPS
                         ? temp_storage.partial_sums[NODE][threadIdx.x]
                         : gpu_gpair());

        if (threadIdx.x == 0) {
          reduction.node_sums[NODE] = sum;
        }
      }
    }
  }

  __device__ __forceinline__ void ProcessRegion(const bst_uint &segment_begin,
                                                const bst_uint &segment_end) {
    // Current position
    bst_uint offset = segment_begin;

    ResetReduction();
    ResetPartials();

    __syncthreads();

    // Process full tiles
    while (offset < segment_end) {
      ProcessTile(offset, segment_end - offset);
      offset += ParamsT::TILE_ITEMS;
    }

    __syncthreads();

    ReducePartials();

    __syncthreads();
  }
};

template <typename ParamsT, typename ReductionT>
struct FindSplitEnactorMultiscan {
  typedef cub::BlockScan<BitFlagSet, ParamsT::BLOCK_THREADS> FlagsBlockScanT;

  typedef cub::WarpReduce<Split> WarpSplitReduceT;

  typedef cub::WarpReduce<float> WarpReduceT;

  typedef cub::WarpScan<gpu_gpair> WarpScanT;

  struct _TempStorage {
    union {
      typename WarpSplitReduceT::TempStorage warp_split_reduce;
      typename FlagsBlockScanT::TempStorage flags_scan;
      typename WarpScanT::TempStorage warp_gpair_scan[ParamsT::N_WARPS];
      typename WarpReduceT::TempStorage warp_reduce[ParamsT::N_WARPS];
    };

    Split warp_best_splits[ParamsT::N_NODES][ParamsT::N_WARPS];
    gpu_gpair partial_sums[ParamsT::N_NODES][ParamsT::N_WARPS];
    gpu_gpair top_level_sum[ParamsT::N_NODES];  // Sum of current partial sums
    gpu_gpair tile_carry[ParamsT::N_NODES];    // Contains top-level sums from
                                               // previous tiles
    Split best_splits[ParamsT::N_NODES];
    // Cache current level nodes into shared memory
    float node_root_gain[ParamsT::N_NODES];
    gpu_gpair node_parent_sum[ParamsT::N_NODES];
  };

  struct TempStorage : cub::Uninitialized<_TempStorage> {};

  // Thread local member variables
  const ItemIter item_iter;
  Split *d_split_candidates_out;
  const Node *d_nodes;
  _TempStorage &temp_storage;
  gpu_gpair gpair;
  float fvalue;
  NodeIdT node_id;
  NodeIdT node_id_adjusted;
  const NodeIdT node_begin;
  const GPUTrainingParam &param;
  const ReductionT &reduction;
  const int level;
  FlagPrefixCallbackOp flag_prefix_op;

  __device__ __forceinline__ FindSplitEnactorMultiscan(
      TempStorage &temp_storage, const ItemIter item_iter, // NOLINT
      Split *d_split_candidates_out, const Node *d_nodes,
      const NodeIdT node_begin, const GPUTrainingParam &param,
      const ReductionT reduction, const int level)
      : temp_storage(temp_storage.Alias()), item_iter(item_iter),
        d_split_candidates_out(d_split_candidates_out), d_nodes(d_nodes),
        node_begin(node_begin), param(param), reduction(reduction),
        level(level), flag_prefix_op() {}

  __device__ __forceinline__ void UpdateTileCarry() {
    if (threadIdx.x < ParamsT::N_NODES) {
      temp_storage.tile_carry[threadIdx.x] +=
          temp_storage.top_level_sum[threadIdx.x];
    }
  }

  __device__ __forceinline__ void ResetTileCarry() {
    if (threadIdx.x < ParamsT::N_NODES) {
      temp_storage.tile_carry[threadIdx.x] = gpu_gpair();
    }
  }

  __device__ __forceinline__ void ResetPartials() {
    if (threadIdx.x < ParamsT::N_WARPS) {
      for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++) {
        temp_storage.partial_sums[NODE][threadIdx.x] = gpu_gpair();
      }
    }

    if (threadIdx.x < ParamsT::N_NODES) {
      temp_storage.top_level_sum[threadIdx.x] = gpu_gpair();
    }
  }

  __device__ __forceinline__ void ResetSplits() {
    if (threadIdx.x < ParamsT::N_WARPS) {
      for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++) {
        temp_storage.warp_best_splits[NODE][threadIdx.x] = Split();
      }
    }

    if (threadIdx.x < ParamsT::N_NODES) {
      temp_storage.best_splits[threadIdx.x] = Split();
    }
  }

  // Cache d_nodes array for this level into shared memory
  __device__ __forceinline__ void CacheNodes() {
    // Get pointer to nodes on the current level
    const Node *d_nodes_level = d_nodes + node_begin;

    if (threadIdx.x < ParamsT::N_NODES) {
      temp_storage.node_root_gain[threadIdx.x] =
          d_nodes_level[threadIdx.x].root_gain;
      temp_storage.node_parent_sum[threadIdx.x] =
          d_nodes_level[threadIdx.x].sum_gradients;
    }
  }

  __device__ __forceinline__ void LoadTile(bst_uint offset,
                                           bst_uint num_remaining) {
    if (threadIdx.x < num_remaining) {
      bst_uint i = offset + threadIdx.x;
      gpair = thrust::get<0>(item_iter[i]);
      fvalue = thrust::get<1>(item_iter[i]);
      node_id = thrust::get<2>(item_iter[i]);
      node_id_adjusted = node_id - node_begin;
    } else {
      node_id = -1;
      node_id_adjusted = -1;
      fvalue = -FLT_MAX;
      gpair = gpu_gpair();
    }
  }

  // Is this node being processed by current kernel iteration?
  __device__ __forceinline__ bool NodeActive() {
    return node_id_adjusted < ParamsT::N_NODES && node_id_adjusted >= 0;
  }

  // Is current fvalue different from left fvalue
  __device__ __forceinline__ bool
  LeftMostFvalue(const bst_uint &segment_begin, const bst_uint &offset,
                 const bst_uint &num_remaining) {
    int left_index = offset + threadIdx.x - 1;
    float left_fvalue = left_index >= static_cast<int>(segment_begin) &&
                                threadIdx.x < num_remaining
                            ? thrust::get<1>(item_iter[left_index])
                            : -FLT_MAX;

    return left_fvalue != fvalue;
  }

  // Prevent splitting in the middle of same valued instances
  __device__ __forceinline__ bool
  CheckSplitValid(const bst_uint &segment_begin, const bst_uint &offset,
                  const bst_uint &num_remaining) {
    BitFlagSet bit_flag = 0;

    bool valid_split = false;

    if (LeftMostFvalue(segment_begin, offset, num_remaining)) {
      valid_split = true;
      // Use MSB bit to flag if fvalue is leftmost
      set_bit(bit_flag, 63);
    }

    // Flag nodeid
    if (NodeActive()) {
      set_bit(bit_flag, node_id_adjusted);
    }

    FlagsBlockScanT(temp_storage.flags_scan)
        .ExclusiveScan(bit_flag, bit_flag, FlagScanOp(), flag_prefix_op);
    __syncthreads();

    if (!valid_split && NodeActive()) {
      if (!check_bit(bit_flag, node_id_adjusted)) {
        valid_split = true;
      }
    }

    return valid_split;
  }

  // Perform warp reduction to find if this lane contains best loss_chg in warp
  __device__ __forceinline__ bool QueryLaneBestLoss(const float &loss_chg) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Possible source of bugs. Not all threads in warp are active here. Not
    // sure if reduce function will behave correctly
    float best = WarpReduceT(temp_storage.warp_reduce[warp_id])
                     .Reduce(loss_chg, cub::Max());

    // Its possible for more than one lane to contain the best value, so make
    // sure only one lane returns true
    unsigned int ballot = __ballot(loss_chg == best);

    if (lane_id == (__ffs(ballot) - 1)) {
      return true;
    } else {
      return false;
    }
  }

  // Which thread in this warp should update the current best split, if any
  // Returns true for one thread or none
  __device__ __forceinline__ bool
  QueryUpdateWarpSplit(const float &loss_chg,
                       volatile const float &warp_best_loss) {
    bool update = false;

    for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++) {
      bool active = node_id_adjusted == NODE;

      unsigned int ballot = __ballot(loss_chg > warp_best_loss && active);

      // No lane has improved loss_chg
      if (__popc(ballot) == 0) {
        continue;
      } else if (__popc(ballot) == 1) {
        // A single lane has improved loss_chg, set true for this lane
        int lane_id = threadIdx.x % 32;

        if (lane_id == __ffs(ballot) - 1) {
          update = true;
        }
      } else {
        // More than one lane has improved loss_chg, perform a reduction.
        if (QueryLaneBestLoss(active ? loss_chg : -FLT_MAX)) {
          update = true;
        }
      }
    }

    return update;
  }

  __device__ void PrintTileScan(int block_id, bool thread_active,
                                float loss_chg, gpu_gpair missing) {
    if (blockIdx.x != block_id) {
      return;
    }

    for (int warp = 0; warp < ParamsT::N_WARPS; warp++) {
      if (threadIdx.x / 32 == warp) {
        for (int lane = 0; lane < 32; lane++) {
          gpu_gpair g = cub::ShuffleIndex(gpair, lane);
          gpu_gpair missing_broadcast = cub::ShuffleIndex(missing, lane);
          float fvalue_broadcast = __shfl(fvalue, lane);
          bool thread_active_broadcast = __shfl(thread_active, lane);
          float loss_chg_broadcast = __shfl(loss_chg, lane);
          NodeIdT node_id_broadcast = cub::ShuffleIndex(node_id, lane);
          if (threadIdx.x == 32 * warp) {
            printf("tid %d, nid %d, fvalue %1.2f, active %c, loss %1.2f, scan ",
                   threadIdx.x + lane, node_id_broadcast, fvalue_broadcast,
                   thread_active_broadcast ? 'y' : 'n',
                   loss_chg_broadcast < 0.0f ? 0 : loss_chg_broadcast);
            g.print();
          }
        }
      }

      __syncthreads();
    }
  }

  __device__ __forceinline__ void
  EvaluateSplits(const bst_uint &segment_begin, const bst_uint &offset,
                 const bst_uint &num_remaining) {
    bool valid_split = CheckSplitValid(segment_begin, offset, num_remaining);

    bool thread_active =
        NodeActive() && valid_split && threadIdx.x < num_remaining;

    const int warp_id = threadIdx.x / 32;

    gpu_gpair parent_sum = thread_active
                               ? temp_storage.node_parent_sum[node_id_adjusted]
                               : gpu_gpair();
    float parent_gain =
        thread_active ? temp_storage.node_root_gain[node_id_adjusted] : 0.0f;
    gpu_gpair missing = thread_active
                            ? parent_sum - reduction.node_sums[node_id_adjusted]
                            : gpu_gpair();

    bool missing_left;

    float loss_chg = thread_active
                         ? loss_chg_missing(gpair, missing, parent_sum,
                                            parent_gain, param, missing_left)
                         : -FLT_MAX;

    // PrintTileScan(64, thread_active, loss_chg, missing);

    float warp_best_loss =
        thread_active
            ? temp_storage.warp_best_splits[node_id_adjusted][warp_id].loss_chg
            : 0.0f;

    if (QueryUpdateWarpSplit(loss_chg, warp_best_loss)) {
      float fvalue_split = fvalue - FVALUE_EPS;

      if (missing_left) {
        gpu_gpair left_sum = missing + gpair;
        gpu_gpair right_sum = parent_sum - left_sum;
        temp_storage.warp_best_splits[node_id_adjusted][warp_id].Update(
            loss_chg, missing_left, fvalue_split, blockIdx.x, left_sum,
            right_sum, param);
      } else {
        gpu_gpair left_sum = gpair;
        gpu_gpair right_sum = parent_sum - left_sum;
        temp_storage.warp_best_splits[node_id_adjusted][warp_id].Update(
            loss_chg, missing_left, fvalue_split, blockIdx.x, left_sum,
            right_sum, param);
      }
    }
  }

  __device__ __forceinline__ void BlockExclusiveScan() {
    ResetPartials();

    __syncthreads();
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++) {
      bool node_active = node_id_adjusted == NODE;

      unsigned int ballot = __ballot(node_active);

      gpu_gpair warp_sum = gpu_gpair();
      gpu_gpair scan_result = gpu_gpair();

      if (ballot > 0) {
        WarpScanT(temp_storage.warp_gpair_scan[warp_id])
            .InclusiveScan(node_active ? gpair : gpu_gpair(), scan_result,
                           cub::Sum(), warp_sum);
      }

      if (node_active) {
        gpair = scan_result - gpair;
      }

      if (lane_id == 0) {
        temp_storage.partial_sums[NODE][warp_id] = warp_sum;
      }
    }

    __syncthreads();

    if (threadIdx.x < 32) {
      for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++) {
        gpu_gpair top_level_sum;
        bool warp_active = threadIdx.x < ParamsT::N_WARPS;
        gpu_gpair scan_result;
        WarpScanT(temp_storage.warp_gpair_scan[warp_id])
            .InclusiveScan(warp_active
                               ? temp_storage.partial_sums[NODE][threadIdx.x]
                               : gpu_gpair(),
                           scan_result, cub::Sum(), top_level_sum);

        if (warp_active) {
          temp_storage.partial_sums[NODE][threadIdx.x] =
              scan_result - temp_storage.partial_sums[NODE][threadIdx.x];
        }

        if (threadIdx.x == 0) {
          temp_storage.top_level_sum[NODE] = top_level_sum;
        }
      }
    }

    __syncthreads();

    if (NodeActive()) {
      gpair += temp_storage.partial_sums[node_id_adjusted][warp_id] +
               temp_storage.tile_carry[node_id_adjusted];
    }

    __syncthreads();

    UpdateTileCarry();

    __syncthreads();
  }

  __device__ __forceinline__ void ProcessTile(const bst_uint &segment_begin,
                                              const bst_uint &offset,
                                              const bst_uint &num_remaining) {
    LoadTile(offset, num_remaining);
    BlockExclusiveScan();
    EvaluateSplits(segment_begin, offset, num_remaining);
  }

  __device__ __forceinline__ void ReduceSplits() {
    for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++) {
      if (threadIdx.x < 32) {
        Split s = Split();
        if (threadIdx.x < ParamsT::N_WARPS) {
          s = temp_storage.warp_best_splits[NODE][threadIdx.x];
        }
        Split best = WarpSplitReduceT(temp_storage.warp_split_reduce)
                         .Reduce(s, split_reduce_op());
        if (threadIdx.x == 0) {
          temp_storage.best_splits[NODE] = best;
        }
      }
    }
  }

  __device__ __forceinline__ void WriteBestSplits() {
    const int nodes_level = 1 << level;

    if (threadIdx.x < ParamsT::N_NODES) {
      d_split_candidates_out[blockIdx.x * nodes_level + threadIdx.x] =
          temp_storage.best_splits[threadIdx.x];
    }
  }

  __device__ __forceinline__ void ProcessRegion(const bst_uint &segment_begin,
                                                const bst_uint &segment_end) {
    // Current position
    bst_uint offset = segment_begin;

    ResetTileCarry();
    ResetSplits();
    CacheNodes();
    __syncthreads();

    // Process full tiles
    while (offset < segment_end) {
      ProcessTile(segment_begin, offset, segment_end - offset);
      __syncthreads();
      offset += ParamsT::TILE_ITEMS;
    }

    __syncthreads();
    ReduceSplits();

    __syncthreads();
    WriteBestSplits();
  }
};

template <typename FindSplitParamsT, typename ReduceParamsT>
__global__ void
#if __CUDA_ARCH__ <= 530
__launch_bounds__(1024, 2)
#endif
    find_split_candidates_multiscan_kernel(
        const ItemIter items_iter, Split *d_split_candidates_out,
        const Node *d_nodes, const int node_begin, bst_uint num_items,
        int num_features, const int *d_feature_offsets,
        const GPUTrainingParam param, const int *d_feature_flags,
        const int level) {
  if (num_items <= 0 || d_feature_flags[blockIdx.x] != 1) {
    return;
  }

  int segment_begin = d_feature_offsets[blockIdx.x];
  int segment_end = d_feature_offsets[blockIdx.x + 1];

  typedef ReduceEnactorMultiscan<ReduceParamsT> ReduceT;
  typedef FindSplitEnactorMultiscan<FindSplitParamsT,
                                    typename ReduceT::_Reduction>
      FindSplitT;

  __shared__ union {
    typename ReduceT::TempStorage reduce;
    typename FindSplitT::TempStorage find_split;
  } temp_storage;

  __shared__ typename ReduceT::Reduction reduction;

  ReduceT(temp_storage.reduce, reduction, items_iter, node_begin)
      .ProcessRegion(segment_begin, segment_end);
  __syncthreads();

  FindSplitT find_split(temp_storage.find_split, items_iter,
                        d_split_candidates_out, d_nodes, node_begin, param,
                        reduction.Alias(), level);
  find_split.ProcessRegion(segment_begin, segment_end);
}

template <int N_NODES>
void find_split_candidates_multiscan_variation(GPUData *data, const int level) {
  const int node_begin = (1 << level) - 1;
  const int BLOCK_THREADS = 512;

  CHECK(BLOCK_THREADS / 32 < 32)
      << "Too many active warps. See FindSplitEnactor - ReduceSplits.";

  typedef FindSplitParamsMultiscan<BLOCK_THREADS, N_NODES, false>
      find_split_params;
  typedef ReduceParamsMultiscan<BLOCK_THREADS, N_NODES, false> reduce_params;
  int grid_size = data->n_features;

  find_split_candidates_multiscan_kernel<
      find_split_params,
      reduce_params><<<grid_size, find_split_params::BLOCK_THREADS>>>(
      data->items_iter, data->split_candidates.data(), data->nodes.data(),
      node_begin, data->fvalues.size(), data->n_features, data->foffsets.data(),
      data->param, data->feature_flags.data(), level);

  dh::safe_cuda(cudaDeviceSynchronize());
}

void find_split_candidates_multiscan(GPUData *data, const int level) {
  // Select templated variation of split finding algorithm
  switch (level) {
  case 0:
    find_split_candidates_multiscan_variation<1>(data, level);
    break;
  case 1:
    find_split_candidates_multiscan_variation<2>(data, level);
    break;
  case 2:
    find_split_candidates_multiscan_variation<4>(data, level);
    break;
  case 3:
    find_split_candidates_multiscan_variation<8>(data, level);
    break;
  case 4:
    find_split_candidates_multiscan_variation<16>(data, level);
    break;
  }
}
}  // namespace tree
}  // namespace xgboost
