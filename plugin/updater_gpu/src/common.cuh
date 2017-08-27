/*!
 * Copyright 2017 XGBoost contributors
 */
#pragma once
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../../src/common/random.h"
#include "../../../src/tree/param.h"
#include "cub/cub.cuh"
#include "device_helpers.cuh"
#include "types.cuh"

namespace xgboost {
namespace tree {

template <typename gpair_t>
__device__ inline float device_calc_loss_chg(
    const GPUTrainingParam& param, const gpair_t& scan, const gpair_t& missing,
    const gpair_t& parent_sum, const float& parent_gain, bool missing_left) {
  gpair_t left = scan;

  if (missing_left) {
    left += missing;
  }

  gpair_t right = parent_sum - left;

  float left_gain = CalcGain(param, left.grad, left.hess);
  float right_gain = CalcGain(param, right.grad, right.hess);
  return left_gain + right_gain - parent_gain;
}

template <typename gpair_t>
__device__ float inline loss_chg_missing(const gpair_t& scan,
                                         const gpair_t& missing,
                                         const gpair_t& parent_sum,
                                         const float& parent_gain,
                                         const GPUTrainingParam& param,
                                         bool& missing_left_out) {  // NOLINT
  float missing_left_loss =
      device_calc_loss_chg(param, scan, missing, parent_sum, parent_gain, true);
  float missing_right_loss = device_calc_loss_chg(
      param, scan, missing, parent_sum, parent_gain, false);

  if (missing_left_loss >= missing_right_loss) {
    missing_left_out = true;
    return missing_left_loss;
  } else {
    missing_left_out = false;
    return missing_right_loss;
  }
}

// Total number of nodes in tree, given depth
__host__ __device__ inline int n_nodes(int depth) {
  return (1 << (depth + 1)) - 1;
}

// Number of nodes at this level of the tree
__host__ __device__ inline int n_nodes_level(int depth) { return 1 << depth; }

// Whether a node is currently being processed at current depth
__host__ __device__ inline bool is_active(int nidx, int depth) {
  return nidx >= n_nodes(depth - 1);
}

__host__ __device__ inline int parent_nidx(int nidx) { return (nidx - 1) / 2; }

__host__ __device__ inline int left_child_nidx(int nidx) {
  return nidx * 2 + 1;
}

__host__ __device__ inline int right_child_nidx(int nidx) {
  return nidx * 2 + 2;
}

__host__ __device__ inline bool is_left_child(int nidx) {
  return nidx % 2 == 1;
}

// Copy gpu dense representation of tree to xgboost sparse representation
inline void dense2sparse_tree(RegTree* p_tree,
  const dh::dvec<DeviceDenseNode>& nodes,
                              const TrainParam& param) {
  RegTree& tree = *p_tree;
  std::vector<DeviceDenseNode> h_nodes = nodes.as_vector();

  int nid = 0;
  for (int gpu_nid = 0; gpu_nid < h_nodes.size(); gpu_nid++) {
    const DeviceDenseNode& n = h_nodes[gpu_nid];
    if (!n.IsUnused() && !n.IsLeaf()) {
      tree.AddChilds(nid);
      tree[nid].set_split(n.fidx, n.fvalue, n.dir == LeftDir);
      tree.stat(nid).loss_chg = n.root_gain;
      tree.stat(nid).base_weight = n.weight;
      tree.stat(nid).sum_hess = n.sum_gradients.hess;
      tree[tree[nid].cleft()].set_leaf(0);
      tree[tree[nid].cright()].set_leaf(0);
      nid++;
    } else if (n.IsLeaf()) {
      tree[nid].set_leaf(n.weight * param.learning_rate);
      tree.stat(nid).sum_hess = n.sum_gradients.hess;
      nid++;
    }
  }
}

// Set gradient pair to 0 with p = 1 - subsample
inline void subsample_gpair(dh::dvec<bst_gpair>* p_gpair, float subsample,
                            int offset) {
  if (subsample == 1.0) {
    return;
  }

  dh::dvec<bst_gpair>& gpair = *p_gpair;

  auto d_gpair = gpair.data();
  dh::BernoulliRng rng(subsample, common::GlobalRandom()());

  dh::launch_n(gpair.device_idx(), gpair.size(), [=] __device__(int i) {
    if (!rng(i + offset)) {
      d_gpair[i] = bst_gpair();
    }
  });
}

// Set gradient pair to 0 with p = 1 - subsample
inline void subsample_gpair(dh::dvec<bst_gpair>* p_gpair, float subsample) {
  int offset = 0;
  subsample_gpair(p_gpair, subsample, offset);
}

inline std::vector<int> col_sample(std::vector<int> features, float colsample) {
  CHECK_GT(features.size(), 0);
  int n = std::max(1, static_cast<int>(colsample * features.size()));

  std::shuffle(features.begin(), features.end(), common::GlobalRandom());
  features.resize(n);

  return features;
}
struct GpairCallbackOp {
  // Running prefix
  bst_gpair_precise running_total;
  // Constructor
  __device__ GpairCallbackOp() : running_total(bst_gpair_precise()) {}
  __device__ bst_gpair_precise operator()(bst_gpair_precise block_aggregate) {
    bst_gpair_precise old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

/**
 * @brief Helper function to sort the pairs using cub's segmented RadixSortPairs
 * @param tmp_mem cub temporary memory info
 * @param keys keys double-buffer array
 * @param vals the values double-buffer array
 * @param nVals number of elements in the array
 * @param nSegs number of segments
 * @param offsets the segments
 */
template <typename T1, typename T2>
void segmentedSort(dh::CubMemory* tmp_mem, dh::dvec2<T1>* keys,
                   dh::dvec2<T2>* vals, int nVals, int nSegs,
                   const dh::dvec<int>& offsets, int start = 0,
                   int end = sizeof(T1) * 8) {
  size_t tmpSize;
  dh::safe_cuda(cub::DeviceSegmentedRadixSort::SortPairs(
      NULL, tmpSize, keys->buff(), vals->buff(), nVals, nSegs, offsets.data(),
      offsets.data() + 1, start, end));
  tmp_mem->LazyAllocate(tmpSize);
  dh::safe_cuda(cub::DeviceSegmentedRadixSort::SortPairs(
      tmp_mem->d_temp_storage, tmpSize, keys->buff(), vals->buff(), nVals,
      nSegs, offsets.data(), offsets.data() + 1, start, end));
}

/**
 * @brief Helper function to perform device-wide sum-reduction
 * @param tmp_mem cub temporary memory info
 * @param in the input array to be reduced
 * @param out the output reduced value
 * @param nVals number of elements in the input array
 */
template <typename T>
void sumReduction(dh::CubMemory& tmp_mem, dh::dvec<T>& in, dh::dvec<T>& out,
                  int nVals) {
  size_t tmpSize;
  dh::safe_cuda(
      cub::DeviceReduce::Sum(NULL, tmpSize, in.data(), out.data(), nVals));
  tmp_mem.LazyAllocate(tmpSize);
  dh::safe_cuda(cub::DeviceReduce::Sum(tmp_mem.d_temp_storage, tmpSize,
                                       in.data(), out.data(), nVals));
}

/**
 * @brief Fill a given constant value across all elements in the buffer
 * @param out the buffer to be filled
 * @param len number of elements i the buffer
 * @param def default value to be filled
 */
template <typename T, int BlkDim = 256, int ItemsPerThread = 4>
void fillConst(int device_idx, T* out, int len, T def) {
  dh::launch_n<ItemsPerThread, BlkDim>(device_idx, len,
                                       [=] __device__(int i) { out[i] = def; });
}

/**
 * @brief gather elements
 * @param out1 output gathered array for the first buffer
 * @param in1 first input buffer
 * @param out2 output gathered array for the second buffer
 * @param in2 second input buffer
 * @param instId gather indices
 * @param nVals length of the buffers
 */
template <typename T1, typename T2, int BlkDim = 256, int ItemsPerThread = 4>
void gather(int device_idx, T1* out1, const T1* in1, T2* out2, const T2* in2,
            const int* instId, int nVals) {
  dh::launch_n<ItemsPerThread, BlkDim>(device_idx, nVals,
                                       [=] __device__(int i) {
                                         int iid = instId[i];
                                         T1 v1 = in1[iid];
                                         T2 v2 = in2[iid];
                                         out1[i] = v1;
                                         out2[i] = v2;
                                       });
}

/**
 * @brief gather elements
 * @param out output gathered array
 * @param in input buffer
 * @param instId gather indices
 * @param nVals length of the buffers
 */
template <typename T, int BlkDim = 256, int ItemsPerThread = 4>
void gather(int device_idx, T* out, const T* in, const int* instId, int nVals) {
  dh::launch_n<ItemsPerThread, BlkDim>(device_idx, nVals,
                                       [=] __device__(int i) {
                                         int iid = instId[i];
                                         out[i] = in[iid];
                                       });
}

}  // namespace tree
}  // namespace xgboost
