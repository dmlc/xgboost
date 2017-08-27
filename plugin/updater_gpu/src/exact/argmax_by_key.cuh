/*
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "../../../../src/tree/param.h"
#include "../common.cuh"
#include "node.cuh"
#include "../types.cuh"

namespace xgboost {
namespace tree {
namespace exact {

/**
 * @enum ArgMaxByKeyAlgo best_split_evaluation.cuh
 * @brief Help decide which algorithm to use for multi-argmax operation
 */
enum ArgMaxByKeyAlgo {
  /** simplest, use gmem-atomics for all updates */
  ABK_GMEM = 0,
  /** use smem-atomics for updates (when number of keys are less) */
  ABK_SMEM
};

/** max depth until which to use shared mem based atomics for argmax */
static const int MAX_ABK_LEVELS = 3;

HOST_DEV_INLINE Split maxSplit(Split a, Split b) {
  Split out;
  if (a.score < b.score) {
    out.score = b.score;
    out.index = b.index;
  } else if (a.score == b.score) {
    out.score = a.score;
    out.index = (a.index < b.index) ? a.index : b.index;
  } else {
    out.score = a.score;
    out.index = a.index;
  }
  return out;
}

DEV_INLINE void atomicArgMax(Split* address, Split val) {
  unsigned long long* intAddress = (unsigned long long*)address;
  unsigned long long old = *intAddress;
  unsigned long long assumed;
  do {
    assumed = old;
    Split res = maxSplit(val, *(Split*)&assumed);
    old = atomicCAS(intAddress, assumed, *(uint64_t*)&res);
  } while (assumed != old);
}


DEV_INLINE void argMaxWithAtomics(
    int id, Split* nodeSplits, const bst_gpair* gradScans,
    const bst_gpair* gradSums, const float* vals, const int* colIds,
    const node_id_t* nodeAssigns, const DeviceDenseNode* nodes, int nUniqKeys,
    node_id_t nodeStart, int len, const  GPUTrainingParam& param) {
  int nodeId = nodeAssigns[id];
  ///@todo: this is really a bad check! but will be fixed when we move
  ///   to key-based reduction
  if ((id == 0) ||
      !((nodeId == nodeAssigns[id - 1]) && (colIds[id] == colIds[id - 1]) &&
        (vals[id] == vals[id - 1]))) {
    if (nodeId != UNUSED_NODE) {
      int sumId = abs2uniqKey(id, nodeAssigns, colIds, nodeStart, nUniqKeys);
      bst_gpair colSum = gradSums[sumId];
      int uid = nodeId - nodeStart;
      DeviceDenseNode n = nodes[nodeId];
      bst_gpair parentSum = n.sum_gradients;
      float parentGain = n.root_gain;
      bool tmp;
      Split s;
      bst_gpair missing = parentSum - colSum;
      s.score = loss_chg_missing(gradScans[id], missing, parentSum, parentGain,
                                 param, tmp);
      s.index = id;
      atomicArgMax(nodeSplits + uid, s);
    }  // end if nodeId != UNUSED_NODE
  }    // end if id == 0 ...
}


__global__ void atomicArgMaxByKeyGmem(
    Split* nodeSplits, const bst_gpair* gradScans, const bst_gpair* gradSums,
    const float* vals, const int* colIds, const node_id_t* nodeAssigns,
    const DeviceDenseNode* nodes, int nUniqKeys, node_id_t nodeStart, int len,
    const TrainParam param) {
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  const int stride = blockDim.x * gridDim.x;
  for (; id < len; id += stride) {
    argMaxWithAtomics(id, nodeSplits, gradScans, gradSums, vals, colIds,
                      nodeAssigns, nodes, nUniqKeys, nodeStart, len,  GPUTrainingParam(param));
  }
}


__global__ void atomicArgMaxByKeySmem(
    Split* nodeSplits, const bst_gpair* gradScans, const bst_gpair* gradSums,
    const float* vals, const int* colIds, const node_id_t* nodeAssigns,
    const DeviceDenseNode* nodes, int nUniqKeys, node_id_t nodeStart, int len,
    const TrainParam param) {
  extern __shared__ char sArr[];
  Split* sNodeSplits = reinterpret_cast<Split*>(sArr);
  int tid = threadIdx.x;
  Split defVal;
#pragma unroll 1
  for (int i = tid; i < nUniqKeys; i += blockDim.x) {
    sNodeSplits[i] = defVal;
  }
  __syncthreads();
  int id = tid + (blockIdx.x * blockDim.x);
  const int stride = blockDim.x * gridDim.x;
  for (; id < len; id += stride) {
    argMaxWithAtomics(id, sNodeSplits, gradScans, gradSums, vals, colIds,
                      nodeAssigns, nodes, nUniqKeys, nodeStart, len, param);
  }
  __syncthreads();
  for (int i = tid; i < nUniqKeys; i += blockDim.x) {
    Split s = sNodeSplits[i];
    atomicArgMax(nodeSplits + i, s);
  }
}

/**
 * @brief Performs argmax_by_key functionality but for cases when keys need not
 *  occur contiguously
 * @param nodeSplits will contain information on best split for each node
 * @param gradScans exclusive sum on sorted segments for each col
 * @param gradSums gradient sum for each column in DMatrix based on to node-ids
 * @param vals feature values
 * @param colIds column index for each element in the feature values array
 * @param nodeAssigns node-id assignments to each element in DMatrix
 * @param nodes pointer to all nodes for this tree in BFS order
 * @param nUniqKeys number of unique node-ids in this level
 * @param nodeStart start index of the node-ids in this level
 * @param len number of elements
 * @param param training parameters
 * @param algo which algorithm to use for argmax_by_key
 */
template < int BLKDIM = 256, int ITEMS_PER_THREAD = 4>
void argMaxByKey(Split* nodeSplits, const bst_gpair* gradScans,
                 const bst_gpair* gradSums, const float* vals,
                 const int* colIds, const node_id_t* nodeAssigns,
                 const DeviceDenseNode* nodes, int nUniqKeys,
                 node_id_t nodeStart, int len, const TrainParam param,
                 ArgMaxByKeyAlgo algo) {
  fillConst<Split, BLKDIM, ITEMS_PER_THREAD>(dh::get_device_idx(param.gpu_id),
                                             nodeSplits, nUniqKeys, Split());
  int nBlks = dh::div_round_up(len, ITEMS_PER_THREAD * BLKDIM);
  switch (algo) {
    case ABK_GMEM:
      atomicArgMaxByKeyGmem<<<nBlks, BLKDIM>>>(
          nodeSplits, gradScans, gradSums, vals, colIds, nodeAssigns, nodes,
          nUniqKeys, nodeStart, len, param);
      break;
    case ABK_SMEM:
      atomicArgMaxByKeySmem<<<nBlks, BLKDIM, sizeof(Split) * nUniqKeys>>>(
          nodeSplits, gradScans, gradSums, vals, colIds, nodeAssigns, nodes,
          nUniqKeys, nodeStart, len, param);
      break;
    default:
      throw std::runtime_error("argMaxByKey: Bad algo passed!");
  }
}

}  // namespace exact
}  // namespace tree
}  // namespace xgboost
