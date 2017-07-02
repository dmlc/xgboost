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

#include "../common.cuh"

namespace xgboost {
namespace tree {
namespace exact {

/**
 * @struct Pair fused_scan_reduce_by_key.cuh
 * @brief Pair used for key basd scan operations on bst_gpair
 */
struct Pair {
  int key;
  bst_gpair value;
};

/** define a key that's not used at all in the entire boosting process */
static const int NONE_KEY = -100;

/**
 * @brief Allocate temporary buffers needed for scan operations
 * @param tmpScans gradient buffer
 * @param tmpKeys keys buffer
 * @param size number of elements that will be scanned
 */
template <int BLKDIM_L1L3 = 256>
int scanTempBufferSize(int size) {
  int nBlks = dh::div_round_up(size, BLKDIM_L1L3);
  return nBlks;
}

struct AddByKey {
  template <typename T>
  HOST_DEV_INLINE T operator()(const T& first, const T& second) const {
    T result;
    if (first.key == second.key) {
      result.key = first.key;
      result.value = first.value + second.value;
    } else {
      result.key = second.key;
      result.value = second.value;
    }
    return result;
  }
};

/**
* @brief Gradient value getter function
* @param id the index into the vals or instIds array to which to fetch
* @param vals the gradient value buffer
* @param instIds instance index buffer
* @return the expected gradient value
*/
HOST_DEV_INLINE bst_gpair get(int id, const bst_gpair* vals, const int* instIds) {
  id = instIds[id];
  return vals[id];
}

template <typename node_id_t, int BLKDIM_L1L3>
__global__ void cubScanByKeyL1(bst_gpair* scans, const bst_gpair* vals,
                               const int* instIds, bst_gpair* mScans,
                               int* mKeys, const node_id_t* keys, int nUniqKeys,
                               const int* colIds, node_id_t nodeStart,
                               const int size) {
  Pair rootPair = {NONE_KEY, bst_gpair(0.f, 0.f)};
  int myKey;
  bst_gpair myValue;
  typedef cub::BlockScan<Pair, BLKDIM_L1L3> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  Pair threadData;
  int tid = blockIdx.x * BLKDIM_L1L3 + threadIdx.x;
  if (tid < size) {
    myKey = abs2uniqKey(tid, keys, colIds, nodeStart, nUniqKeys);
    myValue = get(tid, vals, instIds);
  } else {
    myKey = NONE_KEY;
    myValue = 0.f;
  }
  threadData.key = myKey;
  threadData.value = myValue;
  // get previous key, especially needed for the last thread in this block
  // in order to pass on the partial scan values.
  // this statement MUST appear before the checks below!
  // else, the result of this shuffle operation will be undefined
  int previousKey = __shfl_up(myKey, 1);
  // Collectively compute the block-wide exclusive prefix sum
  BlockScan(temp_storage)
      .ExclusiveScan(threadData, threadData, rootPair, AddByKey());
  if (tid < size) {
    scans[tid] = threadData.value;
  } else {
    return;
  }
  if (threadIdx.x == BLKDIM_L1L3 - 1) {
    threadData.value =
        (myKey == previousKey) ? threadData.value : bst_gpair(0.0f, 0.0f);
    mKeys[blockIdx.x] = myKey;
    mScans[blockIdx.x] = threadData.value + myValue;
  }
}

template <int BLKSIZE>
__global__ void cubScanByKeyL2(bst_gpair* mScans, int* mKeys, int mLength) {
  typedef cub::BlockScan<Pair, BLKSIZE, cub::BLOCK_SCAN_WARP_SCANS> BlockScan;
  Pair threadData;
  __shared__ typename BlockScan::TempStorage temp_storage;
  for (int i = threadIdx.x; i < mLength; i += BLKSIZE - 1) {
    threadData.key = mKeys[i];
    threadData.value = mScans[i];
    BlockScan(temp_storage).InclusiveScan(threadData, threadData, AddByKey());
    mScans[i] = threadData.value;
    __syncthreads();
  }
}

template <typename node_id_t, int BLKDIM_L1L3>
__global__ void cubScanByKeyL3(bst_gpair* sums, bst_gpair* scans,
                               const bst_gpair* vals, const int* instIds,
                               const bst_gpair* mScans, const int* mKeys,
                               const node_id_t* keys, int nUniqKeys,
                               const int* colIds, node_id_t nodeStart,
                               const int size) {
  int relId = threadIdx.x;
  int tid = (blockIdx.x * BLKDIM_L1L3) + relId;
  // to avoid the following warning from nvcc:
  //   __shared__ memory variable with non-empty constructor or destructor
  //     (potential race between threads)
  __shared__ char gradBuff[sizeof(bst_gpair)];
  __shared__ int s_mKeys;
  bst_gpair* s_mScans = reinterpret_cast<bst_gpair*>(gradBuff);
  if (tid >= size) return;
  // cache block-wide partial scan info
  if (relId == 0) {
    s_mKeys = (blockIdx.x > 0) ? mKeys[blockIdx.x - 1] : NONE_KEY;
    s_mScans[0] = (blockIdx.x > 0) ? mScans[blockIdx.x - 1] : bst_gpair();
  }
  int myKey = abs2uniqKey(tid, keys, colIds, nodeStart, nUniqKeys);
  int previousKey = tid == 0 ? NONE_KEY : abs2uniqKey(tid - 1, keys, colIds,
                                                      nodeStart, nUniqKeys);
  bst_gpair myValue = scans[tid];
  __syncthreads();
  if (blockIdx.x > 0 && s_mKeys == previousKey) {
    myValue += s_mScans[0];
  }
  if (tid == size - 1) {
    sums[previousKey] = myValue + get(tid, vals, instIds);
  }
  if ((previousKey != myKey) && (previousKey >= 0)) {
    sums[previousKey] = myValue;
    myValue = bst_gpair(0.0f, 0.0f);
  }
  scans[tid] = myValue;
}

/**
 * @brief Performs fused reduce and scan by key functionality. It is assumed
 * that
 *  the keys occur contiguously!
 * @param sums the output gradient reductions for each element performed
 * key-wise
 * @param scans the output gradient scans for each element performed key-wise
 * @param vals the gradients evaluated for each observation.
 * @param instIds instance ids for each element
 * @param keys keys to be used to segment the reductions. They need not occur
 *  contiguously in contrast to scan_by_key. Currently, we need one key per
 *  value in the 'vals' array.
 * @param size number of elements in the 'vals' array
 * @param nUniqKeys max number of uniq keys found per column
 * @param nCols number of columns
 * @param tmpScans temporary scan buffer needed for cub-pyramid algo
 * @param tmpKeys temporary key buffer needed for cub-pyramid algo
 * @param colIds column indices for each element in the array
 * @param nodeStart index of the leftmost node in the current level
 */
template <typename node_id_t, int BLKDIM_L1L3 = 256, int BLKDIM_L2 = 512>
void reduceScanByKey(bst_gpair* sums, bst_gpair* scans, const bst_gpair* vals,
                     const int* instIds, const node_id_t* keys, int size,
                     int nUniqKeys, int nCols, bst_gpair* tmpScans,
                     int* tmpKeys, const int* colIds, node_id_t nodeStart) {
  int nBlks = dh::div_round_up(size, BLKDIM_L1L3);
  cudaMemset(sums, 0, nUniqKeys * nCols * sizeof(bst_gpair));
  cubScanByKeyL1<node_id_t, BLKDIM_L1L3><<<nBlks, BLKDIM_L1L3>>>(
      scans, vals, instIds, tmpScans, tmpKeys, keys, nUniqKeys, colIds,
      nodeStart, size);
  cubScanByKeyL2<BLKDIM_L2><<<1, BLKDIM_L2>>>(tmpScans, tmpKeys, nBlks);
  cubScanByKeyL3<node_id_t, BLKDIM_L1L3><<<nBlks, BLKDIM_L1L3>>>(
      sums, scans, vals, instIds, tmpScans, tmpKeys, keys, nUniqKeys, colIds,
      nodeStart, size);
}

}  // namespace exact
}  // namespace tree
}  // namespace xgboost
