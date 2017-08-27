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
#include "node.cuh"

namespace xgboost {
namespace tree {
namespace exact {

/**
 * @brief Helper function to update the child node based on the current status
 *  of its parent node
 * @param nodes the nodes array in which the position at 'nid' will be updated
 * @param nid the nodeId in the 'nodes' array corresponding to this child node
 * @param grad gradient sum for this child node
 * @param minChildWeight minimum child weight for the split
 * @param alpha L1 regularizer for weight updates
 * @param lambda lambda as in xgboost
 * @param maxStep max weight step update
 */

DEV_INLINE void updateOneChildNode(DeviceDenseNode* nodes, int nid,
                                   const bst_gpair& grad,
                                   const TrainParam& param) {
  nodes[nid].sum_gradients = grad;
  nodes[nid].root_gain = CalcGain(param, grad.grad, grad.hess);
  nodes[nid].weight = CalcWeight(param, grad.grad, grad.hess);
  nodes[nid].idx = nid;
}

/**
 * @brief Helper function to update the child nodes based on the current status
 *  of their parent node
 * @param nodes the nodes array in which the position at 'nid' will be updated
 * @param pid the nodeId of the parent
 * @param gradL gradient sum for the left child node
 * @param gradR gradient sum for the right child node
 * @param param the training parameter struct
 */

DEV_INLINE void updateChildNodes(DeviceDenseNode* nodes, int pid,
                                 const bst_gpair& gradL, const bst_gpair& gradR,
                                 const TrainParam& param) {
  int childId = (pid * 2) + 1;
  updateOneChildNode(nodes, childId, gradL, param);
  updateOneChildNode(nodes, childId + 1, gradR, param);
}


DEV_INLINE void updateNodeAndChildren(DeviceDenseNode* nodes, const Split& s,
                                      const DeviceDenseNode& n, int absNodeId,
                                      int colId, const bst_gpair& gradScan,
                                      const bst_gpair& colSum, float thresh,
                                      const TrainParam& param) {
  bool missingLeft = true;
  // get the default direction for the current node
  bst_gpair missing = n.sum_gradients - colSum;
  loss_chg_missing(gradScan, missing, n.sum_gradients, n.root_gain, param, missingLeft);
  // get the score/weight/id/gradSum for left and right child nodes
  bst_gpair lGradSum, rGradSum;
  if (missingLeft) {
    lGradSum = gradScan + n.sum_gradients - colSum;
  } else {
    lGradSum = gradScan;
  }
  rGradSum = n.sum_gradients - lGradSum;
  updateChildNodes(nodes, absNodeId, lGradSum, rGradSum, param);
  // update default-dir, threshold and feature id for current node
  nodes[absNodeId].dir = missingLeft ? LeftDir : RightDir;
  nodes[absNodeId].fidx = colId;
  nodes[absNodeId].fvalue = thresh;
}

template < int BLKDIM = 256>
__global__ void split2nodeKernel(
    DeviceDenseNode* nodes, const Split* nodeSplits, const bst_gpair* gradScans,
    const bst_gpair* gradSums, const float* vals, const int* colIds,
    const int* colOffsets, const node_id_t* nodeAssigns, int nUniqKeys,
    node_id_t nodeStart, int nCols, const TrainParam param) {
  int uid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (uid >= nUniqKeys) {
    return;
  }
  int absNodeId = uid + nodeStart;
  Split s = nodeSplits[uid];
  if (s.isSplittable(param.min_split_loss)) {
    int idx = s.index;
    int nodeInstId =
        abs2uniqKey(idx, nodeAssigns, colIds, nodeStart, nUniqKeys);
    updateNodeAndChildren(nodes, s, nodes[absNodeId], absNodeId, colIds[idx],
                          gradScans[idx], gradSums[nodeInstId], vals[idx],
                          param);
  } else {
    // cannot be split further, so this node is a leaf!
    nodes[absNodeId].root_gain = -FLT_MAX;
  }
}

/**
 * @brief function to convert split information into node
 * @param nodes the output nodes
 * @param nodeSplits split information
 * @param gradScans scan of sorted gradients across columns
 * @param gradSums key-wise gradient reduction across columns
 * @param vals the feature values
 * @param colIds column indices for each element in the array
 * @param colOffsets column segment offsets
 * @param nodeAssigns node-id assignment to every feature value
 * @param nUniqKeys number of nodes that we are currently working on
 * @param nodeStart start offset of the nodes in the overall BFS tree
 * @param nCols number of columns
 * @param preUniquifiedKeys whether to uniquify the keys from inside kernel or
 * not
 * @param param the training parameter struct
 */
template < int BLKDIM = 256>
void split2node(DeviceDenseNode* nodes, const Split* nodeSplits,
                const bst_gpair* gradScans, const bst_gpair* gradSums,
                const float* vals, const int* colIds, const int* colOffsets,
                const node_id_t* nodeAssigns, int nUniqKeys,
                node_id_t nodeStart, int nCols, const TrainParam param) {
  int nBlks = dh::div_round_up(nUniqKeys, BLKDIM);
  split2nodeKernel<<<nBlks, BLKDIM>>>(nodes, nodeSplits, gradScans, gradSums,
                                      vals, colIds, colOffsets, nodeAssigns,
                                      nUniqKeys, nodeStart, nCols, param);
}

}  // namespace exact
}  // namespace tree
}  // namespace xgboost
