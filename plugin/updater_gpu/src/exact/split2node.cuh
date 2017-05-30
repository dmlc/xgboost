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

#include "../../../src/tree/param.h"
#include "gradients.cuh"
#include "node.cuh"
#include "loss_functions.cuh"


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
template <typename node_id_t>
DEV_INLINE void updateOneChildNode(Node<node_id_t>* nodes, int nid,
                                   const gpu_gpair& grad,
                                   const TrainParam &param) {
  nodes[nid].gradSum = grad;
  nodes[nid].score = CalcGain(param, grad.g, grad.h);
  nodes[nid].weight = CalcWeight(param, grad.g, grad.h);
  nodes[nid].id = nid;
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
template <typename node_id_t>
DEV_INLINE void updateChildNodes(Node<node_id_t>* nodes, int pid,
                                 const gpu_gpair& gradL, const gpu_gpair& gradR,
                                 const TrainParam &param) {
  int childId = (pid * 2) + 1;
  updateOneChildNode(nodes, childId, gradL, param);
  updateOneChildNode(nodes, childId+1, gradR, param);
}

template <typename node_id_t>
DEV_INLINE void updateNodeAndChildren(Node<node_id_t>* nodes, const Split& s,
                                      const Node<node_id_t>& n, int absNodeId, int colId,
                                      const gpu_gpair& gradScan,
                                      const gpu_gpair& colSum, float thresh,
                                      const TrainParam &param) {
  bool missingLeft = true;
  // get the default direction for the current node
  gpu_gpair missing = n.gradSum - colSum;
  loss_chg_missing(gradScan, missing, n.gradSum, n.score, param, missingLeft);
  // get the score/weight/id/gradSum for left and right child nodes
  gpu_gpair lGradSum, rGradSum;
  if (missingLeft) {
    lGradSum = gradScan + n.gradSum - colSum;
  } else {
    lGradSum = gradScan;
  }
  rGradSum = n.gradSum - lGradSum;
  updateChildNodes(nodes, absNodeId, lGradSum, rGradSum, param);
  // update default-dir, threshold and feature id for current node
  nodes[absNodeId].dir = missingLeft? LeftDir : RightDir;
  nodes[absNodeId].colIdx = colId;
  nodes[absNodeId].threshold = thresh;
}

template <typename node_id_t, int BLKDIM=256>
__global__ void split2nodeKernel(Node<node_id_t>* nodes, const Split* nodeSplits,
                                 const gpu_gpair* gradScans,
                                 const gpu_gpair* gradSums, const float* vals,
                                 const int* colIds, const int* colOffsets,
                                 const node_id_t* nodeAssigns, int nUniqKeys,
                                 node_id_t nodeStart, int nCols,
                                 const TrainParam param) {
  int uid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (uid >= nUniqKeys) {
    return;
  }
  int absNodeId = uid + nodeStart;
  Split s = nodeSplits[uid];
  if (s.isSplittable(param.min_split_loss)) {
    int idx = s.index;
    int nodeInstId = abs2uniqKey(idx, nodeAssigns, colIds, nodeStart,
                                 nUniqKeys);
    updateNodeAndChildren(nodes, s, nodes[absNodeId], absNodeId,
                          colIds[idx], gradScans[idx],
                          gradSums[nodeInstId], vals[idx], param);
  } else {
    // cannot be split further, so this node is a leaf!
    nodes[absNodeId].score = -FLT_MAX;
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
 * @param preUniquifiedKeys whether to uniquify the keys from inside kernel or not
 * @param param the training parameter struct
 */
template <typename node_id_t, int BLKDIM=256>
void split2node(Node<node_id_t>* nodes, const Split* nodeSplits, const gpu_gpair* gradScans,
                const gpu_gpair* gradSums, const float* vals, const int* colIds,
                const int* colOffsets, const node_id_t* nodeAssigns,
                int nUniqKeys, node_id_t nodeStart, int nCols,
                const TrainParam param) {
  int nBlks = dh::div_round_up(nUniqKeys, BLKDIM);
  split2nodeKernel<<<nBlks,BLKDIM>>>(nodes, nodeSplits, gradScans, gradSums,
                                     vals, colIds, colOffsets, nodeAssigns,
                                     nUniqKeys, nodeStart, nCols,
                                     param);
}

}  // namespace exact
}  // namespace tree
}  // namespace xgboost
