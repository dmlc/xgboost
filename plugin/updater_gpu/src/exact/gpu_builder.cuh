/*
 * Copyright (c) 2017, NVIDIA CORPORATION, Xgboost contributors.  All rights
 * reserved.
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

#include <string>
#include <vector>
#include "../../../../src/tree/param.h"
#include "../common.cuh"
#include "argmax_by_key.cuh"
#include "fused_scan_reduce_by_key.cuh"
#include "node.cuh"
#include "split2node.cuh"
#include "xgboost/tree_updater.h"

namespace xgboost {
namespace tree {
namespace exact {


__global__ void initRootNode(Node* nodes, const bst_gpair* sums,
                             const TrainParam param) {
  // gradients already evaluated inside transferGrads
  Node n;
  n.gradSum = sums[0];
  n.score = CalcGain(param, n.gradSum.grad, n.gradSum.hess);
  n.weight = CalcWeight(param, n.gradSum.grad, n.gradSum.hess);
  n.id = 0;
  nodes[0] = n;
}


__global__ void assignColIds(int* colIds, const int* colOffsets) {
  int myId = blockIdx.x;
  int start = colOffsets[myId];
  int end = colOffsets[myId + 1];
  for (int id = start + threadIdx.x; id < end; id += blockDim.x) {
    colIds[id] = myId;
  }
}


__global__ void fillDefaultNodeIds(node_id_t* nodeIdsPerInst,
                                   const Node* nodes, int nRows) {
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  if (id >= nRows) {
    return;
  }
  // if this element belongs to none of the currently active node-id's
  node_id_t nId = nodeIdsPerInst[id];
  if (nId == UNUSED_NODE) {
    return;
  }
  const Node n = nodes[nId];
  node_id_t result;
  if (n.isLeaf() || n.isUnused()) {
    result = UNUSED_NODE;
  } else if (n.isDefaultLeft()) {
    result = (2 * n.id) + 1;
  } else {
    result = (2 * n.id) + 2;
  }
  nodeIdsPerInst[id] = result;
}


__global__ void assignNodeIds(node_id_t* nodeIdsPerInst, int* nodeLocations,
                              const node_id_t* nodeIds, const int* instId,
                              const Node* nodes,
                              const int* colOffsets, const float* vals,
                              int nVals, int nCols) {
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  const int stride = blockDim.x * gridDim.x;
  for (; id < nVals; id += stride) {
    // fusing generation of indices for node locations
    nodeLocations[id] = id;
    // using nodeIds here since the previous kernel would have updated
    // the nodeIdsPerInst with all default assignments
    int nId = nodeIds[id];
    // if this element belongs to none of the currently active node-id's
    if (nId != UNUSED_NODE) {
      const Node n = nodes[nId];
      int colId = n.colIdx;
      // printf("nid=%d colId=%d id=%d\n", nId, colId, id);
      int start = colOffsets[colId];
      int end = colOffsets[colId + 1];
      ///@todo: too much wasteful threads!!
      if ((id >= start) && (id < end) && !(n.isLeaf() || n.isUnused())) {
        node_id_t result = (2 * n.id) + 1 + (vals[id] >= n.threshold);
        nodeIdsPerInst[instId[id]] = result;
      }
    }
  }
}


__global__ void markLeavesKernel(Node* nodes, int len) {
  int id = (blockIdx.x * blockDim.x) + threadIdx.x;
  if ((id < len) && !nodes[id].isUnused()) {
    int lid = (id << 1) + 1;
    int rid = (id << 1) + 2;
    if ((lid >= len) || (rid >= len)) {
      nodes[id].score = -FLT_MAX;  // bottom-most nodes
    } else if (nodes[lid].isUnused() && nodes[rid].isUnused()) {
      nodes[id].score = -FLT_MAX;  // unused child nodes
    }
  }
}

// unit test forward declaration for friend function access

void testSmallData();

void testLargeData();

void testAllocate();

void testMarkLeaves();

void testDense2Sparse();

class GPUBuilder;

std::shared_ptr<xgboost::DMatrix> setupGPUBuilder(
    const std::string& file,
    xgboost::tree::exact::GPUBuilder& builder);


class GPUBuilder {
 public:
  GPUBuilder() : allocated(false) {}

  ~GPUBuilder() {}

  void Init(const TrainParam& p) {
    param = p;
    maxNodes = (1 << (param.max_depth + 1)) - 1;
    maxLeaves = 1 << param.max_depth;
  }

  void UpdateParam(const TrainParam& param) { this->param = param; }

  /// @note: Update should be only after Init!!
  void Update(const std::vector<bst_gpair>& gpair, DMatrix* hMat,
              RegTree* hTree) {
    if (!allocated) {
      setupOneTimeData(*hMat);
    }
    for (int i = 0; i < param.max_depth; ++i) {
      if (i == 0) {
        // make sure to start on a fresh tree with sorted values!
        vals.current_dvec() = vals_cached;
        instIds.current_dvec() = instIds_cached;
        transferGrads(gpair);
      }
      int nNodes = 1 << i;
      node_id_t nodeStart = nNodes - 1;
      initNodeData(i, nodeStart, nNodes);
      findSplit(i, nodeStart, nNodes);
    }
    // mark all the used nodes with unused children as leaf nodes
    markLeaves();
    dense2sparse(hTree);
  }

 private:
  friend void testSmallData();
  friend void testLargeData();
  friend void testAllocate();
  friend void testMarkLeaves();
  friend void testDense2Sparse();
  friend std::shared_ptr<xgboost::DMatrix> setupGPUBuilder(
      const std::string& file, GPUBuilder& builder);

  TrainParam param;
  /** whether we have initialized memory already (so as not to repeat!) */
  bool allocated;
  /** feature values stored in column-major compressed format */
  dh::dvec2<float> vals;
  dh::dvec<float> vals_cached;
  /** corresponding instance id's of these featutre values */
  dh::dvec2<int> instIds;
  dh::dvec<int> instIds_cached;
  /** column offsets for these feature values */
  dh::dvec<int> colOffsets;
  dh::dvec<bst_gpair> gradsInst;
  dh::dvec2<node_id_t> nodeAssigns;
  dh::dvec2<int> nodeLocations;
  dh::dvec<Node> nodes;
  dh::dvec<node_id_t>  nodeAssignsPerInst;
  dh::dvec<bst_gpair> gradSums;
  dh::dvec<bst_gpair> gradScans;
  dh::dvec<Split> nodeSplits;
  int nVals;
  int nRows;
  int nCols;
  int maxNodes;
  int maxLeaves;
  dh::CubMemory tmp_mem;
  dh::dvec<bst_gpair> tmpScanGradBuff;
  dh::dvec<int> tmpScanKeyBuff;
  dh::dvec<int> colIds;
  dh::bulk_allocator<dh::memory_type::DEVICE> ba;

  void findSplit(int level, node_id_t nodeStart, int nNodes) {
    reduceScanByKey(gradSums.data(), gradScans.data(), gradsInst.data(),
                    instIds.current(), nodeAssigns.current(), nVals, nNodes,
                    nCols, tmpScanGradBuff.data(), tmpScanKeyBuff.data(),
                    colIds.data(), nodeStart);
    argMaxByKey(nodeSplits.data(), gradScans.data(), gradSums.data(),
                vals.current(), colIds.data(), nodeAssigns.current(),
                nodes.data(), nNodes, nodeStart, nVals, param,
                level <= MAX_ABK_LEVELS ? ABK_SMEM : ABK_GMEM);
    split2node(nodes.data(), nodeSplits.data(), gradScans.data(),
               gradSums.data(), vals.current(), colIds.data(),
               colOffsets.data(), nodeAssigns.current(), nNodes, nodeStart,
               nCols, param);
  }

  void allocateAllData(int offsetSize) {
    int tmpBuffSize = scanTempBufferSize(nVals);
    ba.allocate(dh::get_device_idx(param.gpu_id), param.silent, &vals, nVals, &vals_cached,
                nVals, &instIds, nVals, &instIds_cached, nVals, &colOffsets,
                offsetSize, &gradsInst, nRows, &nodeAssigns, nVals,
                &nodeLocations, nVals, &nodes, maxNodes, &nodeAssignsPerInst,
                nRows, &gradSums, maxLeaves * nCols, &gradScans, nVals,
                &nodeSplits, maxLeaves, &tmpScanGradBuff, tmpBuffSize,
                &tmpScanKeyBuff, tmpBuffSize, &colIds, nVals);
  }

  void setupOneTimeData(DMatrix& hMat) {
    size_t free_memory = dh::available_memory(dh::get_device_idx(param.gpu_id));
    if (!hMat.SingleColBlock()) {
      throw std::runtime_error("exact::GPUBuilder - must have 1 column block");
    }
    std::vector<float> fval;
    std::vector<int> fId, offset;
    convertToCsc(hMat, fval, fId, offset);
    allocateAllData((int)offset.size());
    transferAndSortData(fval, fId, offset);
    allocated = true;
  }

  void convertToCsc(DMatrix& hMat, std::vector<float>& fval,
                    std::vector<int>& fId, std::vector<int>& offset) {
    MetaInfo info = hMat.info();
    nRows = info.num_row;
    nCols = info.num_col;
    offset.reserve(nCols + 1);
    offset.push_back(0);
    fval.reserve(nCols * nRows);
    fId.reserve(nCols * nRows);
    // in case you end up with a DMatrix having no column access
    // then make sure to enable that before copying the data!
    if (!hMat.HaveColAccess()) {
      const std::vector<bool> enable(nCols, true);
      hMat.InitColAccess(enable, 1, nRows);
    }
    dmlc::DataIter<ColBatch>* iter = hMat.ColIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const ColBatch& batch = iter->Value();
      for (int i = 0; i < batch.size; i++) {
        const ColBatch::Inst& col = batch[i];
        for (const ColBatch::Entry* it = col.data; it != col.data + col.length;
             it++) {
          int inst_id = static_cast<int>(it->index);
          fval.push_back(it->fvalue);
          fId.push_back(inst_id);
        }
        offset.push_back(fval.size());
      }
    }
    nVals = fval.size();
  }

  void transferAndSortData(const std::vector<float>& fval,
                           const std::vector<int>& fId,
                           const std::vector<int>& offset) {
    vals.current_dvec() = fval;
    instIds.current_dvec() = fId;
    colOffsets = offset;
    segmentedSort<float, int>(&tmp_mem, &vals, &instIds, nVals, nCols,
                              colOffsets);
    vals_cached = vals.current_dvec();
    instIds_cached = instIds.current_dvec();
    assignColIds<<<nCols, 512>>>(colIds.data(), colOffsets.data());
  }

  void transferGrads(const std::vector<bst_gpair>& gpair) {
    // HACK
    dh::safe_cuda(cudaMemcpy(gradsInst.data(), &(gpair[0]),
                             sizeof(bst_gpair) * nRows,
                             cudaMemcpyHostToDevice));
    // evaluate the full-grad reduction for the root node
    sumReduction<bst_gpair>(tmp_mem, gradsInst, gradSums, nRows);
  }

  void initNodeData(int level, node_id_t nodeStart, int nNodes) {
    // all instances belong to root node at the beginning!
    if (level == 0) {
      nodes.fill(Node());
      nodeAssigns.current_dvec().fill(0);
      nodeAssignsPerInst.fill(0);
      // for root node, just update the gradient/score/weight/id info
      // before splitting it! Currently all data is on GPU, hence this
      // stupid little kernel
      initRootNode<<<1, 1>>>(nodes.data(), gradSums.data(), param);
    } else {
      const int BlkDim = 256;
      const int ItemsPerThread = 4;
      // assign default node ids first
      int nBlks = dh::div_round_up(nRows, BlkDim);
      fillDefaultNodeIds<<<nBlks, BlkDim>>>(nodeAssignsPerInst.data(),
                                            nodes.data(), nRows);
      // evaluate the correct child indices of non-missing values next
      nBlks = dh::div_round_up(nVals, BlkDim * ItemsPerThread);
      assignNodeIds<<<nBlks, BlkDim>>>(
          nodeAssignsPerInst.data(), nodeLocations.current(),
          nodeAssigns.current(), instIds.current(), nodes.data(),
          colOffsets.data(), vals.current(), nVals, nCols);
      // gather the node assignments across all other columns too
      gather(dh::get_device_idx(param.gpu_id), nodeAssigns.current(),
                        nodeAssignsPerInst.data(), instIds.current(), nVals);
      sortKeys(level);
    }
  }

  void sortKeys(int level) {
    // segmented-sort the arrays based on node-id's
    // but we don't need more than level+1 bits for sorting!
    segmentedSort(&tmp_mem, &nodeAssigns, &nodeLocations, nVals, nCols,
                  colOffsets, 0, level + 1);
    gather<float, int>(dh::get_device_idx(param.gpu_id), vals.other(),
                       vals.current(), instIds.other(), instIds.current(),
                       nodeLocations.current(), nVals);
    vals.buff().selector ^= 1;
    instIds.buff().selector ^= 1;
  }

  void markLeaves() {
    const int BlkDim = 128;
    int nBlks = dh::div_round_up(maxNodes, BlkDim);
    markLeavesKernel<<<nBlks, BlkDim>>>(nodes.data(), maxNodes);
  }

  void dense2sparse(RegTree* p_tree) {
    RegTree& tree = *p_tree;
    std::vector<Node> hNodes = nodes.as_vector();
    int nodeId = 0;
    for (int i = 0; i < maxNodes; ++i) {
      const Node& n = hNodes[i];
      if ((i != 0) && hNodes[i].isLeaf()) {
        tree[nodeId].set_leaf(n.weight * param.learning_rate);
        tree.stat(nodeId).sum_hess = n.gradSum.hess;
        ++nodeId;
      } else if (!hNodes[i].isUnused()) {
        tree.AddChilds(nodeId);
        tree[nodeId].set_split(n.colIdx, n.threshold, n.dir == LeftDir);
        tree.stat(nodeId).loss_chg = n.score;
        tree.stat(nodeId).sum_hess = n.gradSum.hess;
        tree.stat(nodeId).base_weight = n.weight;
        tree[tree[nodeId].cleft()].set_leaf(0);
        tree[tree[nodeId].cright()].set_leaf(0);
        ++nodeId;
      }
    }
  }
};

}  // namespace exact
}  // namespace tree
}  // namespace xgboost
