/*!
 * Copyright 2017 XGBoost contributors
 */
#include <xgboost/tree_updater.h>
#include <utility>
#include <vector>
#include "../../../src/common/sync.h"
#include "../../../src/tree/param.h"
#include "exact/fused_scan_reduce_by_key.cuh"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_gpu);

/**
 * @struct ExactSplitCandidate node.cuh
 * @brief Abstraction of a possible split in the decision tree
 */
struct ExactSplitCandidate {
  /** the optimal gain score for this node */
  float score;
  /** index where to split in the DMatrix */
  int index;

  HOST_DEV_INLINE ExactSplitCandidate() : score(-FLT_MAX), index(INT_MAX) {}

  /**
   * @brief Whether the split info is valid to be used to create a new child
   * @param minSplitLoss minimum score above which decision to split is made
   * @return true if splittable, else false
   */
  HOST_DEV_INLINE bool isSplittable(float minSplitLoss) const {
    return ((score >= minSplitLoss) && (index != INT_MAX));
  }
};

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

HOST_DEV_INLINE ExactSplitCandidate maxSplit(ExactSplitCandidate a,
                                             ExactSplitCandidate b) {
  ExactSplitCandidate out;
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

DEV_INLINE void atomicArgMax(ExactSplitCandidate* address,
                             ExactSplitCandidate val) {
  unsigned long long* intAddress = (unsigned long long*)address;
  unsigned long long old = *intAddress;
  unsigned long long assumed;
  do {
    assumed = old;
    ExactSplitCandidate res = maxSplit(val, *(ExactSplitCandidate*)&assumed);
    old = atomicCAS(intAddress, assumed, *(uint64_t*)&res);
  } while (assumed != old);
}

DEV_INLINE void argMaxWithAtomics(
    int id, ExactSplitCandidate* nodeSplits, const bst_gpair* gradScans,
    const bst_gpair* gradSums, const float* vals, const int* colIds,
    const node_id_t* nodeAssigns, const DeviceDenseNode* nodes, int nUniqKeys,
    node_id_t nodeStart, int len, const GPUTrainingParam& param) {
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
      ExactSplitCandidate s;
      bst_gpair missing = parentSum - colSum;
      s.score = loss_chg_missing(gradScans[id], missing, parentSum, parentGain,
                                 param, tmp);
      s.index = id;
      atomicArgMax(nodeSplits + uid, s);
    }  // end if nodeId != UNUSED_NODE
  }    // end if id == 0 ...
}

__global__ void atomicArgMaxByKeyGmem(
    ExactSplitCandidate* nodeSplits, const bst_gpair* gradScans,
    const bst_gpair* gradSums, const float* vals, const int* colIds,
    const node_id_t* nodeAssigns, const DeviceDenseNode* nodes, int nUniqKeys,
    node_id_t nodeStart, int len, const TrainParam param) {
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  const int stride = blockDim.x * gridDim.x;
  for (; id < len; id += stride) {
    argMaxWithAtomics(id, nodeSplits, gradScans, gradSums, vals, colIds,
                      nodeAssigns, nodes, nUniqKeys, nodeStart, len,
                      GPUTrainingParam(param));
  }
}

__global__ void atomicArgMaxByKeySmem(
    ExactSplitCandidate* nodeSplits, const bst_gpair* gradScans,
    const bst_gpair* gradSums, const float* vals, const int* colIds,
    const node_id_t* nodeAssigns, const DeviceDenseNode* nodes, int nUniqKeys,
    node_id_t nodeStart, int len, const TrainParam param) {
  extern __shared__ char sArr[];
  ExactSplitCandidate* sNodeSplits =
      reinterpret_cast<ExactSplitCandidate*>(sArr);
  int tid = threadIdx.x;
  ExactSplitCandidate defVal;
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
    ExactSplitCandidate s = sNodeSplits[i];
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
template <int BLKDIM = 256, int ITEMS_PER_THREAD = 4>
void argMaxByKey(ExactSplitCandidate* nodeSplits, const bst_gpair* gradScans,
                 const bst_gpair* gradSums, const float* vals,
                 const int* colIds, const node_id_t* nodeAssigns,
                 const DeviceDenseNode* nodes, int nUniqKeys,
                 node_id_t nodeStart, int len, const TrainParam param,
                 ArgMaxByKeyAlgo algo) {
  fillConst<ExactSplitCandidate, BLKDIM, ITEMS_PER_THREAD>(
      dh::get_device_idx(param.gpu_id), nodeSplits, nUniqKeys,
      ExactSplitCandidate());
  int nBlks = dh::div_round_up(len, ITEMS_PER_THREAD * BLKDIM);
  switch (algo) {
    case ABK_GMEM:
      atomicArgMaxByKeyGmem<<<nBlks, BLKDIM>>>(
          nodeSplits, gradScans, gradSums, vals, colIds, nodeAssigns, nodes,
          nUniqKeys, nodeStart, len, param);
      break;
    case ABK_SMEM:
      atomicArgMaxByKeySmem<<<nBlks, BLKDIM,
                              sizeof(ExactSplitCandidate) * nUniqKeys>>>(
          nodeSplits, gradScans, gradSums, vals, colIds, nodeAssigns, nodes,
          nUniqKeys, nodeStart, len, param);
      break;
    default:
      throw std::runtime_error("argMaxByKey: Bad algo passed!");
  }
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
                                   const DeviceDenseNode* nodes, int nRows) {
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  if (id >= nRows) {
    return;
  }
  // if this element belongs to none of the currently active node-id's
  node_id_t nId = nodeIdsPerInst[id];
  if (nId == UNUSED_NODE) {
    return;
  }
  const DeviceDenseNode n = nodes[nId];
  node_id_t result;
  if (n.IsLeaf() || n.IsUnused()) {
    result = UNUSED_NODE;
  } else if (n.dir == LeftDir) {
    result = (2 * n.idx) + 1;
  } else {
    result = (2 * n.idx) + 2;
  }
  nodeIdsPerInst[id] = result;
}

__global__ void assignNodeIds(node_id_t* nodeIdsPerInst, int* nodeLocations,
                              const node_id_t* nodeIds, const int* instId,
                              const DeviceDenseNode* nodes,
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
      const DeviceDenseNode n = nodes[nId];
      int colId = n.fidx;
      // printf("nid=%d colId=%d id=%d\n", nId, colId, id);
      int start = colOffsets[colId];
      int end = colOffsets[colId + 1];
      ///@todo: too much wasteful threads!!
      if ((id >= start) && (id < end) && !(n.IsLeaf() || n.IsUnused())) {
        node_id_t result = (2 * n.idx) + 1 + (vals[id] >= n.fvalue);
        nodeIdsPerInst[instId[id]] = result;
      }
    }
  }
}

__global__ void markLeavesKernel(DeviceDenseNode* nodes, int len) {
  int id = (blockIdx.x * blockDim.x) + threadIdx.x;
  if ((id < len) && !nodes[id].IsUnused()) {
    int lid = (id << 1) + 1;
    int rid = (id << 1) + 2;
    if ((lid >= len) || (rid >= len)) {
      nodes[id].root_gain = -FLT_MAX;  // bottom-most nodes
    } else if (nodes[lid].IsUnused() && nodes[rid].IsUnused()) {
      nodes[id].root_gain = -FLT_MAX;  // unused child nodes
    }
  }
}

class GPUMaker : public TreeUpdater {
 protected:
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
  dh::dvec<DeviceDenseNode> nodes;
  dh::dvec<node_id_t> nodeAssignsPerInst;
  dh::dvec<bst_gpair> gradSums;
  dh::dvec<bst_gpair> gradScans;
  dh::dvec<ExactSplitCandidate> nodeSplits;
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

 public:
  GPUMaker() : allocated(false) {}
  ~GPUMaker() {}

  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) override {
    param.InitAllowUnknown(args);
    maxNodes = (1 << (param.max_depth + 1)) - 1;
    maxLeaves = 1 << param.max_depth;
  }

  void Update(const std::vector<bst_gpair>& gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    GradStats::CheckInfo(dmat->info());
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();

    try {
      // build tree
      for (size_t i = 0; i < trees.size(); ++i) {
        UpdateTree(gpair, dmat, trees[i]);
      }
    } catch (const std::exception& e) {
      LOG(FATAL) << "GPU plugin exception: " << e.what() << std::endl;
    }
    param.learning_rate = lr;
  }
  /// @note: Update should be only after Init!!
  void UpdateTree(const std::vector<bst_gpair>& gpair, DMatrix* hMat,
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
    dense2sparse_tree(hTree, nodes, param);
  }
    // split2node(nodes.data(), nodeSplits.data(), gradScans.data(),
    //           gradSums.data(), vals.current(), colIds.data(),
    //           colOffsets.data(), nodeAssigns.current(), nNodes, nodeStart,
    //           nCols, param);

//__global__ void split2nodeKernel(
//    DeviceDenseNode* nodes, const ExactSplitCandidate* nodeSplits,
//    const bst_gpair* gradScans, const bst_gpair* gradSums, const float* vals,
//    const int* colIds, const int* colOffsets, const node_id_t* nodeAssigns,
//    int nUniqKeys, node_id_t nodeStart, int nCols, const TrainParam param) {

  void split2node(int nNodes, node_id_t nodeStart) {
    auto d_nodes = nodes.data();
    auto d_gradScans = gradScans.data();
    auto d_gradSums = gradSums.data();
    auto d_nodeAssigns = nodeAssigns.current();
    auto d_colIds = colIds.data();
    auto d_vals = vals.current();
    auto d_nodeSplits = nodeSplits.data();
    int nUniqKeys = nNodes;
    float min_split_loss = param.min_split_loss;
    auto gpu_param = GPUTrainingParam(param);

    dh::launch_n(param.gpu_id, nNodes, [=] __device__(int uid) {
      int absNodeId = uid + nodeStart;
      ExactSplitCandidate s = d_nodeSplits[uid];
      if (s.isSplittable(min_split_loss)) {
        int idx = s.index;
        int nodeInstId =
            abs2uniqKey(idx, d_nodeAssigns, d_colIds, nodeStart, nUniqKeys);
        bool missingLeft = true;
        const DeviceDenseNode& n = d_nodes[absNodeId];
        bst_gpair gradScan = d_gradScans[idx];
        bst_gpair gradSum = d_gradSums[nodeInstId];
        float thresh = d_vals[idx];
        int colId = d_colIds[idx];
        // get the default direction for the current node
        bst_gpair missing = n.sum_gradients - gradSum;
        loss_chg_missing(gradScan, missing, n.sum_gradients, n.root_gain, gpu_param,
                         missingLeft);
        // get the score/weight/id/gradSum for left and right child nodes
        bst_gpair lGradSum = missingLeft ? gradScan + missing : gradScan;
        bst_gpair rGradSum = n.sum_gradients - lGradSum;

        // Create children
        d_nodes[left_child_nidx(absNodeId)] =
            DeviceDenseNode(lGradSum, left_child_nidx(absNodeId), gpu_param);
        d_nodes[right_child_nidx(absNodeId)] =
            DeviceDenseNode(rGradSum, right_child_nidx(absNodeId), gpu_param);
        // Set split for parent
        d_nodes[absNodeId].SetSplit(thresh, colId,
                                  missingLeft ? LeftDir : RightDir);
      } else {
        // cannot be split further, so this node is a leaf!
        d_nodes[absNodeId].root_gain = -FLT_MAX;
      }

    });
  }

  void findSplit(int level, node_id_t nodeStart, int nNodes) {
    reduceScanByKey(gradSums.data(), gradScans.data(), gradsInst.data(),
                    instIds.current(), nodeAssigns.current(), nVals, nNodes,
                    nCols, tmpScanGradBuff.data(), tmpScanKeyBuff.data(),
                    colIds.data(), nodeStart);
    argMaxByKey(nodeSplits.data(), gradScans.data(), gradSums.data(),
                vals.current(), colIds.data(), nodeAssigns.current(),
                nodes.data(), nNodes, nodeStart, nVals, param,
                level <= MAX_ABK_LEVELS ? ABK_SMEM : ABK_GMEM);
    // split2node(nodes.data(), nodeSplits.data(), gradScans.data(),
    //           gradSums.data(), vals.current(), colIds.data(),
    //           colOffsets.data(), nodeAssigns.current(), nNodes, nodeStart,
    //           nCols, param);
    split2node(nNodes, nodeStart);
  }

  void allocateAllData(int offsetSize) {
    int tmpBuffSize = scanTempBufferSize(nVals);
    ba.allocate(dh::get_device_idx(param.gpu_id), param.silent, &vals, nVals,
                &vals_cached, nVals, &instIds, nVals, &instIds_cached, nVals,
                &colOffsets, offsetSize, &gradsInst, nRows, &nodeAssigns, nVals,
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
      nodes.fill(DeviceDenseNode());
      nodeAssigns.current_dvec().fill(0);
      nodeAssignsPerInst.fill(0);
      // for root node, just update the gradient/score/weight/id info
      // before splitting it! Currently all data is on GPU, hence this
      // stupid little kernel
      auto d_nodes = nodes.data();
      auto d_sums = gradSums.data();
      auto gpu_params = GPUTrainingParam(param);
      dh::launch_n(param.gpu_id, 1, [=] __device__(int idx) {
        d_nodes[0] = DeviceDenseNode(d_sums[0], 0, gpu_params);
      });
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
};

XGBOOST_REGISTER_TREE_UPDATER(GPUMaker, "grow_gpu")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUMaker(); });

}  // namespace tree
}  // namespace xgboost
